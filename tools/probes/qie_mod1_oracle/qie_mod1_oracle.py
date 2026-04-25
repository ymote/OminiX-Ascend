#!/usr/bin/env python3
"""
Q2.4.5.5.19 — Pure-F32 numpy oracle for QIE-Edit DiT block-0 mod1 dispatch.

Re-orienting after §5.5.17 substep cossim sweep:
  - 04 LN1 cos=1.000 GREEN
  - 05 img_mod1 cos=0.9859 YELLOW (drift first enters here)
  - 07 txt_mod1 cos=0.9800 YELLOW

This oracle replicates the mod1 dispatch in
image_diffusion_engine.cpp:3240-3397:
  1. silu(t_emb)                                          (F16 in/out)
  2. img_mod_params = silu(t_emb) @ img_mod_w^T + img_mod_b   ([1, 6H])
     txt_mod_params = silu(t_emb) @ txt_mod_w^T + txt_mod_b   ([1, 6H])
  3. Split into 6 chunks of size H (legacy native ordering, NOT HF spec):
        [scale1, shift1, gate1, scale2, shift2, gate2]
     (see image_diffusion_engine.cpp:3326-3342 + §5.5.7)
  4. img_LN1_modulated = LN1_img * (1 + scale1_img) + shift1_img
     txt_LN1_modulated = LN1_txt * (1 + scale1_txt) + shift1_txt

Compared end-to-end against native 05_img_mod1.f32 / 07_txt_mod1.f32.

Decision matrix:
  GREEN-A : oracle vs native cos < 0.99 → mod1 bug confirmed in dispatch.
            Identify which sub-component (silu / matmul / split / modulate).
  GREEN-B : oracle vs native cos >= 0.99 → mod1 fine; drift is in t_emb itself,
            i.e. upstream of substep 04 (timestep schedule, embed table,
            time_text_embed.linear chain, etc.). Recommend §5.5.20 t_emb
            oracle.
"""
import argparse
import os
import sys

import numpy as np


DUMP = os.environ.get("QIE_DUMPS", "/tmp/qie_dumps_5516")
GGUF = os.environ.get(
    "QIE_GGUF",
    "/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf",
)

H        = 3072
TXT_SEQ  = 32
IMG_SEQ  = 64


# --- gguf loader helpers (same pattern as qie_q2454l_repack_probe) ---------

def find_gguf_py(repo_root: str) -> None:
    try:
        import gguf  # noqa: F401
        return
    except ImportError:
        pass
    cand = os.path.join(repo_root, "gguf-py")
    if os.path.isdir(cand):
        sys.path.insert(0, cand)


def load_dequant_weight(gguf_path: str, name: str) -> np.ndarray:
    """Dequantize a GGUF tensor to [N, K] F32. (GGUF stores [K, N]; the
    dequantize routine returns flat [n_blocks, elems] which we reshape to
    [N, K] to match the engine's row-major weight layout where rows are
    output channels.)"""
    from gguf import GGUFReader, GGMLQuantizationType, quants

    reader = GGUFReader(gguf_path)
    tensor = None
    for t in reader.tensors:
        if t.name == name:
            tensor = t
            break
    if tensor is None:
        raise RuntimeError(f"tensor not found: {name}")

    K = int(tensor.shape[0])
    N = int(tensor.shape[1])
    qt = tensor.tensor_type
    raw = np.asarray(tensor.data, dtype=np.uint8).reshape(-1)

    if qt == GGMLQuantizationType.F32:
        return raw.view(np.float32).reshape(N, K).astype(np.float32)
    if qt == GGMLQuantizationType.F16:
        return raw.view(np.float16).reshape(N, K).astype(np.float32)
    if qt == GGMLQuantizationType.BF16:
        u16 = raw.view(np.uint16).astype(np.uint32)
        return ((u16 << 16).view(np.float32)).reshape(N, K).astype(np.float32)

    qcls_map = {q.qtype: q for q in quants.__Quant.__subclasses__()
                if hasattr(q, "qtype")}
    qcls = qcls_map.get(qt)
    if qcls is None:
        raise RuntimeError(f"no Python dequant for type {qt}")

    from gguf.constants import GGML_QUANT_SIZES
    elems_per_blk, bytes_per_blk = GGML_QUANT_SIZES[qt]
    n_total = N * K
    n_blocks = n_total // elems_per_blk
    blocks = raw.reshape(n_blocks, bytes_per_blk)
    dq = qcls.dequantize_blocks(blocks)
    return dq.reshape(N, K).astype(np.float32), str(qt).split(".")[-1]


def load_bias(gguf_path: str, name: str):
    from gguf import GGUFReader, GGMLQuantizationType
    reader = GGUFReader(gguf_path)
    for t in reader.tensors:
        if t.name == name:
            data = np.asarray(t.data)
            qt = t.tensor_type
            if qt == GGMLQuantizationType.F32:
                return data.astype(np.float32).reshape(-1).copy(), "F32"
            if qt == GGMLQuantizationType.F16:
                return data.astype(np.float32).reshape(-1), "F16"
            if qt == GGMLQuantizationType.BF16:
                u16 = data.view(np.uint16).astype(np.uint32)
                return ((u16 << 16).view(np.float32)).reshape(-1).astype(np.float32), "BF16"
            raise RuntimeError(f"bias dtype unsupported: {qt}")
    return None, None


def load_weight_with_qt(gguf_path: str, name: str):
    """Same as load_dequant_weight but always returns (W, qt_str)."""
    res = load_dequant_weight(gguf_path, name)
    if isinstance(res, tuple):
        return res
    # F32/F16/BF16 shortcut path returned plain ndarray; tag dtype.
    return res, "FLOAT"


# --- math helpers ----------------------------------------------------------

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def cossim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def f16_round(x):
    return x.astype(np.float16).astype(np.float32)


def stats(label, x):
    print(f"  {label:32s}  mean={x.mean():+.4e}  std={x.std():.4e}  "
          f"min={x.min():+.4e}  max={x.max():+.4e}  absmax={np.abs(x).max():.4e}")


def load_f32(name, expected_n, dump_dir=None):
    if dump_dir is None:
        dump_dir = DUMP
    path = os.path.join(dump_dir, name)
    sz = os.path.getsize(path)
    assert sz == expected_n * 4, f"{name}: size {sz} != {expected_n*4}"
    a = np.fromfile(path, dtype=np.float32, count=expected_n)
    assert a.size == expected_n
    return a


# --- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf",  default=GGUF)
    ap.add_argument("--dumps", default=DUMP)
    ap.add_argument("--block", type=int, default=0)
    args = ap.parse_args()
    dump_dir = args.dumps

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    find_gguf_py(repo_root)

    print(f"GGUF       : {args.gguf}")
    print(f"DUMPS      : {dump_dir}")
    print(f"H={H}  TXT_SEQ={TXT_SEQ}  IMG_SEQ={IMG_SEQ}  block={args.block}")
    print()

    # ------------------------------------------------------------------ 1
    # Load native dumps.
    # ------------------------------------------------------------------
    t_emb       = load_f32("00_t_emb.f32",     H, dump_dir)
    img_LN1     = load_f32("04_img_LN1.f32",   IMG_SEQ * H, dump_dir).reshape(IMG_SEQ, H)
    img_mod1_n  = load_f32("05_img_mod1.f32",  IMG_SEQ * H, dump_dir).reshape(IMG_SEQ, H)
    txt_LN1     = load_f32("06_txt_LN1.f32",   TXT_SEQ * H, dump_dir).reshape(TXT_SEQ, H)
    txt_mod1_n  = load_f32("07_txt_mod1.f32",  TXT_SEQ * H, dump_dir).reshape(TXT_SEQ, H)

    print("--- native dump stats ---")
    stats("00_t_emb",     t_emb)
    stats("04_img_LN1",   img_LN1)
    stats("05_img_mod1",  img_mod1_n)
    stats("06_txt_LN1",   txt_LN1)
    stats("07_txt_mod1",  txt_mod1_n)
    print()

    # ------------------------------------------------------------------ 2
    # Load img_mod / txt_mod weights.
    # ------------------------------------------------------------------
    base = f"transformer_blocks.{args.block}"
    img_w_name = f"{base}.img_mod.1.weight"
    img_b_name = f"{base}.img_mod.1.bias"
    txt_w_name = f"{base}.txt_mod.1.weight"
    txt_b_name = f"{base}.txt_mod.1.bias"

    print("--- loading mod1 weights ---")
    img_W, img_qt = load_weight_with_qt(args.gguf, img_w_name)
    img_b, img_b_qt = load_bias(args.gguf, img_b_name)
    txt_W, txt_qt = load_weight_with_qt(args.gguf, txt_w_name)
    txt_b, txt_b_qt = load_bias(args.gguf, txt_b_name)

    print(f"  img_mod.1.weight  shape={img_W.shape}  qt={img_qt}")
    print(f"  img_mod.1.bias    shape={img_b.shape}  qt={img_b_qt}")
    print(f"  txt_mod.1.weight  shape={txt_W.shape}  qt={txt_qt}")
    print(f"  txt_mod.1.bias    shape={txt_b.shape}  qt={txt_b_qt}")
    if img_W.shape != (6 * H, H):
        print(f"  WARN: img_W shape != (6H, H)={(6*H, H)} — possibly transposed")
    stats("img_W",  img_W)
    stats("img_b",  img_b)
    stats("txt_W",  txt_W)
    stats("txt_b",  txt_b)
    print()

    # ------------------------------------------------------------------ 3
    # Run the mod1 dispatch in pure F32 (mimicking F16 storage at the
    # critical handoff points).
    # ------------------------------------------------------------------
    # silu(t_emb): engine performs this in F16 in-place. We mimic by
    # rounding both input and output to F16.
    t_emb_f16   = f16_round(t_emb)
    silu_t_full = silu(t_emb_f16)
    silu_t_f16  = f16_round(silu_t_full)

    # Engine matmul path: dispatch_matmul_(silu_t, W_q4, scale, b, B=1, H, 6H)
    # Real engine: weight is dequantized to F16 on device, F16 input, F16 bias,
    # F16 output. We mirror that with f16-cast weights/bias.
    img_W_f16 = f16_round(img_W)
    img_b_f16 = f16_round(img_b)
    txt_W_f16 = f16_round(txt_W)
    txt_b_f16 = f16_round(txt_b)

    img_mod_params = silu_t_f16.astype(np.float32) @ img_W_f16.T + img_b_f16
    img_mod_params = f16_round(img_mod_params)            # [6H]
    txt_mod_params = silu_t_f16.astype(np.float32) @ txt_W_f16.T + txt_b_f16
    txt_mod_params = f16_round(txt_mod_params)

    print("--- silu(t_emb) and matmul-out stats ---")
    stats("silu(t_emb)",     silu_t_f16)
    stats("img_mod_params",  img_mod_params)
    stats("txt_mod_params",  txt_mod_params)
    print()

    # Split into 6 chunks (legacy native ordering — see
    # image_diffusion_engine.cpp:3326-3342):
    img_chunks = img_mod_params.reshape(6, H)
    txt_chunks = txt_mod_params.reshape(6, H)
    img_scale1 = img_chunks[0]; img_shift1 = img_chunks[1]; img_gate1 = img_chunks[2]
    img_scale2 = img_chunks[3]; img_shift2 = img_chunks[4]; img_gate2 = img_chunks[5]
    txt_scale1 = txt_chunks[0]; txt_shift1 = txt_chunks[1]; txt_gate1 = txt_chunks[2]
    txt_scale2 = txt_chunks[3]; txt_shift2 = txt_chunks[4]; txt_gate2 = txt_chunks[5]

    print("--- mod chunk stats (legacy ordering: scale1, shift1, gate1, scale2, shift2, gate2) ---")
    for i, lbl in enumerate(["scale1", "shift1", "gate1", "scale2", "shift2", "gate2"]):
        stats(f"img_chunk[{i}]={lbl}", img_chunks[i])
    for i, lbl in enumerate(["scale1", "shift1", "gate1", "scale2", "shift2", "gate2"]):
        stats(f"txt_chunk[{i}]={lbl}", txt_chunks[i])
    print()

    # modulate_(x, scale, shift): x * (1 + scale) + shift  (engine spec)
    def modulate(x, scale, shift):
        # x is [seq, H] F32; engine does this in F16 in/out.
        x_f16 = f16_round(x)
        out = x_f16 * (1.0 + scale[None, :]) + shift[None, :]
        return f16_round(out)

    img_mod1_oracle = modulate(img_LN1, img_scale1, img_shift1)
    txt_mod1_oracle = modulate(txt_LN1, txt_scale1, txt_shift1)

    print("--- oracle vs native (full mod1 chain) ---")
    cos_img = cossim(img_mod1_oracle, img_mod1_n)
    cos_txt = cossim(txt_mod1_oracle, txt_mod1_n)
    print(f"  cos(05_img_mod1 oracle vs native) = {cos_img:.6f}")
    print(f"  cos(07_txt_mod1 oracle vs native) = {cos_txt:.6f}")

    # Also: compare the modulate step in isolation, using "native LN1 + oracle
    # chunk" (already shown above) vs "native LN1 + a perfect chunk" (we do
    # not have a chunk dump, but can infer chunk magnitudes from the cossim
    # gap below).
    diff_img = (img_mod1_oracle - img_mod1_n)
    diff_txt = (txt_mod1_oracle - txt_mod1_n)
    print(f"  img max_abs_diff = {np.abs(diff_img).max():.4e}  mean_abs = {np.abs(diff_img).mean():.4e}")
    print(f"  txt max_abs_diff = {np.abs(diff_txt).max():.4e}  mean_abs = {np.abs(diff_txt).mean():.4e}")
    print()

    # ------------------------------------------------------------------ 4
    # Sub-component diagnostics.
    # ------------------------------------------------------------------
    # (a) LN1 cossim self-check (should be 1.0 since we use the dump itself).
    print("--- diagnostic 1: LN1 sanity (oracle modulate using DUMPED LN1) ---")
    # if LN1 is upstream-clean and modulate is exact, drift is in chunks.
    print(f"  LN1 cossim is trivially 1.0 (we use the dump as input).")
    print()

    # (b) Test alternate configs.
    print("--- diagnostic 2: alternate matmul orientations / silu placement ---")
    alts_img = []
    # NoSilu: matmul of raw t_emb (skip silu)
    alt_params = f16_round(t_emb_f16.astype(np.float32) @ img_W_f16.T + img_b_f16)
    alt_chunks = alt_params.reshape(6, H)
    alt_out = modulate(img_LN1, alt_chunks[0], alt_chunks[1])
    alts_img.append(("no-silu",       cossim(alt_out, img_mod1_n)))

    # silu via x*sigmoid(x) explicit (should be identical to silu impl above)
    sig = 1.0 / (1.0 + np.exp(-t_emb_f16.astype(np.float64)))
    silu_alt = f16_round((t_emb_f16.astype(np.float64) * sig).astype(np.float32))
    alt_params = f16_round(silu_alt.astype(np.float32) @ img_W_f16.T + img_b_f16)
    alt_chunks = alt_params.reshape(6, H)
    alt_out = modulate(img_LN1, alt_chunks[0], alt_chunks[1])
    alts_img.append(("silu=x*sigmoid", cossim(alt_out, img_mod1_n)))

    # W (no transpose): silu_t @ W
    if img_W_f16.shape == (6 * H, H):
        # silu_t is [H], W is [6H, H]; W (no transpose) means silu_t @ W = [H]@[6H, H]
        # — only valid if reshape W to [H, 6H]. Try treating as [H, 6H]:
        W_alt = img_W_f16.reshape(H, 6 * H)  # may be wrong, just to test
        alt_params = f16_round(silu_t_f16.astype(np.float32) @ W_alt + img_b_f16)
        alt_chunks = alt_params.reshape(6, H)
        alt_out = modulate(img_LN1, alt_chunks[0], alt_chunks[1])
        alts_img.append(("W_reshape[H,6H]", cossim(alt_out, img_mod1_n)))

    # silu_t @ W (W is [6H, H], no transpose) — gives [6H]@[6H, H] = [H], wrong shape; skip

    # bias before split vs after split (only matters if bias re-layout).
    # Already includes bias before split — try omitting bias entirely.
    alt_params = f16_round(silu_t_f16.astype(np.float32) @ img_W_f16.T)
    alt_chunks = alt_params.reshape(6, H)
    alt_out = modulate(img_LN1, alt_chunks[0], alt_chunks[1])
    alts_img.append(("no-bias",        cossim(alt_out, img_mod1_n)))

    for lbl, cs in alts_img:
        print(f"  alt {lbl:24s}  img_mod1 cos={cs:.6f}")
    print()

    # (c) Try alternate split orderings (HF spec says
    # [shift1, scale1, gate1, shift2, scale2, gate2]).
    print("--- diagnostic 3: alternate split orderings ---")
    perms = {
        "legacy [s1,sh1,g1,s2,sh2,g2]": (0, 1),   # scale=0, shift=1 (current)
        "HF spec [sh1,s1,g1,sh2,s2,g2]": (1, 0),  # shift=0, scale=1
    }
    for label, (si_scale, si_shift) in perms.items():
        out = modulate(img_LN1, img_chunks[si_scale], img_chunks[si_shift])
        cs = cossim(out, img_mod1_n)
        print(f"  img split {label:38s}  cos={cs:.6f}")
        out = modulate(txt_LN1, txt_chunks[si_scale], txt_chunks[si_shift])
        cs = cossim(out, txt_mod1_n)
        print(f"  txt split {label:38s}  cos={cs:.6f}")
    print()

    # (d) Per-chunk sanity: solve for the "ideal" scale1/shift1 from the
    # native mod1 dump, then cossim those vs our oracle chunks. This tells
    # us whether the matmul itself is wrong vs the modulate step.
    print("--- diagnostic 4: implied scale1/shift1 from native dump ---")
    # mod1[i, j] = LN1[i, j] * (1 + scale1[j]) + shift1[j]
    # For column j: (mod1 - shift1) / LN1 = 1 + scale1, but two unknowns per col.
    # Use least squares on per-column linear regression: mod1 ≈ a*LN1 + b
    # where a = 1+scale1[j], b = shift1[j].
    def implied_scale_shift(LN, MOD):
        H = LN.shape[1]
        seq = LN.shape[0]
        a = np.empty(H, dtype=np.float64)
        b = np.empty(H, dtype=np.float64)
        for j in range(H):
            x = LN[:, j].astype(np.float64)
            y = MOD[:, j].astype(np.float64)
            # least squares: y = a*x + b
            X = np.stack([x, np.ones_like(x)], axis=1)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            a[j], b[j] = coef
        return (a - 1.0).astype(np.float32), b.astype(np.float32)

    impl_scale1_img, impl_shift1_img = implied_scale_shift(img_LN1, img_mod1_n)
    impl_scale1_txt, impl_shift1_txt = implied_scale_shift(txt_LN1, txt_mod1_n)
    print(f"  img implied scale1: cos vs chunk[0]={cossim(impl_scale1_img, img_chunks[0]):.6f}  "
          f"vs chunk[1]={cossim(impl_scale1_img, img_chunks[1]):.6f}  "
          f"vs chunk[2]={cossim(impl_scale1_img, img_chunks[2]):.6f}")
    print(f"  img implied shift1: cos vs chunk[0]={cossim(impl_shift1_img, img_chunks[0]):.6f}  "
          f"vs chunk[1]={cossim(impl_shift1_img, img_chunks[1]):.6f}  "
          f"vs chunk[2]={cossim(impl_shift1_img, img_chunks[2]):.6f}")
    stats("img implied scale1", impl_scale1_img)
    stats("img implied shift1", impl_shift1_img)
    print(f"  txt implied scale1: cos vs chunk[0]={cossim(impl_scale1_txt, txt_chunks[0]):.6f}  "
          f"vs chunk[1]={cossim(impl_scale1_txt, txt_chunks[1]):.6f}  "
          f"vs chunk[2]={cossim(impl_scale1_txt, txt_chunks[2]):.6f}")
    print(f"  txt implied shift1: cos vs chunk[0]={cossim(impl_shift1_txt, txt_chunks[0]):.6f}  "
          f"vs chunk[1]={cossim(impl_shift1_txt, txt_chunks[1]):.6f}  "
          f"vs chunk[2]={cossim(impl_shift1_txt, txt_chunks[2]):.6f}")
    stats("txt implied scale1", impl_scale1_txt)
    stats("txt implied shift1", impl_shift1_txt)
    print()

    # ------------------------------------------------------------------ 5
    # Final verdict.
    # ------------------------------------------------------------------
    print("=== VERDICT ===")
    threshold_green = 0.999
    threshold_amber = 0.99
    if cos_img >= threshold_green and cos_txt >= threshold_green:
        verdict = ("GREEN-B (mod1 dispatch is bit-accurate; drift must be "
                   "in t_emb itself or upstream of substep 04 LN1)")
    elif cos_img >= threshold_amber and cos_txt >= threshold_amber:
        verdict = ("AMBER (mod1 dispatch close but not bit-accurate; "
                   "F16 round-trip noise OR small upstream drift in t_emb)")
    else:
        verdict = ("GREEN-A (mod1 dispatch DRIFTS — see diagnostics 2/3/4 "
                   "above for the misbehaving sub-component)")
    print(f"  img cos = {cos_img:.6f}")
    print(f"  txt cos = {cos_txt:.6f}")
    print(f"  {verdict}")


if __name__ == "__main__":
    main()
