#!/usr/bin/env python3
"""
Q2.4.5.5.38 — Block-1 substep oracle bisect for QIE-Edit DiT (REAL 1024² shape).

Reads block-1 dumps at /tmp/qie_5536_eng_real/block01/, applies F32 numpy
oracles substep-by-substep, and reports per-substep cossim oracle vs engine.

Substeps covered for IMG stream:
  04 LN1, 02 mod_out, 05 mod1, 08 Q/K/V, 09 Q/K rmsnorm, 11 attn_out, 12 to_out_0, 13 resid1.

Block-1 input == block-0 output (verified by md5).

Layout:
  IMG_SEQ = 8192 (1024²/(8*16))
  TXT_SEQ = 212
  H = 3072, NH = 24, HD = 128
  weights stored in GGUF [K, N] (per qie_mod1_oracle.py); we reshape to [N, K]
  to match engine row-major [out_ch, in_ch].
"""
import os
import sys
import time
import argparse
import numpy as np

DUMP_DIR = "/tmp/qie_5536_eng_real/block01"
GGUF = "/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf"
BLOCK = 1

H = 3072
NH = 24
HD = 128
IMG_SEQ = 8192
TXT_SEQ = 212
B = 1


def find_gguf_py(repo_root):
    try:
        import gguf  # noqa
        return
    except ImportError:
        pass
    cand = os.path.join(repo_root, "gguf-py")
    if os.path.isdir(cand):
        sys.path.insert(0, cand)


def load_dequant_weight(gguf_path, name):
    """Return (W_f32_[N,K], qt_str)."""
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
        return raw.view(np.float32).reshape(N, K).astype(np.float32), "F32"
    if qt == GGMLQuantizationType.F16:
        return raw.view(np.float16).reshape(N, K).astype(np.float32), "F16"
    if qt == GGMLQuantizationType.BF16:
        u16 = raw.view(np.uint16).astype(np.uint32)
        return ((u16 << 16).view(np.float32)).reshape(N, K).astype(np.float32), "BF16"

    qcls_map = {q.qtype: q for q in quants.__Quant.__subclasses__() if hasattr(q, "qtype")}
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


def load_bias(gguf_path, name):
    from gguf import GGUFReader, GGMLQuantizationType
    reader = GGUFReader(gguf_path)
    for t in reader.tensors:
        if t.name == name:
            data = np.asarray(t.data)
            qt = t.tensor_type
            if qt == GGMLQuantizationType.F32:
                return data.astype(np.float32).reshape(-1).copy()
            if qt == GGMLQuantizationType.F16:
                return data.astype(np.float32).reshape(-1)
            if qt == GGMLQuantizationType.BF16:
                u16 = data.view(np.uint16).astype(np.uint32)
                return ((u16 << 16).view(np.float32)).reshape(-1).astype(np.float32)
            raise RuntimeError(f"bias dtype unsupported: {qt}")
    return None


def load_f32(name, expected_n):
    path = os.path.join(DUMP_DIR, name)
    sz = os.path.getsize(path)
    if sz != expected_n * 4:
        raise AssertionError(f"{name}: size {sz} != {expected_n*4}")
    return np.fromfile(path, dtype=np.float32, count=expected_n)


def f16r(x):
    return x.astype(np.float16).astype(np.float32)


def cossim(a, b):
    a = np.asarray(a).flatten().astype(np.float64)
    b = np.asarray(b).flatten().astype(np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def absratio(a, b):
    am = float(np.abs(a).max())
    bm = float(np.abs(b).max())
    if bm == 0.0:
        return float("inf") if am != 0.0 else 1.0
    return am / bm


def silu_f32(x):
    return (x.astype(np.float64) / (1.0 + np.exp(-x.astype(np.float64)))).astype(np.float32)


def main():
    global DUMP_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default=None)
    ap.add_argument("--gguf", default=GGUF)
    ap.add_argument("--block", type=int, default=BLOCK)
    ap.add_argument("--no-rope", action="store_true",
                    help="Skip ROPE substep (covered by 09→10 dump cmp directly)")
    args = ap.parse_args()

    if args.dumps is not None:
        DUMP_DIR = args.dumps

    repo_root = "/home/ma-user/work/OminiX-Ascend"
    find_gguf_py(repo_root)

    base = f"transformer_blocks.{args.block}"
    print(f"=== block-{args.block} oracle bisect (REAL 1024² shape) ===")
    print(f"DUMPS: {DUMP_DIR}")
    print(f"GGUF : {args.gguf}")
    print(f"H={H} NH={NH} HD={HD} IMG_SEQ={IMG_SEQ} TXT_SEQ={TXT_SEQ}")
    print()

    results = []  # (substep_name, cos, absratio)

    # ----------- Load engine dumps -----------
    t0 = time.time()
    img_in = load_f32("00_img.f32", IMG_SEQ * H).reshape(IMG_SEQ, H)
    t_emb = load_f32("00_t_emb.f32", H)
    silu_t_eng = load_f32("01_silu_t_emb.f32", H)
    img_mod_out_eng = load_f32("02_img_mod_out.f32", 6 * H)
    img_LN1_eng = load_f32("04_img_LN1.f32", IMG_SEQ * H).reshape(IMG_SEQ, H)
    img_mod1_eng = load_f32("05_img_mod1.f32", IMG_SEQ * H).reshape(IMG_SEQ, H)
    img_Q_eng = load_f32("08_img_Q.f32", IMG_SEQ * H).reshape(IMG_SEQ, NH, HD)
    img_K_eng = load_f32("08_img_K.f32", IMG_SEQ * H).reshape(IMG_SEQ, NH, HD)
    img_V_eng = load_f32("08_img_V.f32", IMG_SEQ * H).reshape(IMG_SEQ, NH, HD)
    img_Q_rms_eng = load_f32("09_img_Q_rmsnorm.f32", IMG_SEQ * H).reshape(IMG_SEQ, NH, HD)
    img_K_rms_eng = load_f32("09_img_K_rmsnorm.f32", IMG_SEQ * H).reshape(IMG_SEQ, NH, HD)
    img_Q_rope_eng = load_f32("10_img_Q_rope.f32", IMG_SEQ * H).reshape(IMG_SEQ, NH, HD)
    img_K_rope_eng = load_f32("10_img_K_rope.f32", IMG_SEQ * H).reshape(IMG_SEQ, NH, HD)
    txt_Q_rope_eng = load_f32("10_txt_Q_rope.f32", TXT_SEQ * H).reshape(TXT_SEQ, NH, HD)
    txt_K_rope_eng = load_f32("10_txt_K_rope.f32", TXT_SEQ * H).reshape(TXT_SEQ, NH, HD)
    txt_V_eng = load_f32("08_txt_V.f32", TXT_SEQ * H).reshape(TXT_SEQ, NH, HD)
    attn_img_eng = load_f32("11_attn_out_img.f32", IMG_SEQ * H).reshape(IMG_SEQ, NH, HD)
    to_out0_eng = load_f32("12_to_out_0.f32", IMG_SEQ * H).reshape(IMG_SEQ, H)
    resid1_eng = load_f32("13_img_resid1.f32", IMG_SEQ * H).reshape(IMG_SEQ, H)
    print(f"[+{time.time()-t0:.1f}s] dumps loaded")

    # ----------- 01 silu sanity (cheap, just confirm) -----------
    silu_oracle = silu_f32(f16r(t_emb))
    silu_oracle = f16r(silu_oracle)
    c = cossim(silu_oracle, silu_t_eng)
    print(f"[01_silu_t_emb]   cos={c:.6f}  absratio={absratio(silu_oracle, silu_t_eng):.4f}")

    # ----------- 02 img_mod_out -----------
    img_mod_W, qt_w = load_dequant_weight(args.gguf, f"{base}.img_mod.1.weight")
    img_mod_b = load_bias(args.gguf, f"{base}.img_mod.1.bias")
    print(f"  img_mod.1.weight qt={qt_w} shape={img_mod_W.shape}")
    img_mod_out_oracle = silu_oracle.astype(np.float32) @ f16r(img_mod_W).T + f16r(img_mod_b)
    img_mod_out_oracle = f16r(img_mod_out_oracle)  # [6H]
    c = cossim(img_mod_out_oracle, img_mod_out_eng)
    print(f"[02_img_mod_out]  cos={c:.6f}  absratio={absratio(img_mod_out_oracle, img_mod_out_eng):.4f}")
    results.append(("02_img_mod_out", c))

    # Free large weight asap
    del img_mod_W

    # Use ENGINE chunks (legacy split: scale1, shift1, gate1, scale2, shift2, gate2)
    chunks_eng = img_mod_out_eng.reshape(6, H)
    img_scale1, img_shift1, img_gate1 = chunks_eng[0], chunks_eng[1], chunks_eng[2]

    # ----------- 04 img_LN1 -----------
    # Engine LN1 is LayerNorm(no affine), eps default 1e-6.
    img_in_f32 = img_in
    mu = img_in_f32.mean(axis=-1, keepdims=True)
    var = img_in_f32.var(axis=-1, keepdims=True)
    LN1_oracle = (img_in_f32 - mu) / np.sqrt(var + 1e-6)
    LN1_oracle = f16r(LN1_oracle.astype(np.float32))
    c = cossim(LN1_oracle, img_LN1_eng)
    print(f"[04_img_LN1]      cos={c:.6f}  absratio={absratio(LN1_oracle, img_LN1_eng):.4f}")
    results.append(("04_img_LN1", c))

    # ----------- 05 img_mod1 -----------
    # mod1 = LN1 * (1 + scale1) + shift1, F16 in/out
    mod1_oracle = f16r(img_LN1_eng) * (1.0 + img_scale1[None, :]) + img_shift1[None, :]
    mod1_oracle = f16r(mod1_oracle)
    c = cossim(mod1_oracle, img_mod1_eng)
    print(f"[05_img_mod1]     cos={c:.6f}  absratio={absratio(mod1_oracle, img_mod1_eng):.4f}")
    results.append(("05_img_mod1", c))

    # ----------- 08 Q/K/V projections -----------
    # input = engine 05_img_mod1 (F16), proj = to_q/k/v (W_q4 + scale, F16 bias)
    # Free older arrays
    del img_in_f32, mu, var, LN1_oracle, mod1_oracle

    mod1_in = f16r(img_mod1_eng)
    for tag, wname, bname, eng_dump in [
        ("Q", f"{base}.attn.to_q.weight", f"{base}.attn.to_q.bias", img_Q_eng),
        ("K", f"{base}.attn.to_k.weight", f"{base}.attn.to_k.bias", img_K_eng),
        ("V", f"{base}.attn.to_v.weight", f"{base}.attn.to_v.bias", img_V_eng),
    ]:
        W, qt = load_dequant_weight(args.gguf, wname)
        b = load_bias(args.gguf, bname)
        Wf16 = f16r(W)
        bf16 = f16r(b)
        # mod1_in [IMG_SEQ, H], W [H, H] reshape from [N=H, K=H]
        out = mod1_in @ Wf16.T + bf16
        out = f16r(out)
        out_resh = out.reshape(IMG_SEQ, NH, HD)
        c = cossim(out_resh, eng_dump)
        print(f"[08_img_{tag}]      cos={c:.6f}  absratio={absratio(out_resh, eng_dump):.4f}  qt={qt}")
        results.append((f"08_img_{tag}", c))
        del W, Wf16, out

    # ----------- 09 RmsNorm Q / K (per-head) -----------
    # rmsnorm: x_norm = x / sqrt(mean(x^2)+eps) * gamma  per HD axis
    eps = 1e-6
    for tag, wname, x_in_eng, eng_dump in [
        ("Q", f"{base}.attn.norm_q.weight", img_Q_eng, img_Q_rms_eng),
        ("K", f"{base}.attn.norm_k.weight", img_K_eng, img_K_rms_eng),
    ]:
        gamma = load_bias(args.gguf, wname)  # [HD]
        # Engine: F16 in/out, F32 internal compute
        x = x_in_eng  # [IMG_SEQ, NH, HD]
        # var = mean(x^2) over last axis
        ms = (x.astype(np.float32) ** 2).mean(axis=-1, keepdims=True)
        out = x.astype(np.float32) / np.sqrt(ms + eps) * gamma[None, None, :]
        out = f16r(out)
        c = cossim(out, eng_dump)
        print(f"[09_img_{tag}_rmsnorm] cos={c:.6f}  absratio={absratio(out, eng_dump):.4f}")
        results.append((f"09_img_{tag}_rmsnorm", c))

    # ----------- 11 attention -----------
    # Joint attention: cat[txt, img] for Q, K, V; output is split back
    Q_joint = np.concatenate([txt_Q_rope_eng, img_Q_rope_eng], axis=0)  # [SEQ, NH, HD]
    K_joint = np.concatenate([txt_K_rope_eng, img_K_rope_eng], axis=0)
    V_joint = np.concatenate([txt_V_eng, img_V_eng], axis=0)

    # Per-head attention in F64; chunk over heads to limit memory
    SEQ = IMG_SEQ + TXT_SEQ
    print(f"[11] running attention (SEQ={SEQ}, NH={NH}, HD={HD})...")
    out_joint = np.empty((SEQ, NH, HD), dtype=np.float32)
    scale = 1.0 / np.sqrt(HD)
    for h in range(NH):
        Qh = Q_joint[:, h, :].astype(np.float64)  # [SEQ, HD]
        Kh = K_joint[:, h, :].astype(np.float64)
        Vh = V_joint[:, h, :].astype(np.float64)
        s = (Qh @ Kh.T) * scale  # [SEQ, SEQ]
        s -= s.max(axis=-1, keepdims=True)
        np.exp(s, out=s)
        s /= s.sum(axis=-1, keepdims=True)
        out_joint[:, h, :] = (s @ Vh).astype(np.float32)
        if h % 8 == 7:
            print(f"  head {h+1}/{NH} done")
    attn_img_oracle = out_joint[TXT_SEQ:]
    c = cossim(attn_img_oracle, attn_img_eng)
    print(f"[11_attn_out_img] cos={c:.6f}  absratio={absratio(attn_img_oracle, attn_img_eng):.4f}")
    results.append(("11_attn_out_img", c))
    del Q_joint, K_joint, V_joint, out_joint

    # ----------- 12 to_out_0 -----------
    W, qt = load_dequant_weight(args.gguf, f"{base}.attn.to_out.0.weight")
    b = load_bias(args.gguf, f"{base}.attn.to_out.0.bias")
    Wf16 = f16r(W)
    bf16 = f16r(b)
    inp = f16r(attn_img_eng).reshape(IMG_SEQ, H)
    to_out_oracle = inp @ Wf16.T + bf16
    to_out_oracle = f16r(to_out_oracle)
    c = cossim(to_out_oracle, to_out0_eng)
    print(f"[12_to_out_0]     cos={c:.6f}  absratio={absratio(to_out_oracle, to_out0_eng):.4f}")
    results.append(("12_to_out_0", c))
    del W, Wf16

    # ----------- 13 resid1 -----------
    # img_hidden (F32) += to_out_0 (F16 src) * gate1 (F16)
    # i.e. resid1 = img_in (F32) + (f16(to_out_0) * f16(gate1)) cast-to-f32
    gate1 = img_gate1  # [H], F16-ish (we store as f32 from engine dump)
    update = f16r(to_out0_eng) * f16r(gate1)[None, :]
    update = update.astype(np.float32)
    resid1_oracle = img_in.astype(np.float32) + update
    c = cossim(resid1_oracle, resid1_eng)
    print(f"[13_img_resid1]   cos={c:.6f}  absratio={absratio(resid1_oracle, resid1_eng):.4f}")
    results.append(("13_img_resid1", c))

    # ----------- Verdict -----------
    print()
    print("=== SUMMARY ===")
    first_div = None
    for name, c in results:
        flag = "PASS" if c >= 0.99 else "DIV"
        print(f"  {name:25s} cos={c:.6f}  {flag}")
        if c < 0.99 and first_div is None:
            first_div = (name, c)

    print()
    if first_div is None:
        print("ENGINE_BIT_ACCURATE_AT_BLOCK_1 (all substeps cos>=0.99)")
    else:
        print(f"ENGINE_HAS_BUG_AT_SUBSTEP_{first_div[0]} (cos={first_div[1]:.6f})")


if __name__ == "__main__":
    main()
