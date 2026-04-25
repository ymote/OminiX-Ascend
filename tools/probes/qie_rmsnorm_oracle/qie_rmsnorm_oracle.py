#!/usr/bin/env python3
"""
Q2.4.5.5.17 — Pure-F32 numpy oracle for QIE-Edit DiT block-0 RmsNorm Q/K.

Final probe of the §5.5.13→§5.5.16→§5.5.18 bisect. Sole remaining suspect
between the F32 CPU reference and native is the RmsNorm-on-Q/K stage:
  - 08_*_Q/K   : pre-RmsNorm (post-projection)
  - 09_*_Q/K_rmsnorm : post-RmsNorm (pre-RoPE)
  - 10_*_Q/K_rope : post-RoPE   (pre-FIA)

Engine code (image_diffusion_engine.cpp:2553 `rms_norm_head_`):
  - F16 input, F16 output, F32 gamma, F32 rstd
  - eps = cfg_.rms_norm_eps = 1e-6f (image_diffusion_engine.h:126)
  - Calls aclnnRmsNorm with t_in=F16, t_g=F32, t_out=F16
  - img stream γ = lw.norm_q_w / lw.norm_k_w
        loaded from `transformer_blocks.{il}.attn.norm_q.weight` /
                    `transformer_blocks.{il}.attn.norm_k.weight`
  - txt stream γ = lw.norm_added_q_w / lw.norm_added_k_w
        loaded from `transformer_blocks.{il}.attn.norm_added_q.weight` /
                    `transformer_blocks.{il}.attn.norm_added_k.weight`

Hypotheses (rank-ordered):
  (1) F16 reduction overflow — Q magnitudes ≤ 723 → x² ≤ 522729; sum HD=128 ≤ 67M;
      mean ≤ 5.2M; sqrt ≤ 2280. F16 max=65504. If aclnnRmsNorm uses F16 accum
      internally, mean(x²) overflows → inf → out = 0/inf = 0 or nan.
  (2) γ-weight indexing swap — engine wires norm_q to txt and norm_added_q to
      img, or vice versa.
  (3) ε mismatch — engine ε differs from CPU reference (1e-6 vs 1e-5 etc).
  (4) Pre/post dump misnumbering.
"""

import numpy as np
import os
import sys

DUMP = os.environ.get("QIE_DUMPS", "/tmp/qie_dumps_5516")
GGUF = os.environ.get(
    "QIE_GGUF",
    "/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf",
)
NH   = 24
HD   = 128
H    = NH * HD          # 3072
TXT  = 32
IMG  = 64
EPS_DEFAULT = 1e-6


def find_gguf_py(repo_root):
    try:
        import gguf  # noqa: F401
        return
    except ImportError:
        pass
    cand = os.path.join(repo_root, "gguf-py")
    if os.path.isdir(cand):
        sys.path.insert(0, cand)


def load_f32(name, expected_n):
    path = os.path.join(DUMP, name)
    sz = os.path.getsize(path)
    assert sz == expected_n * 4, f"{name}: size {sz} != {expected_n*4}"
    a = np.fromfile(path, dtype=np.float32, count=expected_n)
    assert a.size == expected_n
    return a


def load_gamma(gguf_path, name):
    """Load 1-D gamma vector [HD] as F32. gguf-py returns t.data already
    typed for F32/F16/BF16 — just use it."""
    from gguf import GGUFReader, GGMLQuantizationType

    reader = GGUFReader(gguf_path)
    for t in reader.tensors:
        if t.name == name:
            qt = t.tensor_type
            d = t.data
            if qt == GGMLQuantizationType.F32:
                return np.asarray(d, dtype=np.float32).reshape(-1).copy()
            if qt == GGMLQuantizationType.F16:
                return np.asarray(d, dtype=np.float16).astype(np.float32).reshape(-1)
            if qt == GGMLQuantizationType.BF16:
                u16 = np.asarray(d, dtype=np.uint16).astype(np.uint32) << 16
                return u16.view(np.float32).reshape(-1).astype(np.float32)
            raise RuntimeError(f"{name}: unsupported gamma dtype {qt}")
    raise RuntimeError(f"tensor not found: {name}")


def cossim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rmsnorm_f32(x, gamma, eps):
    """x: [..., HD] F32; gamma: [HD] F32. Returns F32."""
    x = x.astype(np.float32)
    var = np.mean(x ** 2, axis=-1, keepdims=True).astype(np.float32)
    rstd = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    return (x * rstd * gamma.astype(np.float32)).astype(np.float32)


def rmsnorm_f16_inputs(x, gamma, eps):
    """Mimic engine: F16 input, F32 gamma, F32 reduction.

    Round x to F16 (storage), then compute reduction in F32 (assumed by
    aclnnRmsNorm spec). Cast result to F16 storage but keep F32 view for
    cosine compare (since dump path = device F16 → host F32 cast).
    """
    x16 = x.astype(np.float16).astype(np.float32)
    var = np.mean(x16 ** 2, axis=-1, keepdims=True).astype(np.float32)
    rstd = (1.0 / np.sqrt(var + eps)).astype(np.float32)
    out = (x16 * rstd * gamma.astype(np.float32)).astype(np.float32)
    # Mimic the dump path: round to F16 then upcast.
    return out.astype(np.float16).astype(np.float32)


def rmsnorm_f16_reduction(x, gamma, eps):
    """Pessimistic: F16 reduction (hypothesis 1). x², mean, sqrt all F16.

    This is what would happen if aclnnRmsNorm internally accumulated in F16.
    """
    x16 = x.astype(np.float16)
    sq = (x16.astype(np.float32) ** 2).astype(np.float16)  # F16 storage
    # mean over last axis in F16
    mean = sq.mean(axis=-1, keepdims=True)  # numpy promotes — force F16
    mean16 = mean.astype(np.float16)
    rstd = (1.0 / np.sqrt(mean16.astype(np.float32) + eps)).astype(np.float16)
    out = (x16 * rstd * gamma.astype(np.float16)).astype(np.float32)
    return out


def report_pair(label, oracle, native, NH=NH, HD=HD):
    g = cossim(oracle, native)
    diff = (oracle - native).astype(np.float64)
    ma  = float(np.abs(diff).mean())
    mx  = float(np.abs(diff).max())
    print(f"  {label:<30s} cos={g:.6f}  mean_abs={ma:.4f}  max_abs={mx:.4f}")
    return g


def per_head(oracle, native, label):
    """oracle/native shape [seq, NH, HD]."""
    print(f"  per-head {label}:")
    head_cos = []
    for h in range(NH):
        c = cossim(oracle[:, h, :], native[:, h, :])
        head_cos.append((h, c))
    head_cos.sort(key=lambda x: x[1])
    for h, c in head_cos[:3]:
        print(f"    worst head {h:2d} cos={c:.4f}")
    for h, c in head_cos[-3:][::-1]:
        print(f"    best  head {h:2d} cos={c:.4f}")


def main():
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    find_gguf_py(repo_root)

    print(f"=== QIE RmsNorm Q/K oracle — {DUMP} ===")
    print(f"GGUF: {GGUF}")
    print(f"shapes: TXT={TXT}, IMG={IMG}, NH={NH}, HD={HD}")

    # --- load pre/post dumps ---
    img_Q_pre  = load_f32("08_img_Q.f32",          IMG * H).reshape(IMG, NH, HD)
    img_K_pre  = load_f32("08_img_K.f32",          IMG * H).reshape(IMG, NH, HD)
    txt_Q_pre  = load_f32("08_txt_Q.f32",          TXT * H).reshape(TXT, NH, HD)
    txt_K_pre  = load_f32("08_txt_K.f32",          TXT * H).reshape(TXT, NH, HD)

    img_Q_post = load_f32("09_img_Q_rmsnorm.f32",  IMG * H).reshape(IMG, NH, HD)
    img_K_post = load_f32("09_img_K_rmsnorm.f32",  IMG * H).reshape(IMG, NH, HD)
    txt_Q_post = load_f32("09_txt_Q_rmsnorm.f32",  TXT * H).reshape(TXT, NH, HD)
    txt_K_post = load_f32("09_txt_K_rmsnorm.f32",  TXT * H).reshape(TXT, NH, HD)

    # --- magnitude diagnostics ---
    print("\n--- magnitude diagnostics (08 pre-RmsNorm) ---")
    for nm, t in [("img_Q", img_Q_pre), ("img_K", img_K_pre),
                  ("txt_Q", txt_Q_pre), ("txt_K", txt_K_pre)]:
        absmax = np.abs(t).max()
        # per-row x² mean (HD-axis)
        rmean = np.mean(t.astype(np.float64) ** 2, axis=-1)
        print(f"  {nm}: |x|_max={absmax:.1f}  mean(x²) max-row={rmean.max():.1f}  "
              f"sqrt={np.sqrt(rmean.max()):.1f}  F16max=65504")

    print("\n--- magnitude diagnostics (09 post-RmsNorm) ---")
    for nm, t in [("img_Q", img_Q_post), ("img_K", img_K_post),
                  ("txt_Q", txt_Q_post), ("txt_K", txt_K_post)]:
        absmax = np.abs(t).max()
        absmin = np.abs(t).min()
        n_zero = int((np.abs(t) == 0.0).sum())
        n_nan  = int(np.isnan(t).sum())
        n_inf  = int(np.isinf(t).sum())
        print(f"  {nm}: |x|_max={absmax:.4f}  |x|_min={absmin:.6f}  "
              f"#zero={n_zero}  #nan={n_nan}  #inf={n_inf}  total={t.size}")

    # --- load gammas ---
    print("\n--- loading gammas ---")
    g_norm_q       = load_gamma(GGUF, "transformer_blocks.0.attn.norm_q.weight")
    g_norm_k       = load_gamma(GGUF, "transformer_blocks.0.attn.norm_k.weight")
    g_norm_added_q = load_gamma(GGUF, "transformer_blocks.0.attn.norm_added_q.weight")
    g_norm_added_k = load_gamma(GGUF, "transformer_blocks.0.attn.norm_added_k.weight")
    for nm, g in [("norm_q", g_norm_q), ("norm_k", g_norm_k),
                  ("norm_added_q", g_norm_added_q),
                  ("norm_added_k", g_norm_added_k)]:
        assert g.shape == (HD,), f"{nm}: shape {g.shape}"
        print(f"  {nm}: shape={g.shape}  mean={g.mean():.4f}  "
              f"std={g.std():.4f}  min={g.min():.4f}  max={g.max():.4f}")

    # --- primary oracle: pure F32 with stated wiring ---
    print(f"\n--- primary oracle (pure F32, eps={EPS_DEFAULT}) ---")
    o_img_Q = rmsnorm_f32(img_Q_pre, g_norm_q,       EPS_DEFAULT)
    o_img_K = rmsnorm_f32(img_K_pre, g_norm_k,       EPS_DEFAULT)
    o_txt_Q = rmsnorm_f32(txt_Q_pre, g_norm_added_q, EPS_DEFAULT)
    o_txt_K = rmsnorm_f32(txt_K_pre, g_norm_added_k, EPS_DEFAULT)
    cQi = report_pair("img_Q  (γ=norm_q)",       o_img_Q, img_Q_post)
    cKi = report_pair("img_K  (γ=norm_k)",       o_img_K, img_K_post)
    cQt = report_pair("txt_Q  (γ=norm_added_q)", o_txt_Q, txt_Q_post)
    cKt = report_pair("txt_K  (γ=norm_added_k)", o_txt_K, txt_K_post)
    g_oracle = np.concatenate([
        np.concatenate([o_img_Q.flatten(), o_txt_Q.flatten()]),
        np.concatenate([o_img_K.flatten(), o_txt_K.flatten()]),
    ])
    g_native = np.concatenate([
        np.concatenate([img_Q_post.flatten(), txt_Q_post.flatten()]),
        np.concatenate([img_K_post.flatten(), txt_K_post.flatten()]),
    ])
    g_global = cossim(g_oracle, g_native)
    print(f"  GLOBAL Q+K cos = {g_global:.6f}")

    per_head(o_img_Q, img_Q_post, "img_Q")
    per_head(o_txt_Q, txt_Q_post, "txt_Q")

    # --- alt-config: γ swap (hypothesis 2) ---
    print("\n--- alt-config: γ swap (img↔txt) ---")
    a_img_Q = rmsnorm_f32(img_Q_pre, g_norm_added_q, EPS_DEFAULT)
    a_img_K = rmsnorm_f32(img_K_pre, g_norm_added_k, EPS_DEFAULT)
    a_txt_Q = rmsnorm_f32(txt_Q_pre, g_norm_q,       EPS_DEFAULT)
    a_txt_K = rmsnorm_f32(txt_K_pre, g_norm_k,       EPS_DEFAULT)
    report_pair("img_Q  (γ=norm_added_q)", a_img_Q, img_Q_post)
    report_pair("img_K  (γ=norm_added_k)", a_img_K, img_K_post)
    report_pair("txt_Q  (γ=norm_q)",       a_txt_Q, txt_Q_post)
    report_pair("txt_K  (γ=norm_k)",       a_txt_K, txt_K_post)

    # --- alt-config: alt eps (hypothesis 3) ---
    print("\n--- alt-config: alt-ε (img_Q with γ=norm_q) ---")
    for e in (1e-6, 1e-5, 5e-6, 1e-4, 1e-7, 0.0):
        oe = rmsnorm_f32(img_Q_pre, g_norm_q, e)
        c = cossim(oe, img_Q_post)
        print(f"  eps={e:>9.1e}  cos={c:.6f}")

    # --- alt-config: F16 input rounding (engine path) ---
    print(f"\n--- alt-config: F16-input/F32-reduction (engine path), eps={EPS_DEFAULT} ---")
    f_img_Q = rmsnorm_f16_inputs(img_Q_pre, g_norm_q,       EPS_DEFAULT)
    f_img_K = rmsnorm_f16_inputs(img_K_pre, g_norm_k,       EPS_DEFAULT)
    f_txt_Q = rmsnorm_f16_inputs(txt_Q_pre, g_norm_added_q, EPS_DEFAULT)
    f_txt_K = rmsnorm_f16_inputs(txt_K_pre, g_norm_added_k, EPS_DEFAULT)
    report_pair("img_Q (F16-in,F32-red)", f_img_Q, img_Q_post)
    report_pair("img_K (F16-in,F32-red)", f_img_K, img_K_post)
    report_pair("txt_Q (F16-in,F32-red)", f_txt_Q, txt_Q_post)
    report_pair("txt_K (F16-in,F32-red)", f_txt_K, txt_K_post)

    # --- alt-config: F16 reduction (hypothesis 1) ---
    print(f"\n--- alt-config: F16-reduction (overflow hypothesis), eps={EPS_DEFAULT} ---")
    h_img_Q = rmsnorm_f16_reduction(img_Q_pre, g_norm_q,       EPS_DEFAULT)
    h_img_K = rmsnorm_f16_reduction(img_K_pre, g_norm_k,       EPS_DEFAULT)
    h_txt_Q = rmsnorm_f16_reduction(txt_Q_pre, g_norm_added_q, EPS_DEFAULT)
    h_txt_K = rmsnorm_f16_reduction(txt_K_pre, g_norm_added_k, EPS_DEFAULT)
    report_pair("img_Q (F16 reduction)", h_img_Q, img_Q_post)
    report_pair("img_K (F16 reduction)", h_img_K, img_K_post)
    report_pair("txt_Q (F16 reduction)", h_txt_Q, txt_Q_post)
    report_pair("txt_K (F16 reduction)", h_txt_K, txt_K_post)

    # --- summary verdict ---
    print("\n=== SUMMARY ===")
    print(f"primary oracle global cos = {g_global:.6f}")
    print(f"  per-stream: img_Q={cQi:.6f}  img_K={cKi:.6f}  "
          f"txt_Q={cQt:.6f}  txt_K={cKt:.6f}")
    if g_global >= 0.999:
        verdict = ("GREEN-B — RmsNorm bit-accurate; bisect failed. "
                   "Re-run substep cossim sweep, suspect block-1+ or hidden-state.")
    elif g_global >= 0.99:
        verdict = "AMBER — RmsNorm mostly accurate, drift below 0.99"
    else:
        verdict = "GREEN-A — RmsNorm divergence confirmed; check alt-configs above"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
