#!/usr/bin/env python3
"""
Q2.4.5.5.18 — Pure-F32 numpy oracle for QIE-Edit DiT 3D-axial RoPE.

Reads post-RmsNorm Q/K (substep 09) for both txt and img streams, recomputes
3D-axial RoPE in pure F32 numpy matching the engine's
`compute_qwen_rope_pe_host` + `apply_rope_host_` contract, and cosine-compares
against the native RoPE-applied dumps (substep 10).

RoPE contract (verified in image_diffusion_engine.cpp):
- Layout per pos: pe[pos, d_pair, 2, 2] with
    pe[pos, d_pair, 0, :] = [ cos(theta), -sin(theta) ]
    pe[pos, d_pair, 1, :] = [ sin(theta),  cos(theta) ]
- Pair convention is INTERLEAVED (NOT NEOX-half-rotation):
    For pair index dp in [0, head_dim/2):
      x_even = x[..., 2*dp]
      x_odd  = x[..., 2*dp + 1]
      y_even = x_even * cos + x_odd  * sin
      y_odd  = x_odd  * cos - x_even * sin
  (i.e., y[..., 2dp]   = x_even * pe00 + x_odd * pe10
         y[..., 2dp+1] = x_even * pe01 + x_odd * pe11
         where pe00=cos, pe01=-sin, pe10=sin, pe11=cos)

- 3D-axial: head_dim=128 split into axes_t=16, axes_h=56, axes_w=56 contiguous
  pair groups (8 + 28 + 28 = 64 pairs).
- Per-axis omega: linspace(0, (axis_dim-2)/axis_dim, axis_dim/2) followed by
    omega[i] = 1 / theta^scale, where scale = end_scale * i / (half_axis-1).
- theta = 10000 (cfg.rope_theta default).

Position IDs:
- txt stream rows i in [0, txt_seq): t=h=w=p where p = txt_start + i, and
    txt_start = max(h_len, w_len) = 64 (for h_len=w_len=64 patch grid).
  These are diagonal (all three axes get the same scalar position).
- img stream rows: r,c in [0,h_len) x [0,w_len), pos = ctx_len + r*w_len + c.
  Coords: t_id=0, h_id = (-h_len/2) + r, w_id = (-w_len/2) + c.
  But the engine indexes only the first img_seq pe rows (e.g. 64 rows).
  Engine uses pe_row_offset = cfg.max_txt_seq (NOT actual txt_seq).

Variant A: img_pe_off = max_txt_seq = 256 (default smoke run).
Variant B: img_pe_off = txt_seq = 32 (QIE_RUNTIME_MAX_TXT_SEQ=1 override).

Decision matrix:
  GREEN-A: cos < 0.99 globally OR concentrated in head/pos → RoPE bug.
  GREEN-B: cos >= 0.99 → RoPE fine; bug is upstream RmsNorm.
  AMBER  : 0.95 <= cos < 0.99 → both potentially.
"""

import numpy as np
import os, sys

DUMP = os.environ.get("QIE_DUMPS", "/tmp/qie_dumps_5516")
NH   = 24
HD   = 128
H    = NH * HD          # 3072
TXT  = 32
IMG  = 64

# 3D-axial config (cfg defaults).
AXES_T = 16
AXES_H = 56
AXES_W = 56
THETA  = 10000.0

# Patch grid: pe table built with max_img_seq=4096 → h_len=w_len=64.
H_LEN  = 64
W_LEN  = 64
TXT_START = max(H_LEN, W_LEN)  # 64 per gen_qwen_image_ids
H_START = -H_LEN // 2          # -32
W_START = -W_LEN // 2          # -32


def load_f32(name, expected_n):
    path = os.path.join(DUMP, name)
    sz = os.path.getsize(path)
    assert sz == expected_n * 4, f"{name}: size {sz} != {expected_n*4}"
    a = np.fromfile(path, dtype=np.float32, count=expected_n)
    assert a.size == expected_n
    return a


def cossim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def axis_omega(axis_dim, theta):
    """linspace(0, (d-2)/d, d/2); omega[i] = 1 / theta^scale."""
    half = axis_dim // 2
    if half == 0: return np.empty((0,), dtype=np.float64)
    if half == 1: return np.array([1.0], dtype=np.float64)
    end_scale = (axis_dim - 2.0) / axis_dim
    i = np.arange(half, dtype=np.float64)
    scale = end_scale * i / (half - 1)
    return 1.0 / np.power(float(theta), scale)


def build_angles_for_pos(t_id, h_id, w_id, omega_t, omega_h, omega_w):
    """Returns angles array of length head_dim/2, in axial order [t-pairs, h-pairs, w-pairs]."""
    return np.concatenate([
        t_id * omega_t,
        h_id * omega_h,
        w_id * omega_w,
    ])


def apply_rope_interleaved(x, cos_pos, sin_pos):
    """
    x:        [..., head_dim] F64
    cos_pos:  [seq, head_dim/2] F64
    sin_pos:  [seq, head_dim/2] F64
    where the seq axis of x is the second-to-last (x: [seq, NH, HD]).

    Interleaved pair convention:
       y[..., 2dp]   = x[..., 2dp]   * cos + x[..., 2dp+1] * sin
       y[..., 2dp+1] = x[..., 2dp+1] * cos - x[..., 2dp]   * sin
    Equivalent matrix: pe = [[cos,-sin],[sin,cos]] applied to row [x_even, x_odd].
    """
    # x shape: [seq, NH, HD]
    seq = x.shape[0]
    HD_ = x.shape[-1]
    half = HD_ // 2
    x_even = x[..., 0::2]   # [seq, NH, half]
    x_odd  = x[..., 1::2]   # [seq, NH, half]
    # cos_pos/sin_pos are [seq, half] → broadcast over NH.
    cos_b = cos_pos[:, None, :]  # [seq, 1, half]
    sin_b = sin_pos[:, None, :]
    y_even = x_even * cos_b + x_odd * sin_b
    y_odd  = x_odd  * cos_b - x_even * sin_b
    out = np.empty_like(x)
    out[..., 0::2] = y_even
    out[..., 1::2] = y_odd
    return out


def build_txt_cos_sin(txt_seq, omega_t, omega_h, omega_w):
    """txt rows: pos i in [0,txt_seq), p = TXT_START + i, t=h=w=p."""
    half = HD // 2
    cos_t = np.zeros((txt_seq, half), dtype=np.float64)
    sin_t = np.zeros((txt_seq, half), dtype=np.float64)
    for i in range(txt_seq):
        p = float(TXT_START + i)
        ang = build_angles_for_pos(p, p, p, omega_t, omega_h, omega_w)
        cos_t[i] = np.cos(ang)
        sin_t[i] = np.sin(ang)
    return cos_t, sin_t


def build_img_cos_sin(img_seq, omega_t, omega_h, omega_w):
    """img rows: row-major over (r,c) in (h_len, w_len). For img_seq=64 = first row only."""
    half = HD // 2
    cos_t = np.zeros((img_seq, half), dtype=np.float64)
    sin_t = np.zeros((img_seq, half), dtype=np.float64)
    # row-major: pos within img-block = r*w_len + c
    for k in range(img_seq):
        r = k // W_LEN
        c = k %  W_LEN
        h_id = float(H_START + r)
        w_id = float(W_START + c)
        t_id = 0.0
        ang = build_angles_for_pos(t_id, h_id, w_id, omega_t, omega_h, omega_w)
        cos_t[k] = np.cos(ang)
        sin_t[k] = np.sin(ang)
    return cos_t, sin_t


def per_head_cossim(oracle, native):
    # shapes: [seq, NH, HD]
    nh = oracle.shape[1]
    return [cossim(oracle[:, h, :], native[:, h, :]) for h in range(nh)]


def per_pos_cossim(oracle, native):
    seq = oracle.shape[0]
    return [cossim(oracle[s], native[s]) for s in range(seq)]


def main():
    omega_t = axis_omega(AXES_T, THETA)
    omega_h = axis_omega(AXES_H, THETA)
    omega_w = axis_omega(AXES_W, THETA)
    assert omega_t.size + omega_h.size + omega_w.size == HD // 2, \
        f"axes sum {omega_t.size + omega_h.size + omega_w.size} != HD/2 {HD//2}"

    # Load 09 (input) and 10 (native output).
    txt_Q_in  = load_f32("09_txt_Q_rmsnorm.f32", TXT * H).reshape(TXT, NH, HD).astype(np.float64)
    txt_K_in  = load_f32("09_txt_K_rmsnorm.f32", TXT * H).reshape(TXT, NH, HD).astype(np.float64)
    img_Q_in  = load_f32("09_img_Q_rmsnorm.f32", IMG * H).reshape(IMG, NH, HD).astype(np.float64)
    img_K_in  = load_f32("09_img_K_rmsnorm.f32", IMG * H).reshape(IMG, NH, HD).astype(np.float64)

    txt_Q_nat = load_f32("10_txt_Q_rope.f32",    TXT * H).reshape(TXT, NH, HD).astype(np.float64)
    txt_K_nat = load_f32("10_txt_K_rope.f32",    TXT * H).reshape(TXT, NH, HD).astype(np.float64)
    img_Q_nat = load_f32("10_img_Q_rope.f32",    IMG * H).reshape(IMG, NH, HD).astype(np.float64)
    img_K_nat = load_f32("10_img_K_rope.f32",    IMG * H).reshape(IMG, NH, HD).astype(np.float64)

    print(f"[5518] dumps: TXT={TXT} IMG={IMG} NH={NH} HD={HD}  axes=({AXES_T},{AXES_H},{AXES_W}) theta={THETA}")
    print(f"[5518] omega_t[0:3]={omega_t[:3]} omega_h[0:3]={omega_h[:3]} omega_w[0:3]={omega_w[:3]}")
    print()

    # ---- TXT stream oracle (interleaved 3D-axial) ----
    cos_txt, sin_txt = build_txt_cos_sin(TXT, omega_t, omega_h, omega_w)
    txt_Q_or = apply_rope_interleaved(txt_Q_in, cos_txt, sin_txt)
    txt_K_or = apply_rope_interleaved(txt_K_in, cos_txt, sin_txt)
    cos_g_txt_Q = cossim(txt_Q_or, txt_Q_nat)
    cos_g_txt_K = cossim(txt_K_or, txt_K_nat)

    # ---- IMG stream oracle ----
    cos_img, sin_img = build_img_cos_sin(IMG, omega_t, omega_h, omega_w)
    img_Q_or = apply_rope_interleaved(img_Q_in, cos_img, sin_img)
    img_K_or = apply_rope_interleaved(img_K_in, cos_img, sin_img)
    cos_g_img_Q = cossim(img_Q_or, img_Q_nat)
    cos_g_img_K = cossim(img_K_or, img_K_nat)

    print("=== Primary config: 3D-axial INTERLEAVED, txt p=TXT_START+i, img (h,w) row-major ===")
    print(f"[primary] global cossim:  txt_Q={cos_g_txt_Q:.6f}  txt_K={cos_g_txt_K:.6f}")
    print(f"[primary]                 img_Q={cos_g_img_Q:.6f}  img_K={cos_g_img_K:.6f}")

    Q_g = cossim(np.concatenate([txt_Q_or, img_Q_or], 0), np.concatenate([txt_Q_nat, img_Q_nat], 0))
    K_g = cossim(np.concatenate([txt_K_or, img_K_or], 0), np.concatenate([txt_K_nat, img_K_nat], 0))
    print(f"[primary] joint global:   Q={Q_g:.6f}  K={K_g:.6f}")
    print()

    # Per-head cossim for img_Q (the most likely offender) and txt_Q.
    ph_img_Q = per_head_cossim(img_Q_or, img_Q_nat)
    ph_txt_Q = per_head_cossim(txt_Q_or, txt_Q_nat)
    print(f"[primary] per-head img_Q min={min(ph_img_Q):.6f} max={max(ph_img_Q):.6f} "
          f"mean={np.mean(ph_img_Q):.6f}")
    print(f"[primary] per-head txt_Q min={min(ph_txt_Q):.6f} max={max(ph_txt_Q):.6f} "
          f"mean={np.mean(ph_txt_Q):.6f}")
    # Worst 3 per-pos for img_Q
    pp_img_Q = per_pos_cossim(img_Q_or, img_Q_nat)
    worst = sorted(range(len(pp_img_Q)), key=lambda i: pp_img_Q[i])[:3]
    print(f"[primary] img_Q worst-3 positions: " + ", ".join(f"pos{w}={pp_img_Q[w]:.6f}" for w in worst))
    pp_txt_Q = per_pos_cossim(txt_Q_or, txt_Q_nat)
    worst = sorted(range(len(pp_txt_Q)), key=lambda i: pp_txt_Q[i])[:3]
    print(f"[primary] txt_Q worst-3 positions: " + ", ".join(f"pos{w}={pp_txt_Q[w]:.6f}" for w in worst))
    print()

    # ---- Alt-config sweep (only relevant if primary RED) ----
    primary_min = min(Q_g, K_g)
    if primary_min >= 0.99:
        print(f"[verdict] PRIMARY GREEN-B: min(Q,K)={primary_min:.6f} >= 0.99")
        print(f"[verdict] RoPE is fine. Bug is upstream in 09 RmsNorm.")
        print(f"[verdict] Recommended next: §5.5.17 RmsNorm oracle.")
        return

    print("=== Alt-config sweep (primary < 0.99) ===")

    def try_config(label, txt_pos_fn, img_pos_fn, axis_order=("t","h","w"),
                   pair_mode="interleaved"):
        """axis_order: permutation of ('t','h','w'). pair_mode in {interleaved, neox}."""
        # rebuild cos/sin tables
        def angles(t_id, h_id, w_id):
            mp = {"t": (omega_t, t_id), "h": (omega_h, h_id), "w": (omega_w, w_id)}
            return np.concatenate([mp[a][1] * mp[a][0] for a in axis_order])

        cos_txt2 = np.zeros((TXT, HD // 2)); sin_txt2 = np.zeros((TXT, HD // 2))
        for i in range(TXT):
            t_id, h_id, w_id = txt_pos_fn(i)
            ang = angles(t_id, h_id, w_id)
            cos_txt2[i] = np.cos(ang); sin_txt2[i] = np.sin(ang)
        cos_img2 = np.zeros((IMG, HD // 2)); sin_img2 = np.zeros((IMG, HD // 2))
        for i in range(IMG):
            t_id, h_id, w_id = img_pos_fn(i)
            ang = angles(t_id, h_id, w_id)
            cos_img2[i] = np.cos(ang); sin_img2[i] = np.sin(ang)

        if pair_mode == "interleaved":
            tq = apply_rope_interleaved(txt_Q_in, cos_txt2, sin_txt2)
            tk = apply_rope_interleaved(txt_K_in, cos_txt2, sin_txt2)
            iq = apply_rope_interleaved(img_Q_in, cos_img2, sin_img2)
            ik = apply_rope_interleaved(img_K_in, cos_img2, sin_img2)
        elif pair_mode == "interleaved_signflip":
            # y_even = x_even*cos - x_odd*sin; y_odd = x_odd*cos + x_even*sin
            def f(x, c, s):
                ce = x[..., 0::2]; co = x[..., 1::2]
                cb = c[:, None, :]; sb = s[:, None, :]
                ye = ce * cb - co * sb
                yo = co * cb + ce * sb
                out = np.empty_like(x); out[..., 0::2] = ye; out[..., 1::2] = yo
                return out
            tq = f(txt_Q_in, cos_txt2, sin_txt2); tk = f(txt_K_in, cos_txt2, sin_txt2)
            iq = f(img_Q_in, cos_img2, sin_img2); ik = f(img_K_in, cos_img2, sin_img2)
        elif pair_mode == "neox":
            def f(x, c, s):
                hd = x.shape[-1]; h = hd // 2
                x1 = x[..., :h]; x2 = x[..., h:]
                cb = c[:, None, :]; sb = s[:, None, :]
                r1 = x1 * cb - x2 * sb
                r2 = x1 * sb + x2 * cb
                return np.concatenate([r1, r2], axis=-1)
            tq = f(txt_Q_in, cos_txt2, sin_txt2); tk = f(txt_K_in, cos_txt2, sin_txt2)
            iq = f(img_Q_in, cos_img2, sin_img2); ik = f(img_K_in, cos_img2, sin_img2)
        else:
            raise ValueError(pair_mode)

        Q = cossim(np.concatenate([tq, iq], 0), np.concatenate([txt_Q_nat, img_Q_nat], 0))
        K = cossim(np.concatenate([tk, ik], 0), np.concatenate([txt_K_nat, img_K_nat], 0))
        print(f"[alt {label}]  Q={Q:.6f}  K={K:.6f}  "
              f"(txt_Q={cossim(tq, txt_Q_nat):.4f} img_Q={cossim(iq, img_Q_nat):.4f} "
              f"txt_K={cossim(tk, txt_K_nat):.4f} img_K={cossim(ik, img_K_nat):.4f})")
        return Q, K

    # Position-fn variants:
    # primary already tried txt p=TXT_START+i, img (h_id, w_id).
    # A1: txt p=i (pe_off=0 with txt_start=0, no max(h,w) shift).
    try_config("A1: txt p=i, img unchanged",
               lambda i: (i, i, i),
               lambda k: (0.0, H_START + k // W_LEN, W_START + k % W_LEN))
    # A2: img with t=h=w=k (no axial split, treat img like txt-style positions).
    try_config("A2: img p=ctx_len+k diagonal",
               lambda i: (TXT_START + i, TXT_START + i, TXT_START + i),
               lambda k: (256 + k, 256 + k, 256 + k))
    # A3: img positions from row 256 onward (max_txt_seq=256 default offset).
    #     ctx_len=256 means img rows are at pe rows 256..256+IMG. pos within img-block
    #     is k = r*w_len+c with k=0..63 → still r=0, c=0..63. Same as primary.
    #     Skip: equivalent to primary (img_pos_fn is already row-major within img block).

    # Pair-mode variants (in case pair convention is different):
    print("[alt pair conv]")
    try_config("B1: NEOX half-rotation",
               lambda i: (TXT_START + i, TXT_START + i, TXT_START + i),
               lambda k: (0.0, H_START + k // W_LEN, W_START + k % W_LEN),
               pair_mode="neox")
    try_config("B2: interleaved sign-flipped",
               lambda i: (TXT_START + i, TXT_START + i, TXT_START + i),
               lambda k: (0.0, H_START + k // W_LEN, W_START + k % W_LEN),
               pair_mode="interleaved_signflip")

    # Axis-order variants (in case t/h/w slot order in head_dim is different):
    print("[alt axis order]")
    try_config("C1: order=(h,w,t)",
               lambda i: (TXT_START + i, TXT_START + i, TXT_START + i),
               lambda k: (0.0, H_START + k // W_LEN, W_START + k % W_LEN),
               axis_order=("h","w","t"))
    try_config("C2: order=(w,h,t)",
               lambda i: (TXT_START + i, TXT_START + i, TXT_START + i),
               lambda k: (0.0, H_START + k // W_LEN, W_START + k % W_LEN),
               axis_order=("w","h","t"))

    # txt_start variants:
    print("[alt txt_start]")
    try_config("D1: txt_start=0 (p=i)",
               lambda i: (i, i, i),
               lambda k: (0.0, H_START + k // W_LEN, W_START + k % W_LEN))
    try_config("D2: txt_start=256",
               lambda i: (256 + i, 256 + i, 256 + i),
               lambda k: (0.0, H_START + k // W_LEN, W_START + k % W_LEN))

    # img coord variants:
    print("[alt img coords]")
    try_config("E1: img unscaled (h_start=0,w_start=0)",
               lambda i: (TXT_START + i, TXT_START + i, TXT_START + i),
               lambda k: (0.0, k // W_LEN, k % W_LEN))
    # Maybe img is laid out by column-major
    try_config("E2: img col-major",
               lambda i: (TXT_START + i, TXT_START + i, TXT_START + i),
               lambda k: (0.0, H_START + k % H_LEN, W_START + k // H_LEN))

    print()
    print(f"[verdict] primary global Q={Q_g:.6f} K={K_g:.6f}")
    if max(Q_g, K_g) < 0.95:
        print("[verdict] GREEN-A — RoPE bug confirmed. See alt-config table for likely fix.")
    elif primary_min >= 0.95:
        print("[verdict] AMBER — partial drift, both RoPE and RmsNorm potentially involved.")
    else:
        print("[verdict] AMBER — partial drift.")


if __name__ == "__main__":
    main()
