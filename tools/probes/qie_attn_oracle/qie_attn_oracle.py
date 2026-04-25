#!/usr/bin/env python3
"""
Q2.4.5.5.16 — Pure-F32 numpy oracle for QIE-Edit DiT block-0 joint attention.

Reads RoPE-applied Q/K (substep 10) and V (substep 08) for both txt and img
streams from the native dump dir, computes joint scaled-dot-product attention
in pure F32 (Q·K^T/sqrt(HD) → softmax → ·V), and cosine-compares the result
against the native FIA output dump (substep 11).

Layout contract (verified via image_diffusion_engine.cpp dump call sites):
  - per-stream dumps shape: [seq, NH*HD] row-major = [seq, NH, HD] flat
  - joint sequence = [txt(32), img(64)] concat on seq axis (txt at offset 0)
  - NH=24, HD=128, hidden=NH*HD=3072
  - FIA layout = BSND, B=1, S=seq_total=96
  - 11_attn_out_{txt,img} are raw FIA output, BEFORE to_out_0 / to_add_out
  - default native path: QIE_ATTN_SOFTMAX_F32=0 → scale=1/sqrt(HD), no kv_trick

Decision matrix:
  GREEN-A: cos < 0.95 globally  → FIA fused-kernel drift (escalate to FIA wrapper)
  GREEN-B: cos >= 0.99          → FIA OK; bug is upstream V/Q/K (escalate §5.5.17/18)
  AMBER  : 0.95 <= cos < 0.99   → mixed; both oracles needed next
  RED    : oracle disagrees w/ self (shape error)
"""

import numpy as np
import os, sys

DUMP = os.environ.get("QIE_DUMPS", "/tmp/qie_dumps_5516")
NH   = 24
HD   = 128
H    = NH * HD          # 3072
TXT  = 32
IMG  = 64
SEQ  = TXT + IMG        # 96


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


def main():
    # Load RoPE-applied Q/K (substep 10) and V (substep 08).
    txt_Q = load_f32("10_txt_Q_rope.f32", TXT * H).reshape(TXT, NH, HD)
    txt_K = load_f32("10_txt_K_rope.f32", TXT * H).reshape(TXT, NH, HD)
    txt_V = load_f32("08_txt_V.f32",      TXT * H).reshape(TXT, NH, HD)
    img_Q = load_f32("10_img_Q_rope.f32", IMG * H).reshape(IMG, NH, HD)
    img_K = load_f32("10_img_K_rope.f32", IMG * H).reshape(IMG, NH, HD)
    img_V = load_f32("08_img_V.f32",      IMG * H).reshape(IMG, NH, HD)

    # Joint = txt || img on seq axis.
    Q = np.concatenate([txt_Q, img_Q], axis=0)  # [SEQ, NH, HD]
    K = np.concatenate([txt_K, img_K], axis=0)
    V = np.concatenate([txt_V, img_V], axis=0)

    # Native split outputs.
    nat_txt = load_f32("11_attn_out_txt.f32", TXT * H).reshape(TXT, NH, HD)
    nat_img = load_f32("11_attn_out_img.f32", IMG * H).reshape(IMG, NH, HD)
    NATIVE = np.concatenate([nat_txt, nat_img], axis=0)  # [SEQ, NH, HD]

    # Reorganize for per-head computation: [NH, SEQ, HD].
    Qh = Q.transpose(1, 0, 2).astype(np.float64)
    Kh = K.transpose(1, 0, 2).astype(np.float64)
    Vh = V.transpose(1, 0, 2).astype(np.float64)

    scale = 1.0 / np.sqrt(HD)

    # Scores: [NH, SEQ_q, SEQ_k]
    scores = np.einsum("hsi,hti->hst", Qh, Kh) * scale

    # Softmax in F64 along last axis.
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)

    # Attention out: [NH, SEQ, HD] → transpose back to [SEQ, NH, HD]
    oracle_h = np.einsum("hst,htv->hsv", p, Vh)
    oracle = oracle_h.transpose(1, 0, 2).astype(np.float32)

    # Diagnostics ---------------------------------------------------------
    diff = oracle.astype(np.float64) - NATIVE.astype(np.float64)
    g_cos = cossim(oracle, NATIVE)
    mean_abs = float(np.abs(diff).mean())
    max_abs  = float(np.abs(diff).max())

    print(f"=== QIE attn oracle — {DUMP} ===")
    print(f"shape: SEQ={SEQ} (txt={TXT} || img={IMG})  NH={NH}  HD={HD}")
    print(f"global cos      = {g_cos:.6f}")
    print(f"mean abs diff   = {mean_abs:.6f}")
    print(f"max  abs diff   = {max_abs:.6f}")
    print(f"oracle  stats   : mean_abs={np.abs(oracle).mean():.4f}  max_abs={np.abs(oracle).max():.4f}")
    print(f"native  stats   : mean_abs={np.abs(NATIVE).mean():.4f}  max_abs={np.abs(NATIVE).max():.4f}")

    # Per-stream cossim.
    cos_txt = cossim(oracle[:TXT], NATIVE[:TXT])
    cos_img = cossim(oracle[TXT:], NATIVE[TXT:])
    print(f"per-stream cos  : txt={cos_txt:.6f}  img={cos_img:.6f}")

    # Per-head cossim.
    print("per-head cos:")
    head_cos = []
    for h in range(NH):
        oh = oracle[:, h, :]
        nh = NATIVE[:, h, :]
        c = cossim(oh, nh)
        head_cos.append((h, c))
        print(f"  head {h:2d} cos = {c:.4f}")

    head_cos.sort(key=lambda x: x[1])
    print("\nworst-3 heads:")
    for h, c in head_cos[:3]:
        print(f"  head {h:2d} cos = {c:.4f}")
    print("best-3 heads:")
    for h, c in head_cos[-3:][::-1]:
        print(f"  head {h:2d} cos = {c:.4f}")

    # Decision.
    print()
    if g_cos < 0.5:
        verdict = "RED — oracle/native catastrophically diverge (likely shape/layout mismatch in oracle)"
    elif g_cos < 0.95:
        verdict = "GREEN-A — FIA fused-kernel drift confirmed; fix is in the FIA wrapper / aclnnFIA dtype/scale args"
    elif g_cos < 0.99:
        verdict = "AMBER — partial drift; need both V-proj (§5.5.17) and RoPE (§5.5.18) oracles"
    else:
        verdict = "GREEN-B — FIA kernel OK; real bug is upstream (V-proj or RoPE on Q/K)"
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
