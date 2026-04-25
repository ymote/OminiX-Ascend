#!/usr/bin/env python3
"""
Q2.4.5.4l Step 1 — Q4_0 repack round-trip probe.

Goal
----
Discriminate among three suspects for the QKV-projection RED
(`08_img_Q cos = -0.0014`, see `docs/qie_q2_phase4_smoke.md` §5.5.11):

  S1. `repack_q4_0_upload` (engine.cpp:256-373) emits the wrong [K, N]
      packed-INT4 buffer (or wrong scale buffer).
  S2. `dispatch_matmul_` tensor-view orientation / transposeB.
  S3. Scale tensor [BLK, N] vs [N, BLK] orientation.

This script targets S1 + S3 head-on, *purely on the host*:

  - Loads the raw Q4_0 bytes for `transformer_blocks.0.attn.to_q.weight`
    from the GGUF (gguf-py reader; no ggml dequant pass).
  - Replicates `repack_q4_0_upload` byte-for-byte in NumPy: emits the
    same `out_w` (K*N/2 INT4-packed bytes) and `out_s` ([BLK, N] F16
    scales) host buffers the engine uploads to NPU.
  - Builds the WQBMMv3 weight semantics from those buffers: read nibble
    at linear index `n*K + k`, two's-complement signed in [-8,+7],
    multiply by `scale[b, n]` where `b = k/32`.  Result is W_repack[K, N].
  - Builds the canonical CPU reference: dequantize each block as
    `(u - 8) * d`, lay out as W_ref[N, K] (GGUF native row-major).
  - Compares W_repack vs W_ref.T at three levels:
      a) max-abs / mean-abs absolute element diff;
      b) cos_sim (W_repack.flatten, W_ref.T.flatten);
      c) per-row matmul probe: y_repack = e_k @ W_repack should equal
         W_ref[:, k] for every k (this is what WQBMMv3 actually computes
         when Y = X @ W with W = W_repack[K, N]).

Verdict
-------
- Both diff and cos_sim near-perfect → S1 is INNOCENT (repack is correct
  and scale layout is correct). Bug must be S2 (dispatch orientation).
- One or more bands disagree → S1 is GUILTY. The output prints the
  exact failure mode (sign-flip / row-shuffle / scale-shuffle / etc.)
  and recommends the patch.

Usage
-----
  cd ~/work/OminiX-Ascend
  python3 tools/probes/qie_q2454l_repack_probe/repack_roundtrip.py \
      --gguf /home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf

Defaults to that GGUF path on ac03 if --gguf omitted.
"""

import argparse
import os
import sys
import struct
import numpy as np


# Block-Q4_0 layout (matches ggml/src/ggml-common.h:170-174):
#   2 bytes ggml_half scale `d`, then 16 bytes of nibble quants `qs[0..15]`.
QK4_0 = 32
BLK_BYTES = 18


def find_gguf_py(repo_root: str) -> None:
    """Add gguf-py to sys.path if not already importable."""
    try:
        import gguf  # noqa: F401
        return
    except ImportError:
        pass
    cand = os.path.join(repo_root, "gguf-py")
    if os.path.isdir(cand):
        sys.path.insert(0, cand)


def read_q4_tensor_raw(gguf_path: str, tensor_name: str):
    """Return (raw_bytes: np.ndarray uint8, ne0_K: int, ne1_N: int)."""
    from gguf import GGUFReader, GGMLQuantizationType

    reader = GGUFReader(gguf_path)
    tensor = None
    for t in reader.tensors:
        if t.name == tensor_name:
            tensor = t
            break
    if tensor is None:
        raise RuntimeError(f"tensor not found: {tensor_name}")
    if tensor.tensor_type != GGMLQuantizationType.Q4_0:
        raise RuntimeError(f"tensor not Q4_0: type={tensor.tensor_type}")

    # GGUF tensor.shape is in *logical* dims order [ne0, ne1, ...] where
    # ne[0] is the inner (K, blocked) dim and ne[1] is N (output channels).
    # tensor.data has shape (n_blocks_total, BLK_BYTES) as uint8 in newer
    # gguf-py; we just want the raw flat bytes.
    K = int(tensor.shape[0])
    N = int(tensor.shape[1])
    expected_blocks = (K // QK4_0) * N
    raw = np.asarray(tensor.data, dtype=np.uint8).reshape(-1)
    expected_bytes = expected_blocks * BLK_BYTES
    if raw.size != expected_bytes:
        raise RuntimeError(
            f"raw byte size mismatch: got {raw.size} expected "
            f"{expected_bytes} ({expected_blocks} blocks × {BLK_BYTES} bytes); "
            f"K={K} N={N}")
    return raw, K, N


def cpu_reference_dequant(raw: np.ndarray, K: int, N: int) -> np.ndarray:
    """
    Canonical Q4_0 dequant per `dequantize_row_q4_0` in
    ggml/src/ggml-quants.c:307-325.

    Input layout: raw bytes are `block_q4_0[N][BLK]` row-major where each
    row is one output channel's K elements split into K/QK4_0 blocks.
    Within a block: 2 bytes scale (le ggml_half), then 16 bytes qs[].
    qs[j] low nibble = element j; qs[j] high nibble = element j+16.
    Dequant: y = (u - 8) * d where u is the bias-8 unsigned nibble.

    Returns W_ref of shape [N, K], dtype float32.
    """
    BLK = K // QK4_0
    W = np.zeros((N, K), dtype=np.float32)
    # Each row of W has BLK*BLK_BYTES bytes in the raw buffer.
    raw2 = raw.reshape(N, BLK, BLK_BYTES)
    for n in range(N):
        for b in range(BLK):
            blk = raw2[n, b]
            # le ggml_half
            d = np.frombuffer(blk[0:2].tobytes(), dtype=np.float16)[0]
            qs = blk[2:18]  # 16 bytes
            # Vectorised low/high nibble per byte.
            u_lo = (qs & 0x0f).astype(np.int32)
            u_hi = ((qs >> 4) & 0x0f).astype(np.int32)
            # Element index j → low nibble at b*32+j; high at b*32+j+16.
            base = b * QK4_0
            W[n, base : base + 16] = (u_lo - 8).astype(np.float32) * float(d)
            W[n, base + 16 : base + 32] = (u_hi - 8).astype(np.float32) * float(d)
    return W


def replicate_repack_q4_0_upload(raw: np.ndarray, K: int, N: int):
    """
    Replicate `repack_q4_0_upload` from
    tools/qwen_image_edit/native/image_diffusion_engine.cpp:284-373
    byte-for-byte. Returns:
      out_w   : np.ndarray uint8, len = K*N/2 (packed signed nibbles)
      out_s   : np.ndarray uint16, shape [BLK, N] (F16 bit pattern)
    """
    BLK = K // QK4_0
    out_w = np.zeros(K * N // 2, dtype=np.uint8)
    out_s = np.zeros(BLK * N, dtype=np.uint16)

    raw2 = raw.reshape(N, BLK, BLK_BYTES)
    for n in range(N):
        n_base_nib = n * K  # linear index of element [k=0, n]
        for b in range(BLK):
            blk = raw2[n, b]
            # Scale: 2 le bytes -> uint16
            d_u16 = struct.unpack("<H", blk[0:2].tobytes())[0]
            out_s[b * N + n] = d_u16

            qs = blk[2:18]
            block_nib_base = n_base_nib + b * QK4_0
            for j in range(QK4_0 // 2):  # 0..15
                byte = int(qs[j])
                # bias-8 unsigned -> signed two's-complement nibble.
                s_lo = (byte & 0x0f) ^ 0x08
                s_hi = ((byte >> 4) & 0x0f) ^ 0x08
                lin_lo = block_nib_base + j
                lin_hi = block_nib_base + j + (QK4_0 // 2)
                # write_nib(lin_lo, s_lo)
                if (lin_lo & 1) == 0:
                    out_w[lin_lo // 2] = (out_w[lin_lo // 2] & 0xf0) | (s_lo & 0x0f)
                else:
                    out_w[lin_lo // 2] = (out_w[lin_lo // 2] & 0x0f) | ((s_lo & 0x0f) << 4)
                # write_nib(lin_hi, s_hi)
                if (lin_hi & 1) == 0:
                    out_w[lin_hi // 2] = (out_w[lin_hi // 2] & 0xf0) | (s_hi & 0x0f)
                else:
                    out_w[lin_hi // 2] = (out_w[lin_hi // 2] & 0x0f) | ((s_hi & 0x0f) << 4)
    return out_w, out_s.reshape(BLK, N)


def materialise_repack_view(out_w: np.ndarray, out_s_blkN: np.ndarray,
                            K: int, N: int) -> np.ndarray:
    """
    Build the matrix WQBMMv3 logically computes from out_w + out_s with
    weight view shape [K, N] strides (1, K), i.e. nibble at linear index
    `n*K + k` is element [k, n], scale[b, n] applies to element [k, n]
    where b = k // 32.

    Returns W_repack of shape [K, N], dtype float32.
    """
    BLK = K // QK4_0
    W = np.zeros((K, N), dtype=np.float32)
    # Vectorise across nibble linear indices.
    lin = np.arange(K * N, dtype=np.int64)
    byte_idx = lin // 2
    is_lo = (lin & 1) == 0
    bytes_ = out_w[byte_idx]
    nibs_unsigned = np.where(is_lo, bytes_ & 0x0f, (bytes_ >> 4) & 0x0f)
    # signed two's-complement 4-bit: [0..7] -> 0..7, [8..15] -> -8..-1.
    nibs_signed = np.where(nibs_unsigned >= 8,
                           nibs_unsigned.astype(np.int32) - 16,
                           nibs_unsigned.astype(np.int32))
    # element [k, n] at lin = n*K + k -> n = lin // K, k = lin % K
    n_idx = lin // K
    k_idx = lin % K
    b_idx = k_idx // QK4_0
    # scale[b, n] from out_s_blkN
    scales_f16_bits = out_s_blkN[b_idx, n_idx]
    scales_f32 = np.frombuffer(scales_f16_bits.astype(np.uint16).tobytes(),
                                dtype=np.float16).astype(np.float32)
    vals = nibs_signed.astype(np.float32) * scales_f32
    W[k_idx, n_idx] = vals
    return W


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(af, bf) / (na * nb))


def report(name: str, A: np.ndarray, B: np.ndarray) -> bool:
    """Print element-diff stats and return True iff PASS (cos>0.9999, max<1e-3)."""
    diff = (A.astype(np.float64) - B.astype(np.float64))
    max_abs = float(np.max(np.abs(diff)))
    mean_abs = float(np.mean(np.abs(diff)))
    cs = cos_sim(A, B)
    rel = max_abs / max(1e-12, float(np.max(np.abs(A))))
    ok = cs > 0.9999 and rel < 1e-3
    verdict = "PASS" if ok else "FAIL"
    print(f"  {name:32s}  cos_sim={cs:>10.6f}  max_abs={max_abs:>10.4e}  "
          f"mean_abs={mean_abs:>10.4e}  rel={rel:>9.3e}  [{verdict}]")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf",
                    default="/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf",
                    help="path to Qwen-Image-Edit Q4_0 GGUF")
    p.add_argument("--tensor",
                    default="transformer_blocks.0.attn.to_q.weight",
                    help="GGUF tensor name to round-trip")
    p.add_argument("--out-dir",
                    default="/tmp/qie_q2454l_repack_probe",
                    help="dump directory for repacked buffers + reference matrix")
    args = p.parse_args()

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    find_gguf_py(repo_root)

    print(f"GGUF      : {args.gguf}")
    print(f"tensor    : {args.tensor}")
    print()

    raw, K, N = read_q4_tensor_raw(args.gguf, args.tensor)
    print(f"raw shape : K={K}  N={N}  bytes={raw.size}")
    print(f"           BLK={K//QK4_0}  expected_packed_W_bytes={K*N//2}  "
          f"scale_elems={(K//QK4_0)*N}")
    print()

    # --- Step A: CPU reference dequantization (ground truth) -------------
    print("[A] CPU reference dequant ((u-8)*d, [N, K] layout)…")
    W_ref = cpu_reference_dequant(raw, K, N)
    print(f"    W_ref shape     = {W_ref.shape}  dtype={W_ref.dtype}")
    print(f"    W_ref absmax    = {float(np.max(np.abs(W_ref))):.4e}")
    print(f"    W_ref mean_abs  = {float(np.mean(np.abs(W_ref))):.4e}")
    print()

    # --- Step B: replicate repack_q4_0_upload byte-for-byte -------------
    print("[B] replicating repack_q4_0_upload (host buffers only)…")
    out_w, out_s = replicate_repack_q4_0_upload(raw, K, N)
    print(f"    out_w shape     = {out_w.shape}  dtype={out_w.dtype}")
    print(f"    out_s shape     = {out_s.shape}  dtype={out_s.dtype} "
          f"(F16 bit pattern, BLK x N)")
    print()

    os.makedirs(args.out_dir, exist_ok=True)
    out_w.tofile(os.path.join(args.out_dir, "py_repack_w.bin"))
    out_s.tofile(os.path.join(args.out_dir, "py_repack_s.bin"))
    W_ref.astype(np.float32).tofile(os.path.join(args.out_dir, "py_W_ref_NK.f32"))
    print(f"    dumped to        {args.out_dir}/")
    print()

    # --- Step C: materialise repack view as [K, N] float matrix ---------
    print("[C] materialise WQBMMv3 weight view from out_w + out_s "
          "([K,N] strides (1,K), nibble-signed * scale[b,n])…")
    W_repack = materialise_repack_view(out_w, out_s, K, N)
    print(f"    W_repack shape  = {W_repack.shape}  dtype={W_repack.dtype}")
    print(f"    W_repack absmax = {float(np.max(np.abs(W_repack))):.4e}")
    print()

    # --- Step D: compare ------------------------------------------------
    print("[D] compare W_repack vs W_ref.T (both should be the SAME [K,N] matrix)")
    ok_main = report("W_repack vs W_ref.T", W_repack, W_ref.T)

    # Also try the OTHER possible interpretation of the [K, N] strides
    # (1, K) view: maybe the engine actually emits [N, K] strides (1, N).
    # If THAT matches, the repack is laid out for the wrong shape contract.
    print()
    print("[D2] sanity views (should all FAIL if repack is correct):")
    # If we (mis)interpret out_w as packing in N-major order, lin = k*N + n.
    # That would mean nibble at index k*N+n is element (k, n).
    W_alt_kN = np.zeros((K, N), dtype=np.float32)
    lin = np.arange(K * N, dtype=np.int64)
    byte_idx = lin // 2
    is_lo = (lin & 1) == 0
    bytes_ = out_w[byte_idx]
    nibs_u = np.where(is_lo, bytes_ & 0x0f, (bytes_ >> 4) & 0x0f).astype(np.int32)
    nibs_s = np.where(nibs_u >= 8, nibs_u - 16, nibs_u)
    # alt: lin = k*N + n
    k_alt = lin // N
    n_alt = lin % N
    b_alt = k_alt // QK4_0
    s_bits = out_s[b_alt, n_alt]
    s_f32 = np.frombuffer(s_bits.astype(np.uint16).tobytes(),
                            dtype=np.float16).astype(np.float32)
    W_alt_kN[k_alt, n_alt] = nibs_s.astype(np.float32) * s_f32
    report("W_alt(lin=k*N+n)  vs W_ref.T", W_alt_kN, W_ref.T)

    # alt scale orientation: read scale[n, b] instead of [b, n]
    if (K // QK4_0) == N:
        # Only meaningful if BLK == N — for to_q (3072x3072 with BLK=96) this
        # is FALSE, so we can't run this sanity case. Skip.
        pass
    else:
        # Try interpreting out_s as [N, BLK] row-major (i.e. scale_idx
        # = n*BLK + b instead of b*N + n).
        out_s_NB = out_s.reshape(K // QK4_0, N).T  # [N, BLK] view of same data
        # Wait — the data laid out *as if* it were [N, BLK]. We must rebuild
        # from scratch. Actually out_s is a FLAT length-(BLK*N) buffer.
        flat = out_s.reshape(-1)  # length BLK*N
        # If engine ACTUALLY wrote [N, BLK] row-major, scale[n, b] = flat[n*BLK + b].
        # Re-evaluate the matmul with that interpretation:
        BLK = K // QK4_0
        s_alt_bits = flat[(n_idx := lin // K) * BLK + (k_idx := lin % K) // QK4_0]
        s_alt_f32 = np.frombuffer(s_alt_bits.astype(np.uint16).tobytes(),
                                   dtype=np.float16).astype(np.float32)
        n_idx2 = lin // K
        k_idx2 = lin % K
        nibs_u2 = np.where((lin & 1) == 0, out_w[lin // 2] & 0x0f,
                            (out_w[lin // 2] >> 4) & 0x0f).astype(np.int32)
        nibs_s2 = np.where(nibs_u2 >= 8, nibs_u2 - 16, nibs_u2)
        W_alt_scale = np.zeros((K, N), dtype=np.float32)
        W_alt_scale[k_idx2, n_idx2] = nibs_s2.astype(np.float32) * s_alt_f32
        report("W_repack (scale=[N,BLK]) vs W_ref.T", W_alt_scale, W_ref.T)

    # --- Step E: matmul probe with one-hot inputs ----------------------
    print()
    print("[E] one-hot matmul probe: y = e_k @ W_repack should equal W_ref[:, k] "
          "(K=in feature)")
    # y_k = W_repack[k, :] (a length-N row); should equal W_ref[:, k] (length-N column).
    sample_ks = [0, 1, 16, 17, 31, 32, 33, K // 2, K - 1]
    fail_count = 0
    for k in sample_ks:
        a = W_repack[k, :]
        b = W_ref[:, k]
        cs = cos_sim(a, b)
        ok = cs > 0.9999
        verdict = "PASS" if ok else "FAIL"
        if not ok:
            fail_count += 1
        print(f"    k={k:5d}  cos={cs:>10.6f}  absmax_a={float(np.max(np.abs(a))):.4e}  "
              f"absmax_b={float(np.max(np.abs(b))):.4e}  [{verdict}]")
    print()

    # --- Verdict ---------------------------------------------------------
    if ok_main and fail_count == 0:
        print("VERDICT: GREEN — repack_q4_0_upload buffers are correct end-to-end.")
        print("         Bug is NOT in S1 (repack) or S3 (scale layout).")
        print("         Proceed to Step 2: dispatch_matmul_ orientation probe.")
        sys.exit(0)
    else:
        print("VERDICT: RED — repack/scale layout is wrong. Inspect [D] and [D2] "
              "rows above to identify the failure mode.")
        sys.exit(2)


if __name__ == "__main__":
    main()
