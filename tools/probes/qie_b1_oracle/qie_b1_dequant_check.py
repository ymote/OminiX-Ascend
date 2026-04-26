#!/usr/bin/env python3
"""
Q2.4.5.5.38 sub-investigation — Q4_0 dequant orientation check.

At block 1, oracle 02_img_mod_out cos=0.04 vs engine. At block 0 (Q5_K) it's
cos=1.000. Hypothesis: GGUF Python dequant for Q4_0 produces different
on-disk layout than for Q5_K, OR the engine has a Q4_0-specific bug.

Tests:
  A. weight reshaped as [N, K] (current oracle)        → cos
  B. weight reshaped as [K, N] (transposed)            → cos
  C. weight as [N, K] but matmul x @ W (no transpose)  → cos
  D. block-2 same orientation (sanity, both blocks Q4_0)
"""
import os, sys
import numpy as np

sys.path.insert(0, "/home/ma-user/work/OminiX-Ascend/gguf-py")

from gguf import GGUFReader, GGMLQuantizationType, quants
from gguf.constants import GGML_QUANT_SIZES

GGUF = "/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf"

H = 3072
IMG_SEQ = 8192


def f16r(x):
    return x.astype(np.float16).astype(np.float32)


def cossim(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def dequant_raw(gguf_path, name):
    """Return raw dequantized vector with shape preserved.

    Returns dq_flat (n_total,), shape_kn (= (K_in, N_out) per gguf header order)."""
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
        dq = raw.view(np.float32)
        return dq.astype(np.float32), (K, N), str(qt).split(".")[-1]
    if qt == GGMLQuantizationType.F16:
        dq = raw.view(np.float16).astype(np.float32)
        return dq, (K, N), str(qt).split(".")[-1]
    if qt == GGMLQuantizationType.BF16:
        u16 = raw.view(np.uint16).astype(np.uint32)
        dq = ((u16 << 16).view(np.float32)).astype(np.float32)
        return dq, (K, N), str(qt).split(".")[-1]

    qcls_map = {q.qtype: q for q in quants.__Quant.__subclasses__() if hasattr(q, "qtype")}
    qcls = qcls_map.get(qt)
    elems_per_blk, bytes_per_blk = GGML_QUANT_SIZES[qt]
    n_total = N * K
    n_blocks = n_total // elems_per_blk
    blocks = raw.reshape(n_blocks, bytes_per_blk)
    dq = qcls.dequantize_blocks(blocks).reshape(-1).astype(np.float32)
    return dq, (K, N), str(qt).split(".")[-1]


def load_bias(gguf_path, name):
    reader = GGUFReader(gguf_path)
    for t in reader.tensors:
        if t.name == name:
            data = np.asarray(t.data)
            qt = t.tensor_type
            if qt == GGMLQuantizationType.F16:
                return data.astype(np.float32).reshape(-1)
            return data.astype(np.float32).reshape(-1).copy()
    return None


def load_f32(path, expected_n):
    return np.fromfile(path, dtype=np.float32, count=expected_n)


def test_block(blk):
    base = f"transformer_blocks.{blk}"
    # Use img_mod.1.weight to test mod_out matmul.
    dq, (K, N), qt = dequant_raw(GGUF, f"{base}.img_mod.1.weight")
    b = load_bias(GGUF, f"{base}.img_mod.1.bias")
    print(f"\n--- block {blk}: {base}.img_mod.1.weight qt={qt} (K,N)=({K},{N}) bias_shape={b.shape} ---")

    # Engine inputs/outputs.
    dump_dir = f"/tmp/qie_5536_eng_real/block{blk:02d}"
    silu = load_f32(f"{dump_dir}/01_silu_t_emb.f32", H)
    eng_out = load_f32(f"{dump_dir}/02_img_mod_out.f32", N)  # 6H=18432

    # Variant A: reshape as [N, K] then x @ W.T
    W_NK = dq.reshape(N, K)
    out_A = silu.astype(np.float32) @ f16r(W_NK).T + f16r(b)
    out_A = f16r(out_A)
    cA = cossim(out_A, eng_out)

    # Variant B: reshape as [K, N] (gguf header order) then x @ W
    W_KN = dq.reshape(K, N)
    out_B = silu.astype(np.float32) @ f16r(W_KN) + f16r(b)
    out_B = f16r(out_B)
    cB = cossim(out_B, eng_out)

    # Variant C: reshape as [N, K] no-transpose (intentional layout mistake)
    out_C_raw = dq.reshape(N, K)
    # we cannot do x @ W here since shape mismatch. test x@ W.T as variant A is identity.

    print(f"  variant A [N,K] then x@W.T : cos={cA:.6f}")
    print(f"  variant B [K,N] then x@W   : cos={cB:.6f}")

    # absmax of W in both layouts
    print(f"  W_NK  mean_abs={np.abs(W_NK).mean():.4e} max_abs={np.abs(W_NK).max():.4e}")
    print(f"  silu  mean_abs={np.abs(silu).mean():.4e} max_abs={np.abs(silu).max():.4e}")
    print(f"  eng   mean_abs={np.abs(eng_out).mean():.4e} max_abs={np.abs(eng_out).max():.4e}")
    print(f"  oracleA mean_abs={np.abs(out_A).mean():.4e} max_abs={np.abs(out_A).max():.4e}")
    print(f"  oracleB mean_abs={np.abs(out_B).mean():.4e} max_abs={np.abs(out_B).max():.4e}")


if __name__ == "__main__":
    test_block(0)
    test_block(1)
    test_block(2)
