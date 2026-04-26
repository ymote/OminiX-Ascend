#!/usr/bin/env python3
"""5.5.31 bit-exact img_mod_1 weight bisect — vectorized."""
import os, numpy as np
from gguf import GGUFReader, dequantize, GGMLQuantizationType

GGUF_PATH = '/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf'
TARGET_BLOCKS = [0, 1, 2, 16, 30, 59]


def cossim(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def engine_repack_dequant(raw, K, N):
    """Vectorized version of repack_q4_0_upload + dequant via engine view.

    Engine packed-nibble view:
      lin = n * K + k    (col-major over n, row-major within row over k)
      byte = packed_w[lin // 2]
      nibble = (byte & 0x0f) if (lin & 1) == 0 else (byte >> 4) & 0x0f
      signed = nibble ^ 0x08 then interpreted as signed-from-bias-0   <-- engine
      <equivalently> signed = nibble - 8 (if you read directly), since the
        engine writes (u ^ 0x08) and the kernel reads as int4-signed.
      scale = scales[k // 32, n]   (engine scale view: [BLK, N] row-major)
      w[k, n] = signed * scale

    GGUF source layout (per row n):
      row = src[n * BLK * 18 : (n+1) * BLK * 18]
      for each block b:
        d_f16 = row[b*18 : b*18+2]
        qs    = row[b*18+2 : b*18+18]   (16 bytes → 32 nibbles)
        for j in 0..15:
          u_lo = qs[j] & 0x0f       → element k0 = b*32 + j
          u_hi = (qs[j] >> 4) & 0x0f → element k1 = b*32 + j + 16
    """
    QK = 32; BLK = K // QK; BB = 18
    src = np.frombuffer(raw, dtype=np.uint8).reshape(N, BLK, BB)
    # scales: [N, BLK] f16
    scales_f16 = src[:, :, 0:2].copy().view(np.uint16).reshape(N, BLK)
    scales = scales_f16.view(np.float16).astype(np.float32)  # [N, BLK]

    qs = src[:, :, 2:18]  # [N, BLK, 16]
    # extract low and high nibbles → indices in [0..15]
    u_lo = qs & 0x0f          # [N, BLK, 16] → element k = b*32 + j
    u_hi = (qs >> 4) & 0x0f   # [N, BLK, 16] → element k = b*32 + j + 16

    # signed via XOR 0x08, interpreted as signed 4-bit two's complement.
    # Engine writes (u ^ 0x08) and reads back as signed-int4. Equivalent dequant
    # value: (u - 8) * scale  (since (u ^ 0x08) reinterpreted as signed-int4 ==
    # u - 8 for u in [0..15]).
    s_lo = u_lo.astype(np.int8) - 8   # [N, BLK, 16]
    s_hi = u_hi.astype(np.int8) - 8

    # Assemble per-block 32 element dequant: first 16 are s_lo, next 16 are s_hi
    # * scale[N, BLK]
    sc = scales[:, :, None]  # [N, BLK, 1]
    blk_lo = s_lo.astype(np.float32) * sc  # [N, BLK, 16]
    blk_hi = s_hi.astype(np.float32) * sc  # [N, BLK, 16]
    # Concatenate along last dim: [N, BLK, 32]
    blk = np.concatenate([blk_lo, blk_hi], axis=-1)
    # Reshape to [N, K]
    W_NK = blk.reshape(N, BLK * QK)
    # Engine logical layout is [K, N]; transpose
    return W_NK.T.copy()  # [K, N]


def main():
    r = GGUFReader(GGUF_PATH)
    found = {}
    for t in r.tensors:
        for b in TARGET_BLOCKS:
            tn = f'transformer_blocks.{b}.img_mod.1.weight'
            if t.name == tn: found[b] = t

    print(f'{"blk":>4} {"dtype":>6} {"K":>5} {"N":>6} {"absmax":>9} {"o_vs_c":>10} {"o_vs_e":>10} {"c_vs_e":>10} {"max|o-e|":>11}')
    print('-' * 92)
    for b in TARGET_BLOCKS:
        t = found.get(b)
        if not t: print(f'{b:>4}: MISSING'); continue
        K, N = int(t.shape[0]), int(t.shape[1])

        oracle_NK = dequantize(t.data, t.tensor_type).astype(np.float32)
        if oracle_NK.shape == (N, K):
            oracle_KN = oracle_NK.T.copy()
        elif oracle_NK.shape == (K, N):
            oracle_KN = oracle_NK
        else:
            oracle_KN = oracle_NK.reshape(K, N)

        cli_KN = oracle_KN  # CLI uses ggml-quants.c → identical to gguf-py

        absmax = float(np.abs(oracle_KN).max())

        if t.tensor_type == GGMLQuantizationType.Q4_0:
            raw = bytes(t.data.tobytes()) if hasattr(t.data, 'tobytes') else bytes(t.data)
            eng_KN = engine_repack_dequant(raw, K, N)
            ovsc = cossim(oracle_KN, cli_KN)
            ovse = cossim(oracle_KN, eng_KN)
            cvse = cossim(cli_KN, eng_KN)
            mxd = float(np.abs(oracle_KN - eng_KN).max())
            print(f'{b:>4} {"Q4_0":>6} {K:>5} {N:>6} {absmax:>9.4f} {ovsc:>10.6f} {ovse:>10.6f} {cvse:>10.6f} {mxd:>11.4e}')
        else:
            print(f'{b:>4} {t.tensor_type.name:>6} {K:>5} {N:>6} {absmax:>9.4f} {1.0:>10.6f} {1.0:>10.6f} {1.0:>10.6f}        N/A  (Q5_K→F16 fallback)')


if __name__ == '__main__':
    main()
