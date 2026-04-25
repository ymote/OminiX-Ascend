#!/usr/bin/env python3
"""
Q2.4.5.4l — QKV-projection round-trip discriminator.

Updated mission (vs. handoff brief): the bisect blamed `repack_q4_0_upload`
+ WQBMMv3 chain at engine.cpp:3410-3428, but `transformer_blocks.0.attn.to_q.weight`
in `Qwen-Image-Edit-2509-Q4_0.gguf` is actually GGUF type 13 (Q5_K), NOT
Q4_0.  load_matmul_weight_upload() falls through to dequant_upload_f16 →
upload F32->F16 → aclnnMm fallback (engine.cpp:1756-1814).  The Q4_0 repack
path is bypassed entirely on block 0, so a Q4_0 round-trip cannot exercise
the failing codepath.  All seven attn-projection weights of block 0
(to_q/k/v, add_q/k/v, to_out.0) are Q5_K; Q5_K appears only in blocks 0 and
59 — every other block is Q4_0.

This probe does the broader round-trip:

  Y_ref = X @ W.T + b              (PyTorch Linear semantics)

with X = cpu_05_img_mod1.f32 (CPU side input — already validated cos≈0.99
vs native), W = ground-truth dequant of GGUF Q5_K block-0 to_q.weight,
b = bias as stored in GGUF.  Then compares Y_ref against:
  - cpu_08_img_Q.f32   (CPU reference dump — should be cos≈1.0 — sanity)
  - 08_img_Q.f32       (native dump — RED, cos = -0.0014 vs cpu_08_img_Q)

Plus several layout variants to pinpoint whether native is computing
X @ W.T  (correct), X @ W (no transpose), -X @ W.T (sign flip), or
something else.

Usage:
  python3 tools/probes/qie_q4_repack_check.py \\
      [--gguf /home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf] \\
      [--cpu-dir /tmp/qie_block0_outputs] \\
      [--native-dir /tmp/qie_block0_inputs] \\
      [--tensor transformer_blocks.0.attn.to_q.weight]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "gguf-py"))

import gguf  # noqa: E402
from gguf import GGMLQuantizationType, GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402


def load_f32(path: Path, expected_elems: int | None = None) -> np.ndarray:
    sz = path.stat().st_size
    n = sz // 4
    if expected_elems is not None and n != expected_elems:
        raise ValueError(f"{path}: {n} elems != expected {expected_elems}")
    arr = np.fromfile(str(path), dtype=np.float32, count=n)
    if arr.size != n:
        raise ValueError(f"{path}: short read {arr.size}/{n}")
    return arr


def cos_per_row(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = np.maximum(na * nb, 1e-30)
    return np.sum(a * b, axis=1) / denom


def stats(a: np.ndarray, b: np.ndarray, label: str) -> None:
    diff = a - b
    cos = cos_per_row(a, b)
    print(
        f"  {label:42s} | "
        f"cos[mean={cos.mean():+.4f} min={cos.min():+.4f} p10={np.percentile(cos,10):+.4f} max={cos.max():+.4f}] | "
        f"abs[max_a={np.abs(a).max():.3e} max_b={np.abs(b).max():.3e}] | "
        f"max|diff|={np.abs(diff).max():.3e}"
    )


def fetch_tensor_dequant(reader: GGUFReader, name: str) -> tuple[np.ndarray, str, list[int]]:
    t = next((x for x in reader.tensors if x.name == name), None)
    if t is None:
        raise KeyError(f"tensor '{name}' not in GGUF")
    qtype = GGMLQuantizationType(int(t.tensor_type))
    shape = list(int(s) for s in t.shape)
    if qtype == GGMLQuantizationType.F32:
        flat = t.data.view(np.float32).reshape(shape[::-1])  # [N, K] → [N rows, K cols]
        return flat.astype(np.float32, copy=True), qtype.name, shape
    if qtype == GGMLQuantizationType.F16:
        flat = t.data.view(np.float16).reshape(shape[::-1]).astype(np.float32)
        return flat, qtype.name, shape
    if qtype == GGMLQuantizationType.BF16:
        # gguf-py: BF16 stored as uint16
        u = t.data.view(np.uint16).astype(np.uint32) << 16
        flat = u.view(np.float32).reshape(shape[::-1]).astype(np.float32, copy=True)
        return flat, qtype.name, shape
    # Quantised path — gguf-py dequantize() takes the raw bytes view
    deq = dequantize(t.data, qtype).astype(np.float32)
    # gguf-py dequantize returns a 1-D-flat F32 array of length n_elements.
    # Reshape to GGUF logical shape: ne[0]=K is innermost, ne[1]=N is rows.
    K, N = shape[0], shape[1] if len(shape) > 1 else 1
    if deq.size != K * N:
        raise ValueError(
            f"{name}: dequant size {deq.size} != K*N {K*N}; shape={shape}"
        )
    W = deq.reshape(N, K)
    return W, qtype.name, shape


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gguf",
        default="/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf",
    )
    ap.add_argument(
        "--cpu-dir",
        default="/tmp/qie_block0_outputs",
        help="CPU-reference dumps (cpu_05_img_mod1.f32, cpu_08_img_Q.f32).",
    )
    ap.add_argument(
        "--native-dir",
        default="/tmp/qie_block0_inputs",
        help="Native engine dumps (05_img_mod1.f32, 08_img_Q.f32).",
    )
    ap.add_argument(
        "--tensor",
        default="transformer_blocks.0.attn.to_q.weight",
        help="GGUF tensor name (matrix). Bias is derived by replacing "
        "'.weight' with '.bias'.",
    )
    ap.add_argument("--input-stage", default="05_img_mod1")
    ap.add_argument("--output-stage", default="08_img_Q")
    ap.add_argument("--K", type=int, default=3072)
    ap.add_argument("--N", type=int, default=3072)
    ap.add_argument("--seq", type=int, default=64,
                    help="img_seq for the small-smoke run (default 64).")
    ap.add_argument(
        "--dump-weight",
        default=None,
        help="Optional: save dequant weight matrix [N, K] F32 to this path.",
    )
    args = ap.parse_args()

    print(f"== Q2.4.5.4l Round-trip discriminator ==")
    print(f"GGUF      : {args.gguf}")
    print(f"Tensor    : {args.tensor}")
    print(f"In stage  : {args.input_stage}  (X)")
    print(f"Out stage : {args.output_stage} (Y)")
    print()

    reader = GGUFReader(args.gguf)
    W, qname, wshape = fetch_tensor_dequant(reader, args.tensor)
    print(f"Weight    : type={qname} shape_gguf={wshape} → reshaped[N, K]={W.shape}")
    if W.shape != (args.N, args.K):
        print(
            f"  !! shape mismatch: got {W.shape} expected ({args.N}, {args.K})",
            file=sys.stderr,
        )
        return 2

    bias_name = args.tensor[: -len(".weight")] + ".bias"
    try:
        b, bname, bshape = fetch_tensor_dequant(reader, bias_name)
        # Bias is a 1-D vector — fetch_tensor_dequant reshaped to [1, N] for
        # 1-D shapes; flatten.
        b = b.reshape(-1)
        print(f"Bias      : type={bname} shape={bshape} (flatten {b.shape})")
    except KeyError:
        print(f"Bias      : NOT PRESENT — using zero bias")
        b = np.zeros(args.N, dtype=np.float32)
        bname = "zero"

    if args.dump_weight:
        W.tofile(args.dump_weight)
        print(f"Dumped W  : {args.dump_weight}  ({W.nbytes} bytes)")

    cpu_dir = Path(args.cpu_dir)
    nat_dir = Path(args.native_dir)

    # Load X (input to projection) — prefer the CPU side since it's used in
    # the CPU forward and is what the gguf Linear sees. Fall back to native.
    x_cpu_path = cpu_dir / f"cpu_{args.input_stage}.f32"
    x_nat_path = nat_dir / f"{args.input_stage}.f32"
    if x_cpu_path.exists():
        X_cpu = load_f32(x_cpu_path).reshape(args.seq, args.K)
        print(f"X (cpu)   : {x_cpu_path}  shape={X_cpu.shape}")
    else:
        X_cpu = None
    X_nat = load_f32(x_nat_path).reshape(args.seq, args.K)
    print(f"X (nat)   : {x_nat_path}  shape={X_nat.shape}")

    Y_cpu_path = cpu_dir / f"cpu_{args.output_stage}.f32"
    Y_nat_path = nat_dir / f"{args.output_stage}.f32"
    Y_cpu = load_f32(Y_cpu_path).reshape(args.seq, args.N)
    Y_nat = load_f32(Y_nat_path).reshape(args.seq, args.N)
    print(f"Y (cpu)   : {Y_cpu_path}  shape={Y_cpu.shape}")
    print(f"Y (nat)   : {Y_nat_path}  shape={Y_nat.shape}")
    print()

    # Reference forward pass — pure NumPy in F32. Matches PyTorch Linear:
    #   Y[m,n] = sum_k X[m,k] * W[n,k] + b[n]   == X @ W.T + b
    X_for_ref = X_cpu if X_cpu is not None else X_nat
    Y_ref = X_for_ref.astype(np.float64) @ W.T.astype(np.float64) + b.astype(np.float64)
    Y_ref = Y_ref.astype(np.float32)
    print(f"Y_ref = X @ W.T + b   shape={Y_ref.shape}  abs.max={np.abs(Y_ref).max():.3e}")

    # Variants — each represents a possible mis-orientation.
    # 1. No-transpose (X @ W) — meaning view-w[K,N] semantics interpreted as
    #    [N, K] direct.  Computes y[m,n] = sum_k X[m,k] * W[k,n]  — i.e.
    #    treats GGUF row n as the column.  For a SQUARE weight this still
    #    runs and produces a different result.
    Y_no_t = (X_for_ref.astype(np.float64) @ W.astype(np.float64) + b.astype(np.float64)).astype(np.float32)
    # 2. Sign flip — bias-8 inversion would flip every nibble's sign, but a
    #    sign-flipped Q4_0 + correct scale would yield -Y_ref + b.  We only
    #    flip the matmul piece; bias is unaffected.
    Y_neg  = (-(X_for_ref.astype(np.float64) @ W.T.astype(np.float64)) + b.astype(np.float64)).astype(np.float32)
    # 3. No bias — debug whether bias mis-add is the culprit.
    Y_nob  = (X_for_ref.astype(np.float64) @ W.T.astype(np.float64)).astype(np.float32)

    print()
    print("--- CPU reference dump (cpu_08_img_Q) vs hypotheses ---")
    stats(Y_cpu, Y_ref,   "cpu_Y  vs  X@W.T + b   (expected cos≈1.0)")
    stats(Y_cpu, Y_no_t,  "cpu_Y  vs  X@W   + b")
    stats(Y_cpu, Y_neg,   "cpu_Y  vs  -X@W.T + b")
    stats(Y_cpu, Y_nob,   "cpu_Y  vs  X@W.T (no bias)")

    print()
    print("--- Native dump (08_img_Q) vs hypotheses ---")
    stats(Y_nat, Y_ref,   "nat_Y  vs  X@W.T + b   (PyTorch Linear)")
    stats(Y_nat, Y_no_t,  "nat_Y  vs  X@W   + b   (NO transpose)")
    stats(Y_nat, Y_neg,   "nat_Y  vs  -X@W.T + b  (sign flip)")
    stats(Y_nat, Y_nob,   "nat_Y  vs  X@W.T       (no bias)")

    # Direct cpu vs native for context
    print()
    print("--- Native vs CPU reference (the original RED) ---")
    stats(Y_nat, Y_cpu,   "nat_Y  vs  cpu_Y       (= §5.5.11 line)")

    print()
    print("--- Verdict ---")
    cos_correct = cos_per_row(Y_nat, Y_ref).mean()
    cos_no_t    = cos_per_row(Y_nat, Y_no_t).mean()
    cos_neg     = cos_per_row(Y_nat, Y_neg).mean()
    if cos_correct > 0.95:
        print("  GREEN: native matches X @ W.T + b — projection is CORRECT.")
        print("         Bug is elsewhere (input alignment? dump-stride bug?)")
    elif cos_no_t > 0.95:
        print("  RED  : native matches X @ W + b (NO transpose).")
        print("         The aclnnMm weight view orientation is wrong:")
        print("         engine.cpp:1764-1765 declares w_strides=(1,K) but should")
        print("         be (N,1) for the GGUF row-major [N, K] buffer to be")
        print("         interpreted directly without transpose.")
    elif cos_neg > 0.95:
        print("  RED  : native matches -X @ W.T + b (sign flip).")
        print("         Some intermediate cast/sign issue.")
    else:
        print("  RED  : native does NOT match any simple-orientation hypothesis.")
        print(
            f"         best_cos: correct={cos_correct:+.4f} "
            f"no_t={cos_no_t:+.4f} neg={cos_neg:+.4f}"
        )
        print("         Inspect: dtype cast (F16->BF16->F16), dispatch_matmul_ "
              "internal scratch buffers, or activation pre-cast to BF16 path.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
