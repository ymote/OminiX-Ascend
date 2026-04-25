#!/usr/bin/env python3
"""
Q2.4.5.4l — QKV matmul oracle.

Critical context (uncovered during Step 1):
  block-0 attention weights in `Qwen-Image-Edit-2509-Q4_0.gguf` are
  **Q5_K**, NOT Q4_0 (28 Q5_K tensors total — block-0 only). So
  `repack_q4_0_upload` is bypassed for the §5.5.11 substep dump and
  the F16-fallback `aclnnMm` path runs instead.

  Therefore the §5.5.11 RED at `08_img_Q cos = -0.0014` cannot be
  blamed on the Q4_0 repack, the WQBMMv3 dispatch, or the Q4 scale.
  The bug lives in EITHER:
    (a) `dequant_upload_f16` mishandling Q5_K (unlikely — same trait
        used everywhere), OR
    (b) `dispatch_matmul_`'s F16 fallback (aclnnMm) tensor-view
        orientation, OR
    (c) the QKV invocation site mis-wiring inputs/outputs.

This probe takes the brute-force decisive path: dequantize the Q5_K
weight to F32 in Python (canonical reference), do the QKV matmul on
the SAME `05_img_mod1` activations the native engine consumed (read
from /tmp/qie_block0_inputs/), and compare against the native engine's
`08_img_Q.f32` output dump (read from the same dir).

Usage
-----
  python3 tools/probes/qie_q2454l_repack_probe/qkv_matmul_oracle.py

Inputs read from /tmp/qie_block0_inputs/:
  05_img_mod1.f32          [img_seq, H] F32 — activations going into to_q
  07_txt_mod1.f32          [txt_seq, H] F32 — activations going into add_q
  08_img_Q.f32 / 08_img_K.f32 / 08_img_V.f32  — what engine produced
  08_txt_Q.f32 / 08_txt_K.f32 / 08_txt_V.f32  — what engine produced

Verdict
-------
  GREEN  — Python oracle matches engine output → bug is downstream of
           QKV projection, OR the dump path itself is correct and the
           §5.5.11 RED was an artefact (unlikely).
  RED    — Python oracle DISAGREES with engine output → bug is in the
           projection itself. The script then reports the failure
           pattern (sign-flip / row-swap / col-swap / scaled / random)
           to localise the orientation issue.
"""

import argparse
import os
import sys
import numpy as np


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
    """Dequantize a GGUF tensor to a [N, K] F32 numpy array."""
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
        out = raw.view(np.float32).reshape(N, K)
        return out.astype(np.float32)
    if qt == GGMLQuantizationType.F16:
        out = raw.view(np.float16).reshape(N, K)
        return out.astype(np.float32)
    if qt == GGMLQuantizationType.BF16:
        # bf16 raw u16 → upcast to f32 by zero-extending mantissa.
        u16 = raw.view(np.uint16).astype(np.uint32)
        u32 = u16 << 16
        return u32.view(np.float32).reshape(N, K).astype(np.float32)

    # Use gguf-py's __Quant subclasses for k-quants.
    qcls_map = {q.qtype: q for q in quants.__Quant.__subclasses__()
                if hasattr(q, "qtype")}
    qcls = qcls_map.get(qt)
    if qcls is None:
        raise RuntimeError(f"no Python dequant for type {qt}")

    # Get block size in elements + bytes per block.
    from gguf.constants import GGML_QUANT_SIZES
    elems_per_blk, bytes_per_blk = GGML_QUANT_SIZES[qt]
    n_total = N * K
    n_blocks = n_total // elems_per_blk
    blocks = raw.reshape(n_blocks, bytes_per_blk)
    dq = qcls.dequantize_blocks(blocks)  # [n_blocks, elems_per_blk]
    return dq.reshape(N, K).astype(np.float32)


def load_bias(gguf_path: str, name: str) -> np.ndarray | None:
    """Load a 1-D bias as F32. Returns None if missing.

    gguf-py returns F32 tensors as np.float32 directly (t.data.dtype already
    set), so we just take a copy. For F16/BF16 we upcast.
    """
    from gguf import GGUFReader, GGMLQuantizationType

    reader = GGUFReader(gguf_path)
    for t in reader.tensors:
        if t.name == name:
            data = np.asarray(t.data)
            if t.tensor_type == GGMLQuantizationType.F32:
                return data.astype(np.float32).reshape(-1).copy()
            if t.tensor_type == GGMLQuantizationType.F16:
                return data.astype(np.float32).reshape(-1)
            if t.tensor_type == GGMLQuantizationType.BF16:
                u16 = data.view(np.uint16).astype(np.uint32)
                return (u16 << 16).view(np.float32).reshape(-1).astype(np.float32)
            raise RuntimeError(f"bias dtype unsupported: {t.tensor_type}")
    return None


def load_f32_dump(path: str, expected_elems: int) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size != expected_elems:
        raise RuntimeError(
            f"{path}: got {arr.size} elems, expected {expected_elems}")
    return arr


def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float64).ravel()
    bf = b.astype(np.float64).ravel()
    na = np.linalg.norm(af)
    nb = np.linalg.norm(bf)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(af, bf) / (na * nb))


def report_match(label: str, A: np.ndarray, B: np.ndarray, threshold: float = 0.99) -> bool:
    diff = (A.astype(np.float64) - B.astype(np.float64))
    cs = cos_sim(A, B)
    abs_max_a = float(np.max(np.abs(A)))
    abs_max_b = float(np.max(np.abs(B)))
    diff_max = float(np.max(np.abs(diff)))
    rel = diff_max / max(1e-12, max(abs_max_a, abs_max_b))
    ratio_max = abs_max_b / max(1e-12, abs_max_a)
    ok = cs > threshold and rel < 0.20  # F16 round-trip can be ~10% rel
    verdict = "PASS" if ok else "FAIL"
    print(f"  {label:36s}  cos={cs:>10.6f}  ratio_max={ratio_max:>6.3f}  "
          f"absmax_oracle={abs_max_a:>9.3e}  absmax_native={abs_max_b:>9.3e}  "
          f"max_abs_diff={diff_max:>9.3e}  [{verdict}]")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gguf",
                    default="/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf")
    p.add_argument("--inputs",
                    default="/tmp/qie_block0_inputs",
                    help="directory containing 05_img_mod1.f32 etc native dumps")
    p.add_argument("--img-seq", type=int, default=64)
    p.add_argument("--txt-seq", type=int, default=32)
    p.add_argument("--H", type=int, default=3072)
    p.add_argument("--block", type=int, default=0,
                    help="transformer block index (default 0 — has Q5_K weights)")
    args = p.parse_args()

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    find_gguf_py(repo_root)

    H = args.H
    img_seq = args.img_seq
    txt_seq = args.txt_seq
    bidx = args.block

    print(f"GGUF       : {args.gguf}")
    print(f"inputs     : {args.inputs}")
    print(f"H={H}  img_seq={img_seq}  txt_seq={txt_seq}  block={bidx}")
    print()

    # Load activations from engine dumps.
    img_in = load_f32_dump(os.path.join(args.inputs, "05_img_mod1.f32"),
                            img_seq * H).reshape(img_seq, H)
    txt_in = load_f32_dump(os.path.join(args.inputs, "07_txt_mod1.f32"),
                            txt_seq * H).reshape(txt_seq, H)

    print(f"  05_img_mod1 absmax  = {float(np.max(np.abs(img_in))):.4e}")
    print(f"  07_txt_mod1 absmax  = {float(np.max(np.abs(txt_in))):.4e}")
    print()

    # Load to_q/k/v + add_q/k/v weights & biases.
    proj_specs = [
        ("img_Q", "to_q",        img_in,  "08_img_Q.f32",  img_seq * H, img_seq),
        ("img_K", "to_k",        img_in,  "08_img_K.f32",  img_seq * H, img_seq),
        ("img_V", "to_v",        img_in,  "08_img_V.f32",  img_seq * H, img_seq),
        ("txt_Q", "add_q_proj",  txt_in,  "08_txt_Q.f32",  txt_seq * H, txt_seq),
        ("txt_K", "add_k_proj",  txt_in,  "08_txt_K.f32",  txt_seq * H, txt_seq),
        ("txt_V", "add_v_proj",  txt_in,  "08_txt_V.f32",  txt_seq * H, txt_seq),
    ]

    all_pass = True
    for label, attn_subname, X, dump_name, expected_dump_elems, M in proj_specs:
        wname = f"transformer_blocks.{bidx}.attn.{attn_subname}.weight"
        bname = f"transformer_blocks.{bidx}.attn.{attn_subname}.bias"
        print(f"--- {label}  ({wname}) ---")
        W = load_dequant_weight(args.gguf, wname)  # [N, K] F32
        b = load_bias(args.gguf, bname)             # [N] F32 or None
        if W.shape != (H, H):
            print(f"  unexpected weight shape {W.shape}")
            all_pass = False
            continue
        # Mimic the engine's F16 cast on weights (dequant_upload_f16
        # calls upload_f32_as_f16 which does fp32→fp16 round-trip).
        W_f16 = W.astype(np.float16).astype(np.float32)
        Y_oracle = X.astype(np.float32) @ W_f16.T  # [M, H]
        if b is not None:
            # Engine casts bias to F16 too (upload_f32_as_f16 path).
            b_f16 = b.astype(np.float16).astype(np.float32)
            Y_oracle = Y_oracle + b_f16
        # Engine writes out F16 (most QKV calls use ACL_FLOAT16 default).
        Y_oracle_f16 = Y_oracle.astype(np.float16).astype(np.float32)

        Y_native = load_f32_dump(os.path.join(args.inputs, dump_name),
                                   expected_dump_elems).reshape(M, H)
        ok = report_match(f"{label} oracle vs native", Y_oracle_f16, Y_native)
        all_pass = all_pass and ok

        # If FAIL, run cheap orientation diagnostics:
        if not ok:
            # 1. Try W (no transpose): Y = X @ W
            Y_alt1 = (X.astype(np.float32) @ W_f16).astype(np.float16).astype(np.float32)
            cs1 = cos_sim(Y_alt1, Y_native)
            # 2. Try X^T inadvertent: Y = X^T @ W^T (only valid if M == H)
            cs2 = float("nan")
            if M == H:
                Y_alt2 = (X.astype(np.float32).T @ W_f16.T).astype(np.float16).astype(np.float32)
                cs2 = cos_sim(Y_alt2, Y_native)
            # 3. Try W ROW-WISE F16 cast vs blockwise: probably won't differ
            #    much; skip for now.
            print(f"      diag: cos(Y=X@W)        = {cs1:>10.6f}")
            if M == H:
                print(f"      diag: cos(Y=X^T@W^T)    = {cs2:>10.6f}")
            # 4. Per-row cosine: which output row(s) are most off?
            row_cs = []
            for m in range(M):
                row_cs.append(cos_sim(Y_oracle_f16[m], Y_native[m]))
            row_cs = np.array(row_cs)
            print(f"      per-row cos: min={row_cs.min():.4f}  "
                  f"median={np.median(row_cs):.4f}  max={row_cs.max():.4f}  "
                  f"std={row_cs.std():.4f}")
            # 5. Per-col cosine
            col_cs = []
            for n in range(H):
                col_cs.append(cos_sim(Y_oracle_f16[:, n], Y_native[:, n]))
            col_cs = np.array(col_cs)
            print(f"      per-col cos: min={col_cs.min():.4f}  "
                  f"median={np.median(col_cs):.4f}  max={col_cs.max():.4f}  "
                  f"std={col_cs.std():.4f}")
        print()

    if all_pass:
        print("VERDICT: GREEN — engine matmul output matches Python oracle "
              "for all 6 QKV projections.")
        print("         The QKV-projection matmul math is CORRECT.")
        print("         §5.5.11 RED must be in the dump path or compare_block0.py.")
        sys.exit(0)
    else:
        print("VERDICT: RED — at least one QKV projection diverges from the "
              "Python oracle. See per-projection diagnostics above to "
              "identify the orientation bug.")
        sys.exit(2)


if __name__ == "__main__":
    main()
