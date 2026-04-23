# Q2.1 Fallback Tensor Audit — Qwen-Image-Edit-2509-Q4_0.gguf

**Author:** Agent QIE-Q2.1-ENUMERATE
**Date:** 2026-04-22
**Subject:** enumeration + classification of the 150 GGUF tensors that miss the Q4_0-resident path in `load_matmul_weight_upload` and fall through to `dequant_upload_f16`, inflating the Q2.1 smoke HBM peak to ~9.51 GiB of F16 fallback weights.
**Source GGUF:** `/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf` on ac03 (11,928,271,392 B = 11.11 GiB; 1,933 tensors total; SHA / mtime: 2026-04-23 06:54).
**Reader:** `gguf.GGUFReader` (Python, `/home/ma-user/anaconda3/envs/PyTorch-2.7.1/bin/python3`).
**Engine code under test:** `tools/qwen_image_edit/native/image_diffusion_engine.cpp` — `load_matmul_weight_upload` (lines 432-459), Q4_0 repack path at `repack_q4_0_to_wqbmmv3` (line 291), fallback branch at line 454 (`stats.f16_fallback_tensors += 1; return dequant_upload_f16(...)`).
**Q2.1 smoke RED commit:** `7b568524`.

---

## §1. Executive summary

Despite the filename declaring `Q4_0`, the exporter applied a **mixed-precision quantization schedule**. Exactly **150 matmul-weight tensors** arrive in non-Q4_0 formats and therefore bypass the compressed-on-device path, landing on HBM as F16 via `dequant_upload_f16`. Totals:

| Category | Count | Stored in GGUF | On HBM as F16 |
|---|---:|---:|---:|
| **Q4_1** (layer 1-58 `mlp.net.2` down-projections) | **116** | 2,610.0 MiB | **8.156 GiB** |
| **Q5_K** (layer 0 + layer 59, ALL matmuls) | **28** | 445.5 MiB | **1.266 GiB** |
| **BF16** (6 global projections: img_in, txt_in, time\_1/\_2, norm_out, proj_out) | **6** | 77.2 MiB | **0.075 GiB** |
| **TOTAL fallback** | **150** | 3,132.75 MiB (3.06 GiB) | **9.497 GiB** |

The 9.497 GiB F16 fallback matches the Q2.1 smoke RED receipt (9.51 GiB) within rounding (difference: 1D-bias inclusion in the engine counter; not material). **No F32 fallthrough**: all 1087 F32 tensors in the GGUF are 1D biases/norms that route through the separate `dequant_upload_f{16,32}` calls, not the matmul path — they are fine, total ~21 MiB.

**Key pattern (exporter's llama.cpp-style quant schedule):**
- **Layer 0 + Layer 59** are treated as "sensitive" and get Q5_K across ALL 14 matmul weights (attn × 8 + mods × 2 + mlp × 4).
- **Layers 1-58** are Q4_0 EXCEPT the two FFN-down projections (`img_mlp.net.2.weight`, `txt_mlp.net.2.weight`), which the exporter bumped to Q4_1 (58 layers × 2 = 116 tensors).
- **Six globals** (I/O projections) stayed in BF16.

This is standard behaviour for most llama.cpp GGUF quantizers (`llama-quantize`, `convert_hf_to_gguf.py`, `stable-diffusion.cpp`) — they refuse to uniformly Q4_0 all weights and bump "important" tensors (first/last layer, FFN-down) to higher precision. **The GGUF is not actually Q4_0-uniform; the name is misleading.**

---

## §2. Full 150-tensor table

Column legend:
- `dtype` — GGUF storage quant type (via `GGMLQuantizationType`)
- `stored_bytes` — bytes in GGUF (what's read)
- `f16_bytes` — bytes on HBM today via `dequant_upload_f16` (= numel × 2)
- `class` — SMALL-F16 / LARGE-QUANTIZED / LARGE-F16 per §3 rules

| name | shape (K×N) | dtype | stored_bytes | f16_bytes | class |
|---|---|---|---:|---:|---|
| transformer_blocks.0.attn.add_k_proj.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.0.attn.add_q_proj.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.0.attn.add_v_proj.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.0.attn.to_add_out.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.0.attn.to_k.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.0.attn.to_out.0.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.0.attn.to_q.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.0.attn.to_v.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.0.img_mlp.net.0.proj.weight | 3072x12288 | Q5_K | 25,952,256 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.0.img_mlp.net.2.weight | 12288x3072 | Q5_K | 25,952,256 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.0.img_mod.1.weight | 3072x18432 | Q5_K | 38,928,384 | 113,246,208 | LARGE-QUANTIZED |
| transformer_blocks.0.txt_mlp.net.0.proj.weight | 3072x12288 | Q5_K | 25,952,256 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.0.txt_mlp.net.2.weight | 12288x3072 | Q5_K | 25,952,256 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.0.txt_mod.1.weight | 3072x18432 | Q5_K | 38,928,384 | 113,246,208 | LARGE-QUANTIZED |
| transformer_blocks.1.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.1.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.2.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.2.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.3.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.3.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.4.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.4.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.5.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.5.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.6.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.6.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.7.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.7.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.8.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.8.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.9.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.9.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.10.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.10.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.11.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.11.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.12.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.12.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.13.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.13.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.14.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.14.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.15.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.15.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.16.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.16.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.17.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.17.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.18.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.18.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.19.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.19.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.20.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.20.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.21.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.21.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.22.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.22.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.23.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.23.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.24.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.24.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.25.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.25.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.26.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.26.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.27.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.27.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.28.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.28.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.29.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.29.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.30.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.30.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.31.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.31.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.32.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.32.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.33.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.33.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.34.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.34.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.35.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.35.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.36.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.36.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.37.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.37.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.38.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.38.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.39.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.39.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.40.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.40.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.41.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.41.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.42.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.42.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.43.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.43.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.44.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.44.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.45.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.45.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.46.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.46.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.47.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.47.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.48.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.48.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.49.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.49.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.50.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.50.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.51.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.51.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.52.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.52.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.53.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.53.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.54.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.54.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.55.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.55.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.56.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.56.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.57.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.57.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.58.img_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.58.txt_mlp.net.2.weight | 12288x3072 | Q4_1 | 23,592,960 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.59.attn.add_k_proj.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.59.attn.add_q_proj.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.59.attn.add_v_proj.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.59.attn.to_add_out.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.59.attn.to_k.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.59.attn.to_out.0.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.59.attn.to_q.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.59.attn.to_v.weight | 3072x3072 | Q5_K | 6,488,064 | 18,874,368 | LARGE-QUANTIZED |
| transformer_blocks.59.img_mlp.net.0.proj.weight | 3072x12288 | Q5_K | 25,952,256 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.59.img_mlp.net.2.weight | 12288x3072 | Q5_K | 25,952,256 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.59.img_mod.1.weight | 3072x18432 | Q5_K | 38,928,384 | 113,246,208 | LARGE-QUANTIZED |
| transformer_blocks.59.txt_mlp.net.0.proj.weight | 3072x12288 | Q5_K | 25,952,256 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.59.txt_mlp.net.2.weight | 12288x3072 | Q5_K | 25,952,256 | 75,497,472 | LARGE-QUANTIZED |
| transformer_blocks.59.txt_mod.1.weight | 3072x18432 | Q5_K | 38,928,384 | 113,246,208 | LARGE-QUANTIZED |
| img_in.weight | 64x3072 | BF16 | 393,216 | 393,216 | SMALL-F16 |
| norm_out.linear.weight | 3072x6144 | BF16 | 37,748,736 | 37,748,736 | LARGE-F16 |
| proj_out.weight | 3072x64 | BF16 | 393,216 | 393,216 | SMALL-F16 |
| time_text_embed.timestep_embedder.linear_1.weight | 256x3072 | BF16 | 1,572,864 | 1,572,864 | LARGE-F16 |
| time_text_embed.timestep_embedder.linear_2.weight | 3072x3072 | BF16 | 18,874,368 | 18,874,368 | LARGE-F16 |
| txt_in.weight | 3584x3072 | BF16 | 22,020,096 | 22,020,096 | LARGE-F16 |

(Row count verified: 150.)

---

## §3. Classification rules

- **SMALL-F16** (< 1 MB stored AND < 1M elements) — acceptable to keep as F16 on HBM. Cost is negligible.
- **LARGE-QUANTIZED** — GGUF stores these in a non-Q4_0 quant format (Q4_1, Q5_K, …). Native repack path needed if we want them compressed on HBM; otherwise they must be dequanted to F16 per-tensor.
- **LARGE-F16** — GGUF stores as F16 or BF16 directly (no quantization applied). Only options: (a) accept the F16 cost, or (b) re-export the GGUF with these quantized, or (c) quantize them on-the-fly at load time (cheap: one-shot SVD-free per-channel scale + round-to-nibble).

**SMALL-F16 threshold rationale:** the smallest BF16 tensors here (`img_in.weight` 64×3072 = 384 KiB, `proj_out.weight` 3072×64 = 384 KiB) are below 1 MiB and contribute < 1 MiB total — not worth a code path. Everything else flagged LARGE-F16 has real HBM cost.

---

## §4. Bytes rollup by category

| Category | Count | Stored in GGUF | F16 on HBM today |
|---|---:|---:|---:|
| SMALL-F16 (BF16 img_in + proj_out) | 2 | 768 KiB | **0.75 MiB** |
| LARGE-F16 (BF16 time_\*, txt_in, norm_out, linear_2) | 4 | 77.2 MiB - 0.75 MiB = ~76.4 MiB | **~76.4 MiB** |
| LARGE-QUANTIZED Q4_1 | 116 | 2,610.0 MiB | **8,156 MiB = 8.156 GiB** |
| LARGE-QUANTIZED Q5_K | 28 | 445.5 MiB | **1,266 MiB = 1.266 GiB** |
| **TOTAL (all 150)** | **150** | **3,132.75 MiB = 3.06 GiB** | **9,497 MiB = 9.50 GiB** |

Sanity check against Q2.1 smoke RED receipt (9.51 GiB): ✓ within rounding; tiny residual comes from engine accounting also including F16 biases (~80 MiB) in `f16_weight_bytes`.

---

## §5. Projected HBM peak after adding native repack paths

**On-HBM layout recipe (each Linear repacked to match the existing Q4_0 layout convention):**

| Source quant | On-HBM storage (per-element avg) | Notes |
|---|---|---|
| Q4_0 (today) | 0.5 B packed + 0.0625 B scale = **0.5625 B** | `[K/32, N]` F16 scales, K×N/2 packed nibbles |
| Q4_1 (new path) | 0.5 B packed + 0.0625 B scale + 0.0625 B min = **0.625 B** | adds F16 min per block-of-32 vs Q4_0 |
| Q5_K (new path) | ~0.6875 B | 5-bit + per-super-block scale/min (stored-equivalent; repack simply keeps GGUF layout or re-tiles to 5-bit+scale tables) |
| BF16 (no compression) | **2.0 B** | bit-convert to F16 identical bytes; no savings |

**Projected bytes under Option (a) — add native repack for Q4_1 and Q5_K, keep BF16 as F16:**

| Category | Count | HBM bytes today (F16) | HBM bytes after native repack | Δ saved |
|---|---:|---:|---:|---:|
| Q4_1 (116 tensors, 4.38 G elems) | 116 | 8.156 GiB | **2.549 GiB** (= numel × 0.625) | **-5.607 GiB** |
| Q5_K (28 tensors, 679 M elems) | 28 | 1.266 GiB | **0.435 GiB** (= stored bytes kept in-place / 5.5 bpw repack) | **-0.831 GiB** |
| BF16 (6 tensors, 40.5 M elems) | 6 | 0.075 GiB | **0.075 GiB** (no change) | 0 |
| **Fallback totals** | **150** | **9.497 GiB** | **3.059 GiB** | **-6.438 GiB** |

Full init-peak projection (integrates engine's `stats_` breakdown; Q4_0 resident path unchanged):

| Bucket | Today (Q2.1 RED) | After Opt (a) |
|---|---:|---:|
| Q4_0 packed (696 tensors, 15.3 G elems × 0.5 B) | 7.14 GiB | 7.14 GiB |
| Q4_0 scales (15.3 G elems × 0.0625 B) | 0.89 GiB | 0.89 GiB |
| Q4_1 native packed + scale/min (0.625 B/elem) | — | 2.55 GiB |
| Q5_K native (~0.687 B/elem keep-stored) | — | 0.44 GiB |
| F16 fallback weights (current 150 tensors) | 9.50 GiB | 0.075 GiB (BF16 only) |
| F16 biases (1D) | ~0.08 GiB | ~0.08 GiB |
| F32 RMSNorm gammas | ~0.02 GiB | ~0.02 GiB |
| RoPE pe | small | small |
| Scratch (fwd buffers) | ~1.3 GiB | ~1.3 GiB |
| **Init peak** | **~18.9 GiB** (over 9 GiB gate) | **~12.4 GiB** (still over 9 GiB gate — see §7) |

Note: even Q4_0 resident alone (696 tensors) costs **8.03 GiB** on HBM. The 9 GiB Q2.1 gate was set against an incorrect "true Q4_0-uniform" assumption; see §7 for re-scope implications.

---

## §6. Per-tensor-pattern summary

| Weight-suffix (per block) | Affected layers | Quant type | # tensors |
|---|---|---|---:|
| `attn.{to_q,to_k,to_v,to_out.0,add_q_proj,add_k_proj,add_v_proj,to_add_out}.weight` | layers 0 and 59 only | Q5_K | 16 |
| `img_mod.1.weight`, `txt_mod.1.weight` | layers 0 and 59 only | Q5_K | 4 |
| `img_mlp.net.0.proj.weight`, `txt_mlp.net.0.proj.weight` | layers 0 and 59 only | Q5_K | 4 |
| `img_mlp.net.2.weight`, `txt_mlp.net.2.weight` (FFN-down) | layers 0, 59 | Q5_K | 4 |
| `img_mlp.net.2.weight`, `txt_mlp.net.2.weight` (FFN-down) | layers 1-58 | **Q4_1** | **116** |
| `img_in.weight`, `txt_in.weight` | global (1 each) | BF16 | 2 |
| `time_text_embed.timestep_embedder.linear_{1,2}.weight` | global | BF16 | 2 |
| `norm_out.linear.weight`, `proj_out.weight` | global | BF16 | 2 |
| **TOTAL** | | | **150** |

**Group-size alignment** (necessary for native repack feasibility):
- All 116 Q4_1 tensors: K ∈ {12288} ✓ divisible by 32
- All 28 Q5_K tensors: K ∈ {3072, 12288, 18432} — wait: Q5_K super-block is 256, so need K%256==0. 3072=256·12 ✓, 12288=256·48 ✓, 18432=256·72 ✓.
- All K dims are group-aligned; no awkward edge cases.

---

## §7. Recommendation

**Adopt Option (a): extend the engine with native Q4_1 and Q5_K repack paths** (and keep BF16 as F16 bit-convert). Reasoning:

1. **The GGUF cannot be re-exported cheaply.** This file was produced by `stable-diffusion.cpp`/`llama.cpp`-family quantizer with a hand-tuned schedule (sensitive-layer Q5_K + FFN-down Q4_1) that mirrors their established quality/size tradeoff for diffusion transformers. Forcing Q4_0-uniform would **degrade end-to-end quality measurably** (the exporter flagged these as needing higher precision for a reason). Producing a genuinely-Q4_0-uniform GGUF also requires rerunning the exporter on the original weights (HF cache on ac01 or remote) — a multi-hour pipeline plus validation — while delivering worse output quality than the mixed-quant we already have.

2. **Q4_1 repack is a small code change.** Q4_1 block layout (`block_q4_1`, `ggml-common.h`) is structurally **identical to Q4_0 plus one extra F16 `m` (min) field per block-of-32**. The current `repack_q4_0_to_wqbmmv3` (`image_diffusion_engine.cpp:291`) can be generalised to `repack_q4x_to_wqbmmv3` with ~30 lines of added code, producing two buffers per tensor instead of two: packed-nibbles, F16-scale, F16-min. The forward-path matmul kernel needs to add one broadcast-add per block-of-32 after the scale-multiply (dequant formula: `x = s·(nib - 8) + m` for Q4_0; `x = s·nib + m` for Q4_1 — conceptually the same "apply scale then add bias-per-block", already inside the scale loop).

3. **Q5_K repack is medium-complexity but high-payoff.** Q5_K is a super-block format (256 elements/super-block with 16-element sub-blocks). The engine can either (a) keep the GGUF bytes as-is on HBM and dequant on-the-fly inside the matmul kernel (simplest — ~0.55 GiB cost vs 1.27 GiB F16), or (b) re-tile into a 5-bit + per-group-32 scale layout consistent with Q4_0/Q4_1 (cleaner forward kernel, same HBM cost). Either saves ~0.83 GiB at the cost of a new aclnn kernel path.

4. **BF16 path stays F16 (no change).** 6 tensors, 75 MiB total — not worth extra code.

5. **Projected init peak after (a): ~12.4 GiB.** Still over the current 9 GiB Q2.1 smoke gate. Therefore the gate itself must be re-scoped: the 9 GiB figure assumed Q4_0-uniform which this GGUF never was. Realistic Q2.1 gate given the exporter's mixed-quant schedule is **≤ 13 GiB init peak** (adds ~0.5 GiB slack for fragmentation). This sits comfortably under the 32 GiB 910B4 HBM with Phase-3 scratch headroom (~15-18 GiB usable).

**Fallback option (c) — accept 9 GiB F16 fallback and re-scope Phase 1 gate** — is acceptable for a smoke milestone but leaves 6.4 GiB on the table permanently; this HBM will be needed for Phase-3 activations, CFG duplication, and aclgraph workspace. Recommend only if Q4_1 repack can't land in this sprint.

**Option (b) — re-export Q4_0-uniform GGUF** — not recommended on quality grounds alone; additionally blocks Phase 2.1 for unknown calendar time while the exporter reruns.

### Concrete action items

1. Generalise `repack_q4_0_to_wqbmmv3` → `repack_q4x_to_wqbmmv3` accepting both `GGML_TYPE_Q4_0` and `GGML_TYPE_Q4_1`; add per-block F16 `m` output buffer alongside the existing scale buffer. Dispatcher in `load_matmul_weight_upload` routes both to this path. (~1 day including unit parity test vs ggml's `dequantize_row_q4_1`.)
2. Add a Q5_K repack (`repack_q5k_to_wqbmmv3`) that keeps Q5_K super-block layout on HBM and a matching kernel path. If kernel work is too heavy for this sprint, fall back to "dequant-to-F16 at load" just for the 28 Q5_K tensors (adds only 1.27 GiB — acceptable for Q2.1 exit; schedule Q5_K kernel for a follow-up). (~2 days full kernel, ~2 hours partial.)
3. Leave the 6 BF16 globals routing through `dequant_upload_f16` as today. Stats counter should be renamed `bf16_to_f16_fallback_tensors` to distinguish from the (now-eliminated) Q4_1/Q5_K fallbacks.
4. Update the Q2.1 smoke gate from `< 9 GiB` to `< 13 GiB`, document the mixed-quant rationale in the engine header comment (overwrite the existing "40.86 GB F16-equiv" stanza which was based on the misleading `Q4_0` filename).

### Minimum-viable Phase-2.1 exit (2-3 days)

- Add Q4_1 native repack (eliminates 8.16 GiB of fallback → 2.55 GiB; net save 5.6 GiB).
- Keep Q5_K as F16-fallback for now (1.27 GiB carried cost).
- Keep BF16 as F16-fallback (75 MiB carried cost).
- Projected init peak: **~13.65 GiB**. Re-scope gate to ≤ 14 GiB.

This exits Q2.1 with 144/150 fallback tensors eliminated (96%) and moves Q5_K kernel to a tracked Phase-2.2 follow-up. If HBM headroom becomes tight in Phase 3, revisit Q5_K.

---

## §8. Reproduction recipe

```bash
# On ac03:
python3 << 'EOF'
from gguf import GGUFReader, GGMLQuantizationType
r = GGUFReader('/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf')
type_name = {q.value: q.name for q in GGMLQuantizationType}

matmul_suffixes = [
    'attn.to_q.weight','attn.to_k.weight','attn.to_v.weight','attn.to_out.0.weight',
    'attn.add_q_proj.weight','attn.add_k_proj.weight','attn.add_v_proj.weight','attn.to_add_out.weight',
    'img_mod.1.weight','txt_mod.1.weight',
    'img_mlp.net.0.proj.weight','img_mlp.net.2.weight',
    'txt_mlp.net.0.proj.weight','txt_mlp.net.2.weight',
]
globals_ = [
    'time_text_embed.timestep_embedder.linear_1.weight',
    'time_text_embed.timestep_embedder.linear_2.weight',
    'img_in.weight','txt_in.weight',
    'norm_out.linear.weight','proj_out.weight',
]
expected = set(globals_)
for il in range(60):
    for s in matmul_suffixes:
        expected.add(f'transformer_blocks.{il}.{s}')

fallback = [t for t in r.tensors
            if t.name in expected and type_name[t.tensor_type.value] != 'Q4_0']
print(f'Fallback count: {len(fallback)}')  # expect 150
EOF
```

## §9. Raw GGUF stats (for traceability)

- Total tensors: 1,933
- By type:
  - F32: 1,087 (all 1D biases/norms — routed through `dequant_upload_f{16,32}`, not the matmul path; total ~21 MiB stored)
  - Q4_0: 696 (Q4-resident matmuls — happy path)
  - Q4_1: 116 (fallback — layer 1-58 FFN-down)
  - Q5_K: 28 (fallback — layer 0 and 59, all matmuls)
  - BF16: 6 (fallback — 6 globals)
- Matmul tensors expected by engine (60 layers × 14 per-layer + 6 globals): 846
  - Q4_0 resident: 696
  - Fallback: 150 ✓
  - Missing: 0 ✓
