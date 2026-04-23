# QIE Q2.1 Load Smoke — ac03

Agent: QIE-Q2.1-SMOKE
Commit under test: `ee452dd9` (`feat(qwen_image_edit): Q2.1 weight-resident load path — replaces F16 preload`)
Host: ac03 (cohabited with A4c re-baseline agent, HBM lock honored)
GGUF: `/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf` (12 GiB on disk)
Env: `GGML_CANN_QUANT_BF16=on`

## Verdict: **RED** (HBM budget)

- Build: **CLEAN** — no patches needed. `cmake --build build-w1 --target qwen_image_edit_native -j 8` finished with zero warnings attributable to Q2.1 (only pre-existing ggml/cann shared-lib links + ccache notice).
- Load:  `init_from_gguf` returned **true**, `is_ready()=true`, zero "missing tensor" aborts, clean teardown, HBM lock properly released.
- Budget: Peak init HBM **17.74 GiB** — **>12 GiB RED threshold** (expected ≤ 9 GiB).

## Receipts

| Field | Expected (task spec) | Actual | Delta |
|---|---|---|---|
| tensors_uploaded | ~870 | 1933 | +122% (includes duplicated per-block tensors in 60-block DiT) |
| q4_tensors | ~720 | **696** | -3% (close) |
| q4_weight_bytes | ~5.11 GiB | **7.14 GiB** | +40% |
| q4_scale_bytes | ~1.27 GiB | **0.89 GiB** | -30% |
| f16_fallback_tensors | 0 (or small) | **150** | NOT small |
| f16_weight_bytes (biases + fallbacks) | — | **9.51 GiB** | dominant HBM cost |
| f32_weight_bytes (RMSNorm gammas) | — | 0.13 MiB | expected |
| rope pe bytes | — | 2.12 MiB | expected |
| scratch | — | 0.20 GiB | expected |
| **Peak init HBM** | **≤ 9 GiB** | **17.74 GiB** | **+97%** (RED) |
| dequant/repack wall | — | 4.4 ms | fast |
| total init wall | — | 101.5 s | most time in GGUF IO + F16 fallback dequant |

`ready_ = true`, `init_from_gguf -> true`, engine dtor clean.

## Finding

The load-path binary works: Q4_0 tensors that hit `load_matmul_weight_upload()` DO route through `repack_q4_0_upload()` and stay compressed (7.14 GiB W4 + 0.89 GiB scale tracks the probe budget reasonably — scale is lighter than projected, weight is heavier but plausible given repack nibble packing at `K*N/2` bytes).

The **blocker is the F16 fallback path**: 150 tensors totaling **9.51 GiB** took the `dequant_upload_f16` branch inside `load_matmul_weight_upload` (`tools/qwen_image_edit/native/image_diffusion_engine.cpp:432-458`). The fallback fires when a GGUF tensor has `type != GGML_TYPE_Q4_0`. The obvious interpretation: the Qwen-Image-Edit-2509-Q4_0 GGUF keeps a large class of tensors (likely the non-Linear matrices — time-embedding projections, image-in/text-in projection weights, modulation projections, potentially token_embd — or the norm/bias variants that are routed through `load_matmul_weight_upload` rather than `try_dequant_upload_f16`) in F16 rather than Q4_0. 150 × avg 65 MiB each ≈ 9.5 GiB.

150 fallbacks crosses the YELLOW threshold ("> 10 F16 fallbacks") *and* the peak HBM crosses the RED threshold, so the overall verdict is **RED**.

## Build output (condensed)

```
-- ggml commit:  ee452dd
-- qwen_image_edit: native engine ENABLED (qwen_image_edit_native) — Phase 1 scaffold
[100%] Building CXX object tools/qwen_image_edit/CMakeFiles/qwen_image_edit_native.dir/native/image_diffusion_engine.cpp.o
[100%] Building CXX object tools/qwen_image_edit/CMakeFiles/qwen_image_edit_native.dir/native/main_native.cpp.o
[100%] Linking CXX executable ../../bin/qwen_image_edit_native
[100%] Built target qwen_image_edit_native
```

No build-fix patch needed; `/tmp/qie_q21_buildfix.patch` was not created.

## Raw log (ac03 `/tmp/qie_q21_smoke.log`, 15 lines verbatim)

```
[qie_native] rope pe: pos=4352 head_dim/2=64 bytes=2228224 (layout=[seq, hd/2, 2, 2] F16)
[qie_native] Phase 2.1 init OK: device=0 gguf=/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf
[qie_native]   tensors uploaded:   1933 (Q4-resident=696, F16-fallback=150)
[qie_native]   Q4 weight bytes:    7662993408 (7.14 GiB)
[qie_native]   Q4 scale  bytes:    957874176 (0.89 GiB)
[qie_native]   F16 weight bytes:   10208759936 (9.51 GiB)  [biases + F16-fallback weights]
[qie_native]   F32 weight bytes:   137216 (0.13 MiB)  [RMSNorm gammas]
[qie_native]   RoPE pe bytes:      2228224 (2.12 MiB)
[qie_native]   Scratch bytes:      218595328 (0.20 GiB)
[qie_native]   Peak init HBM:      19050588288 (17.74 GiB)  [Q2.1 smoke gate: < 9 GiB]
[qie_native]   Dequant/repack wall: 4.4 ms
[qie_native]   Total init wall:    101505.7 ms
[qie_native] acquired HBM lock /tmp/ac03_hbm_lock
[qie_native] init_from_gguf returned ready=true (101505.7 ms). Phase 1 scaffold — nothing else to do yet.
[qie_native] released HBM lock /tmp/ac03_hbm_lock
```

## Next step (NOT in this smoke)

Phase 2 (forward-pass matmul dispatch) is gated. Before it makes sense to land:
1. Identify which 150 tensors fell back. Add per-tensor debug logging (or a one-shot `QIE_LOG` inside the `f16_fallback_tensors += 1` branch at `image_diffusion_engine.cpp:454`) and re-run the smoke to enumerate names.
2. Decide per-class: is each fallback (a) a genuinely-small tensor that should stay F16 (biases, projection norms, time embeddings), (b) a large Linear weight whose GGUF export is non-Q4_0 and needs a second quant-class support path (Q8_0? Q5_K?), or (c) exporter oversight that should be filed upstream against the Qwen-Image-Edit-2509-Q4_0 GGUF producer.
3. Re-gate: once the fallback byte-count drops below ~1 GiB (norms + biases only), the ~9 GiB peak-HBM target should fall out naturally (7.14 Q4 + 0.89 scale + ~1 F16 small + 0.2 scratch ≈ 9.4 GiB).

HBM lock discipline: honored. Acquired after A4c released, released before exit.
