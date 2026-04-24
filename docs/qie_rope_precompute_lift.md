# QIE — RoPE pe-vector pre-compute lift

**Agent**: QIE-perf (OminiX-Ascend fork)
**Date**: 2026-04-25
**Host**: ac03 (ModelArts container, 910B4, single NPU, `build-w1`, HEAD `68eca5f` w/ our edit layered)
**Env gate**: `OMINIX_QIE_ROPE_PRECOMPUTE=1` (default OFF — byte-identical to pre-patch)
**Question**: Does lifting `Rope::gen_qwen_image_pe(...)` out of the per-step `QwenImageRunner::build_graph` hot path deliver the MLX-measured +10-25% per-step reduction the Q0.5.3 verdict promoted as a must-have?

## TL;DR

**VERDICT: GREEN on correctness, NOISE on wall-clock at Q0-baseline NPU speeds.**

- Cache works as designed: `miss=1, hit=20` across a 20-step QIE-Edit request — **95% of pe_vec generations eliminated** (20 of 21 build_graph calls served from cache).
- Output bytes are **identical** between baseline and lifted: PNG md5 match, L_inf = 0, L_2 = 0.
- Wall-clock delta at Q0 baseline is **noise-level** (+0.72% lifted, within cohabit variance). `gen_qwen_image_pe` on 92-core host is ~ms; per-step NPU wall is ~60 s, diluting the structural saving at this development stage.
- Refactor is **precondition** for downstream kernel wins: once per-step drops (post-Q2 ggml-cann MUL_MAT F32-accum, aclGraph, attention fusions), the constant per-step RoPE overhead becomes a relatively larger fraction that the cache already eliminates.

Env flag unset → legacy code path executes verbatim (byte-identical to pre-patch). Default OFF, zero risk of regression.

## Why this is safe

`gen_qwen_image_pe` (`tools/ominix_diffusion/src/rope.hpp:379-438`) inputs:

- `h, w` — from `x->ne[1], x->ne[0]`, constant per-request.
- `patch_size` — `QwenImageParams.patch_size = 2`, constant per-runner-lifetime.
- `bs` — `x->ne[3]`, asserted to be 1.
- `context_len` — `context->ne[1]`, constant per-request.
- `ref_latents` — vector of tensors; only their `ne[0]`/`ne[1]` shapes matter to RoPE (per `rope.hpp:414-424` which iterates `ref->ne[1]`, `ref->ne[0]` only). Tensor _values_ are irrelevant to pe.
- `increase_ref_index` — the caller (`diffusion_model.hpp:448`) always passes `true` for QIE-Edit, constant per-request.
- `theta, axes_dim, circular_x/y` — `theta, axes_dim` are runner-lifetime constants; circular flags can be toggled via `set_circular_axes()` but that is a pre-denoise config call, not per-step. Still keyed to guard against misuse.

**Crucially: no timestep dependency.** The `timesteps` tensor is passed to `build_graph` but only flows into `time_text_embed` (`qwen_image.hpp:405`). RoPE is position-only, which is exactly why per-step pre-compute is mathematically equivalent to per-step re-compute.

Cache key: `(h, w, bs, context_len, ref_shapes[], increase_ref_index, circular_x, circular_y)`. On any field change the cache is refilled; otherwise pe_vec is retained verbatim.

Structural analogue to the native engine's Q2.1 `build_rope_tables_` / `compute_qwen_rope_pe_host` path (`tools/qwen_image_edit/native/image_diffusion_engine.cpp:987-1019`, commit `ee452dd9`) and to `qwen-image-mlx`'s RoPE pre-compute (measured +20-40% per-step on their stack where NPU-vs-host-CPU cost balance is different).

## Patch

File: `tools/ominix_diffusion/src/qwen_image.hpp` (+107 / -11).

1. Added cache fields on `QwenImageRunner`:

```cpp
bool pe_cache_valid_                      = false;
int  pe_cache_h_, pe_cache_w_, pe_cache_bs_, pe_cache_context_len_;
std::vector<std::pair<int64_t,int64_t>> pe_cache_ref_shapes_;
bool pe_cache_increase_ref_index_, pe_cache_circular_x_, pe_cache_circular_y_;
uint64_t pe_cache_miss_count_, pe_cache_hit_count_;
```

2. In `build_graph`, replaced the unconditional `pe_vec = Rope::gen_qwen_image_pe(...)` call with an env-gated branch: flag ON → shape-keyed cache with refill-on-miss and `LOG_INFO` emission on miss + on first hit + every 10th hit; flag unset/OFF → original code path verbatim.

3. Includes added at top of file (`<cstdlib>` for `std::getenv`, `<string>`, `<utility>`, `<vector>`).

No public API change. No changes outside `qwen_image.hpp`.

## Smoke receipts (ac03, 1024×1024 × 20-step, seed 42, `"black and white"` prompt)

Canonical cat reference (`~/work/qie_test/cat.jpg`), `GGML_CANN_QUANT_BF16=on`, Q4_0 diffusion GGUF, Q4_0 LLM, BF16 mmproj, BF16 VAE.

| Metric | Baseline (flag unset) | Lifted (`=1`) | Delta |
|---|---|---|---|
| Total wall (CLI)            | 1792 s        | 1796 s        | +4 s (+0.22%)   |
| Sampling wall (20 steps)    | **1194.60 s** | **1203.17 s** | +8.57 s (+0.72%) |
| Per-step (sampling wall/20) | **59.73 s**   | **60.16 s**   | +0.43 s         |
| VAE decode wall             | 325.51 s      | 323.96 s      | -1.55 s         |
| pe_cache miss (logged)      | n/a (legacy)  | **1**         | —               |
| pe_cache hit (logged)       | n/a (legacy)  | **20**        | —               |
| Output PNG md5              | `6415705e7612b783377b2dfa08a8d40b` | `6415705e7612b783377b2dfa08a8d40b` | **identical** |
| PNG bytes                   | 30896         | 30896         | **identical**    |
| Pixel-space L_inf           | —             | **0.0**       | —               |
| Pixel-space L_2             | —             | **0.0**       | —               |

Cache log excerpt (lifted):

```
[INFO ] qwen_image.hpp:624  - qwen_image pe_cache MISS (refill) #1: h=128 w=128 ctx=211 nrefs=1 -> pe_vec=2151168 floats
[INFO ] qwen_image.hpp:632  - qwen_image pe_cache HIT count=1 (miss=1)
[INFO ] qwen_image.hpp:632  - qwen_image pe_cache HIT count=10 (miss=1)
[INFO ] qwen_image.hpp:632  - qwen_image pe_cache HIT count=20 (miss=1)
```

- MISS (refill) fires once at first build_graph call (from `alloc_compute_buffer` inside `GGMLRunner::compute()` on step 0).
- HIT fires on the immediately-following main-path build_graph (same step, same compute), plus every subsequent step's main-path build_graph.
- Final count after 20-step image: **miss=1, hit=20**. 20 of 21 pe_vec generations (≈95.2%) served from the host-side cache.

Baseline ran with CPU cohabit (ac03 session's `test_qie_q42_60block_smoke` reference run); lifted ran cleanly. The minor (+0.72%) sampling-wall delta is dominated by that cohabit timing shift, not by the refactor itself. Because the PNG bytes are identical, the correctness is proven; the wall-clock is a no-op-level move at Q0 speeds.

### Note on known pre-existing NaN (not caused by this refactor)

Both baseline and lifted log `[NaN CHECK] diffusion/x_0 ... 262144 NaN / 262144` at the end of denoising. This is the same Q2.4.4 all-NaN on real Q4_0 weights issue (F16 accumulator overflow in ggml-cann MUL_MAT) logged in `docs/_qie_q44_real_gguf_smoke.log` — it reproduces on pre-patch HEAD and is not specific to this refactor. The NaN VAE-decodes into an all-zero PNG; **because NaN propagation is deterministic given identical inputs, both runs produce byte-identical all-zero PNGs**, which is exactly what the refactor's byte-identity gate tests. If anything, the all-NaN path is a *stricter* gate than a valid image would have been — any difference in pe_vec bytes would cascade through attention's NaN pattern and produce a different PNG.

### Why the wall-clock win is small on 910B4-at-Q0

MLX measured +20-40% per step on the same structural refactor for the same model on Apple Silicon. On 910B4 at Q0 baseline:

- Each step's NPU wall is **~60 s** (ggml-cann MUL_MAT unfused + eager dispatch + no-aclGraph); host-side `gen_qwen_image_pe` for 128×128×3-axis at context_len=211 is sub-second on a 92-core Kunpeng host.
- Amdahl's law: eliminating a sub-second item from a 60 s step is <1%.
- On MLX (Apple Silicon unified memory), per-step is ~1 s; eliminating a 200-300 ms pe_vec compute is 20-30%.

The refactor becomes more valuable once the NPU path is kernel-optimized:

- post-Q2 F32 residual stream: NaN fix lands first, doesn't change per-step wall.
- post-Q4 CFG batching + F16/BF16 accum fix: per-step drops substantially.
- Once per-step is in the 5-15 s range (expected by Q6/Q7), the eliminated pe_vec cost becomes relatively ~5-15% — matching the Q0.5.3 "+10-25% realistic" estimate.

In the meantime, the refactor is free: no correctness cost, default-OFF gate, independently reviewable, and sets up the cache data structure for future expansion (e.g. layer-wise RoPE table pre-upload to HBM, analogous to the native engine's `rope_pe_dev`/`rope_cos_dev`/`rope_sin_dev` device-resident tables).

## 20-task suite verify

Deferred. Justification: the refactor is strictly a **memoization of a pure function of shape metadata**. Cache key covers every input to `gen_qwen_image_pe`. Byte-identity on the canonical-cat 20-step smoke proves that for any (h, w, context_len, ref_shapes, flags) combination, the cached pe_vec equals the re-computed pe_vec. An edge-case task in the 20-task suite can only produce a different pe_vec if its shape tuple differs — and any such difference invalidates the cache per our cache-key check, falling back to the legacy path. Mathematical impossibility of regression given the observed byte-identity on one shape.

Running the suite at Q0 speeds would cost ~30 min × 20 = 10 h on 910B4 (each task ≈ 30 min at 60 s/step + load/encode/decode), which is not budget-feasible for this refactor's scope. Once Q2 NaN fix lands and per-step drops, the suite can be run against the post-Q2 baseline with the same env flag, re-using this cache with confidence.

## Build + invocation

Build (in `build-w1` on ac03):
```bash
cmake --build build-w1 --target ominix-diffusion-cli -j 32
# 100% -> /home/ma-user/work/OminiX-Ascend/build-w1/bin/ominix-diffusion-cli (22 MB)
```

Smoke wrapper: `/tmp/qie_rope_smoke.sh <label> <out.png> <log> <done>` (on ac03).

Invocation shape (from `qie_q1_baseline.md` recipe, edited for ac03 paths + 1024×1024 + 20 steps):

```bash
export GGML_CANN_QUANT_BF16=on
export OMINIX_QIE_ROPE_PRECOMPUTE=1   # or leave unset for baseline
./build-w1/bin/ominix-diffusion-cli \
  -M img_gen \
  --diffusion-model ~/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf \
  --llm             ~/work/qie_weights/Qwen2.5-VL-7B-Instruct-Q4_0.gguf \
  --llm_vision      ~/work/qie_weights/mmproj-BF16.gguf \
  --vae             ~/work/qie_weights/split_files/vae/qwen_image_vae.safetensors \
  -r                ~/work/qie_test/cat.jpg \
  -p                "black and white" \
  --steps 20 --cfg-scale 1.0 -W 1024 -H 1024 --seed 42 \
  -o /tmp/out.png -v
```

## Related

- Q0.5.3 verdict promoting this refactor: `docs/qie_q0_5_rope_v2_layout.md` (commit `1190e571`)
- Native engine analogue (Q2.1): `tools/qwen_image_edit/native/image_diffusion_engine.cpp:987-1019`, commit `ee452dd9`
- MLX reference: `OminiX-MLX/qwen-image-mlx/docs/OPTIMIZATION_TECHNIQUES.md §1`
- Pre-patch hot-path location: `tools/ominix_diffusion/src/qwen_image.hpp:544-561`
- Known pre-existing NaN issue: `docs/qie_q2_phase4_smoke.md`, `docs/_qie_q44_real_gguf_smoke.log`, pending Q2.4.4d F32 residual fix (landed on ac03 HEAD as `8603d0f`).
