# Qwen3-TTS on Ascend 910B4: Hot-Path-Native Optimization

**From 12.2 fps → 33.8 fps (+177%) on a fixed hardware target, without rewriting the whole stack.**

This document describes the optimization work done in `OminiX-Ascend/tools/qwen_tts/` between the initial llama.cpp baseline and the current native engine. It is written as a design retrospective so the decisions can be audited after the contract closes.

---

## 1. The First Principle

> Locate the hot path. Rewrite only that for the target hardware. Keep everything else portable.

This is not "sacrifice all portability for performance." That framing invites two bad outcomes: (a) maintainers feel permission to fork everything, ballooning the Ascend-only surface area; (b) observers read it as zealotry.

The actual principle is surgical: in an autoregressive generation workload, **80-90% of wall-clock time lives in a narrow inner loop** (transformer forward → codec head → sample → next-step embed). Optimize that loop natively for the fixed target. Everything outside the loop — weight loading, tokenization, audio I/O, API server — stays on portable C++/ggml/standard libraries.

The payoff: native code for the hot loop runs 2-3× faster than generic frameworks (llama.cpp's ggml-cann backend) because it can exploit target-specific data layouts (FRACTAL_NZ), fused attention (FIAS), vendor quantization kernels (aclnnWeightQuantBatchMatmul), and graph capture. The cost: that 10-20% of the codebase is now tied to Ascend.

Under this rule, the question "should I rewrite X for the target?" becomes "does X take >10% of wall-clock?" If no, leave it portable. If yes, evaluate native vs generic trade on that isolated component.

---

## 2. Baseline and Target

| Attribute | Value |
|---|---|
| Model | Qwen3-TTS (Talker 28-layer + Code Predictor 5-layer + decoder conv stack) |
| Hardware | Huawei Ascend 910B4, 32 GB HBM, driver 23.0.6 |
| Runtime | CANN 8.5.0 (initially 8.3.RC1; migrated mid-contract) |
| Baseline path | llama.cpp + ggml-cann backend, unmodified |
| Baseline throughput | **12.2 fps** (codec frames/second, long utterance) |
| Contract goal | **≥ 25 fps end-to-end, ASR-verified content, user-ear-pass audio** |
| Final throughput | **33.8 fps** (W8 + TASK_QUEUE_ENABLE=2, CANN 8.5, cp_groups=8) |

The 2.77× gain was delivered without modifying generation quality — ASR transcription of 3 canonical utterances remained identical throughout.

---

## 3. Rejected Approaches and Why

Before committing to native engines, we considered and rejected:

**(a) Stay on ggml-cann (portable), tune kernels.** The ggml-cann backend JIT-dispatches through a generic tensor abstraction. It cannot consume FRACTAL_NZ weights, cannot dispatch aclnnFusedInferAttentionScoreV2, cannot use aclnnWeightQuantBatchMatmulV3 for W8. Its ceiling on this workload is ~15 fps.

**(b) Wait for vendor to upstream optimizations.** Huawei's llama.cpp-ggml-cann branch moves on its own cadence. Waiting is not a plan.

**(c) Full Rust rewrite.** CANN's C API (aclnn*, acl*) has no mature Rust binding. A Rust port would spend weeks replicating the FFI that's trivial in C++. Same native code under a different language hat.

**(d) Replace the whole pipeline with native code.** Would Ascend-ify the audio I/O, BPE tokenizer, test harness — all of which are not on the hot path. Large maintenance burden, zero performance win.

**(e) Quantize more aggressively (Q4_K_M, Q5_K_M).** GGUF K-quants are supported on CPU/MLX but have no direct Ascend kernel dispatch. A16W8 via aclnnWeightQuantBatchMatmul is the native quantization path; anything else requires dequant-on-load and gives back the speedup.

---

## 4. The Native-Hot-Path Strategy

The hot loop of Qwen3-TTS generation looks like this per codec frame:

```
1. forward(hidden)     — 28-layer Talker transformer, dominant cost
2. codec_head(hidden)  — vocab_size projection
3. sample(logits)      — top-k/top-p with repetition penalty
4. predict_codes(h)    — CpCannEngine, 5-layer transformer, 15 codebooks
5. generation_embed    — sum 16 codec embeddings + text embed
                         → next step's hidden
```

Steps 1 and 4 are 90% of wall time. We replaced them with hand-written engines:

**`TalkerCannEngine`** (`tools/qwen_tts/talker_cann_engine.{h,cpp}`)
- 28-layer transformer, all layers on Ascend
- Dispatches aclnn ops directly: aclnnMm, aclnnFusedInferAttentionScoreV2, aclnnRotaryPositionEmbedding, aclnnRmsNorm, aclnnCast
- Owns its own KV cache, per-layer weight tensors, rope caches
- Runtime symbol loading via dlsym (`cp_cann_symbols.{h,cpp}`) so CANN version differences don't break the build

**`CpCannEngine`** (`tools/qwen_tts/cp_cann_engine.{h,cpp}`)
- 5-layer code predictor for the 15-codebook prediction
- Same aclnn dispatch pattern
- Chunked decode path with dedicated CANN backend (commit `22e3e217`)

Everything else (BPE tokenizer, speaker encoder, decoder conv stack, audio I/O, stft, kissfft) stayed on portable C++/ggml. The speaker encoder even moved to CPU because CANN is 2.7× slower on small Conv1D/SE ops — generic-is-faster for cold-path modules.

---

## 5. The Optimization Stack (landed in order)

Each layer below is **additive on top of all prior layers** and is **gated by an env var** so regressions can be disabled without a rebuild.

### Layer 0 — Native Talker + CpCannEngine (M1-M2.5)

| Stage | Throughput | Δ from baseline |
|---|---|---|
| Baseline ggml-cann (llama.cpp) | 12.2 fps | — |
| Native engine, iterative decode | 18.3 fps | +50% |
| Native engine, batched prefill (FIAS) | 23.2 fps | +90% |

**What changed**: autoregressive loop now stays in aclnn calls end-to-end. No ggml graph rebuild per step, no generic dispatch.

**Key commits**:
- Initial native Talker: M1 series
- Batched prefill fix: commit `5fcd1445` (flip default after Track D confirmed M5 had silently fixed the cos-sim=0.28 regression)

### Layer 1 — Strip llama.cpp from the default build (M3)

- `CMakeLists.txt` exposes `QWEN_TTS_LLAMA` option, default OFF
- `--llama_fallback` CLI flag gated on the option
- Saves ~200 MB RAM, removes JIT dispatch in the common path

**Kept**: the llama.cpp build is still supported as a fallback for xvec mode (MRoPE 4×pos not yet in the native Talker). Users who need xvec compile with `-DQWEN_TTS_LLAMA=ON`.

### Layer 2 — aclGraph capture (M4, parked)

Attempted per-shape graph capture for `forward_decode`:

| Stage | Result |
|---|---|
| aclGraph capture on | **2.3× slower** on single utterance |
| Reason | Capture overhead dominates in one-shot mode |

Only viable in **session mode** (many utterances amortizing the capture cost). Not in v1 default path. Kept behind `TALKER_CANN_GRAPH=1` for future session-API work.

### Layer 3 — FRACTAL_NZ weight layout (M5)

**What it is**: Ascend's native 2D tensor layout for matmul RHS. Transforms a plain row-major `[K, N]` weight into a blocked `[N/16, K/16, 16, 16]` layout that aligns with the 910's cube units.

**How we used it**: On weight load, convert each linear layer's weight to NZ format via `ACL_FORMAT_FRACTAL_NZ` on the descriptor of `mat2` (the RHS). Then call plain `aclnnMm` — it detects the NZ tag and dispatches the fast kernel.

| Stage | Throughput |
|---|---|
| NZ off (ND default) | 22.6 fps median |
| NZ on (mat2 as FRACTAL_NZ) | **25.9 fps** (+15%) |

**Gotcha discovered on CANN 8.3**: plain `aclnnMm` does not consume NZ-tagged operands correctly. Silently produces garbage output ("哎呀!" / "嗯嗯嗯" nonsense). Required migration to CANN 8.5, which fixes operand reordering.

**Gated by**: `TALKER_NZ_WEIGHTS=1`.

### Stretch — A16W8 INT8 quantization (S1)

**What it is**: per-output-channel symmetric INT8 quantization of Q/K/V/O and FFN gate/up/down weights.

**Calibration** (offline, at load):
```
for each linear layer:
    for each output channel c:
        scale_c = max(|W[c,:]|) / 127
        W_int8[c,:] = round(W[c,:] / scale_c).clip(-128, 127)
```

**Dispatch** (at decode call sites):
```
aclnnWeightQuantBatchMatmulV3(activation_f16, W_int8, scale_f16, ...)
```

| Stage | Throughput | Memory |
|---|---|---|
| NZ baseline | 29.7 fps | 6.88 GB |
| W8 on (`TALKER_W8_QUANT=1`) | **33.8 fps** (+14%) | 8.85 GB |

**Memory regression**: W8 keeps F16 weights co-resident (per contract constraint: don't touch `forward_prefill` body). Prefill still dispatches plain aclnnMm on F16 weights. +28% VRAM.

**Gated by**: `TALKER_W8_QUANT=1`.

### Final enabler — TASK_QUEUE_ENABLE=2

Not a code change. A CANN runtime env var.

**What it is**: CANN's task queue mode affects how ops are dispatched to the NPU. On CANN 8.5, the default mode appeared to cause a 27% regression vs 8.3. Track I diagnosed this as **cold-cache / environment drift**, not a real regression. Setting `TASK_QUEUE_ENABLE=2` recovers 8.5 to within noise of 8.3.

**Why it matters**: without this, the CANN 8.5 migration would have looked like a perf loss, and the NZ + W8 stretches (which require 8.5) would have been rejected.

---

## 6. Parallel Quality Track (separable from performance)

Three quality issues surfaced during the perf work. Treated as separate tracks because they have no interaction with throughput.

### (a) "Oh." phantom prefix (commit `69c41884`)

Every utterance produced a leading "Oh." sound that ASR flagged as a 2-3-token prefix junk.

**Root cause**: missing `tokenizer_config.json` caused the BPE tokenizer to fall through silently, treating `<|im_start|>` as raw text and BPE'ing it into 2-3 junk tokens prepended to every utterance.

**Fix**: made the BPE tokenizer **fatal** on missing `tokenizer_config.json`. If the config isn't there, the tokenizer refuses to run instead of silently producing garbage.

### (b) Opening click / prefix noise (commit `2c2c3f6b`, current session)

Each generated utterance had a 50-100 ms noise burst at the ref/target cut point in ICL mode, audible as a click.

**Root cause**: the decoder's convolutional receptive field straddles the ref-to-generated codec seam. The first ~150 ms of the generated portion carry a settle ripple that fade masks can attenuate but not eliminate.

**What we tried first (all bandits)**:
1. 50 ms linear fade — still clicked
2. 120 ms linear fade — RMS reduced to 0.6×, still audible
3. 200 ms cubic fade — first 100 ms down to 0.125×, still a residual at 160 ms

**Root-cause fix**: shift the cut boundary forward by 150 ms so the transient zone falls in the discarded portion. Removed the fade entirely.

Applied symmetrically to ICL, xvec, and customvoice paths.

### (c) Hollow / rumble ("轰隆隆") in long clones

User reported 25-second cloned output sounded hollow with a low-frequency rumble.

**Diagnosis**: spectral analysis showed sub-100 Hz energy 45-95× higher than the reference; 2-5 kHz presence 3× reduced.

**Investigation**: ran F16 (no W8) vs W8 on a fresh 8-second ICL. Both were clean. Ran a fresh 24-second ICL on W8: also clean. Ran the same on 4 non-mayun references (doubao, luoxiang, ellen, maple): all clean.

**Conclusion**: the original rumble-laden files were content-specific / seed-specific, not a systemic bug in W8 or long-generation. No code change needed.

**This is worth calling out as a process lesson**: before coding any "fix," we ran an A/B that exonerated the suspected cause. The first instinct ("it's W8 quantization corrupting codec tokens") was wrong. A 10-minute measurement saved a week of blind rewriting.

---

## 7. The Portability Cost

What is now Ascend-only (compiled against CANN 8.5):

| Component | Lines | Reason |
|---|---|---|
| `TalkerCannEngine` | ~1,500 | Hot path; direct aclnn dispatch |
| `CpCannEngine` | ~800 | Hot path; 5-layer CP transformer |
| `cp_cann_symbols.{h,cpp}` | ~400 | dlsym loader for version-variant aclnn symbols |
| FRACTAL_NZ conversion | ~100 (inline) | Ascend tensor layout |
| A16W8 calibration | ~200 | Ascend quantization kernel dispatch |
| aclGraph capture scaffolding | ~300 (parked) | Ascend graph mode |

**Total native surface: ~3,300 lines.** This is the code that will not run on MLX, CUDA, or CPU without reimplementation.

---

## 8. What We Kept Portable

Everything outside the hot loop:

| Component | Shared with |
|---|---|
| GGUF weight format | MLX, CPU, CUDA, any llama.cpp downstream |
| BPE tokenizer (`qwen_common`) | Qwen ASR, qwen_common consumers |
| Audio I/O (`stft.cpp`, `audio_io.cpp`, kissfft) | Any TTS downstream |
| Speaker encoder | Runs on CPU (Ascend is slower on small ops) |
| Speech tokenizer encoder/decoder | Portable ggml path |
| Generation algorithm (autoregressive + CP sampling + EOS) | Same algorithm on any backend |
| C ABI (`qwen_tts_api.h`) | Any host language via FFI |
| ASR content gate (`scripts/asr_quality_check.sh`) | Portable shell + mlx-whisper |
| DTW quality harness (`scripts/dtw_vs_baseline.py`) | Portable Python |
| Contract / milestone / test structure | Portable project methodology |

Anything porting this stack to a new accelerator would touch the 3,300 lines of native engine code. The other ~80% of the codebase would come over unchanged.

---

## 9. Numbers Recap

| Metric | Baseline | Final | Delta |
|---|---|---|---|
| Throughput (long utt) | 12.2 fps | 33.8 fps | **+177%** |
| Contract goal (§1) | ≥ 25 fps | met at +35% | — |
| VRAM (long utt, peak) | ~6.9 GB | ~8.8 GB | +28% (W8 gate failed) |
| ASR content gate (3 utts) | PASS | PASS | — |
| Prefix noise | 50-100 ms burst | 0 ms (150 ms cut-shift) | eliminated |
| Rumble on long clones | reported | not reproducible on fresh runs | closed (content-specific) |
| Peak memory w/o W8 (NZ baseline) | — | 6.88 GB | — |
| Peak memory w/ W8 (current default) | — | 8.85 GB | — |

---

## 10. Why This Works For Us

This strategy pays off when:
- **Hardware target is fixed** (single deployment platform — here: Ascend 910B4). Native engineering amortizes against many users, not many hardwares.
- **Workload is hot-path-dominated** (autoregressive generation → 90% in one loop). Makes the surgical optimization high-ROI.
- **Generic backends have a known ceiling** (ggml-cann at ~15 fps). Native is the only path above the ceiling.
- **Team can maintain native C++ + vendor SDK expertise**. CANN aclnn isn't a casual API — it requires sustained engagement.

Where it would not work:
- Multi-target SaaS (must support CPU/CUDA/MPS/NPU matrix). The 3,300 native lines become 3,300 × N.
- Flat latency budget workloads (no hot path to optimize).
- Short-lived projects where amortization never pays off.
- Small teams without vendor-SDK expertise.

---

## 11. Lessons

1. **Generic backends have ceilings.** When throughput matters and hardware is fixed, going native is worth it. Benchmark the generic ceiling before assuming the framework is fast enough.

2. **Env-var-gated optimizations let you ship regressions safely.** Every stretch (W8, NZ, aclGraph) was behind a flag. If a regression landed, we flipped the flag off and shipped the previous state — no rebuild, no rollback.

3. **Quality and performance are separable.** The tokenizer-fatal fix and the prefix-trim fix have nothing to do with Ascend or W8. Conflating them would have wasted weeks chasing the wrong root cause.

4. **Measure before "fixing."** The hollow/rumble investigation was going to become a decoder-side rewrite until a 10-minute spectral A/B showed the suspect (W8) was exonerated. The rumble was content-specific, not systemic.

5. **Contracts with stamps enable parallel agent work.** The `§5 milestone / §6 acceptance / Verified-by` structure meant an agent could take an item, prove it landed, and stamp it — without a PM bottleneck.

6. **Root-cause fixes beat bandits.** The fade-mask history (50 ms linear → 120 ms linear → 200 ms cubic) is a sequence of bandits, each better than the last but none correct. The real fix was recognizing that the cut boundary was in the wrong place.

---

## 12. What's Next (post-v1)

Beyond this contract:
- **Dedupe the W8 memory footprint** (stretch S4 — drop F16 weights after prefill).
- **Port MRoPE 4×pos into `TalkerCannEngine`** so xvec mode leaves the llama.cpp fallback path (+35% fps on xvec).
- **Bridge to OminiX-API** (see `ASCEND_API_BRIDGE_CONTRACT.md`) — expose the native engine through a C ABI so the HTTP server consumes it as a library instead of a subprocess.
- **CannFusion research track** — Rust DSL emitting AscendC kernels. Long-horizon; the path toward "write each model once, run on both MLX and Ascend."

---

## Appendix A — Commit Trail (representative)

| Commit | What |
|---|---|
| `69c41884` | BPE tokenizer fatal on missing config (eliminates "Oh." prefix) |
| `5fcd1445` | Flip batched prefill to default (16× prefill speedup) |
| `22e3e217` | Dedicated CANN backend for chunked decode |
| `f09ef578` | Document TASK_QUEUE_ENABLE=2 mitigation |
| `f10508a4` | A16W8 per-channel INT8 quantization landing |
| `2c2c3f6b` | Prefix-click root-cause fix (150 ms cut-shift, removes cubic fade) |
| `ae2832e5` | High-level `qwen_tts_synthesize` C ABI (for OminiX-API bridge) |
| `d88f872d` | `libqwen_tts_api.so` shared library target |

## Appendix B — Key File Index

| File | Purpose |
|---|---|
| `tools/qwen_tts/talker_cann_engine.{h,cpp}` | Native 28-layer Talker |
| `tools/qwen_tts/cp_cann_engine.{h,cpp}` | Native 5-layer Code Predictor |
| `tools/qwen_tts/cp_cann_symbols.{h,cpp}` | dlsym loader for CANN version variance |
| `tools/qwen_tts/qwen_tts.{cpp,h}` | Generation loop + ICL/xvec/customvoice dispatch |
| `tools/qwen_tts/qwen_tts_api.{h,cpp}` | Stable C ABI for host-language FFI |
| `tools/qwen_tts/qwen_tts_api.version` | Linker version script (symbol export control) |
| `tools/qwen_common/bpe_tokenizer.{cpp,h}` | Portable BPE tokenizer (shared with ASR) |
| `NATIVE_TTS_CONTRACT.md` | Delivery contract, source of truth |
| `docs/gguf_quant_exploration.md` | K-quants vs A16W8 trade-off analysis |
