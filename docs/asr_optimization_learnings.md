# ASR Optimization on Ascend — Milestones, Learnings, Forward Work

**Date**: 2026-04-22. **Author**: PM synthesis after A4 landing.
**Source base**: `docs/asr_a0_discovery.md`, `docs/asr_a1a_baseline.md`,
`docs/asr_a1a_data/tier1_q8_cann_clean.json`, A1b-v2 commit messages
(`0dd55cad`), A4 commit message (`525d8a1e`), `docs/contracts/QWEN3_ASR_CONTRACT.md`,
`docs/qwen_tts_optimization_learnings.md` (sibling reference).

**Framing**: this doc mirrors `qwen_tts_optimization_learnings.md` but
the content is different. TTS was a 9-month, 30× arc of vendor-op probing,
cross-layer ports, and fused-op stacking. ASR was a **4-week architecture-
recognition arc** that rode the TTS playbook verbatim plus a W8-loader
unblock. The interesting learning is not what we optimized; it is how much
of TTS transferred at zero additional eng cost, and where the thin novel
surface sat.

## Executive summary

- **Qwen3-ASR-1.7B decoder is architecturally identical to TTS Talker.**
  28L × 2048 × 16Q/8KV GQA × SwiGLU × RoPE θ=1e6 × tied embeddings ×
  vocab=151936. This was not documented; A0 discovery surfaced it. Once
  recognized, the entire TTS native-engine composes verbatim, reducing
  A1b-v2 scope from "port a new decoder" (~4-6 weeks) to "wrap
  `TalkerCannEngine` + port `CpCannEngine::init_lm_head_` /
  `forward_lm_head` / `fetch_logits`" (~1 week with rate-limit disruption).
- **A1a legacy path is still the shipping state** at RTF 0.142 on mayun_ref.
  The native engine arc (A1b-v2 F16 → A4 W8 decode) closed part of the gap
  (0.224 → 0.185) but does not yet beat the legacy number. A4b (prefill
  W8) is the remaining lever; prefill is 73% of mayun_ref's wall
  (1016ms vs 370ms decode).
- **Novel work lived in the split-prefill and the host cohabitation lock.**
  Everything else was composition. The split-prefill pattern —
  `[pre_tokens, audio_features_2048, post_tokens, generated]` fed through
  `batch.embd` into the decoder-only LLM — is cross-modal prefix without
  cross-attention; worth codifying for future audio/VL adapters.
- **TTS-as-ASR self-consistency gate works.** Synthesizing canonical
  phrases via Qwen3-TTS and transcribing with Qwen3-ASR produces
  byte-identical text on 8+ CN phrases. Reusable regression gate that
  catches drift on either side.
- **Meta-process carried over 1:1** from TTS: probe-first, projection
  discount, patch-file push, no Claude coauthor, fork vs origin remote
  discipline, live-result gate trumps offline parity.

## Part 1 — Milestone arc

### A0 — Discovery (1 day, ac03)

- Confirmed ac03 910B4 idle, CANN 8.3.RC1.
- Fetched Qwen3-ASR-1.7B weights via ModelScope (HF proxy blocked;
  `/modelarts` read-only → `/home/ma-user/work/asr_weights/` reroute).
- **Architecture read surprise**: decoder config matched Talker exactly.
  No cross-attention. Audio features injected as a prefix token block
  via `batch.embd`, not via cross-attn K/V.
- Scope was revised downward ~70% on this single finding. Contract §A1
  rewritten from "port a new decoder" to "wrap existing TalkerCannEngine."

### A1a — Legacy ggml-cann Q8_0 baseline (1 week, ac03)

- Build stock `llama.cpp` + CANN backend against Q8_0-quantized decoder
  GGUF. Ran 13-clip tier-1 (8 CN + 5 EN).
- **Gates**: Tier-1 CER=0 byte-identical across all clips. RTF 0.142 on
  mayun_ref, Tier-1 median 0.159.
- Anchored as the reference for regression on every subsequent milestone.
- Receipts at `docs/asr_a1a_data/tier1_q8_cann_clean.json` + RTF calibration
  log.

### A1b-v2 — Native F16 engine (1 week, ac03, rate-limit-interrupted)

- Wrote `AsrTextDecoderCannEngine` (`tools/qwen_asr/native/`) composing
  `TalkerCannEngine` verbatim + porting three methods from
  `CpCannEngine` (`init_lm_head_`, `forward_lm_head`, `fetch_logits`).
  1005 LoC addition.
- `main_native.cpp` CLI driver: audio-encoder (stock ggml-cann path) →
  audio features → split-prefill mixed tokens → native decoder
  autoregressive → lm_head NPU.
- **Gates**: Tier-1 CER=0 across all 13 clips (parity PASS). RTF 0.224
  on mayun_ref.
- Perf regression vs A1a was **predictable bandwidth math**: F16 decoder
  weights 3.88 GB vs Q8_0's 2.06 GB = 1.9× HBM pressure. No engine
  inefficiency — weight-format bound.
- **Blocker surfaced**: `TalkerCannEngine::init_from_gguf` rejected
  GGUF dtype 8 (Q8_0) with "unsupported dtype 8." Only F16/F32/BF16
  supported. TTS's W8 path lives in a separate
  `aclnnWeightQuantBatchMatmulV3` branch with pre-quantized INT8
  buffers, not GGUF Q8_0 dequant. Closing this was A4 scope.
- Commit `0dd55cad` on fork; rate-limit forced manual recovery via ssh
  + bash heredoc (`/tmp/commit_a1b.sh`) to land the commit + patch.

### A4 — W8 decode quant (3 days, ac03)

- Extended `TalkerCannEngine::load_gguf_tensor_f32` + ASR
  `load_embedding_and_lm_head_` to dequantize block-quant tensors via
  `ggml_get_type_traits(type)->to_float` (bit-exact with llama.cpp CPU
  reference). 29 insertions / 8 deletions across two files.
- Defensive branch: only fires when dtype is neither F32 nor F16.
  F16/F32 paths are byte-identical to pre-patch. `TALKER_W8_QUANT=1`
  gate unchanged — enables W8 decode dispatch via existing
  `aclnnWeightQuantBatchMatmulV3`; unset preserves F16.
- **Gates**:
  - Tier-1 CER=0 across 13 clips (parity HARD PASS).
  - RTF mayun_ref: **0.185** (gate ≤ 0.142 MISS; -17% vs A1b-v2).
  - RTF Tier-1 median: **0.262** (gate ≤ 0.159 MISS).
- Root cause of RTF miss: W8 dispatches **only at decode**; prefill
  stays on F16 `aclnnMm` by design. Decomposition on mayun_ref:
  prefill=1016ms, decode=370ms (30 tokens). Prefill is 73% of wall.
- TTS regression verified on ac01: `TALKER_W8_QUANT` unset + F16 Talker
  GGUF → sha256 byte-identical wav pre/post-A4 (wall-clock within
  ±3% noise). Defensive branch proven.
- Commit `525d8a1e` on fork.

### A4b — Prefill W8 (dispatched, ac03 in flight)

- Extend W8 dispatch to prefill's 7 matmul sites per layer × 28 layers
  (Q/K/V/O + gate/up/down).
- Target: RTF ≤ 0.142 on mayun_ref, median ≤ 0.159.
- HARD PARITY GATE: CER=0 must hold. Hard-kill on drift.
- Cohabitates on ac03 with QIE-Q2 scaffold via `/tmp/ac03_hbm_lock`
  semaphore.

## Part 2 — What transferred from TTS (almost everything)

Quasi-verbatim transfers:

| TTS lever | ASR applicability | Notes |
|---|---|---|
| **`TalkerCannEngine` composition** | ✅ | Entire engine reused; ASR wraps it. Decoder architecture identity makes this a 1:1 transfer. |
| **`CpCannEngine` lm_head pattern** | ✅ | Three methods ported: `init_lm_head_`, `forward_lm_head`, `fetch_logits`. vocab=151936 both sides, F16 lm_head weight 593 MB on NPU. |
| **ACLGRAPH pos-keyed capture** | ✅ | ASR decoder autoregression is identical to TTS Talker; same cache pattern. Hasn't been wired yet (A-milestone arc didn't need it; regression risk vs parity priority). |
| **InplaceAddRmsNorm** | ✅ latent | Available as soon as ACLGRAPH lands on ASR path. Not in current ship. |
| **W8 per-channel INT8 + F16 scale** | ✅ | A4's single-file extension unblocked this. Same calibration code. |
| **WSPOOL async-safe retain** | ✅ | Transferred verbatim; no bug surfaced on ASR workload. |
| **Split-prefill via `batch.embd`** | ⚠️ novel | See Part 3. Not a TTS pattern. |
| **GMM-QKV grouped matmul** | ✅ | Not in A-ship; available as drop-in if a future A-milestone needs it. |
| **FIAv2 / V3 / V4** | ✅ | Same 1-16 seq regime as TTS; same GREEN verdict; wired implicitly via TalkerCannEngine. |
| **`aclnnApplyRotaryPosEmbV2`** | ✅ retired | Same retire verdict as TTS (3× closures on wiring). RoPE composed manually in TalkerCannEngine. |
| **Noise-band measurement** | ✅ | N=5 runs on each gate; σ tight (<5% wall). |

Patterns that lived in TTS docs and applied 1:1:
- Probe-first mandate
- Projection discount (÷5)
- Patch-file push (ac03 has no fork creds)
- No Claude coauthor in commit messages
- Fork (`ymote`) vs origin (`OminiX-ai`) discipline — all pushes to fork
- Live-result gate is non-negotiable; offline parity necessary not
  sufficient
- Vendor-op catalog grep at Gate 0

## Part 3 — What was novel (thin surface)

### 3.1 Split-prefill via `batch.embd`

Unlike TTS (pure token prefix) and unlike cross-attention decoders
(K/V side-channel), Qwen3-ASR feeds audio features as part of the
prefix token stream:

```
[<bos>, <lang_id>, <audio_start>, feat_0 ... feat_2047, <audio_end>,
 <task>, <instruction>, <decoding_target>] → decoder
```

Audio features are pre-computed embedding vectors (2048 tokens × 2048
hidden) injected into the prefill via `llama_batch::embd`, bypassing
the token-id → embedding lookup. The decoder does not know it is
cross-modal; everything is a "prefix." No cross-attention at all.

**Why it matters**:
- Structural elegance: no decoder architecture changes vs LLM.
- HBM pressure: 2048-token audio prefix is a real activation budget;
  KV-cache for that prefix stays hot across all generated tokens.
- Regression gate: the feature-tensor-to-`batch.embd` plumbing has no
  TTS parallel. Any future audio/VL adapter reuses this.

### 3.2 TTS-as-ASR self-consistency gate

Synthesize canonical phrases via Qwen3-TTS (shipping 32.2 fps, known
byte-stable wav outputs), transcribe via Qwen3-ASR. Byte-identical
text across 8 CN phrases confirms both ends are stable. Reusable
regression probe for either side.

User directive ("use TTS to test ASR as gate") codified this. Kept in
tier-1 suite at `docs/asr_a1a_data/` as one of the reference anchors.

### 3.3 Host cohabitation lock on ac03

A4 landed → ac03 freed. A4b (prefill W8 extension) + QIE-Q2 (image-edit
scaffold) both dispatched to ac03 in parallel. File-disjoint
(A4b=`tools/qwen_tts/talker_cann_engine.cpp`, QIE-Q2=`tools/qwen_image_edit/`),
but HBM-contended during smoke tests:
- A4b ASR smoke: ~14 GB
- QIE-Q2 image smoke at Q4: ~18-20 GB
- 910B4 total: 32 GB → simultaneous load OOMs

**Solution**: cooperative existence-probe lock at `/tmp/ac03_hbm_lock`.
Agent touches file before smoke, removes after, waits if present.
Coarse-grained but sufficient — smoke tests are minutes not seconds.
`flock(2)` upgrade noted if contention granularity tightens.

**Pattern worth codifying** for future multi-agent shares on 32 GB
cards. 64 GB cards (910B1/B2/B3) would relax this.

## Part 4 — Meta-process wins and misses

### Wins

1. **Architecture recognition as a first-class scope input.** A0's
   finding that ASR decoder ≡ TTS Talker was the largest scope
   reduction in the contract. Always ask "is this actually a new
   arch or is it a reskin of something we have?" before scoping
   port work.
2. **Patch-file + manual heredoc recovery** survived a rate-limit
   mid-milestone. Agent built binary + source; rate-limit killed it
   before commit. Manual ssh + bash heredoc (`/tmp/commit_a1b.sh`)
   landed the commit and generated the patch. Lesson: never assume
   agent owns the commit lifecycle — PM can step in.
3. **Self-consistency gate** (TTS→ASR) was user-dictated and paid off
   immediately. Byte-identical CN transcripts across 8 phrases is a
   stronger gate than WER on noisy external data.
4. **Defensive branch discipline** (A4): TTS F16 GGUF path proven
   unchanged via sha256 post-A4. Same pattern as `TALKER_W8_QUANT`
   and `TALKER_CP_ACLGRAPH` — gate additions behind env vars, keep
   the unset path byte-stable, PM verifies with a cross-workload
   regression test.

### Misses

1. **RTF projection for A1b-v2 was not discounted.** Agent-estimated
   RTF close to A1a; actual was +60% worse due to F16 vs Q8_0
   bandwidth. Projection-discount rule (÷5) should have been applied
   here; specifically, "F16 will regress N% unless W8 lands with it."
   Codify: weight-format regressions must be computed at contract
   scoping, not discovered at gate.
2. **Q8_0 loader gap surfaced at gate, not at contract.** Should have
   been a gate-0 audit finding on A1b-v2 scoping. Contract assumed
   TalkerCannEngine supported GGUF Q8_0 because TTS uses W8 — but TTS's
   W8 is a separate pre-quantized-buffer path, not GGUF dequant. Fix:
   contract-time audit "does our native engine support the checkpoint
   format we plan to ship?" before dispatching port work.
3. **Stray branch `rewire-a2-v2-rope` created by concurrent agents.**
   Multiple agents branched inadvertently from main during A1b-v2 +
   WSX landings. Cleanup cost ~10 min manually. Fix: brief agents
   explicitly to stay on `main` unless asked to branch.

## Part 5 — Open work

### A4b (in flight)
Extend W8 to prefill. Budget 8-10 days. Hard-kill on CER regression.
Target RTF ≤ 0.142. If gate not cleared, decomposition documents A4c
scope (further matmul sites or activation-path wins).

### ACLGRAPH on ASR
Not wired yet. Pos-keyed capture cache pattern works for ASR decoder
autoregression. Skipped in A-arc because A1b-v2 + A4 parity gate was
priority. Post-A4b is the right time to consider; +5-10% decode wall
is the projection based on TTS G2.

### Persistent-engine multi-clip driver
Currently `qwen_asr_native` loads full session (~36s init: lm_head 593
MB + decoder weights + ctx + KV alloc) per clip. Persistent-engine
amortization across calls is pending a CLI driver. Simple refactor;
deferred from A1b-v2 and A4.

### Tier-2 dataset
LibriSpeech (EN) + AISHELL-1 (CN) on ac03 deferred from A1a. Larger WER
anchor for future regression tests. scp-to-ac03 + script pass; no
engine changes. Deferred indefinitely while A4b is the critical path.

### A1b-v2 perf catch-up bigger picture
Even after A4b lands, native engine may not beat A1a's ggml-cann
Q8_0 path on pure RTF. The native engine's value is:
- Amenable to aclGraph capture (next lever)
- Amenable to in-process composition with TTS (shared engine instance)
- Amenable to split-prefill variants for future adapters
- Control of memory lifecycle (WSPOOL, session amortization)

If shipping ASR in production standalone is the only target, A1a
stays. If shipping ASR as part of an OminiX unified inference stack
(TTS + ASR + diffusion sharing one CANN context), native is the
path.

## Part 6 — Vendor asks (inherited from TTS, re-stated)

All valid for ASR; no new ASR-specific asks surfaced:

1. **Open-source `libruntime.so`** — opaque EZ9999 errors blocked
   A4's first loader attempt by hours until traced.
2. **Local CANN SDK for macOS + Linux dev** — Mac LSP sees diagnostic
   noise on every CANN source touch; audit time burned.
3. **English-first op documentation** — FIAv2 / WQBMMv3 /
   GroupedMatmulV3 capability matrices still require in-header grep on
   ac0N hosts.
4. **GGUF Q8_0 / K-quant dispatch in `ggml_cann_mul_mat` +
   `ggml_cann_get_rows`** — same three bugs as QIE Q1. Fixing once
   benefits TTS, ASR, QIE, SD3, Flux, Z-Image, Wan2.

## Part 7 — Transferable patterns for future workloads

Future audio / multimodal / VL / speech workloads can inherit:

1. **Architecture-equivalence audit as scope gate** — check for
   Qwen3-family reskin before assuming net-new port.
2. **Split-prefill via `batch.embd`** as the cross-modal prefix
   pattern — no cross-attention surgery required.
3. **Native engine composition** over net-new port — the Qwen3-family
   decoder (28L × 2048 × 16Q/8KV GQA × SwiGLU × RoPE θ=1e6) is
   single source of truth; wrap + add model-specific pre/post.
4. **Self-consistency gates across sibling models** — TTS→ASR is one;
   VL→TTS→text, ASR→TTS→audio are analogous patterns.
5. **Defensive env-gated extensions** — `TALKER_W8_QUANT`,
   `TALKER_CP_ACLGRAPH`, `TALKER_CP_INPLACE_ADDRMSNORM`. Keep the
   unset path byte-stable; PM verifies with a cross-workload
   regression.
6. **Host cohabitation via cooperative file-lock** on 32 GB cards.
7. **Patch-file handoff from compute hosts to Mac PM** — standard
   workflow; no fork creds on ac hosts.
8. **Manual heredoc commit fallback** when rate-limits kill an
   agent mid-milestone — `/tmp/commit_<milestone>.sh` pattern.

## Part 8 — Numbers summary

| Milestone | Gate | Parity | RTF mayun_ref | RTF Tier-1 median | Status |
|---|---|---|---|---|---|
| A1a legacy Q8_0 | CER=0, RTF ≤ 0.159 | ✅ | 0.142 | 0.159 | **SHIP** |
| A1b-v2 native F16 | CER=0 | ✅ | 0.224 (+60%) | — | LANDED |
| A4 W8 decode | CER=0, RTF ≤ 0.142 | ✅ parity, ❌ RTF | 0.185 (-17% vs A1b-v2) | 0.262 | LANDED |
| A4b prefill W8 | CER=0, RTF ≤ 0.142 | — | — | — | IN FLIGHT |

**Honest current state**: A1a is the shipping path. Native engine is
catching up but not beating. A4b is the last known lever before we
need aclGraph or batching work to close the gap.

**Tracking**: A4 commit `525d8a1e` on `ymote/OminiX-Ascend`. Full
Tier-1 per-clip results at `/tmp/a4_tier1_results.json` (PM local). A1a
anchor at `docs/asr_a1a_data/tier1_q8_cann_clean.json`.
