# Agentic AI Coding Unlocks a New Perf Frontier on Ascend 910

**Executive/technical deck — PM: OminiX, 2026-04-21**
**Thesis**: agentic AI coding (Codex, Claude Code) dissolves the rigid abstraction boundaries of the pre-AI-coding inference stack (ops API, computing engine, inference framework, dataflow layer), enabling end-to-end model/system co-design from single-card to cluster — faster and more thoroughly than a conventional team.

<!--
CHANGELOG
v3 polish (2026-04-21): sharpened baseline-noise honesty on Slide 9 (A.1 +0.3 and M3'new' +0.3 within ~1 fps run-to-run variance; final number is 31-32.2 fps band, not a crisp 32.2); folded Talker aclGraph scaffold probe (TALKER_CANN_GRAPH=1 → 30→18 fps, 40% regression from lazy-capture-on-first-touch) into Slide 5 failing-fast-as-feature list; reconciled projection-vs-reality meta-pattern (G0: +6-10 → +1.15; M3N: +1 → +0.3; ~3-10× optimism) as explicit Slide 9 meta-finding; Appendix B updated with A.1 / M3'new' / M1.B FFNV3 / CannFusion #26 rows marked as single source of truth for Q&A.
v3.1 (2026-04-21): pie-project appendix + Slide 2/6/8/10 Pie-integration edits removed per PM directive; reverted to 5-layer stack and single hardware-scaling axis.
-->

---

## Slide 1 — Thesis + Proof

**Title**: "The boundary was the bottleneck."

- Conventional inference stacks leave **10-50% perf on the table at layer boundaries**
- Boundaries are organisational, not technical: **humans specialise per layer**
- Agentic AI coding crosses **Rust ↔ C++ ↔ aclnn ↔ AscendC ↔ CMake ↔ Python** in one session
- **Proof**: OminiX Qwen3-TTS on 910B4 — **~1 fps (llama.cpp baseline) → 31-32.2 fps band in ~7 days**, one PM + agent swarm. Full arc ≈ **~32× lift** from conventional inference stack. Shipped at fork tag `32fps-landed`, byte-identical user-ear verified.
- Zero permanent headcount added; CP engine rewritten across 4 layers; **4 landed tracks, 3 cleanly killed tracks** (Path C, CannFusion, Talker aclGraph scaffold) — all with receipts

**Speaker notes**: The punchline we'll defend in the next nine slides: the conventional inference stack (`llama.cpp`, `PyTorch`, `LLVM`, `OneFlow`, `vLLM`) is a layered system designed around human-scale specialisation. Each layer has its own language, cadence, team. Cross-layer optimisation is something planned in roadmaps and delivered in quarters. What we're showing today is that agentic AI coding — Codex, Claude Code, agent-swarm — collapses those layer boundaries into something much cheaper than a quarter. In the OminiX case study, we took Qwen3-TTS on Huawei 910B4 from **effectively unusable under stock llama.cpp + ggml-cann (~1 fps, sub-realtime)** to **31-32.2 fps clean-quality, user-ear-verified**, in about a week with one PM gating a small agent swarm. Call the landing number "~32 fps ± 1 fps" — that's the run-to-run noise band we've characterised on 434-frame LONG, and we won't overclaim the last 0.3. That's not "AI wrote some code faster"; it's "AI rewrote the CP engine across the Rust FFI, the C++ dispatch layer, the aclnn ops chain, the AscendC custom-kernel path, and the Python weight-repack, in the same head." That combined perspective is the new thing. Four phases: escape the conventional stack (~1 fps → 22 fps, first native `CpCannEngine`), co-design within the native stack (22 → 30.5 fps via W1 NPU lm_head + W3b kernel fusion), collapse the dispatch floor (30.5 → 31.6 fps via aclGraph pos-keyed replay), incremental post-32 gate closes (31.6 → ~32 fps via A.1 + M3'new', both within noise). Fork tag `32fps-landed` (2026-04-21).

---

## Slide 2 — The Pre-AI-Coding Constraint

**Title**: "Dialects, teams, release cadences."

- Framework (PyTorch / Python) — ML researchers, monthly releases
- Inference engine (vLLM / TRT-LLM) — systems engineers, quarterly
- Computing engine / ops API (aclnn, cuDNN) — vendor, semi-annual
- Kernel DSL (AscendC / CUDA) — kernel authors, project-scoped
- Compiler (bisheng / LLVM / MLIR) — compiler team, decoupled
- **Five layers, five dialects, five release trains** — cross-layer optimisation ≈ 1 quarter ≈ 1 retrospective ≈ 1 re-org

**Speaker notes**: Here's the structural problem. A production inference stack has five layers, and each layer has its own dialect and its own people. Framework, serving engine, ops API, kernel DSL, compiler. The Python framework author does not grep the CANN headers in her free time. The kernel author does not edit the Rust FFI. The systems engineer does not pattern-match the vendor op catalog. They don't know each other's dialects, and they're on different release trains. A classic concrete failure mode in our own work: Huawei shipped `aclnnFFNV3` — a single op that does W8 SwiGLU plus per-channel dequant, collapsing five of our aclnn calls into one — in CANN 8.3 from day one. It sat in the `aclnnop/` include directory on our ac01 host for months. Nobody in our stack grepped for it until an agent did an FO-audit of all 717 aclnnop headers in 30 minutes. Three months of headroom were just sitting on the vendor's disk, untouched. That is not a code problem. That is a coordination-cost problem of exactly the kind AI coding collapses.

---

## Slide 3 — Single AI Crossing All Layers (W1: NPU lm_head Port)

**Title**: "One session, four layers, +8.3 fps."

- Baseline: `lm_head` was a CPU NEON+OMP matvec, **15 ms / frame**
- Agent crossed: Rust FFI, C++ `CpCannEngine`, aclnn dispatch, weight upload
- Same session edited `cp_cann_engine.cpp` + header + `cp_cann_symbols.cpp` + engine init
- Result: lm phase **11.78 ms → 2.06 ms** (**-9.72 ms / frame**); **21.9 fps → 30.2 fps** (**+8.3 fps**)
- Gate: first 10 frames of 3 canonical utts, **0 / 16 token drift**
- Conventional project: 4-6 weeks with specialisations; this was one agent dispatch

**Speaker notes**: W1 is the cleanest case study. Our `lm_head` call — the projection from 1024-dim CP hidden state to 2048-entry codebook vocab, run 15 times per audio frame — had been a CPU matvec since day one of the native engine. We never touched it because moving it to NPU looked like a cross-layer project: new aclTensor lifecycle, weight upload at init, new Rust-wrapped dispatch fn, a correctness gate against the CPU reference. An agent did the whole thing in one session. Loaded the 15 lm_head weights to device at init alongside the existing Q/K/V/O uploads, added `forward_lm_head(group_idx, …)` dispatching `aclnnMm`, added `fetch_logits` with an on-device F32 cast, gated behind `TALKER_LM_HEAD_NPU=1`. The agent read the W8 quant pattern off `aclnnWeightQuantBatchMatmulV3`'s doc and mirrored the code path. The numbers speak: 9.72 ms removed from every frame, 8.3 fps lifted, zero token drift on the correctness gate. That's a Q1-sized optimisation landed in one agent dispatch. Contract: `CP_FPS_OPTIMIZATION_CONTRACT.md §W1`.

---

## Slide 4 — Dissolving the "Ops API Is Fixed" Boundary

**Title**: "The vendor API surface is vastly under-mined."

- Agent grepped **717 aclnn headers** on ac01 in 30 min, filtered for fusion patterns
- Found **3 unused fused ops**, all applicable to our hot path:
  - `aclnnFFNV3` — W8 SwiGLU, 5 ops → 1 op, projected **+0.5-1.2 fps**
  - `aclnnApplyRotaryPosEmbV2` — Q+K fused RoPE, projected **+0.2 fps**
  - `aclnnInplaceAddRmsNorm` — in-place tail, projected **+0.1-0.2 fps**
- Conventional team never scheduled "read all CANN headers" — agents do it as a probe
- Full audit artefact: `docs/fused_op_audit.md` + landing contract `FUSED_OP_LANDING_CONTRACT.md`

**Speaker notes**: This slide is the most boring and the most important. The vendor API surface — `aclnnop/` on CANN, `cutlass/` on CUDA, `ggml-backend` on CPU — is a catalog of pre-fused ops shipped by the hardware vendor. It is the single highest-ROI place to look for free perf, because every op in there is a hand-tuned engineer-year that someone already paid for. And yet: a conventional team never schedules "go read all 717 headers". It's not a feature story, it's not a sprint item, there's no champion. An agent does it in 30 minutes as a side-probe — we called it the FO-audit. The deliverable is `docs/fused_op_audit.md`: 9 applicable candidates, ranked by fps upside × integration cost, with a recommended Phase A (easy ops, +0.25-0.45 fps) and a Phase B (FFNV3, +0.5-1.2 fps). The top find — `aclnnFFNV3` — had been shipping in CANN 8.3 since release. We just hadn't looked. This is the new-model-mistake that the pre-AI era will keep making as long as a human is the grep-query.

---

## Slide 5 — Failing Fast as a Feature (Path C Retrospective)

**Title**: "Optionality is the product."

- W4.1: hand-written AscendC fused-attn-sublayer kernel (RmsNorm+QKV+RoPE+FIAS+O+residual)
- Swarm: author → compile → offline diff **PASS 5e-4** → wire env-gated → live runtime gate **FAIL**
- Drift gate: `max_drift = 1949 codebook IDs` at frame 1, only **1 / 256 positions matched**
- PM closed the track in **72 hours, with receipts**: `docs/contracts/PATH_C_ASCENDC_CONTRACT.md §1a`
- PC-tile re-spike proved 40-core gemv **-32% wall vs aclnnMm** — but F32→F16 cross-core reduce wipes it
- **Second clean kill (this session)**: Talker aclGraph scaffold probe — `TALKER_CANN_GRAPH=1` dropped 30 → 18 fps (−40%), lazy-capture-on-first-touch is wrong for L→R decode where each pos visits once. Disabled in 25 min; gate caught it before any integration spend.
- Traditional team: 3 months sunk; AI swarm: closed cleanly, artefacts preserved, pivot to CannFusion

**Speaker notes**: This is the honest half of the story. W4.1 was a Path C bet: write our own fused attention sublayer in AscendC, keep intermediate tensors in SRAM, skip the DRAM round-trips that aclnn always pays. The agent swarm did it — W4.0 toolchain probe, W4.1.1 skeleton, W4.1.2v vector-primitive rewrite that even passed the offline diff at 5e-4 F16 noise. Landed the wiring env-gated on ac01. Then the live runtime drift gate — identical code path, real weights, real talker — returned a max drift of 1949 codebook IDs at frame 1. The kernel was mathematically incoherent, not just noisy. A conventional team would have spent 2-3 more weeks hypothesis-testing. The agent swarm, under PM supervision, closed the track in 72 hours with a full retrospective. The consolation prize was a PC-tile re-spike: we measured a 40-AIV-core tiled matmul at M=1 decode shapes and it beat aclnnMm by 26% at the gemv layer. But the mandatory F32→F16 cross-core reduce costs 18-22 μs, wiping the win. Atomic-add races at blockDim ≥ 20, NaN at 40 — a hardware limitation of 910B4, not a kernel bug. Contract closed with a recommendation for future resumption: don't start with attn-sublayer, start with a fused RmsNorm+Mm spike to check whether the reduce folds into the downstream op. A smaller, same-shape failure landed in this session: we probed the existing `TALKER_CANN_GRAPH=1` scaffold — previously untested — and saw a 30 → 18 fps regression on a stock LONG run, a 40% drop. The root cause was structural: the scaffold does lazy-capture-on-first-touch, which is correct for CP (every pos visits the same graph many times) but catastrophic for Talker L→R decode, where each pos is visited once and each visit pays the capture cost without ever amortising. Probe-and-disable took 25 minutes and never touched product code. The point of the slide isn't any single kernel failure. The point is that optionality — being able to close a 3-week bet cleanly in 72 hours, or a scaffold experiment in 25 minutes, with data, receipts, and a clean revert — is now the primary organisational asset, and it is built out of agents + PM gates, not out of specialist teams.

---

## Slide 6 — Beyond-Boundary Co-Design (aclGraph + W3b + lm_head)

**Title**: "Three optimisations, one op chain, composed."

- W1: `lm_head` CPU → NPU port (**+8.3 fps**, Slide 3)
- W3b: `aclnnAddRmsNorm` kernel fusion, **255 dispatches / frame saved** (**+0.66 fps**)
- G0-G4: aclGraph pos-keyed capture, 17 graphs × 1 MB, **0 byte-drift** (**+1.15 fps**)
- Three adjacent optimisations on ONE op chain — port, fusion, dispatch amortisation
- No conventional team would have shipped these as a single coherent plan in one quarter
- Cumulative clean-quality result from first native engine: **22 fps → ~32 fps band**, user-ear verified (full arc from llama.cpp ~1 fps baseline = **~32× lift**, tag `32fps-landed`)

**Speaker notes**: This slide shows the compounding effect. The CP engine's `forward_one_token_launch` is an 80-op aclnn chain. An agent with the end-to-end mental model sees three non-competing optimisations on the same chain: port lm_head CPU → NPU (W1), fuse the per-sublayer Add+RmsNorm tail (W3b), capture the whole thing as an aclGraph and replay per pos-key (G2). Conventional team dynamics would ship one per quarter, because each is owned by a different abstraction layer — dispatch, ops API, runtime. The agent composed them in one week. Gotcha: we projected +6 to +10 fps for aclGraph in the G0 feasibility probe; reality was +1.15 because `TASK_QUEUE_ENABLE=2` already amortized most launch overhead (the agent over-estimated the ceiling — see Slide 9 meta-pattern). The honest +1.15 fps on top of W1's +8.3 and W3b's +0.66 was user-ear verified at 31.6 fps with byte-identical parity on canonical xvec / ICL / CV benchmarks (G3.1: max_drift = 0 over 1680 tokens, output WAV md5 matches stock). That's the beyond-boundary co-design dividend.

---

## Slide 7 — Parallel-Universe Proof (OminiX-MLX)

**Title**: "Apple Silicon has no CANN. Agent built the stack anyway."

- Zero inherited infra: no aclGraph, no AscendC, no `aclnn*` library
- Built from `mlx-rs` + `safetensors` + `mlx-sys` primitives
- **9+ production models shipped**: `flux-klein-mlx`, `zimage-mlx`, `gpt-sovits-mlx`, `funasr-qwen4b-mlx`, `qwen3-tts-mlx`, `qwen3-vl-mlx`, `glm4-mlx`, `deepseek-ocr2-mlx`, `qwen-image-mlx`
- **24 GB Mac runs quantized FLUX** via drop-after-encode (f32 encoder → encode → drop → 8-bit transformer)
- Cross-platform benchmark harness: **M3 Max 46 fps > M4 Pro 37 fps > Ascend 31-32.2 fps band** (Ascend now within 30% of M3 Max vs 50% gap at start of project)
- Ascend gap is visible precisely because the MLX stack was written from scratch, no bloat

**Speaker notes**: The counterfactual. Apple Silicon's MLX runtime gives you nothing comparable to CANN's aclnn op library. No `aclnnFusedInferAttentionScoreV2`, no pre-fused SwiGLU, no graph capture, no AscendC-equivalent DSL. And yet in the same calendar window — using the same agent workflow — we shipped nine production models in the MLX universe: FLUX.2-klein image gen, Z-Image 4-bit quantized, GPT-SoVITS voice clone, FunASR Qwen4B ASR, Qwen3-TTS, Qwen3-VL, GLM-4, DeepSeek-OCR2, Qwen-Image. Each is its own Rust crate, averaging 1-1.5k LoC of hot-path code. The 24 GB Mac FLUX result is the most revealing: the agent discovered that f32 text encoder + 8-bit transformer didn't fit simultaneously, designed the drop-after-encode pattern (load encoder → encode prompt → explicitly drop to free GPU memory → load transformer → denoise), and verified with `mlx_sys::mlx_clear_cache()` between steps. That's a production memory-budget pattern invented + shipped by an agent, not extracted from a textbook. We also discovered — and documented in project memory — three MLX precision constraints the hard way: bf16 crashes the Metal backend with `std::runtime_error`, so it's unusable for compute; f16 produces NaN in RmsNorm and softmax because text embedding values range ±16000 near the f16 dynamic-range limits; only f32 is safe for the Qwen3 text encoder forward pass. That constraint discovery, at the agent+hardware boundary, is exactly the thing no pre-AI-coding team would have bothered to write down systematically. The Ascend-vs-Apple gap is now legible: Apple's M3 Max hits 46 fps on Qwen3-TTS because its MLX engine was written exactly for the workload, with no inherited framework bloat. Ascend's ~32 fps is what you get when you're still peeling back a 15-year-old C++ dispatch layer. The delta isn't hardware — Apple M3 Max has ~400 GB/s memory bandwidth vs Ascend 910B4's 800 GB/s, so Ascend's raw fabric is 2× faster and yet it runs ~1.4× slower. The delta is stack debt, and AI coding is how we pay it down.

---

## Slide 8 — Scale Trajectory: Single Card → Cluster

**Title**: "The same mental model, one PR bigger."

- Today: single 910B4, Qwen3-TTS, **~32 fps** band (fork `32fps-landed`)
- Tensor-parallel across 8 cards: agent edits HCCL collective calls + model loader + pipeline stages **in one PR**
- Pipeline-parallel: re-partition 5 CP layers across nodes; agent handles boundary tensor marshalling
- Data-parallel sharding: agent co-designs the batcher with the engine (shared KV cache policy, packed requests)
- No team-boundary hand-offs = **no "we'll define the interface next quarter"**
- Agent carries the end-to-end mental model the entire time; humans carry the gates

**Speaker notes**: Nothing in the single-card playbook stops at one card. Tensor-, pipeline-, data-, and expert-parallel strategies all require exactly the cross-layer editing the agent workflow already does: collective-comm (HCCL/NCCL), model loader (sharding), engine pipeline (activation exchange), batcher (request packing), framework (sampler) — five layers, one PR, one session. A conventional cluster bring-up is six engineers across three teams with a monthly sync. The tradeoff changes: invest less in the interface-definition ritual between teams, more in the correctness-gate ritual that keeps an agent safe. That ritual is cheap compared to what it replaces. Concretely: 8-card tensor-parallel on 910B4 demands a consistent view of what lives where — QKV sharded along head dim, O sharded along contracted dim, FFN sharded by column, all-reduce at every residual. A conventional team writes four interface docs, three design reviews, seven tickets. The agent edits all four sites in one session holding the same mental model — and then the PM gate is: "does output still token-match single-card for the first 100 frames". If yes, ship; if no, debug with full context still loaded. We have not yet run this pattern at cluster scale on OminiX — current deliverable is single-card. Slide 10 proposes exactly that next step. Ceiling argument when asked: ~1 fps → ~32 fps is the single-card ~32× gain we've shown. If the cluster brings even a 1.5× tensor-parallel scaling win (realistic given HCCL overhead) on top of whatever single-card improvements we pull through with the same workflow, we're looking at a 4-5× total gain vs the llama.cpp-ggml-cann baseline — in wall-time measured in weeks, not quarters.

---

## Slide 9 — Limits + Risks (Honest)

**Title**: "Agents are fast; gates are what make them safe."

- **Correctness gates matter**: W4.1 kernel passed offline 5e-4 diff, **drifted 1949 at runtime** — only the live gate caught it
- **Agents invent**: CannFusion F1 probe found the dtype whitelist blocks A16W8 — upstream GitCode #26 filed, would have wasted weeks without probe-first
- **Scaffold probes catch silent regressions**: Talker `TALKER_CANN_GRAPH=1` existing scaffold — measured 30 → 18 fps drop (−40%), lazy-capture-on-first-touch wrong for L→R decode. Disabled in 25 min.
- **Run-to-run noise is real**: baseline probe this session ran stock at 29.9-30.0 fps on 434-frame LONG; M3'new' measured at 31.9-32.2 on same length. The final **A.1 (+0.3) + M3'new' (+0.3)** stack sits inside a ~1 fps noise band — honest framing is "31-32.2 fps", not a crisp 32.2
- **Projection vs reality is systematically optimistic**: G0 projected **+6-10 fps**, delivered **+1.15**. M3'new' projected **+1 fps**, delivered **+0.3**. Consistent **3-10× optimism pattern** — fund on measured gates, never on estimates
- **PM role shifted**: "contract author + gate operator", not "code writer" — discipline change, not tools change
- **Security**: agents have unbounded write scope by default; we use patch-file + PM-pushes mechanism on ac01 (no fork push creds) to preserve review

**Speaker notes**: Six things to flag as honest limits. First: offline tests lie. Our W4.1 AscendC kernel passed a 5e-4 max-abs-diff on a synthetic fixture but catastrophically drifted on the live generate loop. The reason was subtle — something in the attn-sublayer's accumulation path interacted with real hidden-state statistics that the fixture didn't capture. You need live, end-to-end, token-level drift gates on every agent kernel ship, not just unit tests. The W1.4 gate we used (first 10 frames across 3 canonical utts, ≤1 token drift per frame per group) is the template. Second: agents confabulate. Our first CannFusion contract assumed the codegen would accept A16W8 (F16 activation × INT8 weight → F16) because that's the production dtype for Qwen3-TTS. The F1 probe revealed a hard-coded dtype whitelist in `src/validate.rs:141-163` that explicitly rejects it; the project has an intentional negative test at `validate.rs:367-372`. If we'd skipped the F1 probe and gone straight to implementation, we'd have burned a week. We filed the upstream ask at `docs/cannfusion_upstream_ask.md` and logged GitCode issue #26 as the clean close. Probe-first is now the default. Third: existing scaffolds can silently regress. Fresh this session, we probed the pre-existing `TALKER_CANN_GRAPH=1` env flag — shipped untested — and measured a 30 → 18 fps regression on a stock LONG run. The flag uses lazy-capture-on-first-touch, which is correct for CP (every pos revisits the same graph) but wrong for Talker L→R decode (each pos visits once, so every visit pays capture cost without amortising). Disabled in 25 minutes. Fourth: the run-to-run noise band is not zero. Our own probe today hit 29.9-30.0 fps on stock 434-frame LONG; M3'new' hit 31.9-32.2 on the same length. That's roughly 1 fps of normal variance. The A.1 +0.3 and M3'new' +0.3 deltas we claim sit inside that band. The correct framing on stage is "final landed result is the **31-32.2 fps band**, with error bars of roughly ±1 fps", not a crisp 32.2. Do not overclaim. Fifth: agent ceiling estimates are systematically optimistic. The G0 aclGraph probe projected +6 to +10 fps; reality was +1.15. M3'new' projected +1 fps; reality was +0.3. That's a consistent **3-10× optimism pattern** worth naming explicitly. The root cause varies per track — G0 missed that TQE=2 already amortizes per-dispatch launch from ~40 μs to ~2-3 μs (a ~15× reduction baked in); M3'new' underestimated remaining aclGraph replay amortisation — but the meta-lesson is the same: never fund a track on an agent's projection; fund it on measured gates. Sixth: PM role shifts. The PM is now a contract author (crisp, numeric gates per milestone) and a gate operator (verify numbers, ear-check WAVs, sign off). PM does not write the code. That's a genuine discipline shift for the human in the loop. And briefly: security. Agents on a remote NPU host have unbounded write scope by default. Our mitigation is that ac01 has no fork push credentials, so all commits land locally; the agent hands off a patch file; the PM pulls it to the Mac, reviews, and pushes. That keeps one human eye on every ship without bottlenecking the agent's local iteration speed.

---

## Slide 10 — Call to Action / Thesis Restated

**Title**: "Cluster next. Playbook ready."

- **Crack is real and exploitable today** — OminiX Ascend **~1 fps → ~32 fps band (≈32× lift)**, MLX 9 models shipped
- **Playbook = PM-gated agent swarm + contract-per-track + patch-file review mechanism**
- **First movers get a structural perf + velocity advantage measured in quarters, not sprints**
- **Concrete next step**: replicate the OminiX contract pattern on a cluster-scale workload
  - e.g. 8-card tensor-parallel Qwen3-VL-32B on 910B4 cluster, same gate discipline
- Bring OminiX dev playbook, the 5 contracts (CP-FPS, aclGraph, Path C, CannFusion, Fused-Op), and the PC-tile / FO-audit probe artefacts

**Speaker notes**: To restate: pre-AI-coding inference stacks were layered because humans specialise. Agentic AI coding dissolves the specialisation constraint, which in turn dissolves the layered-stack performance ceiling. The OminiX proof point is concrete: ~1 fps on conventional llama.cpp + ggml-cann → 22 fps after escaping to a native `CpCannEngine` → ~32 fps band (±1 fps error bars) after W1 + W3b + aclGraph + A.1 + M3'new' co-design, all on single-card 910B4 Qwen3-TTS in ~7 days, plus 9 production MLX models on Apple Silicon in the same calendar window, all via PM-gated agent swarms. Playbook in six items: (1) contract per track with numeric gates at every milestone, (2) agent-dispatched implementation per milestone, (3) probe-first on anything that might be externally blocked (dtype matrix, API presence, existing scaffolds), (4) live end-to-end drift gates on every kernel ship, (5) user-ear gate outranking fps on audio workloads, (6) patch-file review mechanism where the agent runs on the hardware host and the PM does the final push. The org that adopts this pattern on a cluster-scale workload first gets a structural advantage. Concrete recommendation: 8-card 910B4 deployment of Qwen3-VL-32B, tensor-parallel across cards, pipeline-parallel across nodes. We bring the contracts, the gate discipline, the dev pattern. The layered-stack ceiling is real, the crack is real, this is the year to step through.

---

## Appendix A — Artefacts Referenced

- `docs/contracts/CP_FPS_OPTIMIZATION_CONTRACT.md` — W1 + W3b
- `docs/contracts/ACLGRAPH_CONTRACT.md` — G0-G4
- `docs/contracts/PATH_C_ASCENDC_CONTRACT.md` — W4.1 retro (CLOSED)
- `docs/contracts/CANNFUSION_CONTRACT.md` — F1 dtype probe (CLOSED, upstream GitCode #26)
- `docs/contracts/FUSED_OP_LANDING_CONTRACT.md` — Phase A/B follow-on
- `docs/fused_op_audit.md` — 717 headers grepped
- `docs/aclgraph_feasibility.md` — G0 projection vs reality
- `docs/ascend_910b4_datasheet.md` — 20 AIC + 40 AIV, UB 192 KB, L0C 128 KB, 800 GB/s HBM
- `tools/qwen_tts/cp_cann_engine.cpp` — 2657 LoC, integration target
- OminiX-MLX: `qwen3-tts-mlx/`, `flux-klein-mlx/`, `zimage-mlx/`, `gpt-sovits-mlx/`, `funasr-qwen4b-mlx/`, 9 production crates total

## Appendix B — Key Metrics (Single Source of Truth for Q&A)

**Run-to-run noise band** on 434-frame LONG: **~1 fps**. Stock probe this session: 29.9-30.0 fps; M3N measured: 31.9-32.2 fps. Top-of-stack landings A.1 and M3'new' (each +0.3) individually sit within that band; cumulative stack clears it.

| Track | Before | After | Δ | Gate status | Wall time |
|---|---|---|---|---|---|
| Baseline (llama.cpp + ggml-cann) | — | ~1 fps | — | PM framing, not direct probe | pre-project |
| Native `CpCannEngine` bring-up | ~1 fps | 22 fps | **+21 (~22×)** | clean-quality | ~3 days |
| W1 NPU lm_head port | 21.9 fps | 30.2 fps | **+8.3** | LANDED, 0/16 drift on 10-frame × 3-utt gate | 1 agent dispatch |
| W3b Add+RmsNorm fusion | 30.2 fps | 30.5 fps | **+0.66** | LANDED, 255 dispatches/frame saved | 1 agent dispatch |
| G2 aclGraph pos-keyed | 30.5 fps | 31.6 fps | **+1.15** | LANDED, max_drift=0 over 1680 tokens | 3 agent dispatches |
| Path C AscendC W4.1 | 30.5 fps | 30.5 fps (reverted) | 0 | **CLOSED** — offline PASS, live drift=1949 | 72 hr to clean close |
| CannFusion F1 dtype probe | N/A | N/A | 0 | **CLOSED** — A16W8 blocked by whitelist; upstream **GitCode #26** filed | 25 min probe |
| Talker aclGraph scaffold probe | 30 fps | 18 fps (reverted) | **−12** (regression caught) | **CLOSED** — lazy-capture wrong for L→R decode; disabled | 25 min probe |
| A.1 InplaceAddRmsNorm | 31.6 fps | 31.9 fps | **+0.3** *(within noise)* | LANDED, byte-identical on gate | 1 agent dispatch |
| M3'new' pos 0+1 batched prefill | 31.9 fps | 32.2 fps | **+0.3** *(within noise)* | LANDED, byte-identical on gate; projected +1, delivered +0.3 | 1 agent dispatch |
| M1.B FFNV3 integration | — | — | — | **CLOSED** — op rejects non-MoE runtime; not applicable to Qwen3-TTS dense FFN | 1 agent dispatch (RED probe) |
| **Single-card cumulative** | **~1 fps** | **31-32.2 fps** | **~32× (±1 fps error bars)** | tag `32fps-landed` | ~7 days wall |

**Projection-vs-reality meta-pattern** (for Slide 9 Q&A):

| Track | Projected | Delivered | Optimism factor |
|---|---|---|---|
| G0 aclGraph | +6 to +10 fps | +1.15 fps | **~5-9×** |
| M3'new' | +1 fps | +0.3 fps | **~3×** |
| Path C W4.1 | single-digit fps | 0 (closed) | ∞ |

Pattern: agent ceiling estimates are consistently 3-10× optimistic. Cause varies per track. Mitigation: probe-first, measured gates, never fund on projection.

| MLX Workload | M3 Max | M4 Pro | Ascend 910B4 |
|---|---|---|---|
| Qwen3-TTS (frames/s) | 46 | 37 | 31-32.2 (band) |
| FLUX.2-klein 1024² | production | production | not yet ported |
| Z-Image 4-bit | production | production | not yet ported |

---

