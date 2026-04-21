# CP Codebook-Group Dependency Audit

Agent GD-audit, 2026-04-21. Pure static reading of MLX reference + Ascend
C++ implementations. No code changes, no benchmarks, no ssh.

## Sources consulted

Primary (dependency graph):
- `/Users/yuechen/home/OminiX-MLX/qwen3-tts-mlx/src/talker.rs:308-393`
  — `CodePredictor::generate_codes` (canonical Rust/MLX implementation).
- `/Users/yuechen/home/OminiX-MLX/qwen3-tts-mlx/src/generate.rs:237-250`
  — callsite inside the autoregressive frame loop.
- `/Users/yuechen/home/OminiX-Ascend/tools/qwen_tts/talker.cpp:1571-1899`
  — `TalkerLLM::predict_code_groups` (native CANN, llama.cpp, and CPU
  paths — all three paths).
- `/Users/yuechen/home/OminiX-Ascend/tools/qwen_tts/verify_code_predictor_flow.py`
  — Python reference that re-implements the CP flow two independent
  ways (with and without KV cache) and confirms they match.

Supporting (architecture / shape facts):
- `/Users/yuechen/home/OminiX-Ascend/tools/qwen_tts/export_qwen_tts.py:171-198`
  — CP config: `hidden_size=1024`, `num_hidden_layers=5`,
  `num_attention_heads=16`, `num_kv_heads=8`, `vocab_size=2048`,
  `num_code_groups=16`, `talker_hidden_size=2048`. 15 separate
  `codec_embeddings` and 15 separate `lm_heads`.
- `/Users/yuechen/home/OminiX-Ascend/docs/cp_forward_opt_exploration.md:52-63`
  — prior confirmation that KV cache is shared across groups (growing,
  reset per frame) — corroborates RVQ-style chain.

## Q1 — Token dependency: **YES (strict)**

Each group *i* ∈ [1, 15]'s forward pass receives the **embedding of the
token sampled at group i-1** as its input. No exceptions.

Canonical evidence — `talker.rs:372-389`:

```
for g in 1..15 {
    let token_arr = Array::from_slice(&[codes[g - 1] as i32], &[1, 1]);
    let embed = self.codec_embeddings[g - 1].forward(&token_arr)?;
    let mut current_input = self.small_to_mtp_projection.forward(&embed)?;
    ...
    for (layer, cache) in self.layers.iter_mut().zip(self.caches.iter_mut()) {
        current_input = layer.forward(&current_input, None, cache)?;
    }
    let normed = self.norm.forward(&current_input)?;
    let logits = self.lm_heads[g].forward(&normed)?;
    let token = mlx_rs::ops::indexing::argmax_axis(&logits.squeeze()?, -1, None)?;
    eval(std::iter::once(&token))?;
    codes.push(token.item::<u32>());
}
```

Note: `codec_embeddings[g-1].forward(&token_arr)` with `token_arr =
codes[g-1]` (the *just-sampled* integer token). The argmax is `eval`'d
**inside the loop** before the next iteration — `codes[g-1]` literally
has to be a host-side `u32` before group *g*'s forward can launch.

Corroborated by all three Ascend paths (`talker.cpp`):
- Native CANN loop (lines 1681-1734): `read_embedding_row(...,
  sampled, group_emb.data(), ...)` at 1712-1713, then
  `forward_one_token_launch(group_emb.data(), g+2, ...)` at 1723.
- llama.cpp loop (lines 1813-1837): identical pattern.
- CPU fallback (lines 1879-1896): identical pattern.

And corroborated by the Python reference
(`verify_code_predictor_flow.py:152-173` and 181-207), which
re-implements the flow both with and without KV cache and asserts they
produce **identical tokens** (line 209). That only holds if each
group's input strictly depends on the prior group's sampled token.

## Q2 — Hidden state sharing: **NOT SHARED**

Each group has its own transformer pass. The 5-layer CP transformer is
called **fresh for each group**, consuming a 1-token input
(the projected embedding of the previous group's sampled token) plus a
growing KV cache. The KV cache is what carries state across groups
(not a shared hidden).

Shapes per group forward:
- Input: `[1, 1, cp_hidden=1024]` (the projected previous-token emb).
- Attention attends to `seq_len = g+2` cache slots (pos 0 = talker
  hidden, pos 1 = group-0 emb, pos 2..g+1 = previously-sampled groups).
- Output: `[1, 1, 1024]`, then `lm_heads[g]` (group-specific 1024→2048
  GEMV) → logits → argmax → sampled token.

So "5 layers × 15 group-specific computations" is the correct
characterisation. There is no hoistable shared computation.

## Q3 — Partial-batching feasibility: **NOT FEASIBLE**

Because every group *i* ≥ 1 strictly depends on group *i-1*'s sampled
token (Q1), no intra-frame partial batch is possible. Even "batch
groups 0 and 1" would require sampling group-0's token *and* group-1's
token simultaneously, but group-1's forward needs group-0's
*embedding* (not its hidden state), and that embedding is indexed by
the sampled integer. The sampling is a hard argmax that cannot be
differentiated or softened away.

One theoretical caveat: group-0 logits come from the **Talker** (not
the CP). The CP only produces groups 1..15. That means:
- Group 0 is available at the start of each CP frame (before the CP
  even runs).
- Groups 1..15 are a strict 15-long RVQ chain inside the CP.

So the only "partial" batching option is to batch the 2-token prefill
(talker hidden + group-0 emb) at positions 0 and 1 — which the code
already does (`talker.rs:345-346`, concatenate_axis; `talker.cpp:1784-
1798`, `llama_batch` with `n_tokens=2`). That's already exploited.

## Q4 — RVQ vs parallel codebook: **STRICT RVQ**

Architectural verdict: this is textbook residual vector quantization
with a **depth/MTP transformer** (Meta-style "multi-token prediction"
head). Each group refines / resolves a residual that is conditioned on
the tokens already chosen. The 15 `codec_embeddings` are separate
tables precisely because each one embeds a different residual level's
token space. If codebooks were parallel, the model would share a
single embedding table and head per frame.

Paper anchor: Qwen3-TTS uses the "code predictor" design from the
Qwen2.5-Omni / MoonCast lineage (MTP depth transformer over RVQ
codebooks). The 12-Hz-1.7B-Base model card on HuggingFace describes
this as "residual codec head", which is the same family.

## Q5 — Inter-frame dependency: **CROSS-FRAME DOES NOT HELP**

At frame *t*, the CP takes two pieces of external state:

1. `talker_hidden` at frame *t* — produced by the **Talker**'s current
   forward step. This is the fresh hidden state that comes out of the
   last Talker layer at this frame (`talker.rs:442-470` returns `normed`
   which then flows into `code_predictor.generate_codes`).
2. `group0_token` at frame *t* — sampled from codebook-0 logits at
   frame *t*.

Neither input to the CP is frame *t-1*'s groups. But this doesn't
unlock batching either, because the *intra-frame* dependency is the
blocker (Q1). You can't batch the 15 groups of frame *t* using frame
*t-1*'s tokens as input — they aren't what's fed in.

Frame-level parallelism is independently impossible because the Talker
itself is strictly autoregressive (each frame's text + codec input
depends on the 16 tokens sampled at the previous frame — see
`generate.rs:262` building `input_embed` from `frame` then
`forward_step`).

## Verdict: group-collapse feasibility

- [ ] FULL-COLLAPSE
- [ ] PARTIAL-COLLAPSE
- [x] **NO-COLLAPSE**: strict 15-step RVQ chain. Algorithmic
      group-collapse is **not available** on Qwen3-TTS. The fps lever
      described in the hypothesis (15 GEMVs → 1 GEMM batch=15) cannot
      be realised without retraining the model with a parallel-head
      architecture.

## Engineering estimate

Not applicable for the winning path (there is no winning path in this
direction). Any implementation attempt would produce incorrect audio —
the model's RVQ training objective requires the chain. Verified by the
Python reference's assertion that KV-cached and growing-window
implementations produce **bit-identical** tokens
(`verify_code_predictor_flow.py:209`); both use the chain.

## Recommendation for PM

**Stop pursuing CP group-collapse.** The architecture is strict 15-step
residual vector quantization with per-group codec embeddings and
per-group lm_heads; each group's input is the embedding of the
previous group's freshly-sampled token. This is confirmed in **four
independent sources**: the MLX `CodePredictor::generate_codes`
implementation, all three Ascend C++ paths (native CANN, llama.cpp,
custom CPU), and the two Python reference flows (cached and
growing-window, proven token-identical).

Redirect the +15-20 fps target at the lever described in
`cp_forward_opt_exploration.md` — **within-group kernel work** (W8
quantization, aclGraph capture, NPU-resident lm_head, cross-stream
overlap). Those don't collide with the chain.

One small win that *does* exist and is already taken: the 2-token
prefill at positions 0+1 (talker_hidden + group-0 emb) is already
batched in both MLX (`concatenate_axis`) and Ascend llama.cpp
(`llama_batch` with `n_tokens=2`). Ensuring the native CANN path also
uses a single 2-token launch here rather than two sequential
`forward_one_token_launch` calls (currently it issues two — see
`talker.cpp:1647` and `1667`) is a ~1 ms/frame micro-opt, not a
~+15 fps lever.
