# Q0.5.2 — FIAv2 capability audit at seq=4352 (image-token attention)

**Agent**: QIE-Q0.5
**Date**: 2026-04-22
**Mode**: source-read + prior-art synthesis only. Full wall-time measurement at seq=4352 on ac02 is **Q1-gated** — backend is RED so standalone FIAv2 harness must run without the full model (achievable but not in Q0.5 budget; `fia_v34_probe_verdict.md` shows the harness pattern).
**Question**: Is `aclnnFusedInferAttentionScoreV2` (or V3/V4) a viable substitute for ggml's current `ggml_ext_attention_ext` at QIE's joint-attention sequence length (4096 img + ~256 txt = **4352 tokens**, head_dim=128, num_heads=24, num_kv_heads=24)?

## TL;DR

**VERDICT: LIKELY GREEN on capability axis. Proceed to runtime probe at Q3 (after Q1 backend unblock).** FIAv2 is a standalone fused attention op with no GQA restriction, no MoE gating, no expert-token requirement. It supports MHA directly (which is what QIE uses — `num_heads == num_kv_heads == 24`). The known TTS-side historical failures were specifically GQA packed-UB issues (MHA case is the op's native shape) and one integration/wiring bug — neither applies to QIE.

The uncertainty is on the **runtime-performance axis** at seq=4352: TTS probed at seq ≤ 16, so we have no precedent for how FIAv2 scales into the 4k regime relative to ggml's Flash-style kernel. This is exactly the regime Flash attention is designed for and the regime where naive softmax loses most. A 3-5 day runtime probe at Q3 resolves it.

## Source context

### Current call site

`tools/ominix_diffusion/src/rope.hpp:644-660` is the single QIE attention entry point:

```cpp
__STATIC_INLINE__ struct ggml_tensor* attention(GGMLRunnerContext* ctx,
                                                struct ggml_tensor* q,
                                                struct ggml_tensor* k,
                                                struct ggml_tensor* v,
                                                struct ggml_tensor* pe,
                                                struct ggml_tensor* mask,
                                                float kv_scale        = 1.0f,
                                                bool rope_interleaved = true) {
    // q,k,v: [N, L, n_head, d_head]
    // pe:    [L, d_head/2, 2, 2]
    q = apply_rope(ctx->ggml_ctx, q, pe, rope_interleaved);  // [N*n_head, L, d_head]
    k = apply_rope(ctx->ggml_ctx, k, pe, rope_interleaved);

    auto x = ggml_ext_attention_ext(
        ctx->ggml_ctx, ctx->backend, q, k, v, v->ne[1], mask,
        true, ctx->flash_attn_enabled, kv_scale);   // [N, L, n_head*d_head]
    return x;
}
```

`ggml_ext_attention_ext` with `flash_attn_enabled=true` dispatches ggml-cann's flash path (calls `ggml_cann_flash_attn_ext` internally). That's what we're benchmarking against, not naive softmax.

### QIE attention shape

Per `qwen_image.hpp:350-362` + `:414-420`:
- `num_attention_heads = 24`
- `attention_head_dim = 128`
- `num_kv_heads = num_attention_heads` (**MHA, not GQA**)
- Joint attention: img tokens (e.g. 4096 for 64×64 latent at patch_size=2) + ref-image tokens (edit mode, ~4096 for a same-sized reference) + text tokens (~256).

For Q0.5 target scope (**512×512 output, no ref-image concat for a first baseline**), sequence length is `(512/8 * 512/8) + ~256 = 4352`. Edit mode with a 512×512 reference bumps L to ~8448. Attention is `[batch, n_head=24, L, d_head=128]`.

## FIAv2 / V3 / V4 capability matrix

Synthesised from `docs/fia_v34_probe_verdict.md` (TTS-side probe, 2026-04-21, ac01, CANN 8.3.RC1, byte-identical at seq ≤ 16) and the V2/V3/V4 header diff in that document:

| Capability | QIE requirement | FIAv2 support | Source |
|---|---|---|---|
| `numHeads == numKeyValueHeads` (MHA) | 24 / 24 | Yes (native MHA shape; GQA is the *added* path) | FIA-V34 probe §header-diff, V2 already exposes `numKeyValueHeads` |
| Head dim | 128 | Yes (128 is the canonical head dim across Qwen2.5/3 family; FIA supports 64/80/96/128/192/256 per vendor guidance) | TTS prod shape probe at head_dim=128 passed byte-identical |
| Layout | BSND (`layout=1`) or BNSD | Both accepted | FIA-V34 probe §ship-candidates line 52 |
| Dtype | F16 or BF16 (QIE uses BF16 under `GGML_CANN_QUANT_BF16=on`) | F16 PASS confirmed on 910B4 at CANN 8.3.RC1; BF16 is a documented input dtype in the op header (per FIA-V34 header matrix, additive to V2) | TTS probe is F16; BF16 needs 30-min dtype toggle in Q3 probe |
| Softmax mode | Standard softmax (no sink attention for QIE) | Yes; `learnableSink` is V4-optional, passing `nullptr` degenerates cleanly | FIA-V34 probe §ship-candidates line 52-53 |
| `inputLayout` enum | BSND | Accepted | TTS probe |
| `softmaxLseFlag` | Not required (inference, no training LSE export) | Optional int64 knob; `0` = no-op | FIA-V34 header table |
| `sparseMode` | None (full joint attention, no causal mask on image tokens) | `0` = full attention | FIA-V34 header table |
| Mask | Optional 2D mask (`mask` arg in `attention(...)` is currently usually NULL for image tokens, mask used for text-only padding) | Mask optional; V2 accepts `nullptr` | Standard |
| MLA / decoupled RoPE | Not used (QIE applies RoPE externally via `apply_rope` then feeds rotated Q/K into attention) | N/A (V3 feature, passing `nullptr` to `queryRopeOptional` degenerates) | FIA-V34 probe §header-diff |
| Quant (A16W8 / A8W8 Q dequant) | Not used (QIE weights are quant but attention Q/K/V are activations in F16/BF16) | N/A (V4 feature, optional) | — |

**All capability requirements for QIE attention are satisfied by FIAv2 alone. V3/V4 add nothing QIE needs.**

## Prior-art from TTS / ASR probes (relevant limits)

From `docs/qwen_tts_optimization_learnings.md` and `docs/fia_v34_probe_verdict.md`:

### What PASSED at seq ≤ 16

- `FIAv2` with Q `[1,1,16,128]`, K/V `[1,16,8,128]` on F16 BSND GQA 2:1 → byte-identical vs naive baseline, 76 μs median at CANN 8.3.RC1.
- FIAv2/V3/V4 numerically byte-identical — op family is forward-compatible.

### What FAILED in TTS history (none apply to QIE)

1. **V2 RoPE drift (A.2 contract, ×3 closed):** the `aclnnApplyRotaryPosEmbV2` op, not FIAv2. Re-probed 2026-04-21 (`v2_rope_reopen_verdict.md`) → op is numerically sound; the divergence was a wiring/workspace-lifetime bug. **Not an FIAv2 issue.**
2. **FFNV3 A16W8 rejection:** FFN op family, not FIA. Dtype whitelist is FFN-specific. **Not an FIAv2 issue.**
3. **CannFusion A16W8 whitelist:** compiler-level validator for a different op path. **Not an FIAv2 issue.**
4. **GQA packed-UB (hypothesized H5a in A.2 original):** concern was that V2 RoPE (not FIA) couldn't handle Nq≠Nkv. **Falsified** on the RoPE side (per v2_rope_reopen_verdict YELLOW). QIE is **MHA (Nq==Nkv)** anyway — even if a GQA-only limit existed on some future FIA variant, it wouldn't bite QIE.

### What is unknown for seq=4352

FIAv2 probe coverage on ac01/ac02 is at TTS decode shapes (seq=1..16). We have **no direct measurement** of FIAv2's behavior at 4k-8k sequence length. Flash-style kernels (which FIAv2 descends from based on header taxonomy) typically get **most** of their win vs naive softmax in the 512-8192 regime where HBM traffic for intermediate softmax tensors dominates. The runtime question is whether FIAv2's internal blocking and online-softmax hold up at seq=4352 on 910B4 vs ggml-cann's `ggml_cann_flash_attn_ext`, which is already flash-style.

## Honest performance envelope

Three possible outcomes at Q3 runtime probe (ranked by prior likelihood):

1. **FIAv2 wall within 20% of ggml's flash-attn-ext on 910B4.** Wire it at Q3, picking up dispatch-count reduction (one fused op vs ggml's multi-op chain inside the flash path) and potentially better workspace reuse under WSPOOL. **+4-10% per step** per the contract's Q3 target. This is the contract's expected path.
2. **FIAv2 >30% slower than ggml-flash at seq=4352.** Retire Q3 wiring, document as CANN vendor ask. Likely root cause would be FIAv2's internal block size not tuned for the 4k regime (FIA's lineage is LLM decode / short prefill, not DiT image attention). Vendor ask: "FIAv2 tile config for head_dim=128 seq=4k-8k DiT workloads, or a sibling op `aclnnFusedAttentionScore2D` targeting the image-token regime."
3. **FIAv2 byte-divergent vs ggml-flash.** Extremely unlikely given TTS probe byte-identity; but if it happens, the probe catches it before engine wiring, not after frame-count drift.

## Verdict

**(a) LIKELY GREEN** on the capability axis. Proceed to Q3 runtime probe once Q1 backend is unblocked. No capability blockers — QIE's MHA / F16-or-BF16 / seq=4352 / head_dim=128 / standard-softmax / no-sink / no-MLA shape is exactly what FIAv2 is designed for.

**Do not deprioritize Q3.** Prior-art gives high confidence the op works; the only unknown is perf-delta vs ggml-flash at 4k seq. The 3-5 day Q3 probe is the cheapest way to resolve that.

## Q3 runtime probe spec (1-pager for the Q3 agent)

Recommended harness structure, inheriting the FIA-V34 probe's dlsym-based standalone pattern (no engine deps):

1. Allocate Q/K/V at `[1, 24, 4352, 128]` F16 BSND on ac02.
2. Three paths under same tolerance:
   - Path A: ggml-flash-attn-ext reference wall (measured via `ggml_ext_attention_ext` shim standalone, or via existing ggml_cann dispatcher).
   - Path B: FIAv2 direct dispatch (mirror `fia_v34_probe_verdict.md` harness at new seq len).
   - Path C: FIAv2 at seq=8192 (stretch, edit-mode with 512×512 ref image).
3. 5 warmup + 50 timed iter, median μs, fresh executor per iter (CANN one-shot semantics).
4. Gate: FIAv2 within 20% of ggml-flash → GREEN, wire at Q3. 20-30% slower → YELLOW, park. >30% slower → RED, CANN vendor ask.

Expected wall at seq=4352 on 910B4 (order-of-magnitude sanity): Q·K^T is `24 × 4352 × 4352 × 128` FLOPs × 2 = ~58 GFLOPs per forward per head, ×2 for softmax-times-V. At 910B4's ~400 TFLOPS dense F16, compute floor is ~0.3 ms/head, total 7 ms/attention; 60 blocks × 20 steps × 2 CFG = 2400 calls × 7 ms ≈ 17 s attention-only at compute-floor. HBM-floor is lower (attention is compute-bound at this seq). FIAv2 vs ggml-flash delta should be second-order vs this floor — probe cost is ~5 μs × 50 = 0.25 ms of NPU time per path, total probe wall ~15 sec.

## Residual items

- **BF16 dtype** (QIE's default under `GGML_CANN_QUANT_BF16=on`) — TTS probed F16 only. 30-minute dtype toggle in the Q3 harness closes this. If BF16 has an FIA-family restriction we haven't seen, falls back to F16-attention-with-BF16-weights hybrid.
- **Mask shape at joint text+img attention.** Ggml builds mask internally; FIAv2 takes a flat 2D mask. Wiring task at Q3, not a capability question.
- **Edit-mode seq=8448** (with ref-latent concat). Likely fine given FIA has no documented seq ceiling below 32k; include seq=8192 in Q3 probe as belt-and-braces.
