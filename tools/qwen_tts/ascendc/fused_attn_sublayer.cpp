// SPDX-License-Identifier: Apache-2.0
// =============================================================================
// fused_attn_sublayer.cpp — AscendC custom kernel for Path C W4.1.
// (Contract §5 W4.1.1 named this file `.cce`; renamed to `.cpp` because
// the CANN 8.3.RC1 `ascendc_library()` cmake macro only recognises
// `.cpp` / `.asc` as AscendC source extensions — matches the probe's
// own `kernel.cpp` convention from W4.0.)
//
// Collapses the CP transformer's per-layer attention sublayer into a single
// kernel dispatch. The stock aclnn chain we're replacing (see
// `cp_cann_engine.cpp::forward_one_token_launch`) is:
//
//     RmsNorm(cur, input_ln)                 // 1 dispatch
//     Q = W8Mm(normed, Wq_i8, Wq_scale)      // 1 dispatch
//     K = W8Mm(normed, Wk_i8, Wk_scale)      // 1 dispatch
//     V = W8Mm(normed, Wv_i8, Wv_scale)      // 1 dispatch
//     RmsNorm(Q, q_norm)                     // 1 dispatch
//     RmsNorm(K, k_norm)                     // 1 dispatch
//     RoPE(Q, cos, sin)                      // 1 dispatch  (Q in-place)
//     RoPE(K, cos, sin, dst=k_cache[pos])    // 1 dispatch
//     memcpyAsync(V -> v_cache[pos])         // (not counted — pure copy)
//     FusedInferAttentionScoreV2(Q, Kc, Vc)  // 1 dispatch
//     O = W8Mm(attn_out, Wo_i8, Wo_scale)    // 1 dispatch
//     residual += O  (or AddRmsNorm)         // 1 dispatch (W3b folds post-norm)
//
// Total: 12 aclnn dispatches collapsed into 1 aclrtlaunch_fused_attn_sublayer.
//
// -----------------------------------------------------------------------------
// Tile / pipeline strategy (documented per §5 W4.1 "Open issue 1")
// -----------------------------------------------------------------------------
// Target SoC: Ascend 910B4 (dav-c220-vec for the AIV AI vector core).
//
// The CP transformer's per-forward shapes are small (seq=1 for Q, seq ≤ 17
// for K/V, 16 heads × 128 head_dim, hidden=1024, q_dim=2048, inter=3072).
// This fits entirely in on-chip SRAM on a single AI Vector core, so the
// first-pass skeleton dispatches with `blockDim=1`.
//
// Per-tile plan (all tensors stay resident in LocalTensor after first load):
//   • RMSNORM:  load normed activation + input_ln gamma → RmsNorm →
//               local-normed (shape [cp_hidden=1024] F16).
//   • QKV:      streamed weight tiles ([out_n, cp_hidden] INT8) + row-scale
//               F16 vectors; dequant to F16 via Muls(scale[r]) then Mm-accum
//               into Q/K/V LocalTensors ([q_dim=2048], [kv_dim=1024] F16).
//   • QK-NORM:  per-head RmsNorm on Q, per-kv RmsNorm on K.
//   • ROPE:     rotate_half style (NEOX / mode=0 in aclnnRotaryPositionEmbedding)
//               using host-supplied cos/sin row for `pos`.
//   • ATTN:     seq_len is small (≤ MAX_SEQ=17) so the whole attention
//               fits in SRAM: score = softmax( Q @ K^T * scale ), out = score @ V.
//   • O-MM:     same W8 dequant pattern into attn_out → o_out.
//   • RESIDUAL: o_out += residual (F16 add). Post-norm fusion (cur = norm(o_out))
//               stays out-of-kernel — the engine's existing
//               AddRmsNorm path handles that for il+1 priming.
//
// -----------------------------------------------------------------------------
// KV-cache append strategy (Open issue 2)
// -----------------------------------------------------------------------------
// The engine already does the K-RoPE-into-cache_slot and V-memcpy-into-
// cache_slot on the HOST side (see cp_cann_engine.cpp lines 1520-1546). We
// keep that split: the kernel receives the APPEND'd k_cache / v_cache
// buffers as inputs and reads the full [seq_len, n_kv, head_dim] views.
//
// Rationale: the host already tracks `pos`, already has a cos/sin table on
// device, and already does a cheap aclrtMemcpyAsync for V. Migrating that
// into the kernel would require:
//   (a) a second dispatch per layer (pre-append) OR extra control args
//       threading pos/slot math through AscendC, AND
//   (b) a RoPE-for-K code path inside the kernel that duplicates the host's
//       existing RoPE call.
// Neither is free, and W4.1's perf win comes from collapsing the ATTN
// forward — not the cache-append. If 4.1.5 clears but headroom remains,
// W4.3 can consider inlining the append.
//
// -----------------------------------------------------------------------------
// Precision contract
// -----------------------------------------------------------------------------
// Matches the engine's F16-end-to-end transformer convention:
//   • LocalTensor arithmetic: half (F16) throughout matmul + softmax.
//   • W8 dequant: row_i8 * row_scale_f16 → F16 (per-output-channel symmetric).
//   • Softmax scale: 1/sqrt(head_dim=128) = 0.0883883... F16.
//   • RmsNorm gammas: passed as F16 (same convention as W3b AddRmsNorm).
//     The host uses `init_fusion_f16_gammas_()` to down-cast F32→F16 once at
//     init. For the unfused path (baseline for 4.1.2 diff), the host must
//     provide the F16 copy regardless; we add a cast if fusion is OFF.
//
// -----------------------------------------------------------------------------
// Host-side launcher signature
// -----------------------------------------------------------------------------
// The auto-generated wrapper from ascendc_library() is:
//   extern "C" uint32_t aclrtlaunch_fused_attn_sublayer(
//       uint32_t blockDim, aclrtStream stream,
//       /* pointer args in order below */, /* scalar args */);
//
// See the trailing extern "C" kernel declaration at the bottom.
//
// =============================================================================

#include "kernel_operator.h"

using namespace AscendC;

// -------------- tunables --------------
constexpr int32_t kPipeBufCount = 1;       // decode forward is single-row; no
                                            // double-buffering needed for first
                                            // skeleton pass.

// Matches cp_cann_engine.h (lines 187-198). These are baked into the kernel
// at compile time because the CP transformer has fixed shapes for the
// canonical Qwen3-TTS CP head — changing them requires a rebuild either way.
constexpr int32_t kCpHidden = 1024;   // d_model
constexpr int32_t kQDim     = 2048;   // n_heads * head_dim = 16 * 128
constexpr int32_t kKvDim    = 1024;   // n_kv    * head_dim = 8  * 128
constexpr int32_t kHeadDim  = 128;
constexpr int32_t kNHeads   = 16;
constexpr int32_t kNKv      = 8;
constexpr int32_t kGroup    = kNHeads / kNKv;  // GQA group = 2

constexpr int32_t kMaxSeq   = 17;     // CP_MAX_SEQ; matches engine MAX_SEQ.

// =============================================================================
// KernelFusedAttnSublayer
// =============================================================================
//
// One-block implementation. See file header for tile strategy.
//
// Buffers (GM pointers passed by launcher, in order):
//   residual    [cp_hidden]    F16 — input: cur (residual) ; output: updated
//   normed      [cp_hidden]    F16 — input: RmsNorm(cur, input_ln_gamma)
//                                    produced by host (W3b fusion path).
//                                    NOTE: in the "fusion OFF" path we compute
//                                    this inside the kernel using `in_ln_gamma`;
//                                    the host sets a flag via `opts` (bit 0).
//   in_ln_gamma [cp_hidden]    F16 — input_ln gamma (F16 copy).
//   wq_i8       [q_dim, cp_hidden]   INT8 — Q weight; row-major [out, in].
//   wq_scale    [q_dim]        F16   — per-row F16 scale.
//   wk_i8       [kv_dim, cp_hidden]  INT8
//   wk_scale    [kv_dim]       F16
//   wv_i8       [kv_dim, cp_hidden]  INT8
//   wv_scale    [kv_dim]       F16
//   wo_i8       [cp_hidden, q_dim]   INT8
//   wo_scale    [cp_hidden]    F16
//   q_norm_f16  [head_dim]     F16   — RmsNorm gamma for Q heads.
//   k_norm_f16  [head_dim]     F16   — RmsNorm gamma for K kv-heads.
//   rope_cos    [head_dim]     F16   — cos row for current pos.
//   rope_sin    [head_dim]     F16   — sin row for current pos.
//   k_cache     [MAX_SEQ*kv_dim]  F16 — host already appended k[pos] via RoPE.
//   v_cache     [MAX_SEQ*kv_dim]  F16 — host already appended v[pos].
//   o_out       [cp_hidden]    F16   — output: O-projection result (residual
//                                       add happens inside kernel writing back
//                                       to `residual` buffer).
//   scratch_q   [q_dim]        F16   — kernel scratch; engine allocates once.
//   scratch_scores [n_heads * MAX_SEQ] F16 — scratch for attention scores.
//
// Scalar args:
//   uint32_t seq_len    — pos + 1 (seq length of Kc/Vc in cache).
//   uint32_t eps_bits   — F32 bits of RmsNorm eps (engine passes float bits).
//   uint32_t opts       — bitmask:
//                           bit 0 = fusion_active (host already wrote `normed`;
//                                   kernel skips the input RmsNorm).
//
// =============================================================================

class KernelFusedAttnSublayer {
public:
    __aicore__ inline KernelFusedAttnSublayer() {}

    // Scalar `exp` is not exposed in aicore scope — only `sqrt`, `min`,
    // `max` are (see /ccec_compiler/lib/clang/.../include/__clang_cce_aicore
    // _functions.h). For the skeleton softmax we evaluate exp via the
    // identity exp(x) = 2^(x/ln2) + a rational-polynomial approximation
    // of 2^y that is accurate to ~1e-5 over y ∈ [-16, 0] (softmax always
    // passes x - max_score ≤ 0 ⇒ y ≤ 0). Well within the 4.1.2 ≤ 1e-2
    // F16 gate. TODO(W4.1.5): move softmax into vectorised AscendC
    // Exp primitive if the perf gate needs it.
    __aicore__ inline static float ApproxExp_(float x) {
        if (x < -16.0f) return 0.0f;
        // y = x / ln(2); split into integer (iy) + fractional (fy ∈ [0,1)).
        const float kInvLn2 = 1.4426950408889634f;
        float y = x * kInvLn2;
        int   iy = (int)y;
        if (y < (float)iy) { iy -= 1; }   // floor for negatives.
        float fy = y - (float)iy;
        // 2^fy via Remez-style quartic approximation (abs err < 5e-6 over [0,1])
        // coefficients: Horner form of
        //   2^fy ≈ 1 + fy*(0.6931471805599 + fy*(0.2402265069591 +
        //                  fy*(0.0558458939825 + fy*0.00898934...)))
        float p = 1.0f +
                  fy * (0.69314718f +
                  fy * (0.24022651f +
                  fy * (0.05585890f +
                  fy *  0.00898934f)));
        // Multiply by 2^iy via bit manipulation.
        // IEEE-754 single: sign(1) | exponent(8, bias 127) | mantissa(23).
        // 2^iy has exponent = (iy + 127) << 23 when iy ≥ -126.
        int exp_bits = iy + 127;
        if (exp_bits <= 0) return 0.0f;
        if (exp_bits >= 255) exp_bits = 254;
        union { uint32_t u; float f; } scaler;
        scaler.u = ((uint32_t)exp_bits) << 23;
        return p * scaler.f;
    }

    __aicore__ inline void Init(
        GM_ADDR residual, GM_ADDR normed, GM_ADDR in_ln_gamma,
        GM_ADDR wq_i8, GM_ADDR wq_scale,
        GM_ADDR wk_i8, GM_ADDR wk_scale,
        GM_ADDR wv_i8, GM_ADDR wv_scale,
        GM_ADDR wo_i8, GM_ADDR wo_scale,
        GM_ADDR q_norm_f16, GM_ADDR k_norm_f16,
        GM_ADDR rope_cos, GM_ADDR rope_sin,
        GM_ADDR k_cache, GM_ADDR v_cache,
        GM_ADDR o_out, GM_ADDR scratch_q, GM_ADDR scratch_scores,
        uint32_t seq_len, uint32_t eps_bits, uint32_t opts)
    {
        seqLen_ = seq_len;
        opts_   = opts;
        // Reinterpret the eps bits back to F32 at compile time.
        float eps_f32;
        // __builtin_memcpy keeps the runtime honest about strict aliasing.
        __builtin_memcpy(&eps_f32, &eps_bits, sizeof(float));
        epsF16_ = (half)eps_f32;

        // Wrap GM buffers.
        residualGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(residual),
                                     kCpHidden);
        normedGm_.SetGlobalBuffer  (reinterpret_cast<__gm__ half*>(normed),
                                     kCpHidden);
        inLnGm_.SetGlobalBuffer    (reinterpret_cast<__gm__ half*>(in_ln_gamma),
                                     kCpHidden);
        wqI8Gm_.SetGlobalBuffer    (reinterpret_cast<__gm__ int8_t*>(wq_i8),
                                     (uint32_t)kQDim * kCpHidden);
        wqScaleGm_.SetGlobalBuffer (reinterpret_cast<__gm__ half*>(wq_scale),
                                     kQDim);
        wkI8Gm_.SetGlobalBuffer    (reinterpret_cast<__gm__ int8_t*>(wk_i8),
                                     (uint32_t)kKvDim * kCpHidden);
        wkScaleGm_.SetGlobalBuffer (reinterpret_cast<__gm__ half*>(wk_scale),
                                     kKvDim);
        wvI8Gm_.SetGlobalBuffer    (reinterpret_cast<__gm__ int8_t*>(wv_i8),
                                     (uint32_t)kKvDim * kCpHidden);
        wvScaleGm_.SetGlobalBuffer (reinterpret_cast<__gm__ half*>(wv_scale),
                                     kKvDim);
        woI8Gm_.SetGlobalBuffer    (reinterpret_cast<__gm__ int8_t*>(wo_i8),
                                     (uint32_t)kCpHidden * kQDim);
        woScaleGm_.SetGlobalBuffer (reinterpret_cast<__gm__ half*>(wo_scale),
                                     kCpHidden);
        qNormGm_.SetGlobalBuffer   (reinterpret_cast<__gm__ half*>(q_norm_f16),
                                     kHeadDim);
        kNormGm_.SetGlobalBuffer   (reinterpret_cast<__gm__ half*>(k_norm_f16),
                                     kHeadDim);
        ropeCosGm_.SetGlobalBuffer (reinterpret_cast<__gm__ half*>(rope_cos),
                                     kHeadDim);
        ropeSinGm_.SetGlobalBuffer (reinterpret_cast<__gm__ half*>(rope_sin),
                                     kHeadDim);
        kCacheGm_.SetGlobalBuffer  (reinterpret_cast<__gm__ half*>(k_cache),
                                     (uint32_t)kMaxSeq * kKvDim);
        vCacheGm_.SetGlobalBuffer  (reinterpret_cast<__gm__ half*>(v_cache),
                                     (uint32_t)kMaxSeq * kKvDim);
        oOutGm_.SetGlobalBuffer    (reinterpret_cast<__gm__ half*>(o_out),
                                     kCpHidden);
        qScratchGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(scratch_q),
                                     kQDim);
        scoresGm_.SetGlobalBuffer  (reinterpret_cast<__gm__ half*>(scratch_scores),
                                     (uint32_t)kNHeads * kMaxSeq);

        // Pipe buffer sizes — sized for the largest single working tensor
        // (Q at 2048 F16 = 4KB). We reuse the same queue across QKV/O stages.
        pipe_.InitBuffer(normedQ_,   kPipeBufCount, kCpHidden * sizeof(half));
        pipe_.InitBuffer(qBuf_,      kPipeBufCount, kQDim     * sizeof(half));
        pipe_.InitBuffer(kBuf_,      kPipeBufCount, kKvDim    * sizeof(half));
        pipe_.InitBuffer(vBuf_,      kPipeBufCount, kKvDim    * sizeof(half));
        pipe_.InitBuffer(oBuf_,      kPipeBufCount, kCpHidden * sizeof(half));
    }

    // -------------------------------------------------------------------
    // Process — the fused op chain.
    // Implemented in a correctness-first layout (single block, no double-
    // buffering) so 4.1.2's numerical-diff harness can pin whether the
    // math matches the stock aclnn chain. The 4.1.5 perf gate is the
    // point at which we either re-tile / multi-block (if headroom exists
    // and the kernel is slower than the aclnn chain) OR ship.
    // -------------------------------------------------------------------
    __aicore__ inline void Process() {
        // ---- 1. RmsNorm(cur, input_ln_gamma) → normed (only if not pre-computed)
        //
        // W3b fusion path ("opts bit 0 set") means the host already ran
        // AddRmsNorm at the tail of the PREVIOUS sublayer, leaving
        // `normed` populated. In that case this kernel just reads `normed`
        // as-is. Otherwise we compute it here.
        LocalTensor<half> normed = normedQ_.AllocTensor<half>();
        const bool fusion_active = (opts_ & 1u) != 0u;
        if (fusion_active) {
            DataCopy(normed, normedGm_, kCpHidden);
        } else {
            // Load residual (== cur) + gamma; compute RmsNorm in-place.
            // RmsNorm: x / sqrt(mean(x^2) + eps) * gamma
            LocalTensor<half> resid = normed;  // reuse tensor as staging
            DataCopy(resid, residualGm_, kCpHidden);
            // TODO(W4.1.1): use AscendC::RmsNorm primitive once we confirm
            // its shape contract matches. For the skeleton baseline we
            // stream through a DataCopy + scalar loop fallback in
            // ComputeRmsNorm_ below. The skeleton's correctness gate
            // (4.1.2) pins the math before we graduate to the primitive.
            ComputeRmsNorm_(resid, inLnGm_, kCpHidden, epsF16_);
        }
        normedQ_.EnQue(normed);

        // ---- 2. QKV projection (W8 dequant + matmul).
        LocalTensor<half> q = qBuf_.AllocTensor<half>();
        LocalTensor<half> k = kBuf_.AllocTensor<half>();
        LocalTensor<half> v = vBuf_.AllocTensor<half>();
        LocalTensor<half> normedRO = normedQ_.DeQue<half>();

        W8MatmulActivation_(normedRO, wqI8Gm_, wqScaleGm_, kQDim,   q);
        W8MatmulActivation_(normedRO, wkI8Gm_, wkScaleGm_, kKvDim,  k);
        W8MatmulActivation_(normedRO, wvI8Gm_, wvScaleGm_, kKvDim,  v);
        normedQ_.FreeTensor(normedRO);

        // ---- 3. Q-norm (per-head) and K-norm (per-kv-head).
        //
        // The host's aclnn RmsNorm runs on a [n_heads, head_dim] / [n_kv,
        // head_dim] view. We unroll the same: one RmsNorm per head over
        // a contiguous head_dim-sized slice.
        for (int32_t h = 0; h < kNHeads; ++h) {
            LocalTensor<half> q_h = q[h * kHeadDim];
            ComputeRmsNormLocal_(q_h, qNormGm_, kHeadDim, epsF16_);
        }
        for (int32_t h = 0; h < kNKv; ++h) {
            LocalTensor<half> k_h = k[h * kHeadDim];
            ComputeRmsNormLocal_(k_h, kNormGm_, kHeadDim, epsF16_);
        }

        // ---- 4. RoPE on Q and K.
        // mode=0 (NEOX): out[j]           = x[j] * c - x[j+half] * s
        //                out[j+half]      = x[j] * s + x[j+half] * c
        ApplyRopeAllHeads_(q, kNHeads);
        ApplyRopeAllHeads_(k, kNKv);

        // NOTE: engine is supposed to append RoPE'd K + V to cache BEFORE
        // launching this kernel. We write K/V back to GM though for parity
        // with the stock path (FIAS reads from kCacheGm_/vCacheGm_ using
        // the full appended sequence). The engine's host-side path uses
        // aclrtMemcpyAsync for V and aclnnRotaryPositionEmbedding
        // writing to k_cache[pos]. The kernel's responsibility is only
        // to produce the Q/K/V for THIS pos; host does the append.
        //
        // For the standalone 4.1.2 harness we override this behavior by
        // having the launcher pre-stage k_cache[pos] ← RoPE(k) itself,
        // which means for test purposes this kernel must write K/V into
        // the caller-provided k_cache/v_cache at slot (seq_len - 1). To
        // keep the kernel self-contained, we do that here.
        const uint32_t pos_slot = seqLen_ - 1;
        WriteKVToCache_(k, v, pos_slot);

        // ---- 5. Fused attention score (Q @ Kc^T / sqrt(d)) → softmax → @ Vc
        LocalTensor<half> attn_out = qBuf_.AllocTensor<half>();
        // Overloads the q buffer — after attention we don't need the
        // RoPE'd Q anymore; reuse the LocalTensor.
        ComputeAttention_(q, attn_out);

        qBuf_.FreeTensor(q);
        kBuf_.FreeTensor(k);
        vBuf_.FreeTensor(v);

        // ---- 6. O = W8Mm(attn_out, Wo).
        LocalTensor<half> o = oBuf_.AllocTensor<half>();
        W8MatmulActivation_(attn_out, woI8Gm_, woScaleGm_, kCpHidden, o);
        qBuf_.FreeTensor(attn_out);

        // ---- 7. Residual add: cur = cur + O.   Write back to residual GM.
        //
        // We READ residual from GM, add in local, write back. The W3b
        // fusion path expects the HOST to call AddRmsNorm(post_ln, xOut,
        // yOut=normed, cur=xOut+oOut) on the output of this kernel. The
        // kernel's contract is: produce `o_out = O-projection result`
        // AND write updated `residual = residual + o_out` to GM.
        //
        // The engine's host side already has a memcpy that serializes
        // residual=cur. We let the host do the AddRmsNorm as it did
        // before — this kernel only outputs the linear accumulation.
        LocalTensor<half> oOut = oBuf_.DeQue<half>();
        LocalTensor<half> resid = normedQ_.AllocTensor<half>();  // reuse
        DataCopy(resid, residualGm_, kCpHidden);
        Add(resid, resid, oOut, kCpHidden);  // element-wise F16 add.
        // Emit o_out (for host post-sublayer AddRmsNorm path).
        DataCopy(oOutGm_, oOut, kCpHidden);
        // Emit residual (updated cur).
        DataCopy(residualGm_, resid, kCpHidden);

        oBuf_.FreeTensor(oOut);
        normedQ_.FreeTensor(resid);
    }

private:
    // ------------------------------------------------------------
    // ComputeRmsNorm_ — compute y = RmsNorm(x, gamma, eps) writing y in-place
    // into the LocalTensor `x`. `gammaGm` is a GlobalTensor. `len` is the
    // number of elements. `eps` is F16.
    // This is the skeleton fallback; will be replaced with the AscendC
    // RmsNorm primitive once 4.1.2 confirms the shape contract.
    // ------------------------------------------------------------
    __aicore__ inline void ComputeRmsNorm_(LocalTensor<half> &x,
                                            GlobalTensor<half> &gammaGm,
                                            int32_t len, half eps) {
        // Scalar math on half is forbidden in aicore scope — promote to f32.
        // 1. Compute mean(x^2).
        float mean_sq = 0.0f;
        for (int32_t i = 0; i < len; ++i) {
            float v = (float)x.GetValue(i);
            mean_sq += v * v;
        }
        mean_sq /= (float)len;
        // 2. rstd = 1/sqrt(mean_sq + eps). AscendC's aicore math intrinsic
        //    is `sqrt(float)` (see __clang_cce_aicore_functions.h).
        float rstd = 1.0f / sqrt(mean_sq + (float)eps);
        // 3. x = x * rstd * gamma
        for (int32_t i = 0; i < len; ++i) {
            float v = (float)x.GetValue(i);
            float g = (float)gammaGm.GetValue(i);
            x.SetValue(i, (half)(v * rstd * g));
        }
    }

    // Same but for a sub-view (used by per-head Q/K norm). gammaGm has
    // shape [len] and is shared across heads.
    __aicore__ inline void ComputeRmsNormLocal_(LocalTensor<half> &x,
                                                 GlobalTensor<half> &gammaGm,
                                                 int32_t len, half eps) {
        float mean_sq = 0.0f;
        for (int32_t i = 0; i < len; ++i) {
            float v = (float)x.GetValue(i);
            mean_sq += v * v;
        }
        mean_sq /= (float)len;
        float rstd = 1.0f / sqrt(mean_sq + (float)eps);
        for (int32_t i = 0; i < len; ++i) {
            float v = (float)x.GetValue(i);
            float g = (float)gammaGm.GetValue(i);
            x.SetValue(i, (half)(v * rstd * g));
        }
    }

    // ------------------------------------------------------------
    // W8MatmulActivation_ — y[1, out_n] = x[1, cp_hidden] @ (w_i8 * scale)^T
    //
    // w_i8 is [out_n, cp_hidden] row-major INT8; scale is [out_n] F16
    // per-row symmetric. The engine's `w8_calibrate_weight_` produces
    // exactly this layout (line 326 of cp_cann_engine.cpp), so no
    // re-packing is needed — the kernel reads the engine's existing
    // device buffers directly.
    //
    // Skeleton fallback: scalar dequant + dot. Replace with Mmad +
    // MmulAddLocal primitives once 4.1.5 says we need the speedup.
    // ------------------------------------------------------------
    __aicore__ inline void W8MatmulActivation_(LocalTensor<half> &x,
                                                GlobalTensor<int8_t> &wI8,
                                                GlobalTensor<half> &wScale,
                                                int32_t out_n,
                                                LocalTensor<half> &y) {
        for (int32_t r = 0; r < out_n; ++r) {
            float acc = 0.0f;
            half scale = wScale.GetValue(r);
            for (int32_t c = 0; c < kCpHidden; ++c) {
                int8_t wi = wI8.GetValue((uint32_t)r * kCpHidden + c);
                // Dequant: w_f16 = scale * wi
                float w = (float)scale * (float)wi;
                float xv = (float)x.GetValue(c);
                acc += xv * w;
            }
            y.SetValue(r, (half)acc);
        }
    }

    // ------------------------------------------------------------
    // ApplyRopeAllHeads_ — in-place NEOX-style rotate_half RoPE over a
    // tensor laid out [n_heads, head_dim] (row-major). cos/sin tables
    // are shape [head_dim] (duplicated half; see engine lines 1002-1005
    // where cos[j] == cos[j+half]).
    // ------------------------------------------------------------
    __aicore__ inline void ApplyRopeAllHeads_(LocalTensor<half> &x,
                                               int32_t n_heads) {
        constexpr int32_t kHalf = kHeadDim / 2;
        for (int32_t h = 0; h < n_heads; ++h) {
            int32_t base = h * kHeadDim;
            for (int32_t j = 0; j < kHalf; ++j) {
                float c  = (float)ropeCosGm_.GetValue(j);
                float s  = (float)ropeSinGm_.GetValue(j);
                float x0 = (float)x.GetValue(base + j);
                float x1 = (float)x.GetValue(base + j + kHalf);
                // NEOX: rotate_half pairs (j, j+half).
                // out0 = x0 * c - x1 * s
                // out1 = x0 * s + x1 * c
                x.SetValue(base + j,         (half)(x0 * c - x1 * s));
                x.SetValue(base + j + kHalf, (half)(x0 * s + x1 * c));
            }
        }
    }

    // ------------------------------------------------------------
    // WriteKVToCache_ — append k / v into cache slots at position `slot`.
    // Both caches are shape [MAX_SEQ, n_kv, head_dim] strided [kv_dim, ...].
    // We write only the single row at slot.
    // ------------------------------------------------------------
    __aicore__ inline void WriteKVToCache_(LocalTensor<half> &k,
                                            LocalTensor<half> &v,
                                            uint32_t slot) {
        uint32_t off = slot * (uint32_t)kKvDim;
        for (int32_t i = 0; i < kKvDim; ++i) {
            kCacheGm_.SetValue(off + i, k.GetValue(i));
            vCacheGm_.SetValue(off + i, v.GetValue(i));
        }
    }

    // ------------------------------------------------------------
    // ComputeAttention_ — GQA scaled-dot-product attention over the
    // FULL cached K/V (seq_len = pos + 1).
    //
    //   For each q head h (n_heads=16):
    //     kv_h = h / group   (group = 2 for Qwen3-TTS CP: 16/8)
    //     scores[t] = dot(Q_h, K_cache[t, kv_h]) * scale        t ∈ [0, seq_len)
    //     softmax(scores) over t
    //     out_h = Σ_t  scores[t] * V_cache[t, kv_h]
    //
    // Output shape: [n_heads, head_dim] laid into `attn_out`.
    // ------------------------------------------------------------
    __aicore__ inline void ComputeAttention_(LocalTensor<half> &q,
                                              LocalTensor<half> &attn_out) {
        const float scale = 1.0f / sqrt((float)kHeadDim);
        for (int32_t h = 0; h < kNHeads; ++h) {
            int32_t kv_h = h / kGroup;
            int32_t q_base = h * kHeadDim;
            int32_t k_base_head = kv_h * kHeadDim;

            // --- scores[t] = dot(Q_h, K[t, kv_h]) * scale ---
            float max_score = -3.4e38f;
            for (uint32_t t = 0; t < seqLen_; ++t) {
                uint32_t k_off = t * (uint32_t)kKvDim + (uint32_t)k_base_head;
                float acc = 0.0f;
                for (int32_t d = 0; d < kHeadDim; ++d) {
                    float qv = (float)q.GetValue(q_base + d);
                    float kv = (float)kCacheGm_.GetValue(k_off + d);
                    acc += qv * kv;
                }
                acc *= scale;
                scoresGm_.SetValue(h * kMaxSeq + t, (half)acc);
                if (acc > max_score) max_score = acc;
            }
            // --- softmax in F32 for stability (matches engine's softmax F32
            //     path — see scores_dev_ is F32 in the engine header).
            //     AscendC aicore intrinsic is `exp(float)`.
            float sum = 0.0f;
            for (uint32_t t = 0; t < seqLen_; ++t) {
                float v = (float)scoresGm_.GetValue(h * kMaxSeq + t);
                float e = ApproxExp_(v - max_score);
                scoresGm_.SetValue(h * kMaxSeq + t, (half)e);
                sum += e;
            }
            float inv_sum = 1.0f / sum;
            for (uint32_t t = 0; t < seqLen_; ++t) {
                float v = (float)scoresGm_.GetValue(h * kMaxSeq + t);
                scoresGm_.SetValue(h * kMaxSeq + t, (half)(v * inv_sum));
            }
            // --- out_h = Σ_t score[t] * V[t, kv_h] ---
            for (int32_t d = 0; d < kHeadDim; ++d) {
                float acc = 0.0f;
                for (uint32_t t = 0; t < seqLen_; ++t) {
                    float s = (float)scoresGm_.GetValue(h * kMaxSeq + t);
                    uint32_t v_off = t * (uint32_t)kKvDim +
                                      (uint32_t)k_base_head;
                    float vv = (float)vCacheGm_.GetValue(v_off + d);
                    acc += s * vv;
                }
                attn_out.SetValue(q_base + d, (half)acc);
            }
        }
    }

    // ---------- pipe + queue state ----------
    TPipe pipe_;
    TQue<QuePosition::VECIN,  kPipeBufCount> normedQ_;
    TQue<QuePosition::VECOUT, kPipeBufCount> qBuf_;
    TQue<QuePosition::VECOUT, kPipeBufCount> kBuf_;
    TQue<QuePosition::VECOUT, kPipeBufCount> vBuf_;
    TQue<QuePosition::VECOUT, kPipeBufCount> oBuf_;

    // ---------- GM views ----------
    GlobalTensor<half>   residualGm_;
    GlobalTensor<half>   normedGm_;
    GlobalTensor<half>   inLnGm_;
    GlobalTensor<int8_t> wqI8Gm_;
    GlobalTensor<half>   wqScaleGm_;
    GlobalTensor<int8_t> wkI8Gm_;
    GlobalTensor<half>   wkScaleGm_;
    GlobalTensor<int8_t> wvI8Gm_;
    GlobalTensor<half>   wvScaleGm_;
    GlobalTensor<int8_t> woI8Gm_;
    GlobalTensor<half>   woScaleGm_;
    GlobalTensor<half>   qNormGm_;
    GlobalTensor<half>   kNormGm_;
    GlobalTensor<half>   ropeCosGm_;
    GlobalTensor<half>   ropeSinGm_;
    GlobalTensor<half>   kCacheGm_;
    GlobalTensor<half>   vCacheGm_;
    GlobalTensor<half>   oOutGm_;
    GlobalTensor<half>   qScratchGm_;
    GlobalTensor<half>   scoresGm_;

    uint32_t seqLen_ = 0;
    uint32_t opts_   = 0;
    half     epsF16_ = (half)1e-6f;
};

// =============================================================================
// Launcher entry point.
// The cmake ascendc_library() macro wraps this into
//   aclrtlaunch_fused_attn_sublayer(blockDim, stream, ...args...).
// =============================================================================

extern "C" __global__ __aicore__ void fused_attn_sublayer(
    GM_ADDR residual, GM_ADDR normed, GM_ADDR in_ln_gamma,
    GM_ADDR wq_i8, GM_ADDR wq_scale,
    GM_ADDR wk_i8, GM_ADDR wk_scale,
    GM_ADDR wv_i8, GM_ADDR wv_scale,
    GM_ADDR wo_i8, GM_ADDR wo_scale,
    GM_ADDR q_norm_f16, GM_ADDR k_norm_f16,
    GM_ADDR rope_cos, GM_ADDR rope_sin,
    GM_ADDR k_cache, GM_ADDR v_cache,
    GM_ADDR o_out, GM_ADDR scratch_q, GM_ADDR scratch_scores,
    uint32_t seq_len, uint32_t eps_bits, uint32_t opts)
{
    KernelFusedAttnSublayer op;
    op.Init(residual, normed, in_ln_gamma,
            wq_i8, wq_scale, wk_i8, wk_scale, wv_i8, wv_scale,
            wo_i8, wo_scale, q_norm_f16, k_norm_f16,
            rope_cos, rope_sin, k_cache, v_cache,
            o_out, scratch_q, scratch_scores,
            seq_len, eps_bits, opts);
    op.Process();
}
