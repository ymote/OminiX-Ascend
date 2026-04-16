/**
 * tts_transformer.cpp — Standalone Qwen3-TTS transformer backbone.
 *
 * 28-layer transformer with MRoPE matching MLX exactly:
 *   - Consecutive-pair rotation (interleaved=true)
 *   - Only temporal_section=24 pairs rotated (48/128 dims)
 *   - Frequency: base^(2i/head_dim) with head_dim=128
 *   - Remaining 40 pairs get identity (cos=1, sin=0)
 *
 * Architecture per layer:
 *   RMSNorm → Q/K/V proj → QK norm → MRoPE → GQA attention → O proj → residual
 *   → RMSNorm → gate_proj + up_proj → SiLU → down_proj → residual
 *
 * Uses CPU F32 computation. For CANN acceleration, the individual ops
 * (matmul, softmax) can be replaced with CANN equivalents later.
 */

#include "tts_transformer.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

// ============================================================================
// Configuration (Qwen3-TTS 1.7B talker)
// ============================================================================

static constexpr int N_LAYERS = 28;
static constexpr int HIDDEN = 2048;
static constexpr int N_HEADS = 16;
static constexpr int N_KV_HEADS = 8;
static constexpr int HEAD_DIM = 128;   // HIDDEN / N_HEADS
static constexpr int KV_DIM = 1024;    // N_KV_HEADS * HEAD_DIM
static constexpr int FF_DIM = 6144;    // intermediate_size
static constexpr float RMS_EPS = 1e-6f;
static constexpr float ROPE_BASE = 1000000.0f;
static constexpr int TEMPORAL_SECTION = 24;  // MRoPE: only 24 pairs rotated
static constexpr int MAX_SEQ = 4096;

// ============================================================================
// Weight storage
// ============================================================================

struct LayerWeights {
    std::vector<float> attn_norm;    // [HIDDEN]
    std::vector<float> q_proj;       // [N_HEADS*HEAD_DIM, HIDDEN]
    std::vector<float> k_proj;       // [N_KV_HEADS*HEAD_DIM, HIDDEN]
    std::vector<float> v_proj;       // [N_KV_HEADS*HEAD_DIM, HIDDEN]
    std::vector<float> o_proj;       // [HIDDEN, N_HEADS*HEAD_DIM]
    std::vector<float> q_norm;       // [HEAD_DIM]
    std::vector<float> k_norm;       // [HEAD_DIM]
    std::vector<float> ffn_norm;     // [HIDDEN]
    std::vector<float> gate_proj;    // [FF_DIM, HIDDEN]
    std::vector<float> up_proj;      // [FF_DIM, HIDDEN]
    std::vector<float> down_proj;    // [HIDDEN, FF_DIM]
};

struct TtsTransformer {
    std::vector<LayerWeights> layers;
    std::vector<float> final_norm;   // [HIDDEN]
    int n_threads;

    // KV cache: [N_LAYERS][max_cached_tokens * KV_DIM]
    std::vector<std::vector<float>> k_cache;
    std::vector<std::vector<float>> v_cache;
    int cached_len;

    // Precomputed RoPE cos/sin for temporal section
    // cos_cache[pos][pair] and sin_cache[pos][pair], pair=0..TEMPORAL_SECTION-1
    std::vector<std::vector<float>> rope_cos;
    std::vector<std::vector<float>> rope_sin;
};

// ============================================================================
// GGUF weight loading (F16 → F32)
// ============================================================================

#include "ggml.h"
#include "gguf.h"

static std::vector<float> load_f32_tensor(gguf_context *gguf_ctx, ggml_context *ggml_ctx,
                                           const char *name, size_t expected_elements) {
    int idx = gguf_find_tensor(gguf_ctx, name);
    if (idx < 0) {
        fprintf(stderr, "[tts_transformer] tensor not found: %s\n", name);
        return {};
    }
    ggml_tensor *t = ggml_get_tensor(ggml_ctx, name);
    if (!t) {
        fprintf(stderr, "[tts_transformer] ggml tensor not found: %s\n", name);
        return {};
    }
    size_t n = ggml_nelements(t);
    if (expected_elements > 0 && n != expected_elements) {
        fprintf(stderr, "[tts_transformer] tensor %s: expected %zu elements, got %zu\n",
                name, expected_elements, n);
    }
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t *src = (const ggml_fp16_t *)t->data;
        for (size_t i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        fprintf(stderr, "[tts_transformer] unsupported type for %s: %d\n", name, t->type);
    }
    return out;
}

// ============================================================================
// Math primitives
// ============================================================================

static void rms_norm(const float *x, const float *w, float *out, int dim) {
    float sum = 0;
    for (int i = 0; i < dim; i++) sum += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum / dim + RMS_EPS);
    for (int i = 0; i < dim; i++) out[i] = x[i] * scale * w[i];
}

// out[M] = W^T @ x[N], where W is stored column-major as [N, M] (GGML layout)
// GGML stores weights transposed: W[in_features, out_features]
// So out[m] = sum(W[n*M + m] * x[n]) for n in [0,N)
static void matvec(const float *W, const float *x, float *out, int M, int N) {
    memset(out, 0, M * sizeof(float));
    for (int n = 0; n < N; n++) {
        const float xn = x[n];
        const float *col = W + (size_t)n * M;
        for (int m = 0; m < M; m++) {
            out[m] += col[m] * xn;
        }
    }
}

static void silu(float *x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

// ============================================================================
// MRoPE (matching MLX exactly)
// ============================================================================

static void init_rope_cache(TtsTransformer *t) {
    t->rope_cos.resize(MAX_SEQ);
    t->rope_sin.resize(MAX_SEQ);

    // Precompute inv_freq: 1 / base^(2i / head_dim) for i = 0..TEMPORAL_SECTION-1
    std::vector<float> inv_freq(TEMPORAL_SECTION);
    for (int i = 0; i < TEMPORAL_SECTION; i++) {
        inv_freq[i] = 1.0f / powf(ROPE_BASE, 2.0f * i / HEAD_DIM);
    }

    for (int pos = 0; pos < MAX_SEQ; pos++) {
        t->rope_cos[pos].resize(TEMPORAL_SECTION);
        t->rope_sin[pos].resize(TEMPORAL_SECTION);
        for (int i = 0; i < TEMPORAL_SECTION; i++) {
            float angle = pos * inv_freq[i];
            t->rope_cos[pos][i] = cosf(angle);
            t->rope_sin[pos][i] = sinf(angle);
        }
    }
}

// Apply MRoPE to Q or K vector (one head, HEAD_DIM elements)
// Consecutive-pair rotation: (d[2i], d[2i+1]) rotated by cos/sin[i]
// Only first TEMPORAL_SECTION pairs rotated, rest identity
static void apply_mrope(float *vec, int pos, const TtsTransformer *t) {
    const float *cos = t->rope_cos[pos].data();
    const float *sin = t->rope_sin[pos].data();

    for (int i = 0; i < TEMPORAL_SECTION; i++) {
        float x0 = vec[2 * i];
        float x1 = vec[2 * i + 1];
        vec[2 * i]     = x0 * cos[i] - x1 * sin[i];
        vec[2 * i + 1] = x1 * cos[i] + x0 * sin[i];
    }
    // Pairs TEMPORAL_SECTION..HEAD_DIM/2-1: identity (no rotation)
}

// ============================================================================
// Attention (GQA: 16 query heads, 8 KV heads)
// ============================================================================

static void attention_layer(
    TtsTransformer *t, int layer_idx,
    const float *normed, int seq_len, int cache_offset,
    float *attn_out)
{
    auto &lw = t->layers[layer_idx];
    int gqa_ratio = N_HEADS / N_KV_HEADS;

    // Project Q, K, V for all positions
    std::vector<float> Q(seq_len * N_HEADS * HEAD_DIM);
    std::vector<float> K(seq_len * N_KV_HEADS * HEAD_DIM);
    std::vector<float> V(seq_len * N_KV_HEADS * HEAD_DIM);

    for (int s = 0; s < seq_len; s++) {
        matvec(lw.q_proj.data(), normed + s * HIDDEN, Q.data() + s * N_HEADS * HEAD_DIM,
               N_HEADS * HEAD_DIM, HIDDEN);
        matvec(lw.k_proj.data(), normed + s * HIDDEN, K.data() + s * N_KV_HEADS * HEAD_DIM,
               N_KV_HEADS * HEAD_DIM, HIDDEN);
        matvec(lw.v_proj.data(), normed + s * HIDDEN, V.data() + s * N_KV_HEADS * HEAD_DIM,
               N_KV_HEADS * HEAD_DIM, HIDDEN);
    }

    // QK RMS norm + RoPE per head
    for (int s = 0; s < seq_len; s++) {
        int pos = cache_offset + s;
        for (int h = 0; h < N_HEADS; h++) {
            float *q = Q.data() + s * N_HEADS * HEAD_DIM + h * HEAD_DIM;
            // Per-head QK norm
            float sum = 0;
            for (int d = 0; d < HEAD_DIM; d++) sum += q[d] * q[d];
            float scale = 1.0f / sqrtf(sum / HEAD_DIM + RMS_EPS);
            for (int d = 0; d < HEAD_DIM; d++) q[d] = q[d] * scale * lw.q_norm[d];
            // MRoPE
            apply_mrope(q, pos, t);
        }
        for (int h = 0; h < N_KV_HEADS; h++) {
            float *k = K.data() + s * N_KV_HEADS * HEAD_DIM + h * HEAD_DIM;
            float sum = 0;
            for (int d = 0; d < HEAD_DIM; d++) sum += k[d] * k[d];
            float scale = 1.0f / sqrtf(sum / HEAD_DIM + RMS_EPS);
            for (int d = 0; d < HEAD_DIM; d++) k[d] = k[d] * scale * lw.k_norm[d];
            apply_mrope(k, pos, t);
        }
    }

    // Update KV cache
    auto &kc = t->k_cache[layer_idx];
    auto &vc = t->v_cache[layer_idx];
    for (int s = 0; s < seq_len; s++) {
        memcpy(kc.data() + (cache_offset + s) * KV_DIM,
               K.data() + s * N_KV_HEADS * HEAD_DIM,
               KV_DIM * sizeof(float));
        memcpy(vc.data() + (cache_offset + s) * KV_DIM,
               V.data() + s * N_KV_HEADS * HEAD_DIM,
               KV_DIM * sizeof(float));
    }

    int total_len = cache_offset + seq_len;
    float attn_scale = 1.0f / sqrtf((float)HEAD_DIM);

    // Per-query-head attention
    memset(attn_out, 0, seq_len * HIDDEN * sizeof(float));
    for (int s = 0; s < seq_len; s++) {
        for (int qh = 0; qh < N_HEADS; qh++) {
            int kv_h = qh / gqa_ratio;
            const float *q = Q.data() + s * N_HEADS * HEAD_DIM + qh * HEAD_DIM;

            // Compute scores against all cached K
            std::vector<float> scores(total_len);
            for (int t2 = 0; t2 < total_len; t2++) {
                // Causal: only attend to t2 <= cache_offset + s
                if (t2 > cache_offset + s) {
                    scores[t2] = -1e9f;
                    continue;
                }
                const float *k = kc.data() + t2 * KV_DIM + kv_h * HEAD_DIM;
                float dot = 0;
                for (int d = 0; d < HEAD_DIM; d++) dot += q[d] * k[d];
                scores[t2] = dot * attn_scale;
            }

            // Softmax
            float max_score = -1e9f;
            for (int t2 = 0; t2 <= cache_offset + s; t2++)
                if (scores[t2] > max_score) max_score = scores[t2];
            float sum_exp = 0;
            for (int t2 = 0; t2 <= cache_offset + s; t2++) {
                scores[t2] = expf(scores[t2] - max_score);
                sum_exp += scores[t2];
            }
            for (int t2 = 0; t2 <= cache_offset + s; t2++)
                scores[t2] /= sum_exp;

            // Weighted sum of V
            float *out = attn_out + s * HIDDEN + qh * HEAD_DIM;
            for (int t2 = 0; t2 <= cache_offset + s; t2++) {
                const float *v = vc.data() + t2 * KV_DIM + kv_h * HEAD_DIM;
                float w = scores[t2];
                for (int d = 0; d < HEAD_DIM; d++)
                    out[d] += w * v[d];
            }
        }
    }

    // O projection
    std::vector<float> tmp(HIDDEN);
    for (int s = 0; s < seq_len; s++) {
        matvec(lw.o_proj.data(), attn_out + s * HIDDEN, tmp.data(), HIDDEN, HIDDEN);
        memcpy(attn_out + s * HIDDEN, tmp.data(), HIDDEN * sizeof(float));
    }
}

// ============================================================================
// Full forward pass
// ============================================================================

static void forward_layer(TtsTransformer *t, int il,
                          float *hidden, int seq_len, int cache_offset) {
    auto &lw = t->layers[il];
    std::vector<float> normed(seq_len * HIDDEN);
    std::vector<float> attn_out(seq_len * HIDDEN);

    // Attention norm + attention
    for (int s = 0; s < seq_len; s++)
        rms_norm(hidden + s * HIDDEN, lw.attn_norm.data(), normed.data() + s * HIDDEN, HIDDEN);

    attention_layer(t, il, normed.data(), seq_len, cache_offset, attn_out.data());

    // Residual
    for (int i = 0; i < seq_len * HIDDEN; i++)
        hidden[i] += attn_out[i];

    // FFN norm
    for (int s = 0; s < seq_len; s++)
        rms_norm(hidden + s * HIDDEN, lw.ffn_norm.data(), normed.data() + s * HIDDEN, HIDDEN);

    // SwiGLU FFN
    for (int s = 0; s < seq_len; s++) {
        std::vector<float> gate(FF_DIM), up(FF_DIM), down(HIDDEN);
        matvec(lw.gate_proj.data(), normed.data() + s * HIDDEN, gate.data(), FF_DIM, HIDDEN);
        matvec(lw.up_proj.data(), normed.data() + s * HIDDEN, up.data(), FF_DIM, HIDDEN);
        silu(gate.data(), FF_DIM);
        for (int i = 0; i < FF_DIM; i++) gate[i] *= up[i];
        matvec(lw.down_proj.data(), gate.data(), down.data(), HIDDEN, FF_DIM);
        for (int i = 0; i < HIDDEN; i++) hidden[s * HIDDEN + i] += down[i];
    }
}

// ============================================================================
// Public API
// ============================================================================

extern "C" {

TtsTransformer *tts_transformer_load(const char *gguf_path, int n_threads) {
    struct ggml_context *ggml_ctx = nullptr;
    struct gguf_init_params params = { .no_alloc = false, .ctx = &ggml_ctx };
    struct gguf_context *gguf_ctx = gguf_init_from_file(gguf_path, params);
    if (!gguf_ctx || !ggml_ctx) {
        fprintf(stderr, "[tts_transformer] failed to load GGUF: %s\n", gguf_path);
        return nullptr;
    }

    auto *t = new TtsTransformer();
    t->n_threads = n_threads;
    t->layers.resize(N_LAYERS);
    t->cached_len = 0;

    // Load layer weights
    for (int il = 0; il < N_LAYERS; il++) {
        auto &lw = t->layers[il];
        char name[256];

        auto load = [&](const char *suffix, size_t expected) -> std::vector<float> {
            snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
            return load_f32_tensor(gguf_ctx, ggml_ctx, name, expected);
        };

        lw.attn_norm = load("attn_norm.weight", HIDDEN);
        lw.q_proj = load("attn_q.weight", N_HEADS * HEAD_DIM * HIDDEN);
        lw.k_proj = load("attn_k.weight", N_KV_HEADS * HEAD_DIM * HIDDEN);
        lw.v_proj = load("attn_v.weight", N_KV_HEADS * HEAD_DIM * HIDDEN);
        lw.o_proj = load("attn_output.weight", HIDDEN * N_HEADS * HEAD_DIM);
        lw.q_norm = load("attn_q_norm.weight", HEAD_DIM);
        lw.k_norm = load("attn_k_norm.weight", HEAD_DIM);
        lw.ffn_norm = load("ffn_norm.weight", HIDDEN);
        lw.gate_proj = load("ffn_gate.weight", FF_DIM * HIDDEN);
        lw.up_proj = load("ffn_up.weight", FF_DIM * HIDDEN);
        lw.down_proj = load("ffn_down.weight", HIDDEN * FF_DIM);

        if (lw.attn_norm.empty() || lw.q_proj.empty()) {
            fprintf(stderr, "[tts_transformer] missing weights for layer %d\n", il);
            delete t;
            gguf_free(gguf_ctx);
            return nullptr;
        }
    }

    t->final_norm = load_f32_tensor(gguf_ctx, ggml_ctx, "output_norm.weight", HIDDEN);

    // Init KV cache
    t->k_cache.resize(N_LAYERS);
    t->v_cache.resize(N_LAYERS);
    for (int il = 0; il < N_LAYERS; il++) {
        t->k_cache[il].resize(MAX_SEQ * KV_DIM, 0);
        t->v_cache[il].resize(MAX_SEQ * KV_DIM, 0);
    }

    // Init RoPE cache
    init_rope_cache(t);

    gguf_free(gguf_ctx);

    printf("[tts_transformer] loaded %d layers, hidden=%d, heads=%d/%d, rope_temporal=%d\n",
           N_LAYERS, HIDDEN, N_HEADS, N_KV_HEADS, TEMPORAL_SECTION);
    return t;
}

void tts_transformer_free(TtsTransformer *t) {
    delete t;
}

void tts_transformer_reset(TtsTransformer *t) {
    t->cached_len = 0;
    // No need to zero cache — cached_len tracks valid range
}

int tts_transformer_forward(TtsTransformer *t,
                            const float *embeddings, int seq_len,
                            float *hidden_out) {
    if (!t || seq_len <= 0) return -1;
    if (t->cached_len + seq_len > MAX_SEQ) return -2;

    // Copy input to working buffer
    std::vector<float> hidden(seq_len * HIDDEN);
    memcpy(hidden.data(), embeddings, seq_len * HIDDEN * sizeof(float));

    int cache_offset = t->cached_len;

    // Forward through all layers
    for (int il = 0; il < N_LAYERS; il++) {
        forward_layer(t, il, hidden.data(), seq_len, cache_offset);
    }

    t->cached_len += seq_len;

    // Final RMS norm on last position
    rms_norm(hidden.data() + (seq_len - 1) * HIDDEN,
             t->final_norm.data(), hidden_out, HIDDEN);

    return 0;
}

int tts_transformer_hidden_size(const TtsTransformer *t) {
    return HIDDEN;
}

} // extern "C"
