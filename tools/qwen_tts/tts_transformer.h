/**
 * tts_transformer.h — Standalone Qwen3-TTS transformer backbone.
 *
 * Implements the 28-layer transformer with correct MRoPE
 * (consecutive-pair rotation, temporal-only, matching MLX exactly).
 * Bypasses llama.cpp entirely — uses raw ggml ops.
 *
 * Usage:
 *   TtsTransformer *t = tts_transformer_load("model.gguf", n_threads);
 *   float *hidden = tts_transformer_forward(t, embeddings, seq_len);
 *   tts_transformer_reset(t);
 *   tts_transformer_free(t);
 */

#ifndef TTS_TRANSFORMER_H
#define TTS_TRANSFORMER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TtsTransformer TtsTransformer;

/**
 * Load transformer backbone from GGUF file.
 * Returns NULL on failure.
 */
TtsTransformer *tts_transformer_load(const char *gguf_path, int n_threads);

/**
 * Free all resources.
 */
void tts_transformer_free(TtsTransformer *t);

/**
 * Reset KV cache (call before each new generation).
 */
void tts_transformer_reset(TtsTransformer *t);

/**
 * Forward pass through 28-layer transformer.
 *
 * embeddings: [seq_len, 2048] float32 (pre-constructed by caller)
 * seq_len: number of input positions
 * hidden_out: [2048] float32 — post-norm hidden from LAST position
 *
 * Returns 0 on success.
 */
int tts_transformer_forward(TtsTransformer *t,
                            const float *embeddings, int seq_len,
                            float *hidden_out);

/**
 * Get model hidden size (2048).
 */
int tts_transformer_hidden_size(const TtsTransformer *t);

#ifdef __cplusplus
}
#endif

#endif /* TTS_TRANSFORMER_H */
