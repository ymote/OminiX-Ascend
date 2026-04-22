#pragma once
// ============================================================================
// AsrTextDecoderCannEngine — native Qwen3-ASR text decoder on Ascend NPU.
//
// Qwen3-ASR-1.7B's text decoder is a standalone Qwen3 decoder-only LLM
// (28 layers, hidden=2048, 16 Q heads / 8 KV heads, head_dim=128, SwiGLU,
// RoPE theta=1e6, vocab=151936, tied embeddings). Audio features from
// audio_encoder.cpp (F32 [num_frames, 2048]) are injected as input
// embeddings via the same batch.embd path the llama.cpp baseline uses —
// the decoder attends to them via standard causal self-attention over
// the concatenated `[pre_tokens, audio, post_tokens, generated]` stream.
//
// This engine composes `TalkerCannEngine` (from tools/qwen_tts) verbatim
// for the 28-layer transformer body — the Talker backbone and the ASR
// text decoder are architecturally identical (same ops, same dtype
// conventions, same RoPE). What we add on top:
//
//   1. Host-side text-token embedding lookup (tied with lm_head).
//   2. NPU lm_head via aclnnMm (pattern ported from CpCannEngine W1).
//   3. End-to-end `transcribe()` that drives `AudioEncoder → prefill ×3 →
//      decode loop with NPU lm_head → detokenize`.
//
// Persistent-engine pattern: `init_from_gguf()` runs once per process;
// `transcribe(samples, text)` runs per clip. KV cache is cleared between
// calls via `TalkerCannEngine::reset_kv_cache()`.
// ============================================================================

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../qwen_tts/talker.h"               // TalkerConfig
#include "../../qwen_tts/talker_cann_engine.h"   // TalkerCannEngine
#include "../../qwen_tts/cp_cann_symbols.h"      // g_cann
#include "../audio_encoder.h"
#include "../mel_spectrogram.h"
#include "bpe_tokenizer.h"

// Config mirrors tools/qwen_asr/main.cpp flags.
struct AsrConfig {
    std::string model_dir;          // weight root (for vocab/merges)
    std::string audio_encoder_path; // qwen_asr_audio_encoder.gguf
    std::string decoder_path;       // qwen_asr_decoder.gguf (F16, not Q8)
    std::string vocab_path;         // vocab.json
    std::string merges_path;        // merges.txt
    std::string mel_filters_path;   // mel_filters_whisper.npy (optional)
    std::string device = "CANN0";   // audio encoder backend
    int n_threads      = 4;         // audio encoder CPU threads
    int max_new_tokens = 256;

    // Special token IDs (defaults match Qwen3-ASR tokenizer).
    int audio_start_id = 151669;
    int audio_end_id   = 151670;
    int audio_pad_id   = 151676;
    int im_start_id    = 151644;
    int im_end_id      = 151645;
    int endoftext_id   = 151643;
};

class AsrTextDecoderCannEngine {
public:
    AsrTextDecoderCannEngine() = default;
    ~AsrTextDecoderCannEngine();

    // One-time init: loads audio encoder, text decoder (via TalkerCannEngine),
    // tokenizer, and uploads lm_head (tied with token_embd) to the NPU.
    // Safe to call once per process; subsequent transcribe() calls reuse all
    // resources. Returns false on any load failure.
    bool init(const AsrConfig &cfg);

    // End-to-end: audio file path → transcript.
    bool transcribe_file(const std::string &audio_path, std::string &text_out);

    // End-to-end: 16kHz F32 mono samples → transcript.
    bool transcribe(const std::vector<float> &audio_16k, std::string &text_out);

    // Low-level: feed pre-computed audio features ([num_frames, 2048] F32,
    // row-major) directly, skipping mel + encoder. Used for parity tests.
    bool transcribe_from_features(const std::vector<float> &audio_features,
                                   int num_audio_frames,
                                   std::string &text_out);

    bool is_ready() const { return ready_; }

    // Expose last-run timing breakdown (ms). Overwritten by every transcribe.
    struct Timing {
        double mel_ms  = 0.0;
        double enc_ms  = 0.0;
        double dec_ms  = 0.0;  // prefill + gen (matches baseline's "decoder" field)
        double pre_ms  = 0.0;
        double audio_ms = 0.0;
        double post_ms = 0.0;
        double gen_ms  = 0.0;
        int    gen_tokens = 0;
        double total_ms = 0.0;
    };
    const Timing &last_timing() const { return timing_; }

private:
    // --- Persistent state ---
    bool ready_ = false;
    AsrConfig cfg_;
    MelSpectrogram mel_;
    AudioEncoder   audio_encoder_;
    std::unique_ptr<BpeTokenizer> tokenizer_;
    TalkerCannEngine talker_;             // the 28-layer transformer
    TalkerConfig talker_cfg_;             // n_layers=28 etc.
    int n_embd_      = 0;                 // 2048
    int vocab_size_  = 0;                 // 151936

    // Host-side embedding table for text tokens. F32 [vocab, n_embd].
    // Also reused as the (tied) lm_head weight uploaded to NPU below.
    std::vector<float> tok_embd_host_;

    // NPU lm_head — same 3-stage pattern as CpCannEngine::forward_lm_head:
    //   hidden_F32 → Cast→F16 → Mm(W[vocab,h]) → logits_F16 → Cast→F32 stage.
    void *lm_head_w_dev_        = nullptr; // F16 [vocab, n_embd]
    void *lm_head_h_f16_dev_    = nullptr; // F16 [n_embd] (hidden staging)
    void *lm_head_h_f32_dev_    = nullptr; // F32 [n_embd] (hidden upload)
    void *logits_f16_dev_       = nullptr; // F16 [vocab]
    void *logits_f32_dev_       = nullptr; // F32 [vocab] (D2H staging)
    aclrtStream lm_stream_      = nullptr; // shared with talker_'s primary stream
    bool lm_head_ready_         = false;
    void *lm_workspace_dev_     = nullptr; // aclnnMm workspace (grows on demand)
    size_t lm_workspace_size_   = 0;

    // Token decoding (baseline-compatible: vocab.json + byte-level BPE unicode table).
    std::unordered_map<int, std::string> id_to_bytes_;
    uint8_t unicode_to_byte_[512] = {0};

    Timing timing_;

    // --- Internal helpers ---
    // Load tok_embd_host_ from the GGUF at `path` and upload the tied lm_head
    // weight (same [vocab, n_embd] F16 buffer) to the NPU. Called from init()
    // after `talker_.init_from_gguf()` but before any forward call.
    bool load_embedding_and_lm_head_(const std::string &gguf_path);

    // Build [pre_tokens] and [post_tokens] exactly as QwenASR::build_prompt_segments.
    void build_prompt_segments_(std::vector<int> &pre, std::vector<int> &post);

    // Look up `tokens.size()` text-token embeddings into an F32 [N, n_embd]
    // buffer. Returns host-side vector ready to feed into talker_.forward_prefill.
    std::vector<float> embed_tokens_(const std::vector<int> &tokens);

    // One lm_head step: talker_.forward_decode returned `hidden_f32` on host;
    // this function uploads it, runs the Cast+Mm+Cast chain on NPU, downloads
    // logits to `logits_out` (size vocab_size_).
    void forward_lm_head_(const float *hidden_f32, std::vector<float> &logits_out);

    // Detokenize ID sequence via vocab.json byte-level BPE (baseline parity).
    std::string decode_tokens_(const std::vector<int> &ids);

    // Init reverse vocab from vocab.json — replica of QwenASR::init_reverse_vocab.
    void init_reverse_vocab_(const std::string &vocab_path);
    std::string bpe_unicode_to_bytes_(const std::string &s);

    // Grow aclnnMm workspace on demand (caller passes required_bytes).
    void ensure_lm_workspace_(size_t bytes);
};
