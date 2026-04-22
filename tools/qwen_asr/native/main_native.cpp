// ============================================================================
// qwen_asr_native — end-to-end driver for AsrTextDecoderCannEngine.
//
// Mirrors tools/qwen_asr/main.cpp's CLI flags so the existing harness
// (~/asr_a1a/run_asr_harness.py) can target this binary with only
// `--binary` and `--decoder` changes (no new flags).
//
// Runtime modes:
//   --audio <path>                  # single-clip mode; stdout "=== Result ===" etc.
//   --audio_dir <dir>               # batch mode; one `--audio` per wav, prints
//                                    summary at end. (Optional helper for the
//                                    persistent-engine gate.)
// ============================================================================

#include "asr_text_decoder_engine.h"

#include <cstdio>
#include <cstring>
#include <dirent.h>
#include <string>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <chrono>

static void usage(const char *prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Options (mirrors qwen_asr):\n");
    printf("  --model_dir DIR       Model directory\n");
    printf("  --audio FILE          Input WAV\n");
    printf("  --audio_dir DIR       Batch mode: iterate *.wav\n");
    printf("  --encoder FILE        Audio encoder GGUF\n");
    printf("  --decoder FILE        Text decoder GGUF (F16, Qwen3 llama.cpp format)\n");
    printf("  --vocab FILE          vocab.json\n");
    printf("  --merges FILE         merges.txt\n");
    printf("  --mel_filters FILE    mel_filters_whisper.npy (optional)\n");
    printf("  --device DEV          Audio encoder device (default CANN0)\n");
    printf("  --threads N           Audio encoder threads (default 4)\n");
    printf("  --max_tokens N        Max output tokens (default 256)\n");
    printf("  -h, --help            Show this help\n");
}

static std::vector<std::string> list_wavs(const std::string &dir) {
    std::vector<std::string> out;
    DIR *d = opendir(dir.c_str());
    if (!d) return out;
    struct dirent *e;
    while ((e = readdir(d)) != nullptr) {
        std::string n = e->d_name;
        if (n.size() >= 4 && n.compare(n.size()-4, 4, ".wav") == 0) {
            out.push_back(dir + "/" + n);
        }
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}

int main(int argc, char *argv[]) {
    AsrConfig cfg;
    std::string audio_path, audio_dir;

    for (int i = 1; i < argc; ++i) {
        auto S = [&](int k){ return std::string(argv[k]); };
        if      (strcmp(argv[i], "--model_dir") == 0   && i+1<argc) cfg.model_dir = S(++i);
        else if (strcmp(argv[i], "--audio") == 0       && i+1<argc) audio_path    = S(++i);
        else if (strcmp(argv[i], "--audio_dir") == 0   && i+1<argc) audio_dir     = S(++i);
        else if (strcmp(argv[i], "--encoder") == 0     && i+1<argc) cfg.audio_encoder_path = S(++i);
        else if (strcmp(argv[i], "--decoder") == 0     && i+1<argc) cfg.decoder_path       = S(++i);
        else if (strcmp(argv[i], "--vocab") == 0       && i+1<argc) cfg.vocab_path  = S(++i);
        else if (strcmp(argv[i], "--merges") == 0      && i+1<argc) cfg.merges_path = S(++i);
        else if (strcmp(argv[i], "--mel_filters") == 0 && i+1<argc) cfg.mel_filters_path = S(++i);
        else if (strcmp(argv[i], "--device") == 0      && i+1<argc) cfg.device = S(++i);
        else if (strcmp(argv[i], "--threads") == 0     && i+1<argc) cfg.n_threads = atoi(argv[++i]);
        // Accept --gpu_layers for harness compat; native engine is always 28L on NPU.
        else if (strcmp(argv[i], "--gpu_layers") == 0  && i+1<argc) { ++i; }
        else if (strcmp(argv[i], "--max_tokens") == 0  && i+1<argc) cfg.max_new_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            usage(argv[0]); return 1;
        }
    }

    if (cfg.decoder_path.empty() || cfg.audio_encoder_path.empty() ||
        cfg.vocab_path.empty() || cfg.merges_path.empty()) {
        fprintf(stderr, "[qwen_asr_native] --encoder / --decoder / --vocab / --merges required\n");
        usage(argv[0]); return 1;
    }

    AsrTextDecoderCannEngine engine;
    auto t_init_0 = std::chrono::high_resolution_clock::now();
    if (!engine.init(cfg)) {
        fprintf(stderr, "[qwen_asr_native] init failed\n"); return 2;
    }
    auto t_init_1 = std::chrono::high_resolution_clock::now();
    double init_ms = std::chrono::duration<double, std::milli>(t_init_1 - t_init_0).count();
    printf("[qwen_asr_native] init: %.0fms\n", init_ms);

    auto run_one = [&](const std::string &wav) {
        std::string text;
        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok = engine.transcribe_file(wav, text);
        auto t1 = std::chrono::high_resolution_clock::now();
        double wall_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (!ok) {
            fprintf(stderr, "[qwen_asr_native] transcribe failed: %s\n", wav.c_str());
            return;
        }
        printf("=== Result ===\n%s\n", text.c_str());
        printf("[qwen_asr_native] %s wall=%.0fms\n", wav.c_str(), wall_ms);
    };

    if (!audio_dir.empty()) {
        auto wavs = list_wavs(audio_dir);
        printf("[qwen_asr_native] batch mode: %zu wavs\n", wavs.size());
        for (auto &w : wavs) run_one(w);
    } else if (!audio_path.empty()) {
        run_one(audio_path);
    } else {
        fprintf(stderr, "[qwen_asr_native] --audio or --audio_dir required\n");
        return 1;
    }
    return 0;
}
