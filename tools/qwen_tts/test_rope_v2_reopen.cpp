// test_rope_v2_reopen.cpp — A.2-reopen standalone probe.
//
// Goal: settle whether aclnnApplyRotaryPosEmbV2 produces numerically equivalent
// output to aclnnRotaryPositionEmbedding(mode=0, NEOX) on our exact Qwen3-TTS
// talker shape (GQA: 16 Q heads, 8 KV heads, head_dim=128, F16) when we feed
// the two candidate cos/sin preparations MN surfaced in llm_mutil_npu_brief.md:
//
//   Prep A (half-duplicated): cos[p, d] = cos(p * freq[d < half ? d : d - half])
//                              -- our engine's current prep, pairs with v1 NEOX.
//   Prep B (half-half HF):    MoYoYoTech's fill_cos_sin_hf — same index mapping
//                              (pair = d < half ? d : d - half) as Prep A, i.e.
//                              mathematically equivalent. Included to falsify
//                              the "prep is the delta" hypothesis at the source.
//
// Build (ac02):
//   source /usr/local/Ascend/ascend-toolkit/set_env.sh
//   g++ -std=c++17 -O2 -o test_rope_v2_reopen test_rope_v2_reopen.cpp -ldl -lm
//
// Run:
//   ./test_rope_v2_reopen
//
// Exits 0 on GREEN (V2 matches v1 within F16-rounding tolerance on at least
// one prep), non-zero on RED.
//
// All CANN symbols are loaded at runtime via dlsym. No CANN-header linkage
// needed, matching the engine's own `cp_cann_symbols.cpp` pattern — the probe
// binary stays portable across CANN toolkit versions.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <dlfcn.h>
#include <string>
#include <vector>

// ---------- Minimal ACL type/enum declarations (match toolkit headers) ------
using aclrtStream = void*;
using aclTensor   = struct aclTensor_;
using aclOpExecutor = struct aclOpExecutor_;
using aclScalar   = struct aclScalar_;

using aclError    = int;
using aclnnStatus = int;

enum aclDataType   { ACL_FLOAT = 0, ACL_FLOAT16 = 1, ACL_INT8 = 2, ACL_INT32 = 3, ACL_UINT8 = 4, ACL_BF16 = 27 };
enum aclFormat     { ACL_FORMAT_ND = 2 };
enum aclrtMemcpyKind { ACL_MEMCPY_HOST_TO_DEVICE = 1, ACL_MEMCPY_DEVICE_TO_HOST = 2, ACL_MEMCPY_DEVICE_TO_DEVICE = 3 };

#define ACL_CHECK(e) do { int _s = (int)(e); if (_s != 0) { fprintf(stderr, "ACL err at %s:%d: %d\n", __FILE__, __LINE__, _s); std::exit(1); } } while (0)

// --- function pointer typedefs ---
typedef aclError (*aclInit_t)(const char*);
typedef aclError (*aclFinalize_t)();
typedef aclError (*aclrtSetDevice_t)(int);
typedef aclError (*aclrtResetDevice_t)(int);
typedef aclError (*aclrtCreateStream_t)(aclrtStream*);
typedef aclError (*aclrtDestroyStream_t)(aclrtStream);
typedef aclError (*aclrtSynchronizeStream_t)(aclrtStream);
typedef aclError (*aclrtMalloc_t)(void**, size_t, int);
typedef aclError (*aclrtFree_t)(void*);
typedef aclError (*aclrtMemcpy_t)(void*, size_t, const void*, size_t, aclrtMemcpyKind);

typedef aclTensor* (*aclCreateTensor_t)(const int64_t*, uint64_t, aclDataType,
                                         const int64_t*, int64_t, aclFormat,
                                         const int64_t*, uint64_t, void*);
typedef int (*aclDestroyTensor_t)(aclTensor*);

typedef aclnnStatus (*aclnnRotaryPositionEmbeddingGetWorkspaceSize_t)(
    const aclTensor*, const aclTensor*, const aclTensor*, int64_t, aclTensor*,
    uint64_t*, aclOpExecutor**);
typedef aclnnStatus (*aclnnRotaryPositionEmbedding_t)(
    void*, uint64_t, aclOpExecutor*, aclrtStream);

typedef aclnnStatus (*aclnnApplyRotaryPosEmbV2GetWorkspaceSize_t)(
    aclTensor*, aclTensor*, const aclTensor*, const aclTensor*,
    int64_t, char*, uint64_t*, aclOpExecutor**);
typedef aclnnStatus (*aclnnApplyRotaryPosEmbV2_t)(
    void*, uint64_t, aclOpExecutor*, aclrtStream);

// --- globals ---
struct Syms {
    aclInit_t aclInit;
    aclFinalize_t aclFinalize;
    aclrtSetDevice_t aclrtSetDevice;
    aclrtResetDevice_t aclrtResetDevice;
    aclrtCreateStream_t aclrtCreateStream;
    aclrtDestroyStream_t aclrtDestroyStream;
    aclrtSynchronizeStream_t aclrtSynchronizeStream;
    aclrtMalloc_t aclrtMalloc;
    aclrtFree_t aclrtFree;
    aclrtMemcpy_t aclrtMemcpy;
    aclCreateTensor_t aclCreateTensor;
    aclDestroyTensor_t aclDestroyTensor;
    aclnnRotaryPositionEmbeddingGetWorkspaceSize_t aclnnRotaryPositionEmbeddingGetWorkspaceSize;
    aclnnRotaryPositionEmbedding_t aclnnRotaryPositionEmbedding;
    aclnnApplyRotaryPosEmbV2GetWorkspaceSize_t aclnnApplyRotaryPosEmbV2GetWorkspaceSize;
    aclnnApplyRotaryPosEmbV2_t aclnnApplyRotaryPosEmbV2;
} g;

template <typename F>
static bool resolve(void* h, const char* name, F& out) {
    dlerror();
    void* p = dlsym(h, name);
    const char* e = dlerror();
    if (e || !p) { fprintf(stderr, "dlsym %s failed: %s\n", name, e ? e : "(null)"); return false; }
    out = reinterpret_cast<F>(p);
    return true;
}

static bool load_cann() {
    void* hcl = dlopen("libascendcl.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!hcl) { fprintf(stderr, "dlopen libascendcl: %s\n", dlerror()); return false; }
    void* hop = dlopen("libopapi.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!hop) { fprintf(stderr, "dlopen libopapi: %s\n", dlerror()); return false; }
    bool ok = true;
    ok &= resolve(hcl, "aclInit",                   g.aclInit);
    ok &= resolve(hcl, "aclFinalize",               g.aclFinalize);
    ok &= resolve(hcl, "aclrtSetDevice",            g.aclrtSetDevice);
    ok &= resolve(hcl, "aclrtResetDevice",          g.aclrtResetDevice);
    ok &= resolve(hcl, "aclrtCreateStream",         g.aclrtCreateStream);
    ok &= resolve(hcl, "aclrtDestroyStream",        g.aclrtDestroyStream);
    ok &= resolve(hcl, "aclrtSynchronizeStream",    g.aclrtSynchronizeStream);
    ok &= resolve(hcl, "aclrtMalloc",               g.aclrtMalloc);
    ok &= resolve(hcl, "aclrtFree",                 g.aclrtFree);
    ok &= resolve(hcl, "aclrtMemcpy",               g.aclrtMemcpy);
    ok &= resolve(hop, "aclCreateTensor",           g.aclCreateTensor);
    ok &= resolve(hop, "aclDestroyTensor",          g.aclDestroyTensor);
    ok &= resolve(hop, "aclnnRotaryPositionEmbeddingGetWorkspaceSize",
                  g.aclnnRotaryPositionEmbeddingGetWorkspaceSize);
    ok &= resolve(hop, "aclnnRotaryPositionEmbedding",
                  g.aclnnRotaryPositionEmbedding);
    ok &= resolve(hop, "aclnnApplyRotaryPosEmbV2GetWorkspaceSize",
                  g.aclnnApplyRotaryPosEmbV2GetWorkspaceSize);
    ok &= resolve(hop, "aclnnApplyRotaryPosEmbV2",
                  g.aclnnApplyRotaryPosEmbV2);
    return ok;
}

// --- F16 <-> F32 helpers (IEEE 754 binary16, matches talker engine) --------
static uint16_t f32_to_f16(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 0x1;
    int32_t  exp  = ((x >> 23) & 0xFF) - 127;
    uint32_t mant = x & 0x7FFFFF;
    if (exp == 128) { // inf / nan
        return (uint16_t)((sign << 15) | (0x1F << 10) | (mant ? 0x200 : 0));
    }
    if (exp > 15) return (uint16_t)((sign << 15) | (0x1F << 10));         // overflow → inf
    if (exp < -24) return (uint16_t)(sign << 15);                          // underflow → 0
    if (exp < -14) {
        // subnormal
        uint32_t shift = (uint32_t)(-14 - exp);
        uint32_t m = (mant | 0x800000) >> (shift + 13);
        return (uint16_t)((sign << 15) | m);
    }
    uint16_t e16 = (uint16_t)(exp + 15);
    uint16_t m16 = (uint16_t)(mant >> 13);
    return (uint16_t)((sign << 15) | (e16 << 10) | m16);
}
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign << 31;
        else {
            // subnormal
            int e = -14;
            while (!(mant & 0x400)) { mant <<= 1; e--; }
            mant &= 0x3FF;
            f = (sign << 31) | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 0x1F) {
        f = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out; std::memcpy(&out, &f, 4);
    return out;
}

// ----- Simple FNV hash as md5-substitute for tensor dumps --------------------
static uint64_t fnv1a64(const void* data, size_t n) {
    uint64_t h = 14695981039346656037ULL;
    const uint8_t* p = (const uint8_t*)data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

// --- Tensor helpers ----------------------------------------------------------
static aclTensor* make_tensor(void* dev, const std::vector<int64_t>& shape,
                               const std::vector<int64_t>& strides,
                               aclDataType dt = ACL_FLOAT16) {
    return g.aclCreateTensor(shape.data(), (uint64_t)shape.size(), dt,
                              strides.data(), 0, ACL_FORMAT_ND,
                              shape.data(), (uint64_t)shape.size(), dev);
}
static std::vector<int64_t> contig_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> s(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i) s[i] = s[i + 1] * shape[i + 1];
    return s;
}

// --- Device buffer RAII ------------------------------------------------------
struct DevBuf {
    void* ptr = nullptr;
    size_t bytes = 0;
    void alloc(size_t n) { bytes = n; ACL_CHECK(g.aclrtMalloc(&ptr, n, 0)); }
    void from_host(const void* h, size_t n) {
        ACL_CHECK(g.aclrtMemcpy(ptr, bytes, h, n, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    void to_host(void* h, size_t n) {
        ACL_CHECK(g.aclrtMemcpy(h, n, ptr, bytes, ACL_MEMCPY_DEVICE_TO_HOST));
    }
    ~DevBuf() { if (ptr) g.aclrtFree(ptr); }
};

// ----------------------------------------------------------------------------
// Shape constants: our exact talker GQA decode shape.
// ----------------------------------------------------------------------------
static constexpr int64_t B  = 1;
static constexpr int64_t S  = 1;
static constexpr int64_t NQ = 16;
static constexpr int64_t NK = 8;
static constexpr int64_t DH = 128;
static constexpr int64_t HALF = DH / 2;
static constexpr int64_t POS = 5;           // deterministic position
static constexpr float   THETA = 1e6f;      // Qwen3 RoPE theta (talker default)

// Deterministic f16 payload: sin-based so values span ~[-1, 1] like post-QK-norm Q/K.
static std::vector<uint16_t> gen_payload(int64_t n_heads) {
    std::vector<uint16_t> h((size_t)B * S * n_heads * DH);
    for (size_t i = 0; i < h.size(); ++i) {
        float v = std::sin(0.017f * (float)i) * 0.5f;
        h[i] = f32_to_f16(v);
    }
    return h;
}

// Prep A: half-duplicated.  cos[p, j] = cos[p, j+half] = cos(p * freq[j]).
// Matches talker_cann_engine.cpp build_rope_tables_().
static std::vector<uint16_t> gen_cos_prepA(bool is_sin) {
    std::vector<uint16_t> h((size_t)DH, 0);
    for (int j = 0; j < HALF; ++j) {
        float freq  = 1.0f / std::pow(THETA, (float)(2 * j) / (float)DH);
        float angle = (float)POS * freq;
        float v     = is_sin ? sinf(angle) : cosf(angle);
        uint16_t h_v = f32_to_f16(v);
        h[(size_t)j]        = h_v;
        h[(size_t)j + HALF] = h_v;
    }
    return h;
}

// Prep B: MoYoYoTech fill_cos_sin_hf — pair = (d < half) ? d : d - half.
// NB: pair(d) is IDENTICAL to pair(d + half) for d in [0, half), so this
// yields the same table as Prep A. We still compute it independently to
// cross-check, and then also produce a Prep C as a control.
static std::vector<uint16_t> gen_cos_prepB(bool is_sin) {
    std::vector<uint16_t> h((size_t)DH, 0);
    for (int d = 0; d < DH; ++d) {
        int pair = (d < HALF) ? d : (d - HALF);
        float freq  = 1.0f / std::pow(THETA, (float)(2 * pair) / (float)DH);
        float angle = (float)POS * freq;
        float v     = is_sin ? sinf(angle) : cosf(angle);
        h[(size_t)d] = f32_to_f16(v);
    }
    return h;
}

// Prep C (control): GPT-NeoX "stack" layout — cos[0..half] = cos_pair[0..half],
// cos[half..DH] = cos_pair[0..half]. Conceptually equivalent to Prep A but
// written without explicit duplication: computed once for [0,half) and copied.
// This is a byte-identical reference to Prep A and lets us verify our host
// generators are consistent.
static std::vector<uint16_t> gen_cos_prepC(bool is_sin) {
    std::vector<uint16_t> first_half((size_t)HALF, 0);
    for (int j = 0; j < HALF; ++j) {
        float freq  = 1.0f / std::pow(THETA, (float)(2 * j) / (float)DH);
        float angle = (float)POS * freq;
        float v     = is_sin ? sinf(angle) : cosf(angle);
        first_half[(size_t)j] = f32_to_f16(v);
    }
    std::vector<uint16_t> h((size_t)DH);
    for (int j = 0; j < HALF; ++j) { h[j] = first_half[j]; h[j + HALF] = first_half[j]; }
    return h;
}

// ----- Run path v1 (two RotaryPositionEmbedding mode=0 calls) ---------------
// Allocates output buffers; returns them in out_q/out_k.
static void run_v1(aclrtStream stream,
                   void* q_dev, void* k_dev,
                   void* cos_dev, void* sin_dev,
                   void* q_out_dev, void* k_out_dev) {
    // Q and K both 4-D: [1, 1, N, DH] with strides [N*DH, N*DH, DH, 1].
    auto make_4d = [&](void* buf, int64_t n_heads) {
        std::vector<int64_t> shape{B, S, n_heads, DH};
        std::vector<int64_t> str  {n_heads * DH, n_heads * DH, DH, 1};
        return make_tensor(buf, shape, str);
    };
    // cos/sin broadcast to [1,1,1,DH]; strides [DH, DH, DH, 1].
    auto make_cs = [&](void* buf) {
        std::vector<int64_t> shape{1, 1, 1, DH};
        std::vector<int64_t> str  {DH, DH, DH, 1};
        return make_tensor(buf, shape, str);
    };
    // --- Q ---
    {
        aclTensor* tQ  = make_4d(q_dev, NQ);
        aclTensor* tQo = make_4d(q_out_dev, NQ);
        aclTensor* tC  = make_cs(cos_dev);
        aclTensor* tS  = make_cs(sin_dev);
        uint64_t ws = 0; aclOpExecutor* exec = nullptr;
        ACL_CHECK(g.aclnnRotaryPositionEmbeddingGetWorkspaceSize(
            tQ, tC, tS, /*mode=*/0, tQo, &ws, &exec));
        DevBuf wsb; if (ws) wsb.alloc(ws);
        ACL_CHECK(g.aclnnRotaryPositionEmbedding(wsb.ptr, ws, exec, stream));
        g.aclDestroyTensor(tQ); g.aclDestroyTensor(tQo);
        g.aclDestroyTensor(tC); g.aclDestroyTensor(tS);
    }
    // --- K ---
    {
        aclTensor* tK  = make_4d(k_dev, NK);
        aclTensor* tKo = make_4d(k_out_dev, NK);
        aclTensor* tC  = make_cs(cos_dev);
        aclTensor* tS  = make_cs(sin_dev);
        uint64_t ws = 0; aclOpExecutor* exec = nullptr;
        ACL_CHECK(g.aclnnRotaryPositionEmbeddingGetWorkspaceSize(
            tK, tC, tS, /*mode=*/0, tKo, &ws, &exec));
        DevBuf wsb; if (ws) wsb.alloc(ws);
        ACL_CHECK(g.aclnnRotaryPositionEmbedding(wsb.ptr, ws, exec, stream));
        g.aclDestroyTensor(tK); g.aclDestroyTensor(tKo);
        g.aclDestroyTensor(tC); g.aclDestroyTensor(tS);
    }
    ACL_CHECK(g.aclrtSynchronizeStream(stream));
}

// ----- Run path V2 (single ApplyRotaryPosEmbV2 call, in-place) --------------
// Reports GetWorkspaceSize status so we can distinguish "rejected by checker"
// from "accepted but numerically off".
static int run_v2(aclrtStream stream,
                  void* q_dev, void* k_dev,
                  void* cos_dev, void* sin_dev,
                  int64_t layout) {
    auto make_4d = [&](void* buf, int64_t n_heads) {
        std::vector<int64_t> shape{B, S, n_heads, DH};
        std::vector<int64_t> str  {n_heads * DH, n_heads * DH, DH, 1};
        return make_tensor(buf, shape, str);
    };
    auto make_cs = [&](void* buf) {
        std::vector<int64_t> shape{1, S, 1, DH};
        std::vector<int64_t> str  {DH, DH, DH, 1};
        return make_tensor(buf, shape, str);
    };
    aclTensor* tQ = make_4d(q_dev, NQ);
    aclTensor* tK = make_4d(k_dev, NK);
    aclTensor* tC = make_cs(cos_dev);
    aclTensor* tS = make_cs(sin_dev);
    uint64_t ws = 0; aclOpExecutor* exec = nullptr;
    char mode[] = "half";
    aclnnStatus s = g.aclnnApplyRotaryPosEmbV2GetWorkspaceSize(
        tQ, tK, tC, tS, layout, mode, &ws, &exec);
    fprintf(stderr, "[v2] GetWorkspaceSize layout=%ld -> status=%d, ws=%llu\n",
            (long)layout, (int)s, (unsigned long long)ws);
    if (s != 0) {
        g.aclDestroyTensor(tQ); g.aclDestroyTensor(tK);
        g.aclDestroyTensor(tC); g.aclDestroyTensor(tS);
        return (int)s;
    }
    DevBuf wsb; if (ws) wsb.alloc(ws);
    aclnnStatus s2 = g.aclnnApplyRotaryPosEmbV2(wsb.ptr, ws, exec, stream);
    fprintf(stderr, "[v2] exec status=%d\n", (int)s2);
    g.aclDestroyTensor(tQ); g.aclDestroyTensor(tK);
    g.aclDestroyTensor(tC); g.aclDestroyTensor(tS);
    if (s2 != 0) return (int)s2;
    ACL_CHECK(g.aclrtSynchronizeStream(stream));
    return 0;
}

// --- Diff helpers ------------------------------------------------------------
struct DiffStats { double max_abs; double rel_l2; int64_t first_idx; };
static DiffStats diff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
    DiffStats d{0.0, 0.0, -1};
    double l2a = 0, l2d = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        float va = f16_to_f32(a[i]);
        float vb = f16_to_f32(b[i]);
        double dv = (double)va - (double)vb;
        if (std::fabs(dv) > d.max_abs) d.max_abs = std::fabs(dv);
        if (d.first_idx < 0 && std::fabs(dv) > 1e-3) d.first_idx = (int64_t)i;
        l2a += (double)va * va;
        l2d += dv * dv;
    }
    d.rel_l2 = std::sqrt(l2d) / (std::sqrt(l2a) + 1e-30);
    return d;
}

static void dump_first(const char* tag, const std::vector<uint16_t>& v, int n = 8) {
    fprintf(stderr, "%s[0..%d]:", tag, n);
    for (int i = 0; i < n && i < (int)v.size(); ++i) {
        fprintf(stderr, " %+0.5f", f16_to_f32(v[i]));
    }
    fprintf(stderr, "\n");
}

int main() {
    if (!load_cann()) { fprintf(stderr, "CANN symbol load FAILED\n"); return 2; }

    ACL_CHECK(g.aclInit(nullptr));
    ACL_CHECK(g.aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(g.aclrtCreateStream(&stream));

    fprintf(stderr, "[reopen] shape: Q=[%ld,%ld,%ld,%ld] K=[%ld,%ld,%ld,%ld] pos=%ld theta=%g\n",
            (long)B, (long)S, (long)NQ, (long)DH,
            (long)B, (long)S, (long)NK, (long)DH,
            (long)POS, THETA);

    // ---- Host payloads -----------------------------------------------------
    auto hQ = gen_payload(NQ);
    auto hK = gen_payload(NK);
    auto hcosA = gen_cos_prepA(false);
    auto hsinA = gen_cos_prepA(true);
    auto hcosB = gen_cos_prepB(false);
    auto hsinB = gen_cos_prepB(true);
    auto hcosC = gen_cos_prepC(false);
    auto hsinC = gen_cos_prepC(true);

    fprintf(stderr, "[reopen] hQ fnv=%016llx  hK fnv=%016llx\n",
            (unsigned long long)fnv1a64(hQ.data(), hQ.size()*2),
            (unsigned long long)fnv1a64(hK.data(), hK.size()*2));
    fprintf(stderr, "[reopen] cosA fnv=%016llx  cosB fnv=%016llx  cosC fnv=%016llx\n",
            (unsigned long long)fnv1a64(hcosA.data(), hcosA.size()*2),
            (unsigned long long)fnv1a64(hcosB.data(), hcosB.size()*2),
            (unsigned long long)fnv1a64(hcosC.data(), hcosC.size()*2));
    dump_first("  cosA", hcosA);
    dump_first("  cosB", hcosB);
    dump_first("  cosC", hcosC);
    dump_first("  cosA[64..]", std::vector<uint16_t>(hcosA.begin()+64, hcosA.end()));
    dump_first("  cosB[64..]", std::vector<uint16_t>(hcosB.begin()+64, hcosB.end()));

    bool prepA_B_equal = std::memcmp(hcosA.data(), hcosB.data(), hcosA.size()*2) == 0 &&
                         std::memcmp(hsinA.data(), hsinB.data(), hsinA.size()*2) == 0;
    fprintf(stderr, "[reopen] Prep A == Prep B (byte-identical)? %s\n",
            prepA_B_equal ? "YES (hypothesis falsified at host)" : "NO");

    // ---- Device buffers ----------------------------------------------------
    size_t q_bytes = (size_t)NQ * DH * 2;
    size_t k_bytes = (size_t)NK * DH * 2;
    size_t cs_bytes = (size_t)DH * 2;

    // v1 path uses separate out buffers (non-in-place).
    DevBuf dQ_v1, dK_v1, dQo_v1, dKo_v1;
    DevBuf dQ_A, dK_A, dQ_B, dK_B;
    DevBuf dcosA, dsinA, dcosB, dsinB;
    dQ_v1.alloc(q_bytes); dK_v1.alloc(k_bytes);
    dQo_v1.alloc(q_bytes); dKo_v1.alloc(k_bytes);
    dQ_A.alloc(q_bytes);  dK_A.alloc(k_bytes);
    dQ_B.alloc(q_bytes);  dK_B.alloc(k_bytes);
    dcosA.alloc(cs_bytes); dsinA.alloc(cs_bytes);
    dcosB.alloc(cs_bytes); dsinB.alloc(cs_bytes);

    dQ_v1.from_host(hQ.data(), q_bytes);
    dK_v1.from_host(hK.data(), k_bytes);
    dQ_A.from_host(hQ.data(), q_bytes);
    dK_A.from_host(hK.data(), k_bytes);
    dQ_B.from_host(hQ.data(), q_bytes);
    dK_B.from_host(hK.data(), k_bytes);
    dcosA.from_host(hcosA.data(), cs_bytes);
    dsinA.from_host(hsinA.data(), cs_bytes);
    dcosB.from_host(hcosB.data(), cs_bytes);
    dsinB.from_host(hsinB.data(), cs_bytes);

    // ---- Path 1: v1 reference (Prep A) -------------------------------------
    fprintf(stderr, "\n[reopen] === Path 1: v1 RotaryPositionEmbedding mode=0 NEOX, Prep A ===\n");
    run_v1(stream, dQ_v1.ptr, dK_v1.ptr, dcosA.ptr, dsinA.ptr,
                    dQo_v1.ptr, dKo_v1.ptr);
    std::vector<uint16_t> qV1(NQ*DH), kV1(NK*DH);
    dQo_v1.to_host(qV1.data(), q_bytes);
    dKo_v1.to_host(kV1.data(), k_bytes);
    dump_first("  v1 Q_out", qV1);
    dump_first("  v1 K_out", kV1);
    fprintf(stderr, "  v1 Q fnv=%016llx  v1 K fnv=%016llx\n",
            (unsigned long long)fnv1a64(qV1.data(), q_bytes),
            (unsigned long long)fnv1a64(kV1.data(), k_bytes));

    // ---- Path 2: V2 with Prep A (layout=1) ---------------------------------
    fprintf(stderr, "\n[reopen] === Path 2: V2 + Prep A, layout=1 BSND ===\n");
    int rc_A = run_v2(stream, dQ_A.ptr, dK_A.ptr, dcosA.ptr, dsinA.ptr, /*layout=*/1);
    std::vector<uint16_t> qA(NQ*DH), kA(NK*DH);
    if (rc_A == 0) {
        dQ_A.to_host(qA.data(), q_bytes);
        dK_A.to_host(kA.data(), k_bytes);
        dump_first("  V2A Q_out", qA);
        dump_first("  V2A K_out", kA);
        fprintf(stderr, "  V2A Q fnv=%016llx  V2A K fnv=%016llx\n",
                (unsigned long long)fnv1a64(qA.data(), q_bytes),
                (unsigned long long)fnv1a64(kA.data(), k_bytes));
    }

    // ---- Path 3: V2 with Prep B (layout=1) ---------------------------------
    fprintf(stderr, "\n[reopen] === Path 3: V2 + Prep B, layout=1 BSND ===\n");
    int rc_B = run_v2(stream, dQ_B.ptr, dK_B.ptr, dcosB.ptr, dsinB.ptr, /*layout=*/1);
    // (kA/kB capture happens below; move extended probes after core diff.)
    std::vector<uint16_t> qB(NQ*DH), kB(NK*DH);
    if (rc_B == 0) {
        dQ_B.to_host(qB.data(), q_bytes);
        dK_B.to_host(kB.data(), k_bytes);
        dump_first("  V2B Q_out", qB);
        dump_first("  V2B K_out", kB);
        fprintf(stderr, "  V2B Q fnv=%016llx  V2B K fnv=%016llx\n",
                (unsigned long long)fnv1a64(qB.data(), q_bytes),
                (unsigned long long)fnv1a64(kB.data(), k_bytes));
    }

    // ---- Diff ---------------------------------------------------------------
    auto print_diff = [&](const char* tag, const std::vector<uint16_t>& a,
                          const std::vector<uint16_t>& b, int rc) {
        if (rc != 0) { fprintf(stderr, "%-20s REJECTED (rc=%d)\n", tag, rc); return; }
        auto d = diff(a, b);
        fprintf(stderr, "%-20s max_abs=%.6f  rel_l2=%.6e  first_diff_idx=%lld\n",
                tag, d.max_abs, d.rel_l2, (long long)d.first_idx);
    };

    fprintf(stderr, "\n[reopen] === DIFF vs v1 reference ===\n");
    print_diff("V2A vs v1 Q", qA, qV1, rc_A);
    print_diff("V2A vs v1 K", kA, kV1, rc_A);
    print_diff("V2B vs v1 Q", qB, qV1, rc_B);
    print_diff("V2B vs v1 K", kB, kV1, rc_B);

    // ---- Verdict -----------------------------------------------------------
    // F16 rounding between two semantically-equivalent ops is typically
    // max_abs ~ 5e-4 (1 ulp at magnitude 1). We set a generous threshold of
    // 1e-2 which is what MoYoYoTech's test_rope_fused.cpp uses.
    const double kTol = 1e-2;
    bool A_q_ok = rc_A == 0 && diff(qA, qV1).max_abs < kTol;
    bool A_k_ok = rc_A == 0 && diff(kA, kV1).max_abs < kTol;
    bool B_q_ok = rc_B == 0 && diff(qB, qV1).max_abs < kTol;
    bool B_k_ok = rc_B == 0 && diff(kB, kV1).max_abs < kTol;

    // ---- Path 4: prod-wiring simulation — K written to slot inside a large
    //              cache buffer. This exercises the exact descriptor pattern
    //              talker_cann_engine.cpp uses: cache = [MAX_SEQ, kv_dim], K
    //              viewed at offset pos*kv_dim as [1,1,NK,DH] with strides
    //              [kv_dim, kv_dim, DH, 1] (same stride pattern our decode
    //              emits). V2 reads+writes in-place at this view.
    // -------------------------------------------------------------------------
    fprintf(stderr, "\n[reopen] === Path 4: V2 prod-wiring sim — K slot in cache ===\n");
    int rc_C = -1;
    std::vector<uint16_t> kCache_out(NK*DH);
    std::vector<uint16_t> qC_out(NQ*DH);
    {
        const int64_t MAX_SEQ = 16;           // arbitrary; only slot at POS matters
        const int64_t kv_dim  = NK * DH;
        DevBuf dKcache; dKcache.alloc((size_t)MAX_SEQ * kv_dim * 2);
        // Pre-populate with garbage (non-zero) so we can detect "partial write"
        // failure modes. Use a different pattern than K to make it distinguishable.
        std::vector<uint16_t> cacheInit((size_t)MAX_SEQ * kv_dim, f32_to_f16(-99.0f));
        dKcache.from_host(cacheInit.data(), cacheInit.size()*2);
        // Copy our hK (the already-QK-normed K) into slot at POS.
        uint16_t* slot_host = cacheInit.data() + (size_t)POS * kv_dim;
        std::memcpy(slot_host, hK.data(), hK.size()*2);
        dKcache.from_host(cacheInit.data(), cacheInit.size()*2);

        // Q still goes into its own buffer (same as prod).
        DevBuf dQ4; dQ4.alloc(q_bytes);
        dQ4.from_host(hQ.data(), q_bytes);

        // Build tensors. Q: [1,1,NQ,DH] contiguous, straightforward.
        // K: [1,1,NK,DH] at &cache[POS*kv_dim], strides [kv_dim, kv_dim, DH, 1].
        std::vector<int64_t> q_shape{B, S, NQ, DH};
        std::vector<int64_t> q_str  {NQ*DH, NQ*DH, DH, 1};
        std::vector<int64_t> k_shape{B, S, NK, DH};
        std::vector<int64_t> k_str  {kv_dim, kv_dim, DH, 1};
        void* k_slot_dev = (uint8_t*)dKcache.ptr + (size_t)POS * kv_dim * 2;

        std::vector<int64_t> cs_shape{1, S, 1, DH};
        std::vector<int64_t> cs_str  {DH, DH, DH, 1};

        aclTensor* tQ = make_tensor(dQ4.ptr, q_shape, q_str);
        aclTensor* tK = make_tensor(k_slot_dev, k_shape, k_str);
        aclTensor* tC = make_tensor(dcosA.ptr, cs_shape, cs_str);
        aclTensor* tS = make_tensor(dsinA.ptr, cs_shape, cs_str);
        uint64_t ws = 0; aclOpExecutor* exec = nullptr;
        char mode[] = "half";
        aclnnStatus s = g.aclnnApplyRotaryPosEmbV2GetWorkspaceSize(
            tQ, tK, tC, tS, /*layout=*/1, mode, &ws, &exec);
        fprintf(stderr, "[v2-slot] GetWorkspaceSize status=%d ws=%llu\n", (int)s, (unsigned long long)ws);
        if (s == 0) {
            DevBuf wsb; if (ws) wsb.alloc(ws);
            aclnnStatus s2 = g.aclnnApplyRotaryPosEmbV2(wsb.ptr, ws, exec, stream);
            fprintf(stderr, "[v2-slot] exec status=%d\n", (int)s2);
            if (s2 == 0) {
                ACL_CHECK(g.aclrtSynchronizeStream(stream));
                rc_C = 0;
                // Read back the K slot.
                std::vector<uint16_t> cache_readback((size_t)MAX_SEQ * kv_dim);
                dKcache.to_host(cache_readback.data(), cache_readback.size()*2);
                std::memcpy(kCache_out.data(), cache_readback.data() + (size_t)POS * kv_dim,
                             NK*DH*2);
                dQ4.to_host(qC_out.data(), q_bytes);
                dump_first("  V2-slot Q_out", qC_out);
                dump_first("  V2-slot Kslot_out", kCache_out);

                // Check adjacent cache slots (POS-1, POS+1) to detect over-write.
                fprintf(stderr, "  V2-slot cache[POS-1] first 4: %+0.5f %+0.5f %+0.5f %+0.5f (want -99)\n",
                        f16_to_f32(cache_readback[(size_t)(POS-1)*kv_dim + 0]),
                        f16_to_f32(cache_readback[(size_t)(POS-1)*kv_dim + 1]),
                        f16_to_f32(cache_readback[(size_t)(POS-1)*kv_dim + 2]),
                        f16_to_f32(cache_readback[(size_t)(POS-1)*kv_dim + 3]));
                fprintf(stderr, "  V2-slot cache[POS+1] first 4: %+0.5f %+0.5f %+0.5f %+0.5f (want -99)\n",
                        f16_to_f32(cache_readback[(size_t)(POS+1)*kv_dim + 0]),
                        f16_to_f32(cache_readback[(size_t)(POS+1)*kv_dim + 1]),
                        f16_to_f32(cache_readback[(size_t)(POS+1)*kv_dim + 2]),
                        f16_to_f32(cache_readback[(size_t)(POS+1)*kv_dim + 3]));
            }
        }
        g.aclDestroyTensor(tQ); g.aclDestroyTensor(tK);
        g.aclDestroyTensor(tC); g.aclDestroyTensor(tS);
    }

    fprintf(stderr, "\n[reopen] === VERDICT ===\n");
    fprintf(stderr, "  V2 + Prep A: Q %s, K %s\n", A_q_ok?"PASS":"FAIL", A_k_ok?"PASS":"FAIL");
    fprintf(stderr, "  V2 + Prep B: Q %s, K %s\n", B_q_ok?"PASS":"FAIL", B_k_ok?"PASS":"FAIL");
    // Slot-based prod-wiring sim.
    bool C_q_ok = rc_C == 0 && diff(qC_out, qV1).max_abs < kTol;
    bool C_k_ok = rc_C == 0 && diff(kCache_out, kV1).max_abs < kTol;
    fprintf(stderr, "  V2 prod-slot: Q %s, K %s (rc=%d)\n",
            C_q_ok?"PASS":"FAIL", C_k_ok?"PASS":"FAIL", rc_C);
    if (rc_C == 0) {
        auto dq = diff(qC_out, qV1); auto dk = diff(kCache_out, kV1);
        fprintf(stderr, "    slot Q max_abs=%.6f rel_l2=%.6e  K max_abs=%.6f rel_l2=%.6e\n",
                dq.max_abs, dq.rel_l2, dk.max_abs, dk.rel_l2);
    }
    bool green = (A_q_ok && A_k_ok) || (B_q_ok && B_k_ok);
    fprintf(stderr, "  OVERALL: %s\n", green ? "GREEN — V2 works on our GQA shape"
                                              : "RED — neither prep matches v1");

    // ---- Cleanup -----------------------------------------------------------
    ACL_CHECK(g.aclrtDestroyStream(stream));
    ACL_CHECK(g.aclrtResetDevice(0));
    g.aclFinalize();
    return green ? 0 : 1;
}
