// QIE-Q2.2 Q4_1-resident Gate 0 probe — aclnnWeightQuantBatchMatmulV3
// capability test for Q4_1-style (asymmetric, per-block min) quantization
// via the op's antiquantOffset parameter.
//
// Agent: QIE-Q2.2-PROBE (2026-04-22)
// Contract: Q1.10 amendment (mixed-quant repack extension; `afb3919e`).
// Predecessor: Q4_0 probe `test_qie_q4resident_probe.cpp` — GREEN (cos_sim
//              0.999, docs/qie_q2_q4resident_probe.md).
// Fallback audit: `docs/qie_q21_fallback_audit.md` — 116 FFN-down tensors
//                 stored as Q4_1 (per-block F16 d + F16 m + 16B INT4).
//
// Goal: answer ONE question before the 300-LoC Q2.2 engine refactor —
//   Does aclnnWeightQuantBatchMatmulV3 accept a Q4_1-style packing where:
//     - nibble is UNSIGNED 4-bit [0, 15] (no XOR 0x08),
//     - antiquantScale   = per-group F16 d,   shape [K/G, N],
//     - antiquantOffset  = per-group F16 -m/d, shape [K/G, N],
//   and yield cos_sim > 0.99 vs a CPU dequant reference using the Q4_1
//   dequant formula `x = u * d + m` (u ∈ [0, 15])?
//
// If GREEN, Q2.2.1 landing `repack_q4_1_upload` can mirror the existing
// Q4_0 re-tile, differing only in:
//   1) skip the `u ^ 0x08` bias-8 conversion (Q4_1 nibble stays unsigned),
//   2) emit a third buffer — per-group offset F16 = `-m/d` — uploaded
//      alongside the existing scale buffer,
//   3) matmul call site passes `antiquantOffsetOptional=offset_dev`
//      instead of nullptr.
//
// If RED, fall back to F16 dequant for Q4_1 (keeps +1.27 GiB HBM cost) or
// probe an alternative layout (e.g. per-channel offset, or reshaping as
// Q4_0 with zero-shift and losing the `m` term — inaccurate).
//
// Shape (same as Q4_0 probe for direct perf comparability):
//   x            = [M=128, K=3072] F16
//   weight (W4)  = [K=3072, N=3072] INT4, unsigned per-group G=32 along K
//                  → scale  shape [K/G=96, N=3072] F16
//                  → offset shape [K/G=96, N=3072] F16
//   y            = [M=128, N=3072] F16
//
// Dequant reference (Q4_1 flavour):
//   for each group g of 32 elements in column n:
//     d   = (max(w_group) - min(w_group)) / 15.0
//     m   = min(w_group)
//     u[i] = clamp(round((w[i] - m) / d), 0, 15)
//     x_deq[i] = u[i] * d + m
//
// WQBMMv3 wiring (op semantics per CANN docs):
//   w_hat[k, n] = (nibble[k, n] - antiquantOffset[k/G, n]) * antiquantScale[k/G, n]
//   → to reproduce `u*d + m = d*(u - (-m/d))`, set:
//     antiquantScale  = d
//     antiquantOffset = -m/d
//     nibble          = u (unsigned 0..15)
//
// Exit code 0 = GREEN (op works, cos_sim > 0.99, perf ≤ 2.0× F16 baseline).
// Exit code 1 = YELLOW (op works but numerics or perf off).
// Exit code 2 = RED    (op rejects W4+offset config or output grossly wrong).
//
// Build on ac03:
//   bash build_and_run_q4_1.sh

#include <acl/acl.h>
#include <aclnnop/aclnn_weight_quant_batch_matmul_v3.h>
#include <aclnnop/aclnn_matmul.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#define ACL_CHECK(expr) do {                                                     \
    aclError __err = (expr);                                                     \
    if (__err != ACL_SUCCESS) {                                                  \
        fprintf(stderr, "ACL error %d at %s:%d: %s\n",                           \
                (int)__err, __FILE__, __LINE__, #expr);                          \
        std::abort();                                                            \
    }                                                                            \
} while (0)

#define ACL_CHECK_NN(expr) do {                                                  \
    aclnnStatus __st = (expr);                                                   \
    if (__st != 0) {                                                             \
        fprintf(stderr, "aclnn error %d at %s:%d: %s\n",                         \
                (int)__st, __FILE__, __LINE__, #expr);                          \
        std::abort();                                                            \
    }                                                                            \
} while (0)

// ---------- F16 <-> F32 (IEEE 754 half, no subnormal flush) ----------
static inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint16_t res = (uint16_t)(mant >> (14 - exp));
        return (uint16_t)(sign | res);
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    }
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) { out = sign; }
        else {
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

// ---------- Shape constants (per contract §Q1.10 probe spec) ----------
static constexpr int64_t M = 128;
static constexpr int64_t K = 3072;
static constexpr int64_t N = 3072;
static constexpr int64_t G = 32;       // antiquantGroupSize (Q4_1 block)
static constexpr int64_t K_G = K / G;  // 96 scale/offset rows

// ---------- CPU Q4_1-style quantize (asymmetric, per-group d + m) ----------
// For each column n in [0,N), for each group g in [0,K/G):
//   absmin = min(w_group); absmax = max(w_group);
//   d = (absmax - absmin) / 15.0f;  // 4-bit unsigned range
//   m = absmin;                      // per-group offset in source scale
//   u[i] = clamp(round((w[i] - m) / d), 0, 15)
// Dequant ref: x[i] = u[i] * d + m.
//
// WQBMMv3 reproduces this via:
//   w_hat = (u - offset_store) * d  where offset_store = -m / d.
// So we emit:
//   packed:  K*N/2 bytes, column-major nibble packing — unsigned (no XOR).
//   scales:  K_G × N F16, row-major, value = d.
//   offsets: K_G × N F16, row-major, value = -m / d.
//   dequant_ref_f16: K × N F16 of (u * d + m) for CPU ref matmul.
struct Q41Pack {
    std::vector<uint8_t>  packed;
    std::vector<uint16_t> scales;
    std::vector<uint16_t> offsets;
    std::vector<uint16_t> dequant_f16;
};

static Q41Pack cpu_quantize_q4_1_per_group(const std::vector<float>& w_dense,
                                            int64_t k, int64_t n, int64_t g) {
    Q41Pack out;
    out.packed.assign((size_t)k * n / 2, 0);
    out.scales.assign((size_t)(k / g) * n, 0);
    out.offsets.assign((size_t)(k / g) * n, 0);
    out.dequant_f16.assign((size_t)k * n, 0);

    for (int64_t col = 0; col < n; ++col) {
        for (int64_t grp = 0; grp < k / g; ++grp) {
            // 1) find min/max in this group (asymmetric range, Q4_1 style)
            float vmin =  std::numeric_limits<float>::infinity();
            float vmax = -std::numeric_limits<float>::infinity();
            for (int64_t i = 0; i < g; ++i) {
                float v = w_dense[(grp * g + i) * n + col];
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
            float d = (vmax - vmin) / 15.0f;
            float m = vmin;
            if (d == 0.0f) d = 1e-7f;
            out.scales [(size_t)grp * n + col] = f32_to_f16(d);
            // offset_store = -m / d  (so WQBMMv3 computes (u - off)*d = u*d + m).
            out.offsets[(size_t)grp * n + col] = f32_to_f16(-m / d);

            // 2) quantize + pack (unsigned nibble — NO XOR 0x08 for Q4_1)
            for (int64_t i = 0; i < g; ++i) {
                int64_t k_idx = grp * g + i;
                float v = w_dense[k_idx * n + col];
                int u = (int)std::lrintf((v - m) / d);
                if (u < 0)  u = 0;
                if (u > 15) u = 15;
                uint8_t nibble = (uint8_t)(u & 0x0f);

                // Dequant reference uses the exact (u*d + m) formula.
                out.dequant_f16[k_idx * n + col] =
                    f32_to_f16((float)u * d + m);

                // Pack column-major: linear_index = col*K + k_idx
                size_t lin = (size_t)col * k + k_idx;
                size_t byte_i = lin / 2;
                if ((lin & 1) == 0) {
                    out.packed[byte_i] = (out.packed[byte_i] & 0xf0) | nibble;
                } else {
                    out.packed[byte_i] = (out.packed[byte_i] & 0x0f) |
                                         (uint8_t)(nibble << 4);
                }
            }
        }
    }
    return out;
}

// ---------- CPU F32 matmul: y[M,N] = x[M,K] @ w[K,N] (F16 inputs) ----------
static void cpu_matmul_f16(const std::vector<uint16_t>& x,
                           const std::vector<uint16_t>& w,
                           std::vector<float>& y) {
    y.assign((size_t)M * N, 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t kk = 0; kk < K; ++kk) {
            float xv = f16_to_f32(x[(size_t)i * K + kk]);
            for (int64_t j = 0; j < N; ++j) {
                float wv = f16_to_f32(w[(size_t)kk * N + j]);
                y[(size_t)i * N + j] += xv * wv;
            }
        }
    }
}

static double cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / std::sqrt(na * nb);
}

static double max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
    double mm = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double e = std::fabs((double)a[i] - (double)b[i]);
        if (e > mm) mm = e;
    }
    return mm;
}

int main() {
    printf("=== QIE-Q2.2 Q4_1-resident Gate 0 probe ===\n");
    printf("Shape: x[M=%lld, K=%lld] F16  @  w[K=%lld, N=%lld] INT4 (G=%lld, Q4_1)\n",
           (long long)M, (long long)K, (long long)K, (long long)N, (long long)G);
    printf("Scale  shape: [K/G=%lld, N=%lld] F16  (d per group)\n",
           (long long)K_G, (long long)N);
    printf("Offset shape: [K/G=%lld, N=%lld] F16  (-m/d per group)\n\n",
           (long long)K_G, (long long)N);

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    // ---------- 1) Host tensors ----------
    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
    // Q4_1 captures asymmetric distributions — bias weights to positive side
    // to exercise the `m` term meaningfully (symmetric data would degenerate
    // to a Q4_0-like encoding with m ≈ -7.5·d).
    std::normal_distribution<float>       wdist(0.15f, 0.30f);

    std::vector<uint16_t> x_host((size_t)M * K);
    for (auto& v : x_host) v = f32_to_f16(dist(rng));

    std::vector<float> w_dense((size_t)K * N);
    for (auto& v : w_dense) v = wdist(rng);

    printf("[host] Quantizing %lld × %lld weight Q4_1 per-group (G=%lld)...\n",
           (long long)K, (long long)N, (long long)G);
    Q41Pack q = cpu_quantize_q4_1_per_group(w_dense, K, N, G);

    // Report source asymmetry: mean(m/scale) — if near -7.5 the Q4_1
    // advantage is tiny; if ≠ 0 then Q4_1 earns its keep.
    {
        double mean_off = 0.0;
        for (auto h : q.offsets) mean_off += f16_to_f32(h);
        mean_off /= (double)q.offsets.size();
        printf("[host] mean(offset_store = -m/d) = %.3f  "
               "(0 ⇒ symmetric; ~-7.5 ⇒ Q4_0-equivalent)\n", mean_off);
    }

    // ---------- 2) CPU reference ----------
    printf("[cpu]  Computing F32 reference via CPU F16 matmul over dequant...\n");
    std::vector<float> y_cpu;
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    cpu_matmul_f16(x_host, q.dequant_f16, y_cpu);
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();
    printf("[cpu]  Reference matmul done in %.1f ms\n", cpu_ms);

    // ---------- 3) Device upload ----------
    void *x_dev = nullptr, *w_dev = nullptr, *s_dev = nullptr,
         *o_dev = nullptr, *y_dev = nullptr;
    size_t x_bytes = (size_t)M * K * sizeof(uint16_t);
    size_t w_bytes = q.packed.size();
    size_t s_bytes = q.scales.size()  * sizeof(uint16_t);
    size_t o_bytes = q.offsets.size() * sizeof(uint16_t);
    size_t y_bytes = (size_t)M * N * sizeof(uint16_t);
    ACL_CHECK(aclrtMalloc(&x_dev, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&w_dev, w_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&s_dev, s_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&o_dev, o_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&y_dev, y_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(x_dev, x_bytes, x_host.data(), x_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(w_dev, w_bytes, q.packed.data(), w_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(s_dev, s_bytes, q.scales.data(), s_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(o_dev, o_bytes, q.offsets.data(), o_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    printf("[npu]  Uploaded x=%zu B, w_int4=%zu B (%.2f MiB), "
           "scale=%zu B, offset=%zu B, y_out=%zu B\n",
           x_bytes, w_bytes, (double)w_bytes / (1024.0 * 1024.0),
           s_bytes, o_bytes, y_bytes);

    // ---------- 4) Build tensors ----------
    int64_t x_shape[2]   = {M, K};
    int64_t x_strides[2] = {K, 1};
    int64_t x_storage[2] = {M, K};
    aclTensor* t_x = aclCreateTensor(
        x_shape, 2, ACL_FLOAT16, x_strides, 0, ACL_FORMAT_ND,
        x_storage, 2, x_dev);

    // W4 tensor: shape [K, N], strides (1, K); storage K*N INT4 elements.
    int64_t w_shape[2]   = {K, N};
    int64_t w_strides[2] = {1, K};
    int64_t w_storage    = K * N;
    aclTensor* t_w = aclCreateTensor(
        w_shape, 2, ACL_INT4, w_strides, 0, ACL_FORMAT_ND,
        &w_storage, 1, w_dev);

    // Scale: [K/G, N], row-major.
    int64_t s_shape[2]   = {K_G, N};
    int64_t s_strides[2] = {N, 1};
    int64_t s_storage[2] = {K_G, N};
    aclTensor* t_s = aclCreateTensor(
        s_shape, 2, ACL_FLOAT16, s_strides, 0, ACL_FORMAT_ND,
        s_storage, 2, s_dev);

    // Offset: same shape as scale (per-group per-column).
    int64_t o_shape[2]   = {K_G, N};
    int64_t o_strides[2] = {N, 1};
    int64_t o_storage[2] = {K_G, N};
    aclTensor* t_o = aclCreateTensor(
        o_shape, 2, ACL_FLOAT16, o_strides, 0, ACL_FORMAT_ND,
        o_storage, 2, o_dev);

    int64_t y_shape[2]   = {M, N};
    int64_t y_strides[2] = {N, 1};
    int64_t y_storage[2] = {M, N};
    aclTensor* t_y = aclCreateTensor(
        y_shape, 2, ACL_FLOAT16, y_strides, 0, ACL_FORMAT_ND,
        y_storage, 2, y_dev);

    // ---------- 5) Dispatch WQBMMv3 with non-null antiquantOffset ----------
    uint64_t ws_bytes = 0;
    aclOpExecutor* exec = nullptr;
    aclnnStatus st = aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
        t_x, t_w, t_s,
        /*antiquantOffsetOptional*/ t_o,         // <-- Q4_1 per-group offset
        /*quantScaleOptional*/      nullptr,
        /*quantOffsetOptional*/     nullptr,
        /*biasOptional*/            nullptr,
        /*antiquantGroupSize*/      (int)G,
        /*innerPrecise*/            1,
        t_y, &ws_bytes, &exec);

    if (st != 0) {
        fprintf(stderr,
                "[npu]  *** WQBMMv3 GetWorkspaceSize REJECTED W4+G=32+offset "
                "config with status=%d ***\n",
                (int)st);
        fprintf(stderr,
                "[verdict] RED — op does not accept per-group INT4 antiquantOffset.\n"
                "          Next step: try per-channel fallback or keep Q4_1 on F16 path.\n");
        aclDestroyTensor(t_x); aclDestroyTensor(t_w);
        aclDestroyTensor(t_s); aclDestroyTensor(t_o); aclDestroyTensor(t_y);
        aclrtFree(x_dev); aclrtFree(w_dev); aclrtFree(s_dev);
        aclrtFree(o_dev); aclrtFree(y_dev);
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
        return 2;
    }

    printf("[npu]  WQBMMv3 accepted W4+G=32+offset config. "
           "workspace=%llu B (%.2f MiB)\n",
           (unsigned long long)ws_bytes, (double)ws_bytes / (1024.0 * 1024.0));

    void* ws_dev = nullptr;
    if (ws_bytes > 0) {
        ACL_CHECK(aclrtMalloc(&ws_dev, ws_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    const int warmup = 3;
    const int iters  = 20;
    for (int i = 0; i < warmup; ++i) {
        ACL_CHECK_NN(aclnnWeightQuantBatchMatmulV3(ws_dev, ws_bytes, exec, stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK_NN(aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
            t_x, t_w, t_s, t_o, nullptr, nullptr, nullptr,
            (int)G, 1, t_y, &ws_bytes, &exec));
    }

    std::vector<double> q4_times_us;
    q4_times_us.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ACL_CHECK_NN(aclnnWeightQuantBatchMatmulV3(ws_dev, ws_bytes, exec, stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        q4_times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());

        ACL_CHECK_NN(aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
            t_x, t_w, t_s, t_o, nullptr, nullptr, nullptr,
            (int)G, 1, t_y, &ws_bytes, &exec));
    }
    std::sort(q4_times_us.begin(), q4_times_us.end());
    double q4_median_us = q4_times_us[iters / 2];
    double q4_p10_us    = q4_times_us[iters / 10];
    double q4_p90_us    = q4_times_us[(iters * 9) / 10];
    printf("[npu]  W4_1 matmul wall: median=%.1f us  p10=%.1f us  p90=%.1f us "
           "(%d iters)\n",
           q4_median_us, q4_p10_us, q4_p90_us, iters);

    // ---------- 6) Copy result back, compare ----------
    std::vector<uint16_t> y_npu_h((size_t)M * N);
    ACL_CHECK(aclrtMemcpy(y_npu_h.data(), y_bytes, y_dev, y_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> y_npu_f32((size_t)M * N);
    for (size_t i = 0; i < y_npu_h.size(); ++i) y_npu_f32[i] = f16_to_f32(y_npu_h[i]);

    double cos = cosine_sim(y_cpu, y_npu_f32);
    double mae = max_abs_err(y_cpu, y_npu_f32);
    printf("\n[compare] cosine_sim(CPU ref, NPU Q4_1 matmul) = %.6f\n", cos);
    printf("[compare] max_abs_err                           = %.6f\n", mae);

    // ---------- 7) F16 aclnnMm baseline for perf comparison ----------
    void* w_f16_dev = nullptr;
    size_t w_f16_bytes = (size_t)K * N * sizeof(uint16_t);
    ACL_CHECK(aclrtMalloc(&w_f16_dev, w_f16_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(w_f16_dev, w_f16_bytes, q.dequant_f16.data(),
                          w_f16_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    int64_t w16_shape[2]   = {K, N};
    int64_t w16_strides[2] = {N, 1};
    int64_t w16_storage[2] = {K, N};
    aclTensor* t_w16 = aclCreateTensor(
        w16_shape, 2, ACL_FLOAT16, w16_strides, 0, ACL_FORMAT_ND,
        w16_storage, 2, w_f16_dev);

    uint64_t ws_mm = 0;
    aclOpExecutor* exec_mm = nullptr;
    aclnnStatus st_mm = aclnnMatmulGetWorkspaceSize(
        t_x, t_w16, t_y, /*cubeMathType*/ 1, &ws_mm, &exec_mm);
    if (st_mm != 0) {
        fprintf(stderr, "[npu]  aclnnMatmul GetWorkspaceSize failed status=%d "
                        "(non-fatal — skipping baseline)\n", (int)st_mm);
    } else {
        void* ws_mm_dev = nullptr;
        if (ws_mm > 0) ACL_CHECK(aclrtMalloc(&ws_mm_dev, ws_mm, ACL_MEM_MALLOC_HUGE_FIRST));

        for (int i = 0; i < warmup; ++i) {
            ACL_CHECK_NN(aclnnMatmul(ws_mm_dev, ws_mm, exec_mm, stream));
            ACL_CHECK(aclrtSynchronizeStream(stream));
            ACL_CHECK_NN(aclnnMatmulGetWorkspaceSize(
                t_x, t_w16, t_y, 1, &ws_mm, &exec_mm));
        }
        std::vector<double> mm_times_us;
        mm_times_us.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ACL_CHECK_NN(aclnnMatmul(ws_mm_dev, ws_mm, exec_mm, stream));
            ACL_CHECK(aclrtSynchronizeStream(stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            mm_times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            ACL_CHECK_NN(aclnnMatmulGetWorkspaceSize(
                t_x, t_w16, t_y, 1, &ws_mm, &exec_mm));
        }
        std::sort(mm_times_us.begin(), mm_times_us.end());
        double mm_median_us = mm_times_us[iters / 2];
        printf("[npu]  F16 aclnnMm baseline wall: median=%.1f us  (same shape)\n",
               mm_median_us);
        printf("[perf] Q4_1 / F16 ratio = %.2fx (target < 2.0x, reference Q4_0 was 1.70x)\n",
               q4_median_us / mm_median_us);

        if (ws_mm_dev) aclrtFree(ws_mm_dev);
    }

    // ---------- 8) Verdict ----------
    int rc = 0;
    const char* verdict = nullptr;
    if (cos > 0.99) {
        verdict = "GREEN";
        rc = 0;
    } else if (cos > 0.90) {
        verdict = "YELLOW";
        rc = 1;
    } else {
        verdict = "RED";
        rc = 2;
    }
    printf("\n[verdict] %s  (cos_sim = %.6f, mae = %.6f, Q4_1 median = %.1f us)\n",
           verdict, cos, mae, q4_median_us);
    if (rc == 0) {
        printf("          Q2.2.1 repack_q4_1_upload is cleared to proceed.\n");
    }

    // ---------- Cleanup ----------
    aclDestroyTensor(t_x); aclDestroyTensor(t_w);
    aclDestroyTensor(t_s); aclDestroyTensor(t_o);
    aclDestroyTensor(t_y); aclDestroyTensor(t_w16);
    if (ws_dev) aclrtFree(ws_dev);
    aclrtFree(x_dev); aclrtFree(w_dev); aclrtFree(s_dev);
    aclrtFree(o_dev); aclrtFree(y_dev);
    aclrtFree(w_f16_dev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return rc;
}
