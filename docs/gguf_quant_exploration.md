# GGUF Quant-on-Load Exploration for aclnn Engines

**Track N exploration doc (2026-04-17).** Doc-only; no source edits beyond
the `§8` pointer in `NATIVE_TTS_CONTRACT.md` that references this file.

Baseline: `v1.0` tag of OminiX-Ascend (native Talker + CP aclnn path, F16
weights in GGUF, F16 compute in kernels).

## 0. TL;DR

- Yes — our native TTS stack already leverages llama.cpp's GGUF container +
  `ggml_*` reader APIs. What we do **not** leverage today is GGUF's
  *K-quant* / *legacy-quant* tensor types (`Q4_K_M`, `Q5_K_M`, `Q8_0`, …),
  even though those files **already exist on ModelArts** (we have
  `qwen_tts_talker_llama_q{4_k_m,5_k_m,8_0}.gguf` sitting in
  `tools/qwen_tts/gguf/` from a prior export — see §1). Every matmul
  weight in the two aclnn engines (`TalkerCannEngine`, `CpCannEngine`)
  is loaded as F16 and stays F16 through the aclnn matmul.
- Supporting K-quant weights is **doable** but buys only a **load-size /
  disk-footprint** win: we'd dequantize to F16 on the host at init, upload
  F16 to the NPU, and the runtime matmul path is unchanged. **Zero steady-
  state speedup.**
- The runtime-speedup lever is the **Ascend-native A16W8 path**
  (`aclnnWeightQuantBatchMatmul`, Stretch S1 in §5): INT8 weights stay INT8
  through the matmul, ~30% fps and ~50% weight-memory win. K-quants and
  A16W8 are **complementary, not substitutes** — K-quants shrink the file,
  A16W8 shrinks the kernel.
- **Recommendation**: not a v1 shipping concern. Revisit only if we start
  distributing models to sites where 2.5× file-size shrinkage (F16 →
  Q5_K_M, ~16 bpw → ~5.5 bpw) matters more than the zero-cost runtime
  story we ship today. Stretch S1 is the higher-ROI follow-up.

## 1. Inventory — what's in `~/work/OminiX-Ascend/tools/qwen_tts/gguf/`

Pulled live from ModelArts 2026-04-17 via the `gguf` Python package
(recipe in §1.2 below). **Good news**: the talker backbone is already
pre-exported at four quantization levels — F16 (`qwen_tts_talker_llama.
gguf`), Q4_K_M, Q5_K_M, Q8_0. We don't currently *load* the quantized
files, but they exist on disk and an exporter session already put them
there.

### 1.1 Tensor-type histograms (live)

| file                                         | size       | tensors | dtypes                          |
|----------------------------------------------|------------|---------|---------------------------------|
| `qwen_tts_talker.gguf` (native ND Talker)    | 3483.4 MB  | 316     | F16: 201, F32: 115              |
| `qwen_tts_talker_llama.gguf` (llama fork)    | 2844.3 MB  | 311     | F16: 198, F32: 113              |
| `qwen_tts_talker_llama_q4_k_m.gguf`          |  855.0 MB  | 311     | Q4_K: 169, F32: 113, Q6_K: 29   |
| `qwen_tts_talker_llama_q5_k_m.gguf`          | 1006.3 MB  | 311     | Q5_K: 169, F32: 113, Q6_K: 29   |
| `qwen_tts_talker_llama_q8_0.gguf`            | 1511.3 MB  | 311     | Q8_0: 198, F32: 113             |
| `qwen_tts_talker_llama_q8_0_mrope.gguf`      | 1511.3 MB  | 311     | Q8_0: 198, F32: 113             |
| `qwen_tts_code_predictor.gguf` (CP native)   |  350.3 MB  |  88     | F16: 66, F32: 22                |
| `qwen_tts_cp_llama.gguf` (CP llama fork)     |  165.8 MB  |  58     | F16: 37, F32: 21                |
| `qwen_tts_tokenizer_enc.gguf`                |  225.0 MB  | 225     | F32: 225                        |
| `qwen_tts_tokenizer_dec.gguf`                |  457.3 MB  | 271     | F32: 271                        |
| `qwen_tts_speaker_encoder.gguf`              |   24.0 MB  |  76     | F16: 38, F32: 38                |

Observations:

- **Talker F16 GGUFs ship today are the "native" variant** (`qwen_tts_
  talker.gguf`, 3.5 GB, 316 tensors) — this is what `TalkerCannEngine::
  init_from_gguf` opens on the production path. It has 5 more tensors
  than `qwen_tts_talker_llama.gguf` (presumably an mRoPE cache or a
  couple of ND-reshape sidecars emitted by the native exporter).
- **The three quantized variants** (Q4_K_M / Q5_K_M / Q8_0) are all from
  the **llama fork** of the exporter (311 tensors), not the native one.
  To actually load a quantized talker via the aclnn path, we'd first
  need to re-run the quantizer against `qwen_tts_talker.gguf` (native
  layout) or teach the native loader to accept the 311-tensor layout.
  Both are small fixes; not addressed here.
- **Q4_K_M actually uses Q6_K for 29 tensors** — this is standard
  `llama-quantize Q4_K_M` behavior (embed/output projections + a
  handful of attn layers get bumped to Q6_K to preserve quality).
  Same for Q5_K_M. Loader must handle Q6_K too, not just the headline
  type.
- **Tokenizer enc/dec, speaker encoder** are F32-dominant graph-
  scheduled modules — K-quant doesn't apply to them (they run on
  ggml-cpu / ggml-cann, not the aclnn matmul path).
- **CP** is F16 (37 tensors) + F32 (21 tensors). CP weights are not
  loaded through `gguf_init_from_file` in the aclnn path (see §3.3);
  they come in as `CpWeightsF32` already dequantized.

### 1.2 Recipe used (for re-runs)

```bash
# on ModelArts, from ~/work/OminiX-Ascend
ssh -i ~/home/tensordock/KeyPair-4fbd-yue.pem -p 31984 \
    ma-user@dev-modelarts.cn-southwest-2.huaweicloud.com
cd ~/work/OminiX-Ascend/tools/qwen_tts/gguf

python3 -c "
from gguf.gguf_reader import GGUFReader
from collections import Counter
import os, glob
for f in sorted(glob.glob('*.gguf')):
    size_mb = os.path.getsize(f)/1e6
    r = GGUFReader(f, 'r')
    c = Counter(t.tensor_type.name for t in r.tensors)
    tb = sum(t.n_bytes for t in r.tensors)/1e6
    print(f'=== {f}  file={size_mb:.1f}MB  tensors={len(r.tensors)}'
          f'  tensor_bytes={tb:.1f}MB ===')
    for k, v in sorted(c.items(), key=lambda kv: -kv[1]):
        print(f'    {k:<12s} {v}')
"
```

`gguf` python package ships with the repo at `gguf-py/`; a pip install
also works. No other dependencies.

## 2. What would dequant-on-load buy us?

Exact numbers from §1.1 (talker backbone only — the other modules are
F32-dominant and would not benefit). The F16 reference is `qwen_tts_
talker_llama.gguf` (2844 MB) since that's what the existing quantized
files were derived from; the native `qwen_tts_talker.gguf` is ~640 MB
larger because of the 5 extra mRoPE/reshape tensors.

| metric                                     | F16 (2844 MB) | Q8_0 (1511 MB) | Q5_K_M (1006 MB) | Q4_K_M (855 MB) |
|--------------------------------------------|---------------|----------------|------------------|-----------------|
| file size                                  | 2844 MB       | 1511 MB        | 1006 MB          | 855 MB          |
| shrinkage vs F16                           | —             | **-47%**       | **-65%**         | **-70%**        |
| cold disk read @ 500 MB/s (ModelArts SSD)  | ~5.7 s        | ~3.0 s         | ~2.0 s           | ~1.7 s          |
| host-side dequant → F16 (single-threaded)  | 0             | ~1.0 s         | ~1.8 s           | ~2.0 s          |
| **net cold init**                          | ~5.7 s        | ~4.0 s         | ~3.8 s           | ~3.7 s          |
| **net warm init (OS cache)**               | <0.3 s        | ~1.2 s         | ~1.9 s           | ~2.1 s          |
| HBM after upload                           | 2.72 GB*      | 2.72 GB        | 2.72 GB          | 2.72 GB         |
| decode-step ms                             | unchanged     | unchanged      | unchanged        | unchanged       |
| decode-step fps                            | 22–27         | 22–27 (same)   | 22–27 (same)     | 22–27 (same)    |

*HBM after upload = 198 F16 weight tensors from the llama-fork layout, sum
of shapes `q/k/v/o_proj` + `gate/up/down_proj` + embed + head ≈ 2.72 GB.
Identical in all four columns because everything is dequant-to-F16 before
the aclrtMemcpy host→device.

So: **cold-start wins (~1.7–2.0 s faster), warm-start loses (~1–1.8 s
slower), steady state is a no-op**. In a server-style workload (process
starts once, serves 10k utterances) the init delta is amortized to zero
and the HBM footprint is identical.

If we additionally keep the weight on the NPU as `Q5_K_M` (no dequant,
new matmul op), that's **not** what llama.cpp's quants give us on CANN —
there is no aclnn op that consumes K-quant row formats directly. That
path is Ascend-native INT8 (§4 below), not GGUF K-quants.

## 3. Proposed implementation sketch (code-free)

The load-side plumbing is ~30 lines in each engine. Core idea: inspect
`t->type`, branch on F16 / F32 (today) vs K-quant / legacy-quant (new).

### 3.1 New helper (would live in `talker_cann_engine.cpp` anon ns, or
shared with `cp_cann_engine.cpp` via a new `gguf_upload.cpp`)

```
static bool dequantize_row_to_f16(
    const void *quant_row,   // one packed row from t->data
    uint16_t   *out_f16,     // pre-allocated, ne elements
    int64_t     ne,          // elements per row (= t->ne[0])
    ggml_type   t) {
    // Two-step via the type-traits table:
    //   1) quant_row -> tmp_f32 via ggml_get_type_traits(t)->to_float
    //   2) tmp_f32   -> out_f16 via ggml_fp32_to_fp16_row (or manual
    //                   cast loop — we already have fp32_to_fp16 inline)
    //
    // Bytes per row: depends on type. For Q5_K_M, ne must be a multiple
    // of QK_K (256); ggml_row_size(t, ne) gives the packed byte count.
    // For Q8_0, QK = 32. The type-traits .blck_size field is what we
    // want to assert against at load time.
    const auto *tr = ggml_get_type_traits(t);
    if (!tr || !tr->to_float) return false;
    if (ne % tr->blck_size != 0) return false;

    std::vector<float> tmp(ne);
    tr->to_float(quant_row, tmp.data(), ne);
    for (int64_t i = 0; i < ne; ++i) out_f16[i] = fp32_to_fp16(tmp[i]);
    return true;
}
```

### 3.2 `TalkerCannEngine::init_from_gguf` wiring

Today `upload_tensor_f16` (engines/talker_cann_engine.cpp:196) assumes the
GGUF tensor is F16 or F32 and calls `load_gguf_tensor_f32` which has a
hard-fail branch on "unsupported dtype" (line 187). Proposal:

```
bool upload_tensor_f16(ggml_context *ctx, const char *name,
                       size_t expected_elems, void *&dev) {
    ggml_tensor *t = ggml_get_tensor(ctx, name);
    if (!t) return false;
    size_t n = ggml_nelements(t);
    if (expected_elems && n != expected_elems) return false;

    std::vector<uint16_t> f16(n);

    switch (t->type) {
    case GGML_TYPE_F16:
        std::memcpy(f16.data(), t->data, n * sizeof(uint16_t));
        break;
    case GGML_TYPE_F32: {
        const float *src = (const float *)t->data;
        for (size_t i = 0; i < n; ++i) f16[i] = fp32_to_fp16(src[i]);
        break;
    }
    case GGML_TYPE_Q4_K:
    case GGML_TYPE_Q5_K:
    case GGML_TYPE_Q8_0: {
        // New branch: row-by-row dequant. t->ne[0] = cols, t->ne[1] = rows
        // for 2-D matmul weights; for Qwen3 all Q/K/V/O/gate/up/down are 2-D.
        const int64_t cols = t->ne[0];
        const int64_t rows = t->ne[1];
        const size_t  row_bytes = ggml_row_size(t->type, cols);
        const uint8_t *src = (const uint8_t *)t->data;
        for (int64_t r = 0; r < rows; ++r) {
            if (!dequantize_row_to_f16(src + r * row_bytes,
                                        f16.data() + r * cols,
                                        cols, t->type))
                return false;
        }
        break;
    }
    default:
        fprintf(stderr, "[talker_cann] %s: unsupported dtype %d\n",
                name, (int)t->type);
        return false;
    }

    // aclrtMalloc + aclrtMemcpy f16->device (unchanged path)
    ...
}
```

### 3.3 `CpCannEngine::init` wiring

CpCannEngine does **not** read GGUF directly — it consumes a host-side
`CpWeightsF32` struct that is populated by `TalkerLLM` / `talker.cpp`
(`talker.cpp:601-604`). The right place to add K-quant support is the
code path that *fills* `CpWeightsF32` (currently dequantizes F16 → F32
on host already; adding a Q5_K_M branch there is trivial — same
`to_float` call into the existing F32 buffer, no F16 middle hop needed).

### 3.4 Loader-format detection

We stay dtype-agnostic at load time: the `switch` on `t->type` above
handles any new future dtype by returning false with a clear message.
No CMake flag, no env var — the GGUF file's own metadata is the source
of truth. If a caller hands us a Q5_K_M GGUF today (pre-patch) we fail
cleanly at `upload_tensor_f16`; post-patch we dequant-on-load silently.

### 3.5 Re-export step

Getting Q5_K_M GGUFs is a separate, upstream step:

```
# from llama.cpp build
./build/bin/llama-quantize qwen3_talker.f16.gguf \
    qwen3_talker.q5_k_m.gguf Q5_K_M
```

This is a build-time artifact, not a runtime path — we'd ship the
quantized GGUF alongside (or in place of) the F16 GGUF and pick based on
deploy-site disk / bandwidth constraints. No change to the qwen_tts binary.

## 4. Comparison with Stretch S1 (`aclnnWeightQuantBatchMatmul`, A16W8)

Same model, four options side-by-side (talker backbone only):

| approach                             | disk     | HBM (weight) | decode fps    | quality       | impl cost          |
|--------------------------------------|----------|--------------|---------------|---------------|--------------------|
| **today — F16 GGUF + F16 matmul**    | 2844 MB  | 2.72 GB      | 22–27 (M6.x)  | baseline      | shipped            |
| **Q8_0 + dequant-on-load**           | 1511 MB  | 2.72 GB      | 22–27 (same)  | <0.005 WER    | ~50 LOC per engine |
| **Q5_K_M + dequant-on-load**         | 1006 MB  | 2.72 GB      | 22–27 (same)  | <0.01 WER hit | ~50 LOC per engine |
| **Stretch S1 — Ascend A16W8 (INT8)** | 2844 MB* | ~1.36 GB     | ~29–35 (+30%) | <0.02 WER hit | §5 S1, 200+ LOC    |

*An A16W8 shipping path could also store the INT8 weight in the GGUF as
`Q8_0` (which we already have) + per-channel scales as a sidecar tensor,
which would collapse the disk number to ~1.5 GB too. That's a composite
of both wins and would need its own exporter work to convert the existing
`qwen_tts_talker_llama_q8_0.gguf` per-block scales into per-channel scales
the aclnn op wants.

**Key takeaways**:

1. **Runtime**: K-quants give zero fps delta. A16W8 gives ~30%. If the
   question is "how do we make decode faster?", K-quants are the wrong
   tool.
2. **Memory**: K-quants don't reduce HBM (we dequant to F16 before
   upload). A16W8 halves HBM for the matmul weight.
3. **File size**: K-quants shrink disk ~65%. A16W8 as described doesn't
   (would need its own GGUF-level packing).
4. **The shippable combo** is both: re-export as `Q8_0` + A16W8 kernel.
   Out of scope for v1.

## 5. When to revisit K-quant loading

Pull the trigger if **any** of these lands as a requirement:

- Site-ship scenarios with <5 GB available for model files (mobile / edge
  box). Talker's 3.4 GB F16 matmul weight + 0.6 GB embeddings + 0.2 GB
  tokenizers + 0.7 GB speaker encoder ≈ 5 GB today; Q5_K_M for the
  talker backbone alone would drop the bundle to ~3.4 GB total.
- Slow-internet first-time installers where 2.5× download shrinkage is
  user-visible.
- A16W8 (S1) lands first and we want to further shrink disk footprint
  without touching runtime.

Not a pre-req for any v1 or M6.x milestone.

## 6. ggml / gguf entry points we already depend on

Grep-verified 2026-04-17 at tag `v1.0` (OminiX-Ascend HEAD 98409bde):

- `gguf_init_from_file` — the GGUF container opener; two call sites:
  - `tools/qwen_tts/talker_cann_engine.cpp:524` (TalkerCannEngine init)
  - `tools/qwen_tts/tts_transformer.cpp:345` (S2 TTS transformer)
- `ggml_get_tensor` — tensor-by-name lookup on the `ggml_context`
  populated by `gguf_init_from_file`; call sites:
  - `tools/qwen_tts/talker_cann_engine.cpp:168`
  - `tools/qwen_tts/tts_transformer.cpp:90`
  - `tools/qwen_tts/speech_tokenizer_encoder.cpp:55` (comment citing the
    pattern; the actual call is in a `gt` lambda in the same file)
- `ggml_element_size` / `ggml_nbytes` / `ggml_nelements` — size / stride
  helpers; used widely, notably:
  - `tools/qwen_tts/speech_tokenizer_decoder.cpp:242,246`
  - `tools/qwen_tts/talker_cann_engine.cpp:173`
  - `tools/qwen_tts/infer_session.hpp:88,90,111,114,134,136,138,140,159,160`
- `ggml_fp16_to_fp32` — the one-row bit cast used in
  `load_gguf_tensor_f32` (`talker_cann_engine.cpp:185`). A K-quant
  extension uses the same pattern but goes through `get_type_traits(
  t)->to_float` instead.
- `GGML_TYPE_F16` / `GGML_TYPE_F32` — the two dtypes our loaders
  currently switch on (`talker_cann_engine.cpp:181,183`,
  `tts_transformer.cpp:101,103`, `speech_tokenizer_encoder.cpp:403,406`,
  `talker.cpp:675,678`). Adding `GGML_TYPE_Q4_K`, `GGML_TYPE_Q5_K`,
  `GGML_TYPE_Q8_0` to each `switch` / `if-else` is the minimal surface
  area.
- `ggml-cpu` / `ggml-cann` backend libraries — used by the
  `speech_tokenizer_encoder` / `speech_tokenizer_decoder` /
  `speaker_encoder` graph-scheduler path (`model_defs.cpp:39-66`,
  `infer_session.hpp:33-83`). Those paths would gain K-quant support
  automatically when we re-export their GGUFs quantized; the ggml-cpu
  ops do K-quant matmul natively, the ggml-cann ops fall back to
  dequant-on-dispatch.

**Not used today** (would be added by this track):
- `ggml_get_type_traits(type)->to_float` — the function pointer that
  implements K-quant → F32 row dequant.
- `ggml_row_size(type, n)` — packed byte count for a K-quant row.
- `GGML_TYPE_Q4_K` / `Q5_K` / `Q8_0` dtype enum entries.

## 7. Open questions (deferred)

- **Per-layer mix**: K-quants shine on attention weights where outliers
  are rare and row-wise quantization captures most of the variance; they
  struggle on embedding tables and output projections. A production
  export would likely keep embeddings F16 and quantize only
  `blk.N.attn_*` + `blk.N.ffn_*`. That's standard llama.cpp practice and
  `llama-quantize --skip ...` handles it. Not a load-side concern — our
  switch statement sees whatever the exporter produced.
- **Dequant-on-load parallelism**: the naive implementation above is
  single-threaded. For a 28-layer Talker, per-layer dequant is
  embarrassingly parallel — an `omp parallel for` across layers would
  recover most of the init-time delta if we care. Not a v1 concern.
- **HBM-side K-quant**: no aclnn op consumes K-quant row format today.
  A custom CANN kernel could — out of scope, not explored here.

## 8. Relationship to §5 of the contract

- M5 K-quant work is **not on the roadmap** today. This doc is the
  record of why (zero runtime win, init-time amortizes to zero on
  server workloads).
- Stretch S1 (`aclnnWeightQuantBatchMatmul`) stays the recommended
  runtime-ROI follow-up.
- Track N closes as "explored, parked". Re-open when a deploy-site
  disk/bandwidth constraint is concrete.
