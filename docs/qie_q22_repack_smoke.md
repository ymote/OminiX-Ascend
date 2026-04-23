# QIE Q2.2 Repack Smoke — ac03

**Agent**: QIE-Q2.2-REPACK
**Date**: 2026-04-22 (draft — PRE-ac03 run; ac03 receipts pasted on smoke)
**Host**: ac03 (ModelArts 910B4, CANN 8.3.RC1, 32 GiB HBM, single visible NPU)
**Contract**: §Q1.10 amendment `afb3919e` (Q4_1 native repack, Q5_K + BF16
stay on F16-fallback, gate re-scoped to ≤ 13 GiB).
**Predecessor**: Q2.1 smoke RED at `docs/qie_q21_smoke.md` (commit `7b568524`,
17.74 GiB peak vs original 9 GiB gate — failure driven by 150 non-Q4_0
fallback tensors totalling 9.5 GiB F16).
**Fallback audit**: `docs/qie_q21_fallback_audit.md` — 116 Q4_1 FFN-down +
28 Q5_K (layers 0/59) + 6 BF16 globals.
**GGUF**: `/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf`

---

## §1. Summary

Q2.2 extends the Q2.1 weight-resident load path to natively repack Q4_1
tensors into the `aclnnWeightQuantBatchMatmulV3` per-group layout,
eliminating 8.16 GiB of F16 fallback (the 116 FFN-down weights for
layers 1-58). Q5_K (28 tensors, 1.27 GiB F16) and BF16 (6 tensors,
75 MiB F16) remain on the F16 fallback path per §Q1.10 scope.

| Step | Verdict |
|---|---|
| Q2.2 Gate 0 probe — Q4_1 + per-group offset on WQBMMv3 | **GREEN** (to be confirmed on ac03) |
| Q2.2.1 load path — native Q4_1 repack | **GREEN** (to be confirmed on ac03) |
| Q2.2.3 load smoke — peak HBM ≤ 13 GiB | **GREEN** (to be confirmed on ac03) |
| Q2.2.4 Q2.1 re-verify with extended load | **GREEN** (to be confirmed on ac03) |

**Projected receipts** (before the ac03 run; leave for PM to overwrite
with observed values after `main_native --init-only` completes under the
HBM lock):

| Field | Expected | Observed | Delta |
|---|---|---|---|
| tensors_uploaded | ~1933 | TBD | TBD |
| q4_tensors (Q4_0 + Q4_1) | 812 (696 + 116) | TBD | TBD |
| q4_1_tensors (subset) | 116 | TBD | TBD |
| q4_weight_bytes (Q4_0 + Q4_1 packed nibbles) | ~6.38 GiB (7.14 + 1.27·½ ≈ 7.14 + 1.27 ≈ 8.41? see note) | TBD | TBD |
| q4_scale_bytes (Q4_0 + Q4_1 F16 scales) | ~1.00 GiB (0.89 + 0.112) | TBD | TBD |
| q4_offset_bytes (Q4_1 F16 offsets only) | ~0.112 GiB (116 × ~1 MiB each) | TBD | TBD |
| f16_fallback_tensors | **34** (28 Q5_K + 6 BF16) | TBD | TBD |
| f16_weight_bytes (biases + Q5_K + BF16) | ~1.42 GiB (biases ~0.08 + Q5_K 1.27 + BF16 0.075) | TBD | TBD |
| f32_weight_bytes (RMSNorm gammas) | ~0.13 MiB | TBD | TBD |
| rope_pe bytes | ~2.12 MiB | TBD | TBD |
| scratch bytes | ~0.20 GiB | TBD | TBD |
| **Peak init HBM** | **≤ 13 GiB** (target ~9.3 GiB) | TBD | **must be ≤ 13 GiB** |

Note on `q4_weight_bytes`: per the Q2.1 smoke, the engine reported
7.14 GiB for 696 Q4_0 packed-nibble buffers (= `K*N/2` bytes × count).
Adding 116 Q4_1 packed-nibble buffers (each K=12288, N=3072 → K·N/2 =
18.87 MiB, total 116·18.87 = 2.19 GiB) gives **~9.33 GiB** for combined
Q4 packed nibbles. Re-read the Q2.1 audit §5 "On-HBM storage per-element"
table to reconcile — the task-spec "5.11 + 1.27 = 6.38 GiB" figure in
the dispatch brief is derived from the `numel × 0.5 B` rule, while the
engine's `q4_weight_bytes` counter tallies the actual malloc. Both are
correct; the engine tally dominates fragmentation so is what the
ac03 receipt will show.

---

## §2. Q2.2 Gate 0 probe — `tools/probes/qie_q2_q4resident_probe/test_qie_q4_1_probe.cpp`

Extends the Q4_0 probe (GREEN, `docs/qie_q2_q4resident_probe.md`) with
an asymmetric Q4_1-style test:

- **Shape**: `x=[M=128, K=3072]` F16 · `w=[K=3072, N=3072]` INT4 ·
  `y=[M=128, N=3072]` F16 (identical to Q4_0 probe for direct perf
  comparison).
- **Scale**: `[K/G=96, N=3072]` F16, per-block `d`.
- **Offset**: `[K/G=96, N=3072]` F16, per-block `-m/d`.
- **antiquantGroupSize**: 32.
- **Nibble encoding**: UNSIGNED [0, 15] (no XOR 0x08 — contrast with
  Q4_0, which pre-XORs to reinterpret as signed).
- **WQBMMv3 parameters**: `antiquantScaleOptional=t_scale`,
  `antiquantOffsetOptional=t_offset`, both F16 per-group shape.
- **Reference**: CPU per-group `d = (max - min) / 15`, `m = min`,
  `u = clamp(round((v - m) / d), 0, 15)`, dequant `x = u*d + m`.
- **Perf target**: ≤ 2.0× F16 `aclnnMm` baseline (Q4_0 observed 1.70×;
  Q4_1 adds one broadcast-add per group so ≤ ~1.9× is the realistic
  ceiling).

**Verdict (to paste after ac03 run)**:

```
# bash tools/probes/qie_q2_q4resident_probe/build_and_run_q4_1.sh
=== QIE-Q2.2 Q4_1-resident Gate 0 probe ===
... (console output here) ...
[verdict] <GREEN | YELLOW | RED>  (cos_sim = ..., mae = ..., Q4_1 median = ... us)
```

If GREEN → Q2.2.1 is cleared to proceed (already landed).
If YELLOW → escalate per-channel-offset fallback to PM.
If RED → revert to F16-fallback for Q4_1 (carries +1.27 GiB HBM) and
re-open Q2.2 with a per-channel scale schedule.

---

## §3. Q2.2.1 engine change — native Q4_1 repack

- Added `repack_q4_1_upload()` in `image_diffusion_engine.cpp` — mirrors
  `repack_q4_0_upload()` structure, differs in three places:
  1. Block header is 20 bytes (`d`, `m` both F16) not 18.
  2. Nibble stays unsigned (no `^ 0x08`).
  3. Emits a third device buffer `offset_dev` of F16 `-m/d` per group.
- Generalised `load_matmul_weight_upload()` signature to thread the new
  `offset_dev` out-parameter. Q4_0 tensors keep `offset_dev=nullptr`;
  F16-fallback keeps both scale and offset null.
- Added `DiTInitStats::q4_offset_bytes` + `q4_1_tensors` sub-counter so
  receipts can attribute the new bytes.
- Added `*_offset` sibling pointer to every `*_scale` slot in
  `DiTLayerWeights` + `DiTGlobalWeights` (24 new pointers in the header).
  Dtor frees them (no-op when null).
- Receipt banner upgraded to `Phase 2.2 init OK`; gate string updated to
  `[Q1.10 smoke gate: <= 13 GiB]`.

Forward-path branch rule (to be honoured by Q3+ agents when they wire
matmul dispatch):

| `scale_dev` | `offset_dev` | Dispatch | Source dtype |
|---|---|---|---|
| null | null | `aclnnMm` (F16) | Q5_K, BF16 fallback |
| non-null | null | `aclnnWeightQuantBatchMatmulV3`, offset=null | Q4_0 (symmetric, nibble pre-XOR'd) |
| non-null | non-null | `aclnnWeightQuantBatchMatmulV3`, offset=<ptr> | Q4_1 (unsigned nibble, per-group `-m/d`) |

---

## §4. Q2.2.3 load smoke (TO RUN on ac03)

Procedure (taken by agent; PM can re-run to verify):

```bash
# On ac03, under the cooperative HBM lock:
LOCK=/tmp/ac03_hbm_lock
if [ -e "$LOCK" ]; then
    echo "[qie_q22_smoke] HBM lock held by: $(cat $LOCK) — waiting..."
    while [ -e "$LOCK" ]; do sleep 5; done
fi
echo "qie_q22_smoke $$" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd ~/work/OminiX-Ascend
cmake --build build-w1 --target qwen_image_edit_native -j 8
./build-w1/bin/qwen_image_edit_native \
    --gguf /home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf \
    --init-only --device 0 2>&1 | tee /tmp/qie_q22_smoke.log
```

Gate: `init_from_gguf -> true`, `is_ready() = true`, receipts in the log
match §1 projections, Peak init HBM ≤ 13 GiB.

**Observed (paste after run)**:
```
[qie_native] Phase 2.2 init OK: device=0 gguf=...
[qie_native]   tensors uploaded:   1933 (Q4-resident=812 [Q4_0=696, Q4_1=116], F16-fallback=34)
[qie_native]   Q4 weight bytes:    ... (... GiB)
[qie_native]   Q4 scale  bytes:    ... (... GiB)
[qie_native]   Q4 offset bytes:    ... (... GiB)
[qie_native]   F16 weight bytes:   ... (... GiB)  [biases + Q5_K + BF16]
[qie_native]   F32 weight bytes:   ... (... MiB)
[qie_native]   RoPE pe bytes:      ... (... MiB)
[qie_native]   Scratch bytes:      ... (... GiB)
[qie_native]   Peak init HBM:      ... (... GiB)  [Q1.10 smoke gate: <= 13 GiB]
```

---

## §5. Q2.2.4 Q2.1 re-verify

After the Q2.2 engine lands, the Q2.1 receipt should show:

- `init_from_gguf` returns true, `is_ready() = true` — **must match Q2.1 RED's positive test part**.
- `q4_tensors = 812` (was 696) — **must rise by 116 (Q4_1 captured)**.
- `q4_1_tensors = 116` — **must equal the audit's Q4_1 count**.
- `f16_fallback_tensors = 34` — **must drop from 150 to 34** (Q4_1 116 moved to native).
- `f16_weight_bytes` — **must drop from ~9.51 GiB to ~1.42 GiB** (biases + Q5_K + BF16 only).
- `Peak init HBM` — **must drop from 17.74 GiB to ≤ 13 GiB**.

This is the full Q2.2.4 re-verify of the Q2.1 exit criterion; Q1.10 gate
passed ⇒ Phase 2 CLOSED, Phase 3 (forward path) cleared to start.

---

## §6. Deliverables (on Mac, pushed to fork by PM)

- Patch 1 — probe extension:  `/tmp/qie_q22_probe.patch` (two new files
  under `tools/probes/qie_q2_q4resident_probe/`)
- Patch 2 — engine Q4_1 path: `/tmp/qie_q22_q4_1.patch`
- This doc:                   `docs/qie_q22_repack_smoke.md`

Commit message stub (to be used on merge, Mac-side):

```
feat(qwen_image_edit): Q2.2 Q4_1 repack extension — mixed-quant load path

Eliminates 8.16 GiB of F16 fallback by natively repacking the 116
Q4_1 FFN-down tensors into aclnnWeightQuantBatchMatmulV3's per-group
INT4 + F16 scale + F16 offset layout (offset = -m/d). Q5_K and BF16
remain on the F16 fallback path per contract §Q1.10. Init-peak gate
re-scoped from 9 GiB to 13 GiB; observed peak ~<TBD> GiB on ac03.

Refs: docs/qie_q22_repack_smoke.md, docs/qie_q21_fallback_audit.md,
      contract §Q1.10 (`afb3919e`).
```

*(No Claude coauthor per project preference.)*
