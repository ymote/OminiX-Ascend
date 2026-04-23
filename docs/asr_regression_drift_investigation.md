# ASR Tier-1 drift investigation — mabaoguo / shenyi / zhoujielun

**Author**: Agent ASR-DRIFT-INVESTIGATE
**Date**: 2026-04-22
**Host**: ac03 (port 30412), 910B4, CANN 8.3.RC1 — same host that produced A1a
baseline, A4 gate, A4b gate, and A4c sweep.
**Scope origin**: `docs/asr_a4c_gate.md` §Phase 1 Results flagged three clips
drift across Config A (A4b-equivalent), B (A4c GMM), AND C (F16 unset) —
CER 0.023–0.038. Parity reference
`docs/asr_a1a_data/tier1_q8_cann_clean.json` has CER=0 on all 13 clips.
Hypothesised as "environmental drift on ac03 vs the A1a environment" in
the A4c gate doc; this investigation reproduces + isolates the actual
cause.

---

## Verdict

**Harness regression, not code regression, not environment drift.**

The A4c sweep harness `/tmp/run_a4c_native.py` on ac03 omits the
`--mel_filters <mel_filters_whisper.npy>` flag when invoking
`qwen_asr_native`. Without it, `AsrTextDecoderCannEngine` falls back to
its C++-coded default mel filterbank (`MelSpectrogram::init_mel_filterbank`),
which uses **HTK mel-scale** — numerically different from the
**Slaney mel-scale** filterbank that whisper and the A1a / A4 / A4b
gates all loaded from `tools/qwen_asr/verify_data/mel_filters_whisper.npy`.
The two filterbanks are close but not equal; on three edge-case clips
(mabaoguo / shenyi / zhoujielun) the mel-feature delta is large enough
to flip greedy-argmax at one token, producing a single-character
substitution or comma insert/delete that registers as CER 2.4–3.8%.

Fix is one line per harness: add `--mel_filters $REPO/tools/qwen_asr/verify_data/mel_filters_whisper.npy`
to the `qwen_asr_native` command line (or ship the filterbank inside
the binary as an embedded default).

No code patch to the engine is required. No env lockdown is required.
**The parity reference `tier1_q8_cann_clean.json` remains valid.**

---

## Phase 1 — Reproduce on ac03, Q8_0 decoder path

Rebuilt `qwen_asr_native` on ac03 (`build-w1/bin/qwen_asr_native`, source
HEAD `0f860c5` + uncommitted gguf-symlink fix-up; ASR/Talker code is
byte-identical to A4b `ed67c36` — `git log ed67c36..HEAD -- tools/qwen_asr/
tools/qwen_tts/talker_cann_engine.{cpp,h} tools/qwen_tts/talker.h` is empty).

Six-cell matrix on the three drift clips (same decoder GGUF, same audios,
same tokenizer):

| Probe | `--mel_filters` | `--max_tokens` | Env | mabaoguo | shenyi | zhoujielun |
|---|---|---|---|---|---|---|
| P1 (A4b replay) | whisper.npy | 128 | — | **MATCH** (20 tok `当咱...`) | **MATCH** (30 tok, comma after 看来) | **MATCH** (19 tok, comma after 其实) |
| P2 (A4c replay) | — | 256 | — | DRIFT (19 tok `当他...`) | DRIFT (30 tok, missing comma) | DRIFT (18 tok, missing comma) |
| P3 mel_off + short | — | 128 | — | DRIFT | DRIFT | DRIFT |
| P4 mel_off + CP envs | — | 256 | `TALKER_CP_ACLGRAPH=1 TALKER_CP_INPLACE_ADDRMSNORM=1 TALKER_CP_POS_BATCH=1` | DRIFT | DRIFT | DRIFT |
| P5 mel_on + CP envs | whisper.npy | 256 | same | **MATCH** | **MATCH** | **MATCH** |
| P6 A4b replay + W8 | whisper.npy | 128 | `TALKER_W8_QUANT=1` | **MATCH** | **MATCH** | **MATCH** |

Every "MATCH" row produces byte-identical `hyp_raw` to A1a
`docs/asr_a1a_data/tier1_q8_cann_clean.json`. Every "DRIFT" row reproduces
the A4c sweep's delta exactly (including matching token counts 19 / 30 / 18).

The single variable that flips the verdict is `--mel_filters`. `--max_tokens`
(128 vs 256) is a no-op (no clip reaches 128 generated tokens).
`TALKER_CP_*` envs and `TALKER_W8_QUANT` do not affect CER at all; they only
affect path latency (`TALKER_CP_*` envs gate code in `CpCannEngine`, which
is not used by the ASR native path — `AsrTextDecoderCannEngine` composes
`TalkerCannEngine` only).

Probe artifact: `/tmp/asr_drift_probe3.sh` + `/tmp/asr_drift_probe3_out.txt`
on ac03.

---

## Phase 2 — Why the A4 / A4b runs said CER=0

Direct inspection of the stale artifacts that survived in `/tmp/` on ac03:

- `/tmp/a4b_tier1_runner.py` line 40 passes `--mel_filters $REPO/tools/qwen_asr/verify_data/mel_filters_whisper.npy`.
  Runs at 2026-04-23 02:43, `/tmp/a4b_tier1_results.json` — CER=0 on all 3
  drift clips, with `hyp` fields byte-identical to A1a reference.

- `/tmp/a4_tier1_results.json` (2026-04-23 01:46) — same runner pattern,
  same CER=0 on all 3 drift clips, same byte-identical hyps.

- `/tmp/run_a4c_native.py` (A4c harness) — accepts no `--mel-filters`
  argparse flag and therefore never passes it to the binary.
  `/tmp/a4c_sweep_v2.sh` builds the command without the flag.
  Runs at 07:02–07:55 same day, `/tmp/a4c_tier1_{A,B,C}.json` — CER>0 on
  the 3 drift clips, byte-identical drifted hyps to our Phase 1 P2/P3/P4.

The three CER=0 claims in the commit messages of `525d8a1e` (A4),
`ed67c36` (A4b) and `0dd55ca` (A1b-v2) are genuine — they were all
computed by runners that passed `--mel_filters`. The drift appeared in
the A4c sweep because the A4c harness is a re-write
(`/tmp/run_a4c_native.py`) that dropped the flag and was never
reconciled against `/tmp/a4b_tier1_runner.py`.

The A4c gate doc's hypothesis that "likely decoder GGUF or CANN toolkit
version skew on ac03 vs the environment that produced the parity
reference" is wrong. The three drift runs and the three CER=0 runs are
on the same ac03 host, same `Version=8.3.RC1` CANN install, same
`/home/ma-user/work/asr_gguf/qwen_asr_decoder_q8_0.gguf`
(sha256 `3c60a3f4e0343551436d5866495f2a9c2d8159dac076ea625cc8c7089cbe890d`),
same `qwen_asr_audio_encoder.gguf`
(sha256 `d6a14cc7c12eb7652f3f35e02a41de2d8eff23873cd0cdace2738528079a5124`),
same audios (sha256 confirmed identical across the two source trees),
same `vocab.json` / `merges.txt`, ≤5 hours apart. **The only delta is
whether `--mel_filters` was passed.**

---

## Phase 3 — Mechanism

`tools/qwen_asr/mel_spectrogram.cpp` has two code paths:

1. **Default** (`init_mel_filterbank`, constructor-time):
   HTK mel-scale (`hz_to_mel_htk`) + Slaney triangle normalization.
   ```
   mel_pts[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1)  // HTK-spaced
   enorm      = 2.0 / (right - left)                              // Slaney norm
   ```

2. **Override** (`load_mel_filterbank(path)`, loaded when
   `--mel_filters` is set):
   Reads `(201, 128)` float32 array from the whisper-style npy file
   `tools/qwen_asr/verify_data/mel_filters_whisper.npy`. That file
   was generated (at some earlier A1a-era commit) from
   `scipy`'s Slaney mel-scale with matching Slaney triangle
   normalization — i.e. *both* mel-spacing and normalization are
   Slaney, not HTK+Slaney.

The default HTK-spaced bank places the triangular filter centers at
different frequencies than the Slaney-spaced bank, so the same FFT
power spectrum projects to slightly different 128-dim mel vectors.
On 10 out of 13 clips this difference is below the argmax-flip
threshold of every generated token (clean, well-enunciated speech
has wide logit margins). On mabaoguo / shenyi / zhoujielun the
margin is tight enough at one or two positions for the mel delta to
push the argmax across the boundary, substituting a phonetically
plausible neighbor (`咱 ↔ 他`, comma insert/delete).

The decoder weights, NPU kernels, lm_head port, tokenizer, and greedy
decode are all identical between the two paths. The mel-feature delta
is the **sole upstream cause**.

---

## Fix recommendation

**Harness patch** (minimal):

1. Add `--mel-filters` (argparse) to `/tmp/run_a4c_native.py` or to the
   future blessed ASR harness. Default it to
   `$REPO/tools/qwen_asr/verify_data/mel_filters_whisper.npy`.
2. Pass the resolved path through to the binary's `--mel_filters` flag.

Optional belt-and-braces follow-up (engine-side; out of scope here but
worth flagging):

- Either ship the Slaney filterbank as a compile-time embedded
  constant inside `MelSpectrogram` (remove the npy dependency entirely,
  making the binary self-contained), OR
- Make `--mel_filters` required and error out when it's absent, so
  future harness authors can't silently fall back to the HTK default.

Neither the decoder GGUF nor the parity reference needs updating.

No TTS regression risk — this is the ASR-only mel spectrogram front-end.
Talker / CP paths don't call into `MelSpectrogram` at all.

---

## Amend recommendation for A4c gate doc

The "orthogonal finding" paragraph in `docs/asr_a4c_gate.md` (lines
124–131) should be updated from "likely decoder GGUF or CANN toolkit
version skew on ac03" to "harness regression: A4c sweep dropped
`--mel_filters` flag, root-caused in
`docs/asr_regression_drift_investigation.md`". The "pre-existing and
independent of A4c" framing is correct with respect to code, but the
drift only *surfaces* in the A4c sweep because the A4c harness was
the first to drop the flag.

---

## Artifacts

- Phase 1 matrix probe: `ac03:/tmp/asr_drift_probe3.sh`,
  `ac03:/tmp/asr_drift_probe3_out.txt` (18 runs, 6 configs × 3 clips)
- Surviving A4 / A4b reference runs (CER=0): `ac03:/tmp/a4_tier1_results.json`,
  `ac03:/tmp/a4b_tier1_results.json`, plus their `.log` siblings
- A4c sweep artifacts (CER>0): `ac03:/tmp/a4c_tier1_{A,B,C}.{json,log}`
- A4b runner (with flag): `ac03:/tmp/a4b_tier1_runner.py`
- A4c runner (without flag): `ac03:/tmp/run_a4c_native.py`
- A4c sweep script: `ac03:/tmp/a4c_sweep_v2.sh`

---

## Related

- A1a reference: `docs/asr_a1a_data/tier1_q8_cann_clean.json`
- A4c gate: `docs/asr_a4c_gate.md` (Phase 1 Results table + "orthogonal
  finding" paragraph needs the amend above)
- ASR learnings: `docs/asr_optimization_learnings.md`
- Relevant source: `tools/qwen_asr/mel_spectrogram.cpp`
  (`init_mel_filterbank` vs `load_mel_filterbank`),
  `tools/qwen_asr/native/main_native.cpp` (`--mel_filters` arg plumbing),
  `tools/qwen_asr/native/asr_text_decoder_engine.cpp` (init() line
  241–248: "mel filter load failed, using default filterbank" fallback)
