# QIE 20-Task Canonical Eye-Gate Suite

**Agent**: QIE-Q0.5
**Date**: 2026-04-22
**Status**: FROZEN. Used unchanged from Q2.5 onward through Q8 final gate. Deviating mid-contract poisons comparability (per `qie_optimization_learnings.md` §7 New-for-QIE item 8).
**Purpose**: reusable eye-gate benchmark for QIE-2511 optimization. 10 edit categories × 2 languages (CN + EN) = 20 tasks. Reused across Q8 + future Qwen-Image-Edit / SD3 / Flux-Kontext / Z-Image-Edit contracts.

## Suite protocol summary

- **Source image**: each task has a **reference image description + canonical source URL OR local fixture path**. If no local fixture exists at Q1, the image is fetched once at first-landing, hashed with sha256, committed to `tools/ominix_diffusion/testdata/qie_eye_gate/` at Q2 landing, and locked thereafter. Size is pre-normalized to 512×512 for Q2.5-Q7 runs; Q8 adds a 1024×1024 variant.
- **Prompt**: locked wording. No paraphrasing between milestones. CN prompts are **not translations** of EN prompts — each CN task has an independent, language-idiomatic formulation chosen by a native speaker at Q1.
- **Seed**: fixed scheduler seed per task (`seed = <task_id_hash> % 2^31`). Recorded in the baseline output metadata.
- **Baseline output**: captured **once** at Q2 (first native-engine landing, byte-parity vs ggml-cann backend baseline). Stored at `tools/ominix_diffusion/testdata/qie_eye_gate/baseline_q2/<task_id>.png`. Never overwritten. Each subsequent milestone compares to this baseline, not to the prior milestone.
- **Pass/fail criterion**: per-task binary verdict by human reviewer.
  - **PASS**: edit goal achieved (the verbal "expected outcome" is visibly satisfied) AND no visible artifacts introduced (no faces distorted, no obvious color banding, no text corruption unrelated to the requested edit).
  - **FAIL**: edit goal missed, OR visible quality regression vs the Q2 baseline.
- **Rollup**: pass count out of 20 per milestone. Historical: ggml-cann baseline = TBD (currently RED). Native Q2 target = **20/20**. Every subsequent milestone target = **20/20** (no regression tolerance).
- **Regression trigger**: **any ≥ 1 task regression** vs the Q2 baseline trips a HARD-KILL rollback on the milestone's lever. Per `qie_optimization_learnings.md` §7 item 13: eye-gate is binary on edit tasks; no "maybe the next run will be better" slow-death.
- **Reviewer budget**: 2 human-hours per milestone per reviewer, 2 reviewers for consensus on ambiguous calls. Budgeted in contract §M8 gates.
- **Run cadence**: mandatory at Q2 (baseline capture), Q2.5 (CacheDIT tuning), Q3 (FIAv2 wiring), Q4 (CFG batching), Q5 (grouped matmul), Q7 (VAE fusion), Q8 (final gate). Optional at Q6 (weight-upload wiring — no numerical change expected).

## Baseline output storage

```
tools/ominix_diffusion/testdata/qie_eye_gate/
    sources/
        cat_studio.png              # sha256-locked, fetched once
        city_street.png
        ... (see source-image provenance section below)
    baseline_q2/
        qie_01_color_cn.png         # captured at Q2 milestone, never modified
        qie_01_color_en.png
        ... 20 files
    <milestone_tag>/                # e.g. baseline_q2.5, baseline_q4, baseline_q8
        qie_01_color_cn.png
        ... per-milestone captures for diff review
    METADATA.json                   # seed, prompt, source_hash, step_count, resolution per task
```

If no local fixture exists at Q1 time, source images are described + URL'd below and the Q2 landing agent captures them with sha256 verification. Any URL rot is caught there, not at Q5.

## The 20 tasks

Task IDs are `qie_<NN>_<cat>_<lang>` where `cat` is one of {color, style, addobj, remobj, text, bg, weather, time, pose, comp} and `lang` is {cn, en}.

### Category 1 — Color change

| ID | Src image | Prompt (locked) | Expected outcome | Difficulty |
|---|---|---|---|---|
| qie_01_color_cn | cat_studio.png — public-domain photo of a ginger cat in studio lighting, plain background (described; URL locked at Q1) | `把猫的毛色改成黑白相间` | Cat's fur is now black-and-white (tuxedo pattern or similar); face structure, pose, lighting unchanged | easy |
| qie_02_color_en | car_red.png — public-domain photo of a red sedan on a plain road | `Change the car color to metallic blue` | Car is now metallic blue with preserved body contour / shadow; road and background unchanged | easy |

### Category 2 — Style transfer

| ID | Src image | Prompt (locked) | Expected outcome | Difficulty |
|---|---|---|---|---|
| qie_03_style_cn | landscape_mountain.png — public-domain photo of a snowy mountain range | `转换成宋代山水画风格` | Output looks like a Song-dynasty ink landscape (monochrome, ink-wash aesthetic, traditional composition) while keeping mountain shapes | medium |
| qie_04_style_en | portrait_woman.png — public-domain headshot, neutral expression | `Convert to Van Gogh oil painting style with thick brushstrokes` | Output has visible impasto / swirl brushwork in Van Gogh style; face identity roughly preserved | medium |

### Category 3 — Object addition

| ID | Src image | Prompt (locked) | Expected outcome | Difficulty |
|---|---|---|---|---|
| qie_05_addobj_cn | desk_empty.png — wooden desk with a laptop, no other objects | `在桌上加一杯热咖啡` | A coffee cup (ideally with visible steam) appears on the desk at a plausible position; shadow + perspective consistent; laptop unchanged | medium |
| qie_06_addobj_en | beach_plain.png — empty sandy beach with ocean horizon | `Add a beach umbrella and two folding chairs in the foreground` | Umbrella + 2 chairs added in the foreground at a plausible position; sand texture / horizon line unchanged | medium |

### Category 4 — Object removal

| ID | Src image | Prompt (locked) | Expected outcome | Difficulty |
|---|---|---|---|---|
| qie_07_remobj_cn | street_scene.png — city street with one parked car and a few pedestrians | `移除画面中央的那辆汽车` | The central parked car is removed; replaced by plausible street surface; pedestrians + storefront signs unchanged | hard |
| qie_08_remobj_en | portrait_glasses.png — person wearing glasses, plain background | `Remove the glasses from the person's face` | Glasses removed; eye region plausibly reconstructed; hair + skin tone + expression unchanged | hard |

### Category 5 — Text add / edit

| ID | Src image | Prompt (locked) | Expected outcome | Difficulty |
|---|---|---|---|---|
| qie_09_text_cn | storefront_sign_blank.png — a storefront with a blank sign area above the door | `在招牌上加上"春风茶馆"四个字` | The four Chinese characters "春风茶馆" appear on the sign in a plausible traditional signage style; characters legible and correctly stroked | hard |
| qie_10_text_en | tshirt_plain.png — plain white t-shirt on a hanger | `Print the text "HELLO WORLD" across the chest in bold black letters` | "HELLO WORLD" appears in bold black on the chest; letters legible, correctly spelled, correctly sized | hard |

### Category 6 — Background swap

| ID | Src image | Prompt (locked) | Expected outcome | Difficulty |
|---|---|---|---|---|
| qie_11_bg_cn | portrait_park.png — headshot with a park background (trees, bench) | `把背景换成故宫红墙金瓦` | Background becomes Forbidden-City-style red walls + golden roof tiles; subject's face, hair, clothing unchanged; lighting plausibly re-matched | medium |
| qie_12_bg_en | product_plain.png — a ceramic mug on plain white backdrop | `Replace the background with a cozy wooden kitchen counter with soft morning light` | Mug preserved exactly; background swapped to wooden counter + warm light; cast shadow on counter plausible | medium |

### Category 7 — Weather change

| ID | Src image | Prompt (locked) | Expected outcome | Difficulty |
|---|---|---|---|---|
| qie_13_weather_cn | country_road.png — rural road on a sunny clear day | `改成暴雨天气` | Rain streaks visible; wet road reflections; overcast sky; scene geometry unchanged | medium |
| qie_14_weather_en | cityscape_clear.png — city skyline, clear day | `Change the weather to heavy snowfall with snow accumulated on rooftops` | Snow falling; rooftops + ledges have snow accumulation; buildings unchanged; sky overcast gray | medium |

### Category 8 — Time of day

| ID | Src image | Prompt (locked) | Expected outcome | Difficulty |
|---|---|---|---|---|
| qie_15_time_cn | park_noon.png — park scene at midday | `改成夕阳西下的黄昏` | Golden-hour warm lighting; elongated shadows; sky has sunset gradient; trees + paths unchanged in geometry | medium |
| qie_16_time_en | bedroom_day.png — bedroom in natural daylight | `Change the scene to late-night with only a bedside lamp illumination` | Window is dark; only a warm pool of light from a bedside lamp; rest of room in deep shadow; furniture layout unchanged | hard |

### Category 9 — Pose change

| ID | Src image | Prompt (locked) | Expected outcome | Difficulty |
|---|---|---|---|---|
| qie_17_pose_cn | person_standing.png — person standing straight, arms at sides, front view | `让人举起双手向上跳` | Person's pose changes to arms-up jumping; face + clothing identity preserved; background unchanged | hard |
| qie_18_pose_en | dog_sitting.png — dog sitting, facing camera | `Change the dog's pose to running forward with mouth open` | Dog is now in a running pose with open mouth; same breed + fur color; background consistent | hard |

### Category 10 — Composition change

| ID | Src image | Prompt (locked) | Expected outcome | Difficulty |
|---|---|---|---|---|
| qie_19_comp_cn | room_cluttered.png — bedroom with visible bed, desk, bookshelf in current layout | `把书架移到床的左侧，桌子放到窗边` | Bookshelf on left of bed; desk by window; same objects, new arrangement; lighting plausibly re-matched | hard |
| qie_20_comp_en | food_plate.png — plate with a steak, fries on the right, salad on the left | `Swap the positions: fries on the left, salad on the right, keep the steak in the center` | Fries and salad positions swapped; steak still centered; plate + table unchanged | medium |

## Difficulty distribution

| Difficulty | Count | IDs |
|---|---|---|
| Easy | 2 | 01, 02 |
| Medium | 10 | 03, 04, 05, 06, 11, 12, 13, 14, 15, 20 |
| Hard | 8 | 07, 08, 09, 10, 16, 17, 18, 19 |

Rationale: a balanced mix lets regressions be isolated by class. If Q5 trips only the "hard" bucket, it's likely a fine-detail or attention-range issue. If it trips "easy," it's a core semantic / color-handling regression. 8-hard biases the suite toward surfacing subtle regressions (text rendering, multi-object removal, pose change) that a naive 5-per-category distribution would under-cover.

## Source image provenance

**Policy**: every source image is either public-domain OR CC0 OR a project-owned fixture. No personally-identifiable-information images. All sha256-locked in `METADATA.json` at Q2 landing. **The Q1 agent or Q2 agent performs the one-time fetch + hash lock.** Until then, each entry lists a canonical description + a suggested search anchor.

Recommended source pools (in priority order):

1. **`OminiX-MLX/qwen-image-mlx/` existing fixtures** — `/Users/yuechen/home/OminiX-MLX/qwen-image-mlx/` has `boy.png, dorm.png, hollywood.png, horse.png, fp32_cat.png, fp32_fluffy_cat.png, my_image.png` among others. Reuse for overlap (cat, portrait, landscape) where license permits (confirm at Q1).
2. **Wikimedia Commons public-domain or CC0** for landscapes, storefronts, streets, food plates.
3. **Pexels / Unsplash CC0** for studio product shots, portraits (generic), beach / park scenes.
4. **Project-owned synthetic fixtures** (e.g. a plain colored shape via ImageMagick) for trivially-constructible reference like "empty desk" or "plain white backdrop" — safer licensing, zero fetch risk.

The Q1 agent's onboarding task list includes: "Fetch 20 source images per the provenance section of `qie_eye_gate_suite.md`; sha256-lock in METADATA; commit under `tools/ominix_diffusion/testdata/qie_eye_gate/sources/`."

## Scoring harness

Not yet implemented (minimal shell pipeline for Q2):

```bash
# Usage: ./run_eye_gate.sh <milestone_tag> [task_ids...]
./ominix_diffusion \
    --model <qie_model_path> \
    --ref-image testdata/qie_eye_gate/sources/${SRC} \
    -p "${PROMPT}" \
    --seed ${SEED} \
    --steps 20 \
    --resolution 512 \
    -o testdata/qie_eye_gate/${MILESTONE_TAG}/${TASK_ID}.png
```

For each milestone:
1. Run all 20 tasks (estimated wall on 910B4 Q4: 20 × ~8-12 s/image ≈ 3-4 min NPU wall post-Q2).
2. Generate side-by-side grid `<milestone>/grid_vs_baseline.png` with baseline_q2 for reviewer comparison.
3. Two reviewers independently mark PASS/FAIL per task; disagreements reviewed jointly with the reference PM.
4. Result tuple `(milestone_tag, pass_count, failing_task_ids)` logged in `METADATA.json` under `milestones.<tag>.eye_gate`.

Full automation (CLIP-score sanity, structural-similarity regression filter, NR-IQA artifact flagging) is out of scope for Q0.5 — the suite is **human-eye-first**. Automated scoring can be layered on later as **advisory, not gating** (TTS's ear-gate lesson: automation was helpful for catching obvious regressions but not for ear-qualitative verdicts; same applies here).

## Reuse across contracts

This suite is written once for QIE-2511 and reused for:
- QIE-2512 / QIE-2601 future variants
- Flux-Kontext (stable-diffusion.cpp already supports it via the same `-r` ref-image plumbing)
- SD3-Edit / Z-Image-Edit (when contracts open)
- Step-distillation checkpoints when (if) they land publicly — the distilled variant must still pass 20/20 on the same suite.

Per `qie_optimization_learnings.md` §6 item 6: "Quality gate = user-eye on a 20-task canonical suite. Codify the suite once and reuse across Flux / Qwen-Image / Z-Image contracts. Save the 2-hour-per-milestone subjective review cost from re-defining per contract."

## Open items (for Q1 agent onboarding)

1. **Source image acquisition and sha256 locking.** 1-2 hours agent wall including license verification. Commit URLs and hashes to `METADATA.json` before Q2 baseline capture.
2. **Prompt translation sanity check.** CN prompts authored here are language-idiomatic formulations; the Q1 agent (or a Chinese-speaking PM/reviewer) confirms they read naturally and aren't translation artifacts. Re-lock if any need rewording **at Q1 only** — post-Q2 lock is frozen.
3. **Seed selection.** Each task needs a fixed seed. Recommendation: `seed = sha256(task_id)[:8] interpreted as int`. Deterministic, reproducible, committed in METADATA at Q2 landing.
4. **Resolution policy.** Q2.5 through Q7: 512×512. Q8: add 1024×1024 runs (same suite, same seeds) to catch resolution-scaling regressions (VAE decode, attention seq-len ~16k). Doubles the eye-review budget at Q8 only.
5. **Ref-image test coverage.** All 20 tasks use `-r <src>` (edit mode). No T2I-only tasks — QIE is edit-specific. If a T2I contract opens later (qwen-image base model), that gets its own suite with source-image-free prompts.

## Why 20 (not 30, not 10)

- **20 is the budget line** for 2-reviewer × 2-hour eye-gate per milestone. 30 tasks double-dip the review budget and we lose the per-milestone cadence. 10 tasks under-cover the failure-mode space (category × language × difficulty is already 10 × 2 × 3 = 60 cells; 20 samples one-per-language-per-category, which is the minimum for "no category is blind").
- **Evenness > completeness.** Every category has CN + EN, every category is equally weighted. This is intentional — if Q4 CFG batching trips text rendering (qie_09, qie_10) but passes everything else, that's a 90% pass rate but a clear localized regression signal. Don't dilute that signal with more text tasks.
- **CN and EN are separate samples, not translations.** Lexical choices in CN vs EN prompts activate different tokenizer paths and text-encoder embedding regions. Translating EN → CN would collapse this variance. Separate tasks catch "regression appears only on CN prompts" class of bugs (we had one in Qwen3-TTS early work: initial Chinese normalization dropped tone markers).

## HARD-KILL rule (for PM ledger)

Any milestone that produces **less than 20/20** in the eye-gate triggers an immediate rollback of that milestone's lever. The contract's 2× steps/s gate is **conjunctive** with eye-gate — meeting one but not the other is a fail. This is non-negotiable per contract §9 "HARD KILL at eye-gate regression." Eye-gate-before-wall-clock precedence inherited from `qie_optimization_learnings.md` §7 item 13.

## Final status for Q0.5.4

- [x] 20 tasks defined with category + language + difficulty balance.
- [x] Per-task: ID, source-image description + provenance policy, locked CN/EN prompt, expected outcome, difficulty.
- [x] Usage protocol: when to run, how to score, rollup, regression trigger.
- [x] Storage path under `tools/ominix_diffusion/testdata/qie_eye_gate/` (baseline-at-Q2 commit).
- [x] Cross-contract reuse note.
- [ ] Source image fetch + sha256 lock (scheduled for Q1 agent).
- [ ] Seed selection per task (scheduled for Q2 agent landing; formula proposed).
- [ ] Baseline capture (scheduled for Q2 milestone).

Suite is **LOCKED as of this commit.** Any future modification — add/remove/re-word task — requires explicit PM sign-off and a new doc revision noting what changed and why.
