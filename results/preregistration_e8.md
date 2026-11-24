# E8 test plan: the fixes for the Jeff failure, on a fresh test set

Written 24 Nov 2026, after the E8 development run and before any system has run on Jeff test v2.
`scripts/evaluate_few_shot.py` refuses to run without this file.

**Status: post-hoc.** E8 was designed after the pre-registered RQ6 test failed (`results/verdicts.md`).
It is not part of `results/preregistration.md`, and it does not change any pre-registered verdict:
RQ6 stays "not met". E8 answers a different question: having diagnosed why the routers fail on the
Jeff intents, does the fix hold up on queries nobody has seen?

## What happened before this file

1. **Diagnosis** (`results/e7_diagnosis.md`, `scripts/diagnose_jeff.py`): saved predictions and
   Banking77 validation only. S2's confidence stops separating right from wrong answers on Jeff, the
   general encoder matches topic or a salient word rather than the request, and 10 examples per intent
   explain part of the gap.
2. **Design fixed** (`configs/e8_few_shot.yaml`, committed in `0b3e92b` before any development run):
   the 12 variants, the seeds, and the per-tier selection rule.
3. **Development run** (`results/e8_dev.md`, clean tree at `0b3e92b`): every variant on the old Jeff
   test split, which had been read during the diagnosis and so became development data, plus a
   Banking77 validation check. The rule chose one variant per tier (below).
4. **Jeff test v2 built** (`data/jeff/jeff_v2_queries.csv`, `results/jeff_v2_label_check.md`):
   - A new Claude agent wrote 180 queries from the same brief as v1.
   - 54 queries were too close to a v1 query (difflib ratio above 0.70, the v1 test-to-support
     reference), 6 of them word for word. The same agent rewrote them before any system ran, and
     every v2 query is now within the cap.
   - A second agent labelled all 180 blind: same label on 180 of 180, 4 marked unsure.
   - No system, router or LLM, has run on v2.
   - Claude saw fewer than 20 v2 queries: quoted in the writer's replies and in the build report. That
     was after the variants and the selection rule were fixed, and the selection was then made by
     the rule on development data alone.

## Systems run on v2 (frozen)

| tier | system | runs |
| --- | --- | --- |
| control | S2 unchanged: `frozen/examples/nearest` | 1 (deterministic) |
| no training | the rule chose S2 itself, so nothing extra runs | |
| head only | `frozen/descriptions/head`: logistic regression on the frozen MiniLM embeddings of the 140 support examples plus the 14 intent definitions | 1 (deterministic) |
| encoder fine-tune | `tuned/examples/nearest`: the MiniLM encoders fine-tuned on the 140 support examples during development (`checkpoints/e8/jeff/examples/seed0` to `seed4`), nearest-example scoring | 5 seeds |

The support set is the Jeff support split, as in E7 and development. Queries are format-controlled
(`router.preprocess.format_control`), as in E7. No setting changes after this file.

## S4 on v2

- Few-shot S4 (`configs/s4_jeff_fewshot.yaml`, the variant frozen in policy v1), one call per v2
  query: 180 calls, about US$0.15.
- Approved by the owner on 24 Nov 2026 ("Few-shot S4 only, ~US$0.15").
- The spend cap stays at US$13.20 cumulative over the four S4 caches.
- Zero-shot S4 is not run.
- Stop rule: if more than 1% of replies are invalid, stop and report before evaluating.

## Numbers and how each is read

**Primary: the AURC gain over the control** (control AURC minus the fix's AURC).
- Paired bootstrap, 1,000 resamples stratified by in scope and out of scope, 95% interval.
- For the fine-tuned variant, the AURC is the mean over its five seeds in every resample.
- A fix **holds up on fresh data** when the interval's lower bound is above 0.
- **It does not hold up** when the interval includes 0 or lies below it.

**Secondary, all reported:**
- Coverage at matched risk (few-shot S4's error on v2), with an interval: a bootstrap of the seed
  mean, with the matched risk recomputed on every resample (as E7).
- Coverage at 1%, 2% and 5% risk, in-scope accuracy, macro-F1, OOS AUROC, and confidence AUROC.
- **The development threshold, applied unchanged:** each run's threshold from `results/e8_dev.json`.
  Reported: the coverage, the selective risk and the number of out-of-scope queries handled. This
  is what a deployed router would do. Its selective risk is compared with S4's error on v2.
- **The RQ6 yardstick, post-hoc:** the Banking77 test coverage at matched risk (the RQ1 number)
  minus each system's v2 coverage at matched risk, against D = 20 points. Reported for information
  only; the RQ6 verdict stays "not met".

**Control check:** if the control's numbers on v2 are far from its numbers on the old test split,
v2 is not comparable in difficulty, and the report says so.

## Run once

`scripts/evaluate_few_shot.py` runs every system above on v2 in one go. It refuses to run again once
`results/e8_test_v2.json` exists. If a bug is found after that file exists, the fix and the reason
are logged under Deviations below, and only the affected step is re-run.

## Deviations

None so far.

## Later notes (24 Nov 2026, after two independent reviews; the plan above is unchanged)

- **Diagnosis wording.** Step 1's summary of the diagnosis ("the general encoder matches topic or a salient word rather than the request") was later revised. Jeff queries sit far from every labelled example, and at equal distance S2 is no less accurate on Jeff than on Banking77. Why Jeff queries are far is not known (`results/e7_diagnosis.md`, section 5). The systems and the selection were not affected.
- **Coverage at matched risk on v2.** S4 made no errors on v2, so this number is a first-error count, and the percentile bootstrap interval in `results/e8_test_v2.md` is not valid for it. Per-seed counts are reported instead (`results/headline_reanalysis.md`, G).
- **The development threshold, applied unchanged.** `results/e8_test_v2.md` averages selective risk over seeds and counts a seed that handles nothing as 0%. Pooled over seeds the fine-tuned router handled 149 query-runs with 3 wrong (2.01%) (`results/headline_reanalysis.md`, G).
- **Standard deviations.** `results/e8_test_v2.md` used ddof = 0. The sample standard deviation (ddof = 1) of the fine-tuned in-scope accuracy is 0.81 points.
- **Timing.** v2 was written after the candidates and the rule were committed (`0b3e92b`) and before any system ran on it. The development run that applied the rule finished after v2 was written.

`results/e8_test_v2.md` is left as written, because the run-once rule means it is not regenerated.
