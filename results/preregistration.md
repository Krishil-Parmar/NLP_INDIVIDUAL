# Pre-registration

**Status:** in force from the commit that adds this file. That commit predates every test-split prediction; `git log` shows the order.

**Who decided:** Claude Code drafted it at the owner's request ("select whatever you think would be appropriate"). The owner confirmed it on 23 Nov 2026 ("go ahead") and approved the M5 plan.

**State of the test splits before this commit:** nothing had been predicted on, scored on or looked at, for either the Banking77 test split or the Jeff test split.

The machine-readable copy of the thresholds is `configs/preregistration.yaml`. `scripts/report_verdicts.py` applies them to the test result files.

## Thresholds (README section 4)

| ID | Hypothesis | Value | Validation evidence | Why this value |
| --- | --- | --- | --- | --- |
| RQ1 | The best router reaches coverage of at least N% at matched risk on Banking77 | **N = 60** | Best router S2: 76.59% at matched risk 12.22% (95% CI 74.81 to 78.22). Weakest, S0: 70.74%. At a 5% risk bar, S2 reaches 55.56% | 60% is a clear product claim: the router takes at least 6 in 10 queries off the LLM. The margin is there because coverage falls fast if S4's error rate on test is lower than on validation |
| RQ2 | The pairwise matcher beats the softmax encoder on OOS AUROC by at least M points | **M = 1.0** | S3 minus S1: +2.82 points (95% CI +2.21 to +3.56) | S1 is only 3.11 points from a perfect 100, so 1 point is about a third of its remaining room. It sits well below the validation interval |
| RQ3 | With 10 support examples per held-out intent, the matcher is within K accuracy points of a fully retrained softmax | **K = 5** | Not measured (E4 not run) | Being within 5 points means a new tool can be added from about 10 examples without retraining (R5) |
| RQ4 | The system ranking is stable between original and cleaned labels | No threshold | Stable in all four validation cells | Same order by mean macro-F1 |
| RQ5 | The best router's flip rate under validated paraphrases is below F% | **F = 10** | Not measured (E5 not run) | At most 1 in 10 rewordings may change the decision (R6) |
| RQ6 | Coverage at matched risk on the Jeff set drops by no more than D points versus Banking77 | **D = 20** | Not measured | From about 75% on Banking77, a drop of 20 points still leaves the router answering more than half of real-style queries (R7) |
| R4 | The router's p95 latency on CPU is at most L ms | **L = 100** | S2 p95 6.4 ms; S3 p95 159.7 ms (Apple M3 Pro, CPU, batch 1, `results/latency.json`) | About a ninth of Jeff's 900 ms time-to-first-token target |

## How each hypothesis is judged

- **The verdict:** met when the point estimate reaches the threshold, not met otherwise. The 95% bootstrap interval (1,000 resamples) is reported next to it. If the interval crosses the threshold, the verdict is also labelled "not clear-cut".
- **RQ1:**
  - **Queries:** `test_fmt`, which is Banking77 test plus CLINC150 OOS test, format-controlled, 4,079 queries.
  - **Router:** the best router frozen in `results/policy_v1.json` (`s2_all-MiniLM-L6-v2`), even if another system does better on test.
  - **Matched risk:** few-shot S4's error rate on the same queries. The few-shot variant was frozen in `policy_v1.json` on validation.
  - **The RQ1 number:** coverage at matched risk recomputed on test (seed mean).
  - **Its interval:** a stratified bootstrap (in-scope and out-of-scope rows drawn separately), with the matched risk recomputed in every resample.
  - **Also reported:** coverage, selective risk and cost with policy v1's frozen validation thresholds and temperatures applied unchanged, plus ECE on test after the frozen temperatures.
- **RQ2:** S3 minus S1 AUROC on `test_fmt`, each averaged over its 5 seeds, using the paired, stratified bootstrap. OOS recall is reported at each seed's validation threshold.
- **RQ4 (E2 on test):**
  - **Run:** the cleaned-label models (S0, S1 and S3 five seeds each, S2 with its cleaned support set) predict raw Banking77 test once. They are scored against the original test labels, since test was never checked for label errors.
  - **Stable:** the same order by mean macro-F1 for original-trained and cleaned-trained models. Each system's training effect is reported with its paired bootstrap interval, stratified by intent.
- **RQ6 (E7 on the Jeff set):** set out in its own section below.
- **R4:** the best router's p95 in `results/latency.json`, measured at M4 on 200 validation queries. Latency is a property of the model and machine, not of the split.
- **RQ3 and RQ5:** E4 and E5 are stretch work, early in the README's cut order, and are not run at M5. They are reported as "not tested" unless they are built later under this same pre-registration.

## E7: the Jeff set (RQ6)

- **Queries:** the 180 Jeff test queries (14 intents × 10, plus 40 out of scope), format-controlled like E6, split name `jeff_test_fmt`. The set was written by a Claude agent at the owner's request (README 6.1).
- **S2 and S3:**
  - The frozen models: S2 with seed 0, and S3's cross-encoder with seeds 0 to 4, trained on Banking77.
  - Their support set is the 140 Jeff support queries (10 per intent), and the label set is the 14 Jeff intents.
  - No retraining and no setting changes. The support examples are not format-controlled, as in E6.
- **S0 and S1 are not applicable.** They are softmax classifiers over the 77 Banking77 intents and cannot output a Jeff intent. Retraining them from 10 examples per intent is E4's question.
- **S4:**
  - **Model:** the same model and settings as on Banking77, zero-shot and few-shot.
  - **Prompts:** `prompts/s4_jeff_v1.txt` and `prompts/s4_jeff_fewshot_v1.txt`. Each lists the 14 intent names, each followed by its description and "not this" note from `configs/jeff.yaml`, plus the out-of-scope definition from the same file, in personal-finance wording.
  - **Few-shot examples:** 2 Jeff support queries per intent, drawn at random with seed 0 and format-controlled.
  - **Configs:** `configs/s4_jeff.yaml` and `configs/s4_jeff_fewshot.yaml`.
- **Matched risk:** few-shot S4's error rate on the 180 Jeff test queries. The variant choice is carried over from validation.
- **RQ6 number:**
  - The best router's coverage at matched risk on Banking77 test (the RQ1 number) minus its coverage at matched risk on the Jeff test split (recomputed there, with its own bootstrap interval).
  - Met when the drop is at most 20 points. The verdict is labelled not clear-cut if the widest pairing of the two intervals crosses 20 points.
- **Also reported per system:**
  - in-scope accuracy and macro-F1 over the 14 intents
  - OOS AUROC
  - coverage and selective risk at the frozen Banking77 thresholds
  - coverage at the fixed risks of 1%, 2% and 5%
- **Limitation, stated before the run:** the queries are written by a Claude model, and S4 is a Claude model too. S4 may find them easier than real users' messages, which would lower the matched risk and make the bar harder for the routers. Model-written text is also not proof of how real Australian users write.

## Runs, in order (each test split touched once per system)

1. **`scripts/predict.py --split test`:** S0, S1 and S3 with seeds 0 to 4, and S2 MiniLM with seed 0, on raw Banking77 test plus CLINC150 OOS test. S2 mpnet is not a compared system and is not run.
2. **`scripts/predict_format_control.py --split test`:** the same runs on `test_fmt`.
3. **`scripts/predict.py --split test --labels cleaned`:** the E2 cleaned-label runs on raw test.
4. **`scripts/run_s4.py --split test --config s4_llm_fewshot`:** few-shot S4 on `test_fmt`. Zero-shot S4 is not run on Banking77 test: the frozen matched-risk variant is few-shot, and zero-shot would push the total past the cap.
5. **`scripts/predict_jeff.py`:** S2 and S3 on `jeff_test_fmt`.
6. **`scripts/run_s4.py --split test --config s4_jeff` and `--config s4_jeff_fewshot`.**
7. **Evaluations:** `evaluate.py --split test` (E1), `evaluate_oos.py --split test_fmt` (E3), `evaluate_router.py --split test_fmt` (E6), `evaluate_label_noise.py --split test` (E2), `evaluate_jeff.py` (E7), then `report_verdicts.py`.

## S4 spend

- **Approved:** at most US$10 of new spend for M5 (owner, 23 Nov 2026).
- **How it is enforced:** every S4 config's cap is US$13.20, which is the US$3.19 already spent on validation plus the US$10. It is counted over all four S4 cache files.
- **Estimates:**
  - few-shot S4 on 4,079 `test_fmt` queries: about US$6.40
  - both Jeff variants on 180 queries: well under US$0.50

## Stop rules during the runs

- **Stop and report if:**
  - any script errors
  - S4's invalid replies exceed 1% of a split
  - the spend cap stops a run
- **Never re-run a test prediction because of its result.**

## Deviations

Any change after the first test file exists is recorded here: what changed, why, and whether any test number had been seen.

None so far.
