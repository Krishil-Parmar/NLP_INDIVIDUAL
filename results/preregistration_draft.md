# Pre-registration: DRAFT, not yet in force

Drafted 23 Nov 2026 by Claude Code. The owner asked Claude to choose the thresholds ("select whatever you think would be appropriate").

**When it takes effect:** once the owner confirms it, it is saved as `results/preregistration.md` and committed. That has to happen before any test-split run. Until then, every script that scores a test split refuses to run.

**State of the test splits:** nothing has been run on the Banking77 test split or the Jeff test split, and nothing has been looked at in them.

## Thresholds (README section 4)

Each value comes from what the requirement needs, checked against validation. None is set just under a validation result so that it is sure to pass.

| ID | Hypothesis | Value | Validation evidence | Why this value |
| --- | --- | --- | --- | --- |
| RQ1 | The best router reaches coverage of at least N% at matched risk on Banking77 | **N = 60** | Best router S2: 76.59% at matched risk 12.22%. Weakest, S0: 70.74%. At a 5% risk bar, S2 reaches 55.56% | 60% is a clear product claim: the router takes at least 6 in 10 queries off the LLM. The margin is there because coverage falls fast if S4's error rate on test is lower than on validation |
| RQ2 | The pairwise matcher beats the softmax encoder on OOS AUROC by at least M points | **M = 1.0** | S3 minus S1: +2.82 points (95% CI +2.21 to +3.56) | S1 is only 3.11 points from a perfect 100, so 1 point is about a third of its remaining room. It sits well below the validation interval |
| RQ3 | With 10 support examples per held-out intent, the matcher is within K accuracy points of a fully retrained softmax | **K = 5** | Not measured (E4 not run) | Being within 5 points means a new tool can be added from about 10 examples without retraining (R5) |
| RQ4 | The system ranking is stable between original and cleaned labels | No threshold | Stable in all four validation cells | Same order by mean macro-F1 |
| RQ5 | The best router's flip rate under validated paraphrases is below F% | **F = 10** | Not measured (E5 not run) | At most 1 in 10 rewordings may change the decision (R6) |
| RQ6 | Coverage at matched risk on the Jeff set drops by no more than D points versus Banking77 | **D = 20** | Not measured (Jeff set not yet run) | From about 75% on Banking77, a drop of 20 points still leaves the router answering more than half of real-style queries. Below that it is hard to justify running it (R7) |
| R4 | The router's p95 latency on CPU is at most L ms | **L = 100** | S2 p95 6.4 ms; S3 p95 159.7 ms (Apple M3 Pro, CPU, batch 1) | About a ninth of Jeff's 900 ms time-to-first-token target. On the validation numbers S3 would not meet it; S2, the best router, would |

## How each hypothesis is judged

- **The verdict:** met when the point estimate reaches the threshold, not met otherwise. The 95% bootstrap interval is reported next to it. If the interval crosses the threshold, the verdict is also labelled "not clear-cut".
- **RQ1:**
  - **Router:** the best router frozen in `results/policy_v1.json` (`s2_all-MiniLM-L6-v2`).
  - **Queries:** `test_fmt`, which is Banking77 test plus CLINC150 OOS test, format-controlled, 4,079 queries.
  - **Matched risk:** few-shot S4's error rate on those same queries. The few-shot variant was frozen in `policy_v1.json` on validation, so it is not re-chosen on test.
  - **Also reported:** coverage with policy v1's frozen thresholds applied unchanged.
- **RQ2:** S3 minus S1 AUROC on `test_fmt`, each averaged over its 5 seeds, using the paired, stratified bootstrap (1,000 resamples).
- **RQ4 (E2 on test):**
  - **Run:** the training effect only. The cleaned-label models (S0, S1 and S3 five seeds each, S2 with its cleaned support set) predict Banking77 test once. They are scored against the original test labels, since test was never checked for label errors.
  - **Stable:** the same order by mean macro-F1 when trained on original or on cleaned labels. Each system's training effect is reported with its bootstrap interval.
- **RQ6 (E7):** two things are fixed in the M5 plan, before the Jeff test split is run:
  - how each system uses the Jeff support split
  - S4's prompt on the Jeff intents

  The baseline for the drop is the best router's RQ1 coverage on Banking77 test.
- **RQ3 and RQ5:** E4 and E5 are stretch work and are next in the README's cut order. If they are cut, they are reported as "not tested".

## S4 spend at M5 (needs the owner's explicit yes before any call)

- **Banking77 `test_fmt`:** few-shot S4 on 4,079 queries, about US$6.40 at validation's US$0.00157 per call.
- **Jeff test split:** both S4 variants on 180 queries, under US$0.50.
- **Not run:** zero-shot S4 on Banking77 test (about US$3.20). Matched risk uses the frozen few-shot variant, so zero-shot isn't needed, and it would push the total past the cap. E3 reports zero-shot S4 on validation only.
- **Proposed cap: US$10** for all of M5.

## The Jeff set

- **Who wrote it:** a separate Claude agent, at the owner's request. It worked only from the intent definitions in `configs/jeff.yaml`, which come from Jeff's real tool belt. It saw no project code, data, models or results. This changes README section 6.1, which says the owner writes it.
- **Label check:** a second Claude agent labelled every query blind. Agreement is in `results/jeff_label_check.md`, and disagreements go to the owner for review.
- **Splits:** 14 intents × 20 queries, 10 support and 10 test per intent chosen at random (seed 0), plus 40 OOS queries (15 unrelated, 25 finance-adjacent), all in test.
- **Limitation:** the queries are written by a Claude model, and S4 is a Claude model too. S4 may find them easier than real users' messages, which would lower the matched risk and make the bar harder for the routers. Model-written text is also not proof of how real Australian users write (R7).
