# E7 diagnosis: why S2 fails on the Jeff set

Written by `scripts/diagnose_jeff.py`. Do not edit by hand. Reads saved predictions and Banking77 validation only; nothing is tuned and no test prediction is made.

## Provenance

- Generated (UTC): 2026-11-24T05:41:52+00:00
- Git commit: `7328d65c12dc18fb0d8923451904e9d3cf50a6f5`
- Seed: None
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## 1. The right intent is usually on the shortlist

S2 on the Jeff test split (in-scope queries): the true intent scores highest in 56.4% of queries, is in the top two in 75.7% and in the top three in 88.6%. S2 finds the right neighbourhood but picks the wrong neighbour.

## 2. The confidence stops separating right from wrong answers

AUROC here is the chance that a random right answer gets a higher confidence than a random wrong one (50% = no better than a coin flip). Coverage at low risk needs this to be high: the router keeps the most confident answers.

| data | in-scope queries | in-scope accuracy (%) | AUROC right vs wrong, nearest cosine (%) | AUROC right vs wrong, margin (%) | median cosine, right | median cosine, wrong |
| --- | --- | --- | --- | --- | --- | --- |
| Banking77 test_fmt (E6) | 3079 | 92.1 | 74.8 | 89.2 | 0.916 | 0.821 |
| Jeff test (E7) | 140 | 56.4 | 54.3 | 60.7 | 0.567 | 0.549 |

## 3. Support size explains part of the gap, not all of it (Banking77 validation)

S2 re-run on Banking77 validation (format-controlled, with the OOS pool) with its support set cut to k random examples per intent. The first coverage column is at few-shot S4's validation error (12.2%), the bar policy v1 was set on. Mean over five draws (standard deviations in the JSON file).

| support | examples per intent | in-scope accuracy (%) | confidence AUROC (%) | OOS AUROC (%) | coverage at 12.2% risk (%) | coverage at 5% risk (%) | coverage at 1% risk (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full train | all (about 120) | 90.8 | 78.3 | 99.9 | 76.6 | 55.6 | 10.1 |
| 5 random draws | 10 | 80.0 | 76.8 | 99.7 | 54.5 | 27.7 | 6.5 |
| 5 random draws | 20 | 84.0 | 76.6 | 99.8 | 65.1 | 30.3 | 8.3 |
| 5 random draws | 50 | 88.4 | 79.1 | 99.9 | 74.1 | 49.1 | 18.3 |

For comparison, on the Jeff test split (10 support examples per intent) S2 covers 1.7% at 5% risk and 1.7% at 1% risk (`results/e7_jeff.json`). Cutting Banking77 to 10 examples per intent halves the coverage at 5% risk. **Correction (24 Nov 2026):** the first version of this report said the rest of the Jeff gap comes from the Jeff intents themselves. That was not shown, and neither is the opposite: see section 5.

## 5. Accuracy at the same distance from the nearest labelled example (added 24 Nov 2026)

S2's in-scope accuracy by the cosine between a query and its nearest labelled example (S2's confidence), in narrow bands. The first block gives Banking77 the same support size as Jeff (10 per intent), so sparse support is held fixed. If Jeff were as hard as Banking77 at equal distance, its accuracy would match the first block band by band.

Share of in-scope queries with a labelled example at cosine 0.7 or more: Banking77 validation, 10 examples per intent 83.4%, Banking77 test, full support 97.7%, Jeff test 11.4%. So S2 is no less accurate on Jeff than on Banking77 at the same distance and the same support size (Jeff has 14 intents to choose from against 77, which favours it), but far fewer Jeff queries have a close labelled example. Why they are far (more varied wording, or intents that 10 examples cannot cover) cannot be separated with this data.

| data | cosine band | queries | mean cosine | accuracy (%) | 95% Wilson CI |
| --- | --- | --- | --- | --- | --- |
| Banking77 validation, 10 examples per intent | 0.0 to 0.5 | 24 | 0.445 | 20.8 | [9.2, 40.5] |
| Banking77 validation, 10 examples per intent | 0.5 to 0.6 | 152 | 0.567 | 44.7 | [37.1, 52.7] |
| Banking77 validation, 10 examples per intent | 0.6 to 0.7 | 656 | 0.663 | 58.2 | [54.4, 61.9] |
| Banking77 validation, 10 examples per intent | 0.7 to 0.8 | 1525 | 0.754 | 72.3 | [70.0, 74.5] |
| Banking77 validation, 10 examples per intent | 0.8 to 0.9 | 1806 | 0.847 | 89.7 | [88.2, 91.0] |
| Banking77 validation, 10 examples per intent | 0.9 to 1.0 | 837 | 0.934 | 98.1 | [96.9, 98.8] |
| Banking77 test, full support | 0.0 to 0.5 | 3 | 0.463 | 33.3 | [6.1, 79.2] |
| Banking77 test, full support | 0.5 to 0.6 | 10 | 0.563 | 70.0 | [39.7, 89.2] |
| Banking77 test, full support | 0.6 to 0.7 | 59 | 0.662 | 57.6 | [44.9, 69.4] |
| Banking77 test, full support | 0.7 to 0.8 | 292 | 0.761 | 76.4 | [71.2, 80.9] |
| Banking77 test, full support | 0.8 to 0.9 | 978 | 0.858 | 91.0 | [89.0, 92.6] |
| Banking77 test, full support | 0.9 to 1.0 | 1737 | 0.947 | 96.8 | [95.9, 97.6] |
| Jeff test | 0.0 to 0.5 | 43 | 0.437 | 55.8 | [41.1, 69.6] |
| Jeff test | 0.5 to 0.6 | 49 | 0.550 | 51.0 | [37.5, 64.4] |
| Jeff test | 0.6 to 0.7 | 32 | 0.646 | 59.4 | [42.3, 74.5] |
| Jeff test | 0.7 to 0.8 | 14 | 0.739 | 64.3 | [38.8, 83.7] |
| Jeff test | 0.8 to 0.9 | 2 | 0.828 | 100.0 | [34.2, 100.0] |
| Jeff test | 0.9 to 1.0 | 0 |  | n/a |  |

## 4. Where the Jeff errors are

Accuracy per intent (S2, Jeff test split):

| intent | accuracy (%) |
| --- | --- |
| category_trend | 30.0 |
| simulate_monthly_plan | 30.0 |
| afford_purchase | 40.0 |
| find_transactions | 40.0 |
| remember_fact | 40.0 |
| compare_periods | 50.0 |
| month_spending | 50.0 |
| upcoming_bills | 50.0 |
| cashflow_overview | 60.0 |
| recurring_bills | 60.0 |
| project_savings | 70.0 |
| refresh_data | 80.0 |
| safe_to_spend | 90.0 |
| savings_goals | 100.0 |

Most frequent confusions:

| true | predicted | count |
| --- | --- | --- |
| category_trend | month_spending | 4 |
| cashflow_overview | month_spending | 3 |
| simulate_monthly_plan | project_savings | 3 |
| afford_purchase | simulate_monthly_plan | 2 |
| category_trend | find_transactions | 2 |
| compare_periods | upcoming_bills | 2 |
| find_transactions | remember_fact | 2 |
| month_spending | category_trend | 2 |

Content words shared with the matched support example (scikit-learn English stop words removed): 0.90 on average for right answers, 0.92 for wrong ones. So the number of shared words cannot tell a right answer from a wrong one. Reading the errors shows what goes wrong: many wrong matches share the topic or one salient word with the query but answer a different request (`results/error_analysis.md`). Median S2 confidence on the out-of-scope queries: 0.379.
