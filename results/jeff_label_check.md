# Jeff set: blind label check

Written by `scripts/make_jeff_set.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-22T23:47:07+00:00
- Git commit: `bbda18c772a7a7863827578d279ebd8e8a8f53a3`
- Seed: 0
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## Agreement

A second Claude agent labelled every query from the text alone, given only the intent definitions in `configs/jeff.yaml`, without the writer's labels. The writer's label stays the label; this measures how clear the labels are.

**Same label on 320 of 320 queries (100.0%).**

| label | agree |
| --- | --- |
| afford_purchase | 20 of 20 |
| cashflow_overview | 20 of 20 |
| category_trend | 20 of 20 |
| compare_periods | 20 of 20 |
| find_transactions | 20 of 20 |
| month_spending | 20 of 20 |
| oos | 40 of 40 |
| project_savings | 20 of 20 |
| recurring_bills | 20 of 20 |
| refresh_data | 20 of 20 |
| remember_fact | 20 of 20 |
| safe_to_spend | 20 of 20 |
| savings_goals | 20 of 20 |
| simulate_monthly_plan | 20 of 20 |
| upcoming_bills | 20 of 20 |

## Disagreements (for the owner's review)

None.

## Marked unsure by the checker (for the owner's review)

The checker gave a label but thought a careful reader could pick another.

| id | split | text | writer_label | checker_label |
| --- | --- | --- | --- | --- |
| 25 | support | macca's purchases in october | find_transactions | find_transactions |
| 60 | test | What's my savings rate been over the past 6 months? | cashflow_overview | cashflow_overview |
| 72 | support | overall am i better or worse off since i sold the car and started taking the train | compare_periods | compare_periods |
| 188 | support | how long to save 10k if i put 600 away every month | simulate_monthly_plan | simulate_monthly_plan |
| 215 | support | at this rate when do i hit 20k saved | project_savings | project_savings |
| 273 | support | From now on, treat the Bunnings spending as business expenses, it's for my side hustle | remember_fact | remember_fact |

## Caveat

Both agents are Claude models working from the same definitions, so agreement shows the labels are clear to a model reader. It does not show that real users write this clearly: the writer was asked for one clear label per message, so the set is likely cleaner than real traffic.
