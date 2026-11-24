# Jeff test v2: blind label check and overlap with v1

Written by `scripts/make_jeff_set.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-24T00:46:05+00:00
- Git commit: `0b3e92bf2fcf18d265af761d172710f8cfa0aa86`
- Seed: None
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## Agreement

A second Claude agent labelled every query from the text alone, given only the intent definitions in `configs/jeff.yaml`, without the writer's labels. The writer's label stays the label; this measures how clear the labels are.

**Same label on 180 of 180 queries (100.0%).**

| label | agree |
| --- | --- |
| afford_purchase | 10 of 10 |
| cashflow_overview | 10 of 10 |
| category_trend | 10 of 10 |
| compare_periods | 10 of 10 |
| find_transactions | 10 of 10 |
| month_spending | 10 of 10 |
| oos | 40 of 40 |
| project_savings | 10 of 10 |
| recurring_bills | 10 of 10 |
| refresh_data | 10 of 10 |
| remember_fact | 10 of 10 |
| safe_to_spend | 10 of 10 |
| savings_goals | 10 of 10 |
| simulate_monthly_plan | 10 of 10 |
| upcoming_bills | 10 of 10 |

## Disagreements (for the owner's review)

None.

## Marked unsure by the checker (for the owner's review)

The checker gave a label but thought a careful reader could pick another.

| id | split | text | writer_label | checker_label |
| --- | --- | --- | --- | --- |
| 9 | test_v2 | how much went on takeaway in july, felt like a lot of uber eats and maccas | month_spending | month_spending |
| 22 | test_v2 | Been really careful since my birthday in August. Has any of that actually turned into money left over? | cashflow_overview | cashflow_overview |
| 97 | test_v2 | If rent goes up 80 a fortnight, how does that play out over the next 6 months? | simulate_monthly_plan | simulate_monthly_plan |
| 118 | test_v2 | Will I hit my wedding savings goal by next October at this pace? | savings_goals | savings_goals |

## Caveat

Both agents are Claude models working from the same definitions, so agreement shows the labels are clear to a model reader. It does not show that real users write this clearly: the writer was asked for one clear label per message, so the set is likely cleaner than real traffic.

## Closest v1 query for each v2 query

difflib ratio on lower-cased text, against all 320 v1 queries (support and old test). 1.0 = identical. Cap: 0.7. For reference, the closest any v1 test query came to a v1 support example: 0.702. Highest: 0.700. Above 0.8: 0. Median: 0.518. The five highest:

| v2_id | v2_text | v2_intent | ratio | v1_text | v1_split | v1_intent |
| --- | --- | --- | --- | --- | --- | --- |
| 21 | how much came in vs went out over the last 12 months | cashflow_overview | 0.7 | how much came in vs went out last financial year | support | cashflow_overview |
| 6 | Can you break down August by category for me | month_spending | 0.695 | Break down last month's spending by category for me | test | month_spending |
| 46 | total spending per month for the last 6 months | category_trend | 0.69 | income vs spending for the past 12 months | support | cashflow_overview |
| 159 | Can you set up an automatic transfer of $100 every payday into savings? | oos | 0.687 | Can you set up a $100 a month auto transfer into my savings? | test | oos |
| 61 | whats due by friday | upcoming_bills | 0.682 | What's due before payday? | test | upcoming_bills |
