# M6 error analysis: router errors read by hand

Written by `scripts/make_error_sheet.py --summarise`. Do not edit by hand. Rows: `results/error_analysis/sheet.csv`; Claude's category and note per row: `results/error_analysis/claude_categories.csv`.

## Which errors

Banking77: 100 errors drawn at random (seed 0) from all queries S2 handles wrongly on format-controlled test at its frozen policy v1 threshold. Jeff: every in-scope query S2 gets wrong on the old Jeff test split, plus every out-of-scope query it would handle at the frozen Banking77 threshold.

## Categories (Claude, every row)

| category | definition | banking77_test | jeff_old_test |
| --- | --- | --- | --- |
| ambiguous | The query fits both the true and the predicted intent; a careful person could pick either. | 25 (25%) | 0 (0%) |
| label_error | The true label looks wrong and the predicted intent fits better. | 12 (12%) | 0 (0%) |
| same_topic | The predicted intent shares the topic, but the request is different (another action, time window or object); a careful person would pick the true label. | 32 (32%) | 35 (52%) |
| oos_accepted | An out-of-scope query was handled as if it were in scope. | 28 (28%) | 6 (9%) |
| other | None of these; the note says why. | 3 (3%) | 26 (39%) |

## Owner's blind check

Not done yet: `results/error_analysis/owner_check.csv` has empty categories.
