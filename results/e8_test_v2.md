# E8 final test: the fixes on the fresh Jeff test v2

Written by `scripts/evaluate_few_shot.py`. Do not edit by hand. **Post-hoc experiment, not in the main pre-registration**; its plan is `results/preregistration_e8.md`. The pre-registered RQ6 verdict (not met) is unchanged.

## Provenance

- Generated (UTC): 2026-11-24T01:09:41+00:00
- Git commit: `35b1737d0b52247554ef603a463a975510e0755c`
- Seed: 0
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## Setup

180 fresh Jeff queries (40 out of scope), format-controlled, run once. Matched risk: few-shot S4's error on them, 0.00% (in-scope accuracy 100.00%).

## Results (mean ± std over seeds; interval: bootstrap of the seed mean, matched risk recomputed per resample)

| system | tier | seeds | in-scope accuracy (%) | OOS AUROC (%) | AURC (%) | coverage at matched risk (%) | 95% CI | coverage at 1% (%) | coverage at 5% (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen/examples/nearest (control: S2 unchanged) | no training | 1 | 69.29 ± 0.00 | 94.37 | 27.06 ± 0.00 | 2.78 | [1.11, 8.89] | 2.78 | 2.78 |
| frozen/descriptions/head | head only | 1 | 82.14 ± 0.00 | 78.46 | 12.38 ± 0.00 | 17.22 | [12.78, 37.22] | 17.22 | 36.67 |
| tuned/examples/nearest | encoder fine-tune | 5 | 82.43 ± 0.73 | 93.14 | 10.69 ± 0.44 | 24.89 | [20.89, 43.11] | 24.89 | 51.00 |

## Primary number: AURC gain over the control (paired bootstrap; positive = the fix is better)

| system | AURC gain vs control (points) | 95% CI |
| --- | --- | --- |
| frozen/descriptions/head | 14.68 | [7.70, 23.06] |
| tuned/examples/nearest | 16.37 | [9.25, 25.69] |

## The RQ6 yardstick, post-hoc

Banking77 test coverage at matched risk (the RQ1 number): 77.47% [76.44, 78.43]. RQ6's rule, applied to each system on v2. This does not change the pre-registered RQ6 verdict.

| system | coverage at matched risk (%) | drop from Banking77 test (points) | within D = 20.00? |
| --- | --- | --- | --- |
| frozen/examples/nearest | 2.78 | 74.69 | no |
| frozen/descriptions/head | 17.22 | 60.25 | no |
| tuned/examples/nearest | 24.89 | 52.58 | no |

## The development thresholds applied unchanged (what a deployed router would do)

| system | coverage (%) | selective risk (%) | OOS handled (of 40) |
| --- | --- | --- | --- |
| frozen/examples/nearest | 3.33 | 16.67 | 0.0 |
| frozen/descriptions/head | 21.11 | 2.63 | 1.0 |
| tuned/examples/nearest | 16.56 | 0.94 | 0.0 |
