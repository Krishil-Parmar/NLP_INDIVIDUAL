# E7 results: the Jeff set (domain shift)

Written by `scripts/evaluate_jeff.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-23T06:26:50+00:00
- Git commit: `703a0d9f933a537917d214f5c56dd4c9701beccb`
- Seed: 0
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## Setup

180 Jeff test queries (40 out of scope), format-controlled. S2 and S3 are the frozen models with the 140 Jeff support queries as their support set: no retraining, no new settings. S0 and S1 are not applicable: they are softmax classifiers over the 77 Banking77 intents and cannot output a Jeff intent. The Jeff set was written by a Claude agent (README 6.1); S4 is a Claude model too.

## S4 on the Jeff set

| variant | error rate (%) | in-scope accuracy (%) | macro-F1 (%) | OOS recall (%) | invalid | US$ per call | sets matched risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| s4_claude-sonnet-4-6 | 1.11 | 98.57 | 98.56 | 100.00 | 0 | 0.00066 |  |
| s4_claude-sonnet-4-6-fewshot | 0.56 | 99.29 | 99.28 | 100.00 | 0 | 0.00083 | yes |

Matched risk (the frozen few-shot variant's error rate on these queries): **0.56%**.

## Routers at matched risk (threshold recomputed on these queries), mean ± std over seeds

The interval is a bootstrap of the seed-mean coverage, with the matched risk recomputed on every resample.

| system | seeds | in-scope accuracy (%) | macro-F1 (%) | OOS AUROC (%) | coverage at matched risk (%) | 95% CI | selective risk there (%) | AURC (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s2_all-MiniLM-L6-v2 | 1 | 56.43 ± 0.00 | 56.67 ± 0.00 | 90.93 ± 0.00 | 1.67 ± 0.00 | [0.00, 6.67] | 0.00 ± 0.00 | 42.15 ± 0.00 |
| s3_quora-distilroberta-base | 5 | 48.00 ± 2.44 | 47.13 ± 2.59 | 88.35 ± 0.31 | 0.00 ± 0.00 | [0.00, 2.22] | 0.00 ± 0.00 | 53.84 ± 1.56 |

## Routers at the frozen Banking77 thresholds (policy v1 applied unchanged)

| system | coverage (%) | selective risk (%) | OOS handled (of 40) |
| --- | --- | --- | --- |
| s2_all-MiniLM-L6-v2 | 61.67 ± 0.00 | 45.95 ± 0.00 | 6.0 |
| s3_quora-distilroberta-base | 59.44 ± 1.96 | 53.98 ± 2.68 | 6.4 |

## Coverage at fixed risk levels

| system | coverage at risk 1% (%) | coverage at risk 2% (%) | coverage at risk 5% (%) |
| --- | --- | --- | --- |
| s2_all-MiniLM-L6-v2 | 1.67 ± 0.00 | 1.67 ± 0.00 | 1.67 ± 0.00 |
| s3_quora-distilroberta-base | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 |
