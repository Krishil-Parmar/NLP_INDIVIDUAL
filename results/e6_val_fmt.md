# E6 results on val_fmt

Written by `scripts/evaluate_router.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-22T10:00:54+00:00
- Git commit: `4a435052fa33b0c6602c491a604d4a3c379793a1`
- Seed: 0
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Dataset `banking77` revision: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`
- Dataset `clinc150` revision: `155b9c710419136e17307b80d0a13e68cd46b4ec`
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5

## S4, the LLM being replaced

Both S4 variants, same model, temperature 0, format-controlled text. The one with the lower error rate sets the matched risk and the price of an escalation (rule fixed in `configs/e6_router.yaml` before either was scored here).

| variant | error rate (%) | in-scope accuracy (%) | OOS recall (%) | invalid | US$ per call | sets matched risk |
| --- | --- | --- | --- | --- | --- | --- |
| s4_claude-sonnet-4-6 | 18.00 | 76.30 | 98.29 | 0 | 0.00079 |  |
| s4_claude-sonnet-4-6-fewshot | 12.22 | 84.30 | 97.71 | 0 | 0.00157 | yes |

`s4_claude-sonnet-4-6-fewshot` (`au.anthropic.claude-sonnet-4-6` on Amazon Bedrock). Error rate on all queries (the matched risk): **12.22%**. In-scope accuracy 84.30%, OOS recall 97.71%, invalid replies 0. Measured cost US$0.00157 per call, so US$1.571 per 1,000 queries if every query goes to the LLM. API latency p50 682 ms, p95 831 ms (from Sydney).

## Policy v1 at matched risk, mean ± std over seeds

Each router handles the queries at or above its threshold and escalates the rest to S4. The threshold gives the highest coverage whose selective risk is at or below S4's error rate. 350 of the queries are out of scope; "OOS handled" counts those the router answered itself, the dangerous error. Router cost is taken as US$0 (local model); the saving is against sending every query to S4.

| system | seeds | coverage at matched risk (%) | selective risk (%) | HANDLE precision (%) | HANDLE recall (%) | OOS handled (of n) | AURC (%) | cost per 1,000 queries (US$) | saving vs LLM only (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s0_tfidf | 5 | 70.74 ± 0.00 | 12.15 ± 0.00 | 87.85 ± 0.00 | 94.16 ± 0.00 | 61.0 | 9.20 ± 0.00 | 0.460 | 70.74 ± 0.00 |
| s1_distilroberta-base | 5 | 73.38 ± 0.64 | 12.17 ± 0.03 | 87.83 ± 0.03 | 97.32 ± 0.46 | 54.4 | 7.62 ± 0.26 | 0.418 | 73.38 ± 0.64 |
| s2_all-MiniLM-L6-v2 | 1 | 76.59 ± 0.00 | 12.19 ± 0.00 | 87.81 ± 0.00 | 100.00 ± 0.00 | 34.0 | 8.16 ± 0.00 | 0.368 | 76.59 ± 0.00 |
| s3_quora-distilroberta-base | 5 | 76.18 ± 0.25 | 12.19 ± 0.02 | 87.81 ± 0.02 | 99.91 ± 0.05 | 31.6 | 7.80 ± 0.15 | 0.374 | 76.18 ± 0.25 |

**Best router** (highest mean coverage at matched risk, the rule fixed in README 7.2): `s2_all-MiniLM-L6-v2`.

## Coverage at fixed risk levels

| system | coverage at risk 1% (%) | coverage at risk 2% (%) | coverage at risk 5% (%) |
| --- | --- | --- | --- |
| s0_tfidf | 23.19 ± 0.00 | 44.52 ± 0.00 | 56.00 ± 0.00 |
| s1_distilroberta-base | 45.44 ± 3.25 | 54.18 ± 0.87 | 63.33 ± 0.47 |
| s2_all-MiniLM-L6-v2 | 10.07 ± 0.00 | 42.81 ± 0.00 | 55.56 ± 0.00 |
| s3_quora-distilroberta-base | 29.19 ± 8.61 | 42.13 ± 3.44 | 58.40 ± 1.45 |

## Comparisons (paired, stratified bootstrap of coverage at matched risk)

The matched risk is recomputed on every resample. An interval that spans 0 means no clear difference.

| A | B | coverage A - B (points) | 95% CI | resamples |
| --- | --- | --- | --- | --- |
| s3_quora-distilroberta-base | s1_distilroberta-base | +2.80 | [+0.92, +4.16] | 1000 |
| s3_quora-distilroberta-base | s2_all-MiniLM-L6-v2 | -0.41 | [-1.72, +0.76] | 1000 |

## Calibration (in-scope rows, fitted and measured on the same validation set)

Temperature scaling for the softmax systems; ECE with 15 bins. In-sample, so the after-scaling ECE is optimistic; the honest number comes from test at M5.

| system | temperature (mean) | ECE before (%) | ECE after (%) |
| --- | --- | --- | --- |
| s0_tfidf | 0.835 | 4.95 | 4.09 |
| s1_distilroberta-base | 0.969 | 3.79 | 3.68 |
| s3_quora-distilroberta-base | not scaled | 6.00 | - |

## Latency (batch size 1, warm model, Apple M3 Pro, CPU only, 5 torch threads, 200 queries)

| system | p50 (ms) | p95 (ms) |
| --- | --- | --- |
| s0_tfidf | 0.3 | 0.3 |
| s1_distilroberta-base | 14.4 | 18.9 |
| s2_all-MiniLM-L6-v2 | 5.5 | 6.4 |
| s3_quora-distilroberta-base | 95.7 | 159.7 |

## Figure

Risk-coverage curves, first seed of each system, with the matched-risk operating points: `results/figures/e6_val_fmt_risk_coverage.png`
