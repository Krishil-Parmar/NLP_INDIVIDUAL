# E6 results on test_fmt

Written by `scripts/evaluate_router.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-23T06:24:39+00:00
- Git commit: `703a0d9f933a537917d214f5c56dd4c9701beccb`
- Seed: 0
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Dataset `banking77` revision: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`
- Dataset `clinc150` revision: `155b9c710419136e17307b80d0a13e68cd46b4ec`
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## S4, the LLM being replaced

Every S4 variant run on this split, same model, temperature 0, format-controlled text. The variant frozen on validation (`results/policy_v1.json`) sets the matched risk and the price of an escalation.

| variant | error rate (%) | in-scope accuracy (%) | OOS recall (%) | invalid | US$ per call | sets matched risk |
| --- | --- | --- | --- | --- | --- | --- |
| s4_claude-sonnet-4-6-fewshot | 10.27 | 87.17 | 97.60 | 0 | 0.00158 | yes |

`s4_claude-sonnet-4-6-fewshot` (`au.anthropic.claude-sonnet-4-6` on Amazon Bedrock). Error rate on all queries (the matched risk): **10.27%**. In-scope accuracy 87.17%, OOS recall 97.60%, invalid replies 0. Measured cost US$0.00158 per call, so US$1.580 per 1,000 queries if every query goes to the LLM. API latency p50 683 ms, p95 876 ms (from Sydney).

## Policy v1 with the thresholds frozen on validation, mean ± std over seeds

Each router handles the queries at or above the threshold frozen on validation (`results/policy_v1.json`) and escalates the rest to S4. 1000 of the queries are out of scope; "OOS handled" counts those the router answered itself, the dangerous error. Router cost is taken as US$0 (local model); the saving is against sending every query to S4.

| system | seeds | coverage at frozen thresholds (%) | selective risk (%) | HANDLE precision (%) | HANDLE recall (%) | OOS handled (of n) | AURC (%) | cost per 1,000 queries (US$) | saving vs LLM only (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s0_tfidf | 5 | 72.32 ± 0.00 | 11.36 ± 0.00 | 88.64 ± 0.00 | 95.33 ± 0.00 | 162.0 | 7.98 ± 0.00 | 0.437 | 72.32 ± 0.00 |
| s1_distilroberta-base | 5 | 75.37 ± 0.80 | 10.50 ± 0.68 | 89.50 ± 0.68 | 97.36 ± 0.39 | 172.4 | 6.71 ± 0.12 | 0.389 | 75.37 ± 0.80 |
| s2_all-MiniLM-L6-v2 | 1 | 78.11 ± 0.00 | 10.99 ± 0.00 | 89.01 ± 0.00 | 99.96 ± 0.00 | 109.0 | 7.83 ± 0.00 | 0.346 | 78.11 ± 0.00 |
| s3_quora-distilroberta-base | 5 | 77.52 ± 0.43 | 12.02 ± 0.43 | 87.98 ± 0.43 | 99.71 ± 0.08 | 98.2 | 7.30 ± 0.12 | 0.355 | 77.52 ± 0.43 |

**Best router** (frozen on validation, README 7.2): `s2_all-MiniLM-L6-v2`. Highest coverage at matched risk on this split: `s2_all-MiniLM-L6-v2`.

## Coverage at fixed risk levels

| system | coverage at risk 1% (%) | coverage at risk 2% (%) | coverage at risk 5% (%) |
| --- | --- | --- | --- |
| s0_tfidf | 29.88 ± 0.00 | 46.14 ± 0.00 | 60.48 ± 0.00 |
| s1_distilroberta-base | 34.20 ± 3.56 | 50.54 ± 2.01 | 66.54 ± 0.60 |
| s2_all-MiniLM-L6-v2 | 3.09 ± 0.00 | 4.36 ± 0.00 | 64.28 ± 0.00 |
| s3_quora-distilroberta-base | 29.01 ± 2.92 | 42.60 ± 2.82 | 62.95 ± 1.41 |

## Coverage at matched risk, recomputed on this split (the RQ1 number)

The threshold is the one with the highest coverage whose selective risk is at or below S4's error rate on these queries. Stratified bootstrap of the seed-mean coverage, with the matched risk recomputed on every resample.

| system | coverage at matched risk (%) | 95% CI (seed mean) |
| --- | --- | --- |
| s0_tfidf | 70.65 ± 0.00 | [68.84, 72.52] |
| s1_distilroberta-base | 75.07 ± 0.20 | [73.78, 76.33] |
| s2_all-MiniLM-L6-v2 | 77.47 ± 0.00 | [76.44, 78.43] |
| s3_quora-distilroberta-base | 75.71 ± 0.20 | [74.33, 76.94] |

## Comparisons (paired, stratified bootstrap of coverage at matched risk)

The matched risk is recomputed on every resample. An interval that spans 0 means no clear difference.

| A | B | coverage A - B (points) | 95% CI | resamples |
| --- | --- | --- | --- | --- |
| s3_quora-distilroberta-base | s1_distilroberta-base | +0.64 | [-0.41, +1.67] | 1000 |
| s3_quora-distilroberta-base | s2_all-MiniLM-L6-v2 | -1.76 | [-2.69, -0.84] | 1000 |

## Calibration on test (in-scope rows, temperatures frozen on validation)

ECE with 15 bins, before and after the validation temperatures. Out of sample, so this is the honest calibration number.

| system | temperature (mean) | ECE before (%) | ECE after (%) |
| --- | --- | --- | --- |
| s0_tfidf | 0.835 | 4.03 | 2.07 |
| s1_distilroberta-base | 0.969 | 1.81 | 1.64 |
| s3_quora-distilroberta-base | not scaled | 5.69 | - |

## Latency (batch size 1, warm model, Apple M3 Pro, CPU only, 5 torch threads, 200 queries)

| system | p50 (ms) | p95 (ms) |
| --- | --- | --- |
| s0_tfidf | 0.3 | 0.3 |
| s1_distilroberta-base | 14.4 | 18.9 |
| s2_all-MiniLM-L6-v2 | 5.5 | 6.4 |
| s3_quora-distilroberta-base | 95.7 | 159.7 |

## Figure

Risk-coverage curves, first seed of each system, with the matched-risk operating points: `results/figures/e6_test_fmt_risk_coverage.png`
