# The stronger LLM bar on Banking77 test (post-hoc, run once)

Written by `scripts/evaluate_llm_bar_test.py`. Do not edit by hand. Plan: `results/plan_llm_bar_test.md`, committed with the frozen thresholds (`results/policy_llm_bar.json`) before the test run. No pre-registered verdict changes.

## Provenance

- Generated (UTC): 2026-11-24T08:16:37+00:00
- Git commit: `87fe6f264345ad738c3aeec0a8a25ea6e6e9b7e9`
- Seed: 0
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## The bar: S4 with the 10 most similar labelled train queries (4079 format-controlled test queries, 1000 out of scope)

Error rate 5.52% (the frozen few-shot S4: 10.27%). In-scope accuracy 93.96%, macro precision 94.30%, macro recall 93.96%, macro F1 93.95%. OOS recall 96.10%. Invalid replies 0. US$0.00161 per call, US$6.55 for this split.

## Coverage at this bar, recomputed on test

Means over seeds; the interval is a bootstrap of the seed mean with the bar recomputed on every resample.

| router | seeds | coverage (%) | 95% CI | in-scope handled (% of in-scope) | OOS handled |
| --- | --- | --- | --- | --- | --- |
| s0_tfidf | 5 | 61.90 | [59.74, 64.50] | 79.77 | 69.0 |
| s1_distilroberta-base | 5 | 67.65 | [66.12, 69.17] | 87.36 | 69.6 |
| s2_all-MiniLM-L6-v2 | 1 | 67.17 | [62.95, 69.80] | 88.83 | 5.0 |
| s3_quora-distilroberta-base | 5 | 65.35 | [61.62, 68.39] | 86.38 | 6.0 |

RQ1-style yardstick, post-hoc: the policy v1 best router (s2_all-MiniLM-L6-v2) covers 67.17% at this bar, against N = 60.00%. The pre-registered RQ1 verdict is unchanged.

## Thresholds frozen on validation at this bar, applied unchanged (end to end)

The router answers what it handles and the retrieval S4 answers the rest. Paired bootstrap of the error difference (pair minus LLM alone), 1,000 resamples stratified by in-scope and out-of-scope.

| router | coverage (%) | LLM calls | selective risk (%) | router + LLM wrong (%) | LLM alone wrong (%) | difference (points, 95% CI) | reading | OOS answered: pair / LLM alone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s0_tfidf | 63.13 | 1504 | 6.02 | 7.33 | 5.52 | 1.81 [1.25, 2.35] | more errors | 104.0 / 39 |
| s1_distilroberta-base | 69.33 | 1251 | 6.38 | 7.27 | 5.52 | 1.75 [1.28, 2.21] | more errors | 114.2 / 39 |
| s2_all-MiniLM-L6-v2 | 66.73 | 1357 | 5.47 | 6.01 | 5.52 | 0.49 [0.02, 0.96] | more errors | 39.0 / 39 |
| s3_quora-distilroberta-base | 69.95 | 1226 | 6.91 | 6.91 | 5.52 | 1.39 [0.80, 1.89] | more errors | 42.2 / 39 |
