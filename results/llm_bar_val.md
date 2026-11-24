# How strict is the LLM bar? (validation only, post-hoc)

Written by `scripts/evaluate_llm_bar.py`. Do not edit by hand. Validation only: no test split is touched and no verdict changes. The routers' thresholds below are chosen on these same queries, so the coverage numbers are in-sample.

## Provenance

- Generated (UTC): 2026-11-24T06:44:34+00:00
- Git commit: `0099f96e0a2eecb9c5a9dc3dd1b65ec868ceb626`
- Seed: None
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## The three S4 variants (1350 format-controlled validation queries, 350 out of scope)

| S4 variant | error rate on the mix (%) | in-scope accuracy (%) | macro F1 (%) | OOS recall (%) | invalid | US$ per call | S2 right on its in-scope errors |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero-shot | 18.00 | 76.30 | 76.06 | 98.29 | 0 | 0.00079 | 180 of 237 |
| few-shot (frozen bar) | 12.22 | 84.30 | 84.91 | 97.71 | 0 | 0.00157 | 102 of 157 |
| retrieval (10 nearest train queries) | 6.96 | 91.90 | 92.11 | 96.29 | 0 | 0.00163 | 20 of 81 |

## Router coverage at each variant's error rate

For each S4 variant as the bar: each router's threshold is the one with the highest coverage whose selective risk is at or below that variant's error rate; then the router answers what it handles and that S4 variant answers the rest. Means over seeds.

| bar (S4 variant) | bar (%) | router | coverage (%) | in-scope handled (% of in-scope) | OOS handled | router + S4 wrong (%) | S4 alone wrong (%) | OOS answered: pair / S4 alone |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero-shot | 18.00 | s0_tfidf | 78.44 | 95.50 | 104.0 | 16.07 | 18.00 | 105.0 / 6 |
| zero-shot | 18.00 | s1_distilroberta-base | 80.27 | 98.18 | 101.8 | 15.23 | 18.00 | 104.4 / 6 |
| zero-shot | 18.00 | s2_all-MiniLM-L6-v2 | 82.00 | 100.00 | 107.0 | 14.74 | 18.00 | 107.0 / 6 |
| zero-shot | 18.00 | s3_quora-distilroberta-base | 81.60 | 99.90 | 102.6 | 14.81 | 18.00 | 103.8 / 6 |
| few-shot (frozen bar) | 12.22 | s0_tfidf | 70.74 | 89.40 | 61.0 | 12.00 | 12.22 | 64.0 / 8 |
| few-shot (frozen bar) | 12.22 | s1_distilroberta-base | 73.38 | 93.62 | 54.4 | 11.39 | 12.22 | 59.6 / 8 |
| few-shot (frozen bar) | 12.22 | s2_all-MiniLM-L6-v2 | 76.59 | 100.00 | 34.0 | 9.33 | 12.22 | 34.0 / 8 |
| few-shot (frozen bar) | 12.22 | s3_quora-distilroberta-base | 76.18 | 99.68 | 31.6 | 9.51 | 12.22 | 34.4 / 8 |
| retrieval (10 nearest train queries) | 6.96 | s0_tfidf | 62.07 | 81.30 | 25.0 | 8.59 | 6.96 | 35.0 / 13 |
| retrieval (10 nearest train queries) | 6.96 | s1_distilroberta-base | 66.68 | 87.38 | 26.4 | 8.53 | 6.96 | 36.6 / 13 |
| retrieval (10 nearest train queries) | 6.96 | s2_all-MiniLM-L6-v2 | 65.19 | 87.90 | 1.0 | 7.19 | 6.96 | 13.0 / 13 |
| retrieval (10 nearest train queries) | 6.96 | s3_quora-distilroberta-base | 68.19 | 91.80 | 2.6 | 7.17 | 6.96 | 13.6 / 13 |
