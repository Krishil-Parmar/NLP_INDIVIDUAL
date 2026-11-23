# Pre-registered verdicts

Written by `scripts/report_verdicts.py` from the test result files. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-23T06:26:53+00:00
- Git commit: `703a0d9f933a537917d214f5c56dd4c9701beccb`
- Seed: None
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## Verdicts

| id | hypothesis | result | verdict |
| --- | --- | --- | --- |
| RQ1 | best router (s2_all-MiniLM-L6-v2) coverage at matched risk on Banking77 test >= 60% | 77.47% [76.44, 78.43] (matched risk 10.27%) | met |
| RQ2 | S3 minus S1 OOS AUROC on test >= 1 point | +3.08 points [+2.60, +3.61] | met |
| RQ3 | S3 within 5 accuracy points of a retrained softmax (E4) | E4 not run | not tested |
| RQ4 | system ranking stable between original and cleaned training labels | order S1 encoder > S2 embedding > S3 pairwise > S0 TF-IDF; training effect, macro-F1 points: S0 TF-IDF -0.01 [-0.38, +0.37]; S1 encoder -0.26 [-0.50, -0.02]; S2 embedding +0.28 [-0.04, +0.61]; S3 pairwise +0.30 [-0.11, +0.74] | met |
| RQ5 | best router flip rate under paraphrase < 10% (E5) | E5 not run | not tested |
| RQ6 | coverage drop from Banking77 test to the Jeff set <= 20 points | +75.80 points: Banking77 77.47% [76.44, 78.43], Jeff 1.67% [0.00, 6.67] (Jeff matched risk 0.56%) | not met |
| R4 | best router p95 latency <= 100 ms (Apple M3 Pro, CPU only, 5 torch threads, batch 1) | p95 6.4 ms over 200 queries | met |

RQ6's interval is the widest pair of the two coverage intervals (low = Banking77 low minus Jeff high), since the two test sets are independent; it is used only to label a verdict not clear-cut.
