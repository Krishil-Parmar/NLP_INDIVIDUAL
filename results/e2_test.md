# E2 results on test: label noise

Written by `scripts/evaluate_label_noise.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-23T06:24:41+00:00
- Git commit: `703a0d9f933a537917d214f5c56dd4c9701beccb`
- Seed: None
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Dataset `banking77` revision: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`
- Dataset `clinc150` revision: `155b9c710419136e17307b80d0a13e68cd46b4ec`
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## Setup

cleanlab flagged 163 train rows and 0 of the 3079 in-scope test rows (`results/e2_cleaner.md`). "Cleaned" means those rows are removed, never relabelled. Cleaned-label systems keep every other choice frozen. Flagged is not the same as wrong: the audit below measures how often a flag is a real error.

## In-scope macro-F1 (test, raw text), percent, mean ± std over seeds

| system | seeds | original train, all | cleaned train, all |
| --- | --- | --- | --- |
| S0 TF-IDF | 5 | 89.11 ± 0.00 | 89.10 ± 0.00 |
| S1 encoder | 5 | 93.13 ± 0.17 | 92.87 ± 0.23 |
| S2 embedding | 1 | 91.96 ± 0.00 | 92.25 ± 0.00 |
| S3 pairwise | 5 | 90.89 ± 0.22 | 91.19 ± 0.20 |

## In-scope accuracy, same cells

| system | original train, all | cleaned train, all |
| --- | --- | --- |
| S0 TF-IDF | 89.09 ± 0.00 | 89.09 ± 0.00 |
| S1 encoder | 93.14 ± 0.16 | 92.87 ± 0.22 |
| S2 embedding | 91.98 ± 0.00 | 92.27 ± 0.00 |
| S3 pairwise | 90.93 ± 0.22 | 91.21 ± 0.20 |

## Rankings

| cell | ranking (best first, by mean macro-F1) |
| --- | --- |
| original train, all | S1 encoder > S2 embedding > S3 pairwise > S0 TF-IDF |
| cleaned train, all | S1 encoder > S2 embedding > S3 pairwise > S0 TF-IDF |

**Ranking stable in all 2 cells: yes.**

## Training effect: cleaned minus original train labels, all rows

Paired bootstrap, stratified by intent, 1000 resamples, 95% percentile interval.

| system | cleaned - original macro-F1 (points) | 95% CI |
| --- | --- | --- |
| S0 TF-IDF | -0.01 | [-0.38, +0.37] |
| S1 encoder | -0.26 | [-0.50, -0.02] |
| S2 embedding | +0.28 | [-0.04, +0.61] |
| S3 pairwise | +0.30 | [-0.11, +0.74] |

No test row was checked for label errors (cleanlab checked train and validation only), so every row is scored against its original label and the "flagged removed" cells are not reported. S4 has no training data, so it has no cleaned version.

Few-shot S4 prompt examples that are flagged train rows: 2 of 154.

## Audit (blind)

Who judged each item is in the `annotator` column of `results/e2_audit.csv` ("owner" or "claude"). The Claude judge saw only the sheet (query, two options in random order, three examples each), never the key or the owner's verdicts; its verdicts on every item are in `results/e2_audit_claude.csv`. README section 8.1 has the procedure.

| annotator | items | real errors | suggestion fits | neither fits | ambiguous | false flags | real-error share (95% Wilson) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| claude | 89 | 53 | 50 | 3 | 9 | 27 | 60% (49 to 69%) |
| owner | 11 | 8 | 7 | 1 | 1 | 2 | 73% (43 to 90%) |

**All items together:** 61 of 100 are real label errors (61%, 95% Wilson interval 51 to 70%): 57 where the suggested label fits better, 4 where neither fits. Ambiguous (both fit): 10. False flags (the given label fits): 29.

**Owner and Claude judge on the same 11 items:** same verdict on 8 of 11, same scored category on 8 of 11.

| item | owner | Claude judge |
| --- | --- | --- |
| 5 | a (false flag) | b (real error) |
| 7 | both (ambiguous) | a (real error) |
| 11 | b (real error) | both (ambiguous) |
