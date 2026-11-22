# E2 results on val: label noise

Written by `scripts/evaluate_label_noise.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-22T23:18:24+00:00
- Git commit: `f559503d309fecdb8b8984656c21186153df781b`
- Seed: None
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Dataset `banking77` revision: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`
- Dataset `clinc150` revision: `155b9c710419136e17307b80d0a13e68cd46b4ec`
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## Setup

cleanlab flagged 163 train rows and 29 of the 1000 in-scope val rows (`results/e2_cleaner.md`). "Cleaned" means those rows are removed, never relabelled. Cleaned-label systems keep every other choice frozen. Flagged is not the same as wrong: the audit below measures how often a flag is a real error.

## In-scope macro-F1 (val, raw text), percent, mean ± std over seeds

| system | seeds | original train, all | cleaned train, all | original train, flagged removed | cleaned train, flagged removed |
| --- | --- | --- | --- | --- | --- |
| S0 TF-IDF | 5 | 88.95 ± 0.00 | 87.76 ± 0.00 | 90.69 ± 0.00 | 89.70 ± 0.00 |
| S1 encoder | 5 | 91.46 ± 0.44 | 91.58 ± 0.61 | 93.65 ± 0.50 | 93.77 ± 0.56 |
| S2 embedding | 1 | 92.11 ± 0.00 | 92.65 ± 0.00 | 94.04 ± 0.00 | 94.53 ± 0.00 |
| S3 pairwise | 5 | 90.52 ± 0.31 | 91.30 ± 0.45 | 92.49 ± 0.30 | 93.25 ± 0.50 |

## In-scope accuracy, same cells

| system | original train, all | cleaned train, all | original train, flagged removed | cleaned train, flagged removed |
| --- | --- | --- | --- | --- |
| S0 TF-IDF | 89.10 ± 0.00 | 88.00 ± 0.00 | 90.94 ± 0.00 | 90.01 ± 0.00 |
| S1 encoder | 91.10 ± 0.47 | 91.28 ± 0.53 | 93.37 ± 0.50 | 93.55 ± 0.48 |
| S2 embedding | 91.80 ± 0.00 | 92.40 ± 0.00 | 93.82 ± 0.00 | 94.34 ± 0.00 |
| S3 pairwise | 90.22 ± 0.26 | 90.98 ± 0.48 | 92.30 ± 0.27 | 93.00 ± 0.53 |

## Rankings

| cell | ranking (best first, by mean macro-F1) |
| --- | --- |
| original train, all | S2 embedding > S1 encoder > S3 pairwise > S0 TF-IDF |
| cleaned train, all | S2 embedding > S1 encoder > S3 pairwise > S0 TF-IDF |
| original train, flagged removed | S2 embedding > S1 encoder > S3 pairwise > S0 TF-IDF |
| cleaned train, flagged removed | S2 embedding > S1 encoder > S3 pairwise > S0 TF-IDF |

**Ranking stable in all four cells: yes.**

## Training effect: cleaned minus original train labels, all rows

Paired bootstrap, stratified by intent, 1000 resamples, 95% percentile interval.

| system | cleaned - original macro-F1 (points) | 95% CI |
| --- | --- | --- |
| S0 TF-IDF | -1.19 | [-2.22, -0.12] |
| S1 encoder | +0.12 | [-0.39, +0.65] |
| S2 embedding | +0.54 | [+0.01, +1.13] |
| S3 pairwise | +0.79 | [+0.16, +1.48] |

## S4 evaluation effect (format-controlled validation, cached replies, no API calls)

The error rate is on the mixed set (in-scope plus out-of-scope), the same quantity E6 uses as the matched risk. No flagged row is out of scope, so removing flagged rows only removes in-scope rows.

| system | rows (all / kept) | error rate, all | error rate, flagged removed | in-scope accuracy, all | in-scope accuracy, flagged removed |
| --- | --- | --- | --- | --- | --- |
| S4 zero-shot | 1350 / 1321 | 18.00% | 16.81% | 76.30% | 77.75% |
| S4 few-shot | 1350 / 1321 | 12.22% | 10.90% | 84.30% | 85.99% |

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
