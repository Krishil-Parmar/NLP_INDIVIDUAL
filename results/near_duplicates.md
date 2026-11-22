# Near-duplicate check

Written by `scripts/near_duplicates.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-22T02:06:57+00:00
- Git commit: `fcc372772ea990a2b76b60fe8a274494c0215739`
- Seed: None
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5

## Similarity of each query to its nearest train query

Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (cosine similarity of unit-length embeddings). Train is the support set. Exact duplicates were already removed in M0, so these are near-paraphrases.

| split | queries | median cosine | share with cosine >= 0.99 | share with cosine >= 0.95 | share with cosine >= 0.9 |
| --- | --- | --- | --- | --- | --- |
| val | 1000 | 0.911 | 2.2% | 26.1% | 56.2% |
| test | 3079 | 0.917 | 2.9% | 29.0% | 58.5% |

Validation and test have similar near-duplicate rates, so validation is a fair proxy for test when tuning thresholds. Near-duplicates are a property of Banking77 itself. They favour nearest-neighbour systems and are reported as a limitation, not removed (owner decision, 22 Nov 2026).
