# E8 development: candidate fixes for S2 on a small new intent set

Written by `scripts/develop_few_shot.py`. Do not edit by hand. **Post-hoc experiment, not pre-registered.** The pre-registered RQ6 verdict (not met) is unchanged.

## Provenance

- Generated (UTC): 2026-11-24T01:03:10+00:00
- Git commit: `0b3e92bf2fcf18d265af761d172710f8cfa0aa86`
- Seed: None
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## Setup

Variants: encoder (frozen MiniLM, or fine-tuned on the support set SetFit-style) x support (the 140 examples, or those plus the 14 intent definitions) x scoring (nearest-example cosine as S2, top-two margin, or a logistic-regression head). Fine-tuned variants: five seeds, mean ± standard deviation. Jeff development data: the old Jeff test split (180 queries, 40 out of scope), read during the diagnosis and therefore no longer used as a test. Matched risk: few-shot S4's error there, 0.56%.

Check: frozen/examples/nearest (S2 itself) reproduces the saved E7 numbers exactly.

## Jeff development data

| variant | tier | in-scope accuracy (%) | confidence AUROC (%) | OOS AUROC (%) | AURC (%) | coverage at 0.56% risk (%) | coverage at 1% (%) | coverage at 5% (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen/examples/nearest **(chosen)** | no training | 56.43 ± 0.00 | 54.29 | 90.93 | 42.15 ± 0.00 | 1.67 | 1.67 | 1.67 |
| frozen/examples/margin | no training | 56.43 ± 0.00 | 60.72 | 60.98 | 49.05 ± 0.00 | 0.00 | 0.00 | 0.00 |
| frozen/examples/head | head only | 78.57 ± 0.00 | 85.09 | 72.07 | 17.65 ± 0.00 | 14.44 | 14.44 | 16.67 |
| tuned/examples/nearest **(chosen)** | encoder fine-tune | 81.14 ± 1.47 | 79.19 | 88.79 | 14.17 ± 1.63 | 10.22 | 10.22 | 32.00 |
| tuned/examples/margin | encoder fine-tune | 81.14 ± 1.47 | 82.08 | 63.47 | 20.66 ± 0.48 | 8.89 | 8.89 | 9.89 |
| tuned/examples/head | encoder fine-tune | 83.86 ± 1.84 | 83.97 | 78.53 | 14.23 ± 0.35 | 14.89 | 14.89 | 18.56 |
| frozen/descriptions/nearest | no training | 56.43 ± 0.00 | 54.87 | 91.11 | 42.04 ± 0.00 | 1.67 | 1.67 | 1.67 |
| frozen/descriptions/margin | no training | 56.43 ± 0.00 | 61.71 | 62.50 | 46.54 ± 0.00 | 0.56 | 0.56 | 0.56 |
| frozen/descriptions/head **(chosen)** | head only | 79.29 ± 0.00 | 84.84 | 75.27 | 16.67 ± 0.00 | 15.00 | 15.00 | 16.11 |
| tuned/descriptions/nearest | encoder fine-tune | 81.00 ± 1.40 | 75.85 | 88.94 | 14.83 ± 1.30 | 8.33 | 8.33 | 22.44 |
| tuned/descriptions/margin | encoder fine-tune | 81.00 ± 1.40 | 80.34 | 65.62 | 19.97 ± 0.69 | 8.44 | 8.44 | 11.00 |
| tuned/descriptions/head | encoder fine-tune | 83.43 ± 0.53 | 84.05 | 79.98 | 13.80 ± 0.39 | 18.33 | 18.33 | 23.44 |

## Chosen variants (rule fixed in `configs/e8_few_shot.yaml` before this run)

Per tier, the lowest mean AURC; a simpler variant of the same tier within 0.5 AURC points wins instead.

| tier | chosen variant | dev AURC (%) |
| --- | --- | --- |
| no training | frozen/examples/nearest | 42.15 |
| head only | frozen/descriptions/head | 16.67 |
| encoder fine-tune | tuned/examples/nearest | 14.17 |

## Banking77 validation check (report only, not used for selection)

10 random support examples per intent, five draws (fine-tuning seed = draw). Matched risk: few-shot S4's validation error, 12.22%. Descriptions do not exist for Banking77, so only the examples support is run.

| variant | tier | in-scope accuracy (%) | confidence AUROC (%) | OOS AUROC (%) | AURC (%) | coverage at 12.22% risk (%) | coverage at 1% (%) | coverage at 5% (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen/examples/nearest | no training | 79.96 ± 1.37 | 76.79 | 99.74 | 14.34 ± 0.75 | 54.53 | 6.53 | 27.66 |
| frozen/examples/margin | no training | 79.96 ± 1.37 | 85.37 | 83.77 | 14.99 ± 0.69 | 52.56 | 9.97 | 36.79 |
| frozen/examples/head | head only | 80.24 ± 0.34 | 79.35 | 97.95 | 13.82 ± 0.42 | 56.16 | 14.41 | 28.92 |
| tuned/examples/nearest | encoder fine-tune | 79.16 ± 0.76 | 75.90 | 99.42 | 15.77 ± 0.39 | 53.20 | 2.10 | 8.89 |
| tuned/examples/margin | encoder fine-tune | 79.16 ± 0.76 | 76.37 | 70.32 | 21.54 ± 0.72 | 29.42 | 3.23 | 15.72 |
| tuned/examples/head | encoder fine-tune | 78.14 ± 0.84 | 83.00 | 98.27 | 14.15 ± 0.51 | 56.28 | 14.74 | 33.47 |
