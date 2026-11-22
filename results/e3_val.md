# E3 results on val

Written by `scripts/evaluate_oos.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-22T07:33:49+00:00
- Git commit: `d80b6e8710707b0837d7120fee1e4524c6c68f60`
- Seed: 0
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Dataset `banking77` revision: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`
- Dataset `clinc150` revision: `155b9c710419136e17307b80d0a13e68cd46b4ec`
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5

## Out-of-scope detection (val), percent, mean ± std over seeds

**Raw text.** Formatting separates the two sources on its own (surface cues below), so these numbers overstate OOS detection. The primary numbers are in `results/e3_val_fmt.md`.

1000 in-scope Banking77 queries and 350 CLINC150 out-of-scope queries. AUROC: OOS is the positive class, higher is better. FPR at 95% TPR: the threshold keeps at least 95% of in-scope queries; FPR is the share of OOS queries also kept, lower is better (so OOS recall at that threshold is 100 minus this number). Std is the sample standard deviation across seeds.

**Limitation:** the OOS queries come from a different dataset (CLINC150) than the in-scope ones, so rejection looks easier than in real use even with formatting controlled (topics and length still differ). The Jeff set's finance-adjacent OOS queries are the harder test.

| system | seeds | AUROC | FPR at 95% TPR | threshold (mean) |
| --- | --- | --- | --- | --- |
| s0_tfidf | 5 | 94.80 ± 0.00 | 24.57 ± 0.00 | 0.6573 |
| s1_distilroberta-base | 5 | 97.38 ± 0.10 | 15.31 ± 2.00 | 0.4178 |
| s2_all-MiniLM-L6-v2 | 1 | 99.92 ± 0.00 | 0.57 ± 0.00 | 0.2468 |
| s2_all-mpnet-base-v2 | 1 | 99.94 ± 0.00 | 0.29 ± 0.00 | 0.2559 |
| s3_quora-distilroberta-base | 5 | 99.70 ± 0.04 | 1.54 ± 0.38 | 0.5048 |

## Surface cues (no model)

AUROC of a trivial formatting rule on its own, on the text the systems saw. 50 means the rule carries no signal.

| rule (fires = looks like CLINC150) | in-scope share | OOS share | AUROC of the rule alone |
| --- | --- | --- | --- |
| starts lowercase | 5.8 | 100.0 | 97.1 |
| no final punctuation | 13.4 | 100.0 | 93.3 |
| no comma | 86.4 | 99.4 | 56.5 |

## Comparisons (paired, stratified bootstrap)

Each system's AUROC is averaged over its seeds; in-scope and OOS queries are resampled separately, with the same rows for both systems. An interval that spans 0 means no clear difference.

| A | B | AUROC A - B (points) | 95% CI | resamples |
| --- | --- | --- | --- | --- |
| s3_quora-distilroberta-base | s1_distilroberta-base | +2.32 | [+1.78, +2.95] | 1000 |
| s3_quora-distilroberta-base | s2_all-MiniLM-L6-v2 | -0.22 | [-0.37, -0.11] | 1000 |
| s3_quora-distilroberta-base | s0_tfidf | +4.90 | [+3.64, +6.34] | 1000 |

## OOS queries s3_quora-distilroberta-base accepted most confidently (first seed)

The lowest OOS scores among out-of-scope queries: the misses to read first.

| text | pred_label | confidence | matched_example |
| --- | --- | --- | --- |
| can you tell me who else is on my account | topping_up_by_card | 0.968 | Can somebody else top up my account? |
| how do i change my billing address | edit_personal_details | 0.954 | My billing address changed and I need to update my information. |
| which nations were on each side of the conflict in world war ii | country_support | 0.850 | Which countries are represented? |
| how do you store my card information | compromised_card | 0.554 | My card data has been exposed. |
| can you show me my credit card number | supported_cards_and_currencies | 0.542 | I need to know if you will accept my credit card. |
| what is the account number | cancel_transfer | 0.540 | I used the wrong account number! |
| can you restart my phone | lost_or_stolen_phone | 0.421 | I no longer have my phone. |
| what are the pros and cons of credit cards | supported_cards_and_currencies | 0.409 | What other credit cards can I use? |
| is the pnc bank open | change_pin | 0.357 | Can I change my PIN at my local bank? |
| who is the oscar who "the oscar's" is named after | edit_personal_details | 0.327 | How do I change my name? |

## Figure

ROC curves, first seed of each system: `results/figures/e3_val_roc.png`
