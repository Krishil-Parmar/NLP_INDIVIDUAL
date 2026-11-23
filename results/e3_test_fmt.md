# E3 results on test_fmt

Written by `scripts/evaluate_oos.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-23T06:24:23+00:00
- Git commit: `703a0d9f933a537917d214f5c56dd4c9701beccb`
- Seed: 0
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Dataset `banking77` revision: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`
- Dataset `clinc150` revision: `155b9c710419136e17307b80d0a13e68cd46b4ec`
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## Out-of-scope detection (test_fmt), percent, mean ± std over seeds

**Format-controlled text:** every query (in-scope and OOS) was lowercased and stripped of punctuation before prediction (`router.preprocess.format_control`), so formatting cannot give the answer away. These are the primary E3 numbers.

3079 in-scope Banking77 queries and 1000 CLINC150 out-of-scope queries. AUROC: OOS is the positive class, higher is better. FPR at 95% TPR: the threshold keeps at least 95% of in-scope queries; FPR is the share of OOS queries also kept, lower is better . Std is the sample standard deviation across seeds.

**Limitation:** the OOS queries come from a different dataset (CLINC150) than the in-scope ones, so rejection looks easier than in real use even with formatting controlled (topics and length still differ). The Jeff set's finance-adjacent OOS queries are the harder test.

| system | seeds | AUROC | FPR at 95% TPR | OOS recall (val threshold) | in-scope kept (val threshold) | threshold (mean) |
| --- | --- | --- | --- | --- | --- | --- |
| s0_tfidf | 5 | 95.54 ± 0.00 | 20.30 ± 0.00 | 79.60 ± 0.00 | 95.06 ± 0.00 | 0.6560 |
| s1_distilroberta-base | 5 | 96.40 ± 0.20 | 18.60 ± 1.20 | 79.90 ± 1.01 | 95.47 ± 0.22 | 0.4473 |
| s2_all-MiniLM-L6-v2 | 1 | 99.70 ± 0.00 | 0.70 ± 0.00 | 99.30 ± 0.00 | 94.87 ± 0.00 | 0.2548 |
| s3_quora-distilroberta-base | 5 | 99.49 ± 0.00 | 1.48 ± 0.13 | 98.46 ± 0.13 | 95.13 ± 0.26 | 0.4267 |
| s4_claude-sonnet-4-6-fewshot | 1 | 98.65 ± 0.00 | 2.40 ± 0.00 | 97.60 ± 0.00 | 99.71 ± 0.00 | 0.0000 |

## Surface cues (no model)

AUROC of a trivial formatting rule on its own, on the text the systems saw. 50 means the rule carries no signal.

| rule (fires = looks like CLINC150) | in-scope share | OOS share | AUROC of the rule alone |
| --- | --- | --- | --- |
| starts lowercase | 100.0 | 100.0 | 50.0 |
| no final punctuation | 100.0 | 100.0 | 50.0 |
| no comma | 100.0 | 100.0 | 50.0 |

## Comparisons (paired, stratified bootstrap)

Each system's AUROC is averaged over its seeds; in-scope and OOS queries are resampled separately, with the same rows for both systems. An interval that spans 0 means no clear difference.

| A | B | AUROC A - B (points) | 95% CI | resamples |
| --- | --- | --- | --- | --- |
| s3_quora-distilroberta-base | s1_distilroberta-base | +3.08 | [+2.60, +3.61] | 1000 |
| s3_quora-distilroberta-base | s2_all-MiniLM-L6-v2 | -0.21 | [-0.31, -0.13] | 1000 |
| s3_quora-distilroberta-base | s0_tfidf | +3.95 | [+3.32, +4.60] | 1000 |

## OOS queries s3_quora-distilroberta-base accepted most confidently (first seed)

The lowest OOS scores among out-of-scope queries: the misses to read first.

| text | pred_label | confidence | matched_example |
| --- | --- | --- | --- |
| can i deposit a check | top_up_by_cash_or_cheque | 0.999 | Can I deposit a check? |
| i need to update my address | edit_personal_details | 0.998 | I need to update my address. |
| is there a way to change your houses address | edit_personal_details | 0.988 | I need to change my home address. |
| can you tell me what credit cards i have open | supported_cards_and_currencies | 0.975 | What other credit cards can I use? |
| how long do i have to wait to get my refund | Refund_not_showing_up | 0.966 | How long will it take to refund me |
| please take all my money out of my checking account and close the account | terminate_account | 0.955 | Help me terminate and close my account. |
| do you know if it is possible to close my savings account | terminate_account | 0.909 | How can I close my account? |
| can i use direct deposit for my tax refund | direct_debit_payment_not_recognised | 0.901 | I want a refund on a direct debit. |
| how do i make money | receiving_money | 0.876 | How can I receive money? |
| i have too many credit cards is there any way you can help me organize them | supported_cards_and_currencies | 0.855 | What other credit cards can I use? |

## Figure

ROC curves, first seed of each system: `results/figures/e3_test_fmt_roc.png`
