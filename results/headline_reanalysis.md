# Headline re-analysis (post-hoc)

Written by `scripts/reanalyse_headline.py`. Do not edit by hand. Reads saved predictions only: no new prediction, no tuning, and no pre-registered verdict changes. Prompted by an independent critical review on 24 Nov 2026.

## Provenance

- Generated (UTC): 2026-11-24T02:47:40+00:00
- Git commit: `460cf1997a776aef06b4f5fb71be103b6d8b46d9` (uncommitted changes present)
- Seed: 0
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## A. What coverage at matched risk is made of (Banking77 test_fmt)

3079 in-scope and 1000 out-of-scope queries: in-scope share 75.48%. S4's in-scope accuracy is 87.17%, below every router's (`results/e1_test.md`), so at S4's overall error rate a router can handle almost every in-scope query. Coverage above the in-scope share can only come from handling out-of-scope queries, which are always errors under policy v1 (it has no REJECT). Means over seeds.

| system | bar | coverage (%) | in-scope queries handled (% of in-scope) | out-of-scope handled (of 1,000) |
| --- | --- | --- | --- | --- |
| s0_tfidf | 1% risk | 29.88 | 39.36 | 7.0 |
| s0_tfidf | 2% risk | 46.14 | 60.51 | 19.0 |
| s0_tfidf | 5% risk | 60.48 | 78.24 | 58.0 |
| s0_tfidf | 10% risk | 70.14 | 88.47 | 137.0 |
| s0_tfidf | matched risk (10.27%) | 70.65 | 88.96 | 143.0 |
| s0_tfidf | frozen policy v1 | 72.32 | 90.55 | 162.0 |
| s1_distilroberta-base | 1% risk | 34.20 | 45.11 | 6.2 |
| s1_distilroberta-base | 2% risk | 50.54 | 66.28 | 20.6 |
| s1_distilroberta-base | 5% risk | 66.54 | 86.15 | 61.6 |
| s1_distilroberta-base | 10% risk | 74.73 | 93.77 | 161.2 |
| s1_distilroberta-base | matched risk (10.27%) | 75.07 | 94.02 | 167.2 |
| s1_distilroberta-base | frozen policy v1 | 75.37 | 94.25 | 172.4 |
| s2_all-MiniLM-L6-v2 | 1% risk | 3.09 | 4.06 | 1.0 |
| s2_all-MiniLM-L6-v2 | 2% risk | 4.36 | 5.75 | 1.0 |
| s2_all-MiniLM-L6-v2 | 5% risk | 64.28 | 85.06 | 3.0 |
| s2_all-MiniLM-L6-v2 | 10% risk | 77.25 | 99.90 | 75.0 |
| s2_all-MiniLM-L6-v2 | matched risk (10.27%) | 77.47 | 99.90 | 84.0 |
| s2_all-MiniLM-L6-v2 | frozen policy v1 | 78.11 | 99.94 | 109.0 |
| s3_quora-distilroberta-base | 1% risk | 29.01 | 38.37 | 1.8 |
| s3_quora-distilroberta-base | 2% risk | 42.60 | 56.38 | 2.0 |
| s3_quora-distilroberta-base | 5% risk | 62.95 | 83.22 | 5.6 |
| s3_quora-distilroberta-base | 10% risk | 75.39 | 98.36 | 46.6 |
| s3_quora-distilroberta-base | matched risk (10.27%) | 75.71 | 98.63 | 51.4 |
| s3_quora-distilroberta-base | frozen policy v1 | 77.52 | 99.51 | 98.2 |

Check: the coverage column reproduces `results/e6_test_fmt.json` exactly.

## B. Router plus LLM end to end, at the frozen policy v1 thresholds

The router answers what it handles and S4 answers the rest. Paired bootstrap of the error difference, 1,000 resamples stratified by in-scope and out-of-scope. Negative = fewer errors than S4 alone. Means over seeds.

| system | LLM calls | wrong answers | error rate (%) | in-scope wrong | out-of-scope answered | error minus S4 alone (points, 95% CI) |
| --- | --- | --- | --- | --- | --- | --- |
| S4 alone | 4079 | 419.0 | 10.27 | 395.0 | 24.0 |  |
| s0_tfidf + S4 | 1129 | 444.0 | 10.89 | 269.0 | 175.0 | 0.61 [-0.42, 1.64] |
| s1_distilroberta-base + S4 | 1005 | 409.2 | 10.03 | 224.8 | 184.4 | -0.24 [-1.07, 0.67] |
| s2_all-MiniLM-L6-v2 + S4 | 893 | 353.0 | 8.65 | 242.0 | 111.0 | -1.62 [-2.70, -0.64] |
| s3_quora-distilroberta-base + S4 | 917 | 394.6 | 9.67 | 287.8 | 106.8 | -0.60 [-1.65, 0.40] |

## C. S2's in-scope accuracy by distance to its nearest labelled example

S2's confidence is the cosine to the nearest support example. Banking77's support is its train split; the Jeff sets' support is the 140 Jeff support examples.

| data | cosine to nearest example | queries | accuracy (%) | 95% Wilson CI |
| --- | --- | --- | --- | --- |
| Banking77 test | 0.0 to 0.7 | 72 | 58.33 | [46.81, 69.01] |
| Banking77 test | 0.7 to 0.8 | 292 | 76.37 | [71.17, 80.88] |
| Banking77 test | 0.8 to 0.9 | 978 | 91.00 | [89.04, 92.64] |
| Banking77 test | 0.9 to 1.0 | 1737 | 96.83 | [95.90, 97.56] |
| Jeff test (E7) | 0.0 to 0.7 | 124 | 54.84 | [46.07, 63.32] |
| Jeff test (E7) | 0.7 to 0.8 | 14 | 64.29 | [38.76, 83.66] |
| Jeff test (E7) | 0.8 to 0.9 | 2 | 100.00 | [34.24, 100.00] |
| Jeff test (E7) | 0.9 to 1.0 | 0 | n/a |  |
| Jeff test v2 (E8 control) | 0.0 to 0.7 | 121 | 67.77 | [59.01, 75.44] |
| Jeff test v2 (E8 control) | 0.7 to 0.8 | 16 | 75.00 | [50.50, 89.82] |
| Jeff test v2 (E8 control) | 0.8 to 0.9 | 3 | 100.00 | [43.85, 100.00] |
| Jeff test v2 (E8 control) | 0.9 to 1.0 | 0 | n/a |  |

## D. RQ6 at one common risk bar

The pre-registered RQ6 drop (75.80 points) compares coverage at two different bars (S4's error: 10.27% on Banking77, 0.56% on Jeff). Here both sides use the same bar. The RQ6 verdict (not met) is unchanged.

| risk bar (%) | S2 Banking77 test (%) | S2 Jeff test (%) | drop (points) |
| --- | --- | --- | --- |
| 1 | 3.09 | 1.67 | 1.42 |
| 2 | 4.36 | 1.67 | 2.70 |
| 5 | 64.28 | 1.67 | 62.61 |
| 10 | 77.25 | 1.67 | 75.58 |
| 15 | 81.81 | 1.67 | 80.14 |

## E. Pipeline bug: out-of-scope queries that duplicate a Banking77 query after format control

`scripts/make_splits.py` removed out-of-scope queries that also appear in Banking77 by comparing normalised raw text. CLINC150 is lowercase and unpunctuated, so a case or punctuation difference hid the match until format control. Validation pool: 0 []. Test pool: 2 ['can i deposit a check', 'i need to update my address']. S2's confidence rank of the test duplicates (1 = most confident of 4,079): [621, 15]. The splits stay as they are (test sets are not edited after the run); this sensitivity check drops the test duplicates for information only.

| system | risk bar (%) | coverage (%) | without the duplicated rows (%) |
| --- | --- | --- | --- |
| s0_tfidf | 1 | 29.88 | 35.20 |
| s0_tfidf | 2 | 46.14 | 46.38 |
| s0_tfidf | 5 | 60.48 | 60.68 |
| s1_distilroberta-base | 1 | 34.20 | 38.62 |
| s1_distilroberta-base | 2 | 50.54 | 51.36 |
| s1_distilroberta-base | 5 | 66.54 | 66.68 |
| s2_all-MiniLM-L6-v2 | 1 | 3.09 | 3.95 |
| s2_all-MiniLM-L6-v2 | 2 | 4.36 | 4.95 |
| s2_all-MiniLM-L6-v2 | 5 | 64.28 | 64.58 |
| s3_quora-distilroberta-base | 1 | 29.01 | 31.58 |
| s3_quora-distilroberta-base | 2 | 42.60 | 43.65 |
| s3_quora-distilroberta-base | 5 | 62.95 | 63.28 |

## F. S2's most confident errors on Banking77 test

Errors among S2's 300 most confident test queries: 10 (1 out of scope). The first 10, by confidence rank. The predicted intent is always the label of the nearest train example, so a near-copy with a different label is a confident error.

| rank | cosine | query | true | predicted | nearest train example |
| --- | --- | --- | --- | --- | --- |
| 15 | 0.994 | can i deposit a check | oos | top_up_by_cash_or_cheque | Can I deposit a check? |
| 127 | 0.985 | how many transactions can i make with a disposable card | get_disposable_virtual_card | disposable_card_limits | How many transactions can I make on my disposable card? |
| 163 | 0.982 | when will my transfer go through | pending_transfer | balance_not_updated_after_bank_transfer | When will my transfer come through? |
| 179 | 0.981 | why was my payment declined | declined_card_payment | declined_transfer | Why did my payment get declined? |
| 196 | 0.981 | how do i top up my card | transfer_into_account | topping_up_by_card | How do I top up on my card? |
| 204 | 0.98 | why did my top up not work | top_up_failed | top_up_reverted | Why didn't my top up work? |
| 238 | 0.978 | how can i top up | topping_up_by_card | transfer_into_account | How do I top up? |
| 279 | 0.976 | why hasn't my transfer gone through yet | pending_transfer | failed_transfer | Why hasn't my transfer gone through |
| 285 | 0.976 | why didn't the atm give me my money | declined_cash_withdrawal | wrong_amount_of_cash_received | Why didn't the ATM give me all my money |
| 295 | 0.975 | im not sure what this charge is for | card_payment_fee_charged | direct_debit_payment_not_recognised | i do not know what this charge is for |
