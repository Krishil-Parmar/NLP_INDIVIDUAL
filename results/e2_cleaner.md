# E2 step 1: label issues flagged by cleanlab

Written by `scripts/find_label_issues.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-22T11:32:01+00:00
- Git commit: `d7abe76a4617e8e246580f17f4f15d73159ba1a1`
- Seed: 0
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Dataset `banking77` revision: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`
- Dataset `clinc150` revision: `155b9c710419136e17307b80d0a13e68cd46b4ec`
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## Cleaner

Frozen `sentence-transformers/all-mpnet-base-v2` embeddings, then logistic regression with 5-fold stratified cross-validation (seed 0). Every row's probabilities come from the fold that did not see it. C kept: **100** (lowest CV log loss). Embedding took 15 s on mps.

| C | CV log loss | CV accuracy |
| --- | --- | --- |
| 0.1 | 2.0758 | 82.61% |
| 1 | 0.6325 | 90.79% |
| 10 | 0.3042 | 92.48% |
| 100 | 0.2682 | 92.40% |

## Flagged rows

cleanlab `find_label_issues`, `filter_by=prune_by_noise_rate`, ranked by `self_confidence`. The suggested label is the most probable label other than the given one.

| split | rows | flagged | share |
| --- | --- | --- | --- |
| train | 8992 | 163 | 1.81% |
| val | 1000 | 29 | 2.90% |

## Intents with the largest share of train rows flagged (top 15)

| intent | train rows flagged |
| --- | --- |
| top_up_reverted | 9.9% |
| balance_not_updated_after_bank_transfer | 6.5% |
| contactless_not_working | 6.2% |
| reverted_card_payment? | 6.2% |
| pending_top_up | 6.0% |
| declined_transfer | 5.8% |
| wrong_exchange_rate_for_cash_withdrawal | 5.4% |
| verify_my_identity | 5.3% |
| pending_transfer | 4.5% |
| failed_transfer | 4.1% |
| card_delivery_estimate | 4.0% |
| compromised_card | 3.9% |
| card_arrival | 3.6% |
| declined_card_payment | 3.6% |
| unable_to_verify_identity | 3.3% |

## Most frequent (given, suggested) label pairs (top 15)

| given label | suggested label | flagged |
| --- | --- | --- |
| top_up_reverted | top_up_failed | 8 |
| reverted_card_payment? | declined_card_payment | 7 |
| wrong_exchange_rate_for_cash_withdrawal | card_payment_wrong_exchange_rate | 7 |
| pending_top_up | top_up_failed | 5 |
| card_arrival | card_delivery_estimate | 5 |
| declined_transfer | failed_transfer | 5 |
| card_delivery_estimate | card_arrival | 5 |
| balance_not_updated_after_bank_transfer | pending_transfer | 4 |
| why_verify_identity | verify_my_identity | 4 |
| verify_my_identity | why_verify_identity | 4 |
| top_up_reverted | pending_top_up | 4 |
| failed_transfer | declined_transfer | 3 |
| transfer_not_received_by_recipient | failed_transfer | 3 |
| declined_card_payment | reverted_card_payment? | 3 |
| pending_top_up | topping_up_by_card | 3 |

## Stop checks before retraining

All passed.
