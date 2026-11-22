# E1 results on val

Written by `scripts/evaluate.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-22T01:35:35+00:00
- Git commit: `ad405abe5e09fbf82d3cf8aa7d0cc1dcefe109ab`
- Seed: None
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Dataset `banking77` revision: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`
- Dataset `clinc150` revision: `155b9c710419136e17307b80d0a13e68cd46b4ec`
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5

## In-scope classification (val), percent, mean ± std over seeds

Macro averages are primary (each of the 77 intents counts equally). Std is the sample standard deviation across seeds.

| system | seeds | accuracy | macro_precision | macro_recall | macro_f1 | weighted_precision | weighted_recall | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s0_tfidf | 5 | 89.10 ± 0.00 | 89.98 ± 0.00 | 89.03 ± 0.00 | 88.95 ± 0.00 | 89.86 ± 0.00 | 89.10 ± 0.00 | 89.01 ± 0.00 |
| s1_distilroberta-base | 5 | 91.10 ± 0.47 | 92.07 ± 0.48 | 91.47 ± 0.41 | 91.46 ± 0.44 | 91.62 ± 0.49 | 91.10 ± 0.47 | 91.06 ± 0.49 |

## s0_tfidf

10 intents with the lowest F1 (averaged over seeds):

| intent | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| virtual_card_not_working | 1.000 | 0.500 | 0.667 | 4 |
| transfer_into_account | 0.643 | 0.818 | 0.720 | 11 |
| transfer_timing | 0.750 | 0.692 | 0.720 | 13 |
| balance_not_updated_after_bank_transfer | 0.750 | 0.706 | 0.727 | 17 |
| card_not_working | 0.727 | 0.727 | 0.727 | 11 |
| card_payment_wrong_exchange_rate | 0.857 | 0.706 | 0.774 | 17 |
| compromised_card | 0.700 | 0.875 | 0.778 | 8 |
| top_up_by_bank_transfer_charge | 1.000 | 0.636 | 0.778 | 11 |
| topping_up_by_card | 0.875 | 0.700 | 0.778 | 10 |
| verify_my_identity | 0.692 | 0.900 | 0.783 | 10 |

Most frequent confusions (true intent predicted as another, counts summed over seeds):

| true | predicted | count |
| --- | --- | --- |
| card_payment_wrong_exchange_rate | wrong_exchange_rate_for_cash_withdrawal | 15 |
| top_up_by_bank_transfer_charge | top_up_by_card_charge | 15 |
| balance_not_updated_after_bank_transfer | balance_not_updated_after_cheque_or_cash_deposit | 10 |
| balance_not_updated_after_bank_transfer | pending_transfer | 10 |
| card_payment_not_recognised | direct_debit_payment_not_recognised | 10 |
| cash_withdrawal_not_recognised | compromised_card | 10 |
| declined_transfer | failed_transfer | 10 |
| pending_card_payment | declined_card_payment | 10 |
| pending_top_up | top_up_failed | 10 |
| pin_blocked | change_pin | 10 |

## s1_distilroberta-base

10 intents with the lowest F1 (averaged over seeds):

| intent | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| topping_up_by_card | 0.695 | 0.760 | 0.724 | 10 |
| pending_top_up | 0.780 | 0.707 | 0.741 | 15 |
| transfer_not_received_by_recipient | 0.879 | 0.671 | 0.760 | 17 |
| balance_not_updated_after_bank_transfer | 0.776 | 0.800 | 0.787 | 17 |
| pin_blocked | 0.838 | 0.745 | 0.789 | 11 |
| top_up_by_bank_transfer_charge | 0.978 | 0.727 | 0.834 | 11 |
| why_verify_identity | 0.861 | 0.817 | 0.835 | 12 |
| declined_transfer | 0.897 | 0.785 | 0.836 | 13 |
| pending_transfer | 0.806 | 0.880 | 0.841 | 15 |
| balance_not_updated_after_cheque_or_cash_deposit | 0.795 | 0.900 | 0.844 | 18 |

Most frequent confusions (true intent predicted as another, counts summed over seeds):

| true | predicted | count |
| --- | --- | --- |
| pending_top_up | top_up_failed | 13 |
| card_payment_wrong_exchange_rate | wrong_exchange_rate_for_cash_withdrawal | 11 |
| transfer_not_received_by_recipient | balance_not_updated_after_bank_transfer | 11 |
| why_verify_identity | verify_my_identity | 11 |
| declined_transfer | failed_transfer | 10 |
| top_up_by_bank_transfer_charge | top_up_by_card_charge | 10 |
| wrong_exchange_rate_for_cash_withdrawal | card_payment_wrong_exchange_rate | 10 |
| balance_not_updated_after_bank_transfer | pending_transfer | 9 |
| pin_blocked | change_pin | 9 |
| card_payment_not_recognised | direct_debit_payment_not_recognised | 8 |
