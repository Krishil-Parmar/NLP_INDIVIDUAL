# E1 results on val_fmt

Written by `scripts/evaluate.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-22T07:34:09+00:00
- Git commit: `d80b6e8710707b0837d7120fee1e4524c6c68f60`
- Seed: None
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Dataset `banking77` revision: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`
- Dataset `clinc150` revision: `155b9c710419136e17307b80d0a13e68cd46b4ec`
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5

## In-scope classification (val_fmt), percent, mean ± std over seeds

**Format-controlled text:** queries were lowercased and stripped of punctuation before prediction (the E3 control, `router.preprocess.format_control`). The models were trained on the raw text.

Macro averages are primary (each of the 77 intents counts equally). Std is the sample standard deviation across seeds.

| system | seeds | accuracy | macro_precision | macro_recall | macro_f1 | weighted_precision | weighted_recall | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s0_tfidf | 5 | 89.10 ± 0.00 | 89.98 ± 0.00 | 89.03 ± 0.00 | 88.95 ± 0.00 | 89.86 ± 0.00 | 89.10 ± 0.00 | 89.01 ± 0.00 |
| s1_distilroberta-base | 5 | 89.40 ± 1.01 | 90.47 ± 0.87 | 89.56 ± 0.94 | 89.57 ± 0.96 | 90.11 ± 0.94 | 89.40 ± 1.01 | 89.35 ± 1.03 |
| s2_all-MiniLM-L6-v2 | 1 | 90.80 ± 0.00 | 91.60 ± 0.00 | 91.28 ± 0.00 | 91.04 ± 0.00 | 91.43 ± 0.00 | 90.80 ± 0.00 | 90.73 ± 0.00 |
| s2_all-mpnet-base-v2 | 1 | 90.90 ± 0.00 | 91.67 ± 0.00 | 91.33 ± 0.00 | 91.06 ± 0.00 | 91.52 ± 0.00 | 90.90 ± 0.00 | 90.78 ± 0.00 |
| s3_quora-distilroberta-base | 5 | 90.38 ± 0.26 | 91.19 ± 0.24 | 90.75 ± 0.21 | 90.58 ± 0.21 | 91.12 ± 0.26 | 90.38 ± 0.26 | 90.38 ± 0.26 |

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
| topping_up_by_card | 0.651 | 0.740 | 0.691 | 10 |
| transfer_not_received_by_recipient | 0.742 | 0.659 | 0.695 | 17 |
| pending_top_up | 0.758 | 0.707 | 0.731 | 15 |
| balance_not_updated_after_bank_transfer | 0.758 | 0.753 | 0.755 | 17 |
| pin_blocked | 0.821 | 0.745 | 0.781 | 11 |
| transfer_timing | 0.888 | 0.723 | 0.796 | 13 |
| why_verify_identity | 0.834 | 0.767 | 0.797 | 12 |
| country_support | 0.746 | 0.892 | 0.809 | 13 |
| pending_transfer | 0.767 | 0.867 | 0.813 | 15 |
| card_not_working | 0.916 | 0.745 | 0.821 | 11 |

Most frequent confusions (true intent predicted as another, counts summed over seeds):

| true | predicted | count |
| --- | --- | --- |
| card_about_to_expire | country_support | 15 |
| why_verify_identity | verify_my_identity | 13 |
| card_payment_wrong_exchange_rate | wrong_exchange_rate_for_cash_withdrawal | 12 |
| fiat_currency_support | exchange_via_app | 12 |
| transfer_not_received_by_recipient | balance_not_updated_after_bank_transfer | 12 |
| declined_transfer | failed_transfer | 10 |
| pending_top_up | top_up_failed | 10 |
| top_up_by_bank_transfer_charge | top_up_by_card_charge | 10 |
| balance_not_updated_after_bank_transfer | pending_transfer | 9 |
| pin_blocked | change_pin | 9 |

## s2_all-MiniLM-L6-v2

10 intents with the lowest F1 (averaged over seeds):

| intent | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| transfer_not_received_by_recipient | 0.667 | 0.471 | 0.552 | 17 |
| balance_not_updated_after_bank_transfer | 0.609 | 0.824 | 0.700 | 17 |
| pending_top_up | 1.000 | 0.600 | 0.750 | 15 |
| topping_up_by_card | 0.727 | 0.800 | 0.762 | 10 |
| order_physical_card | 0.818 | 0.750 | 0.783 | 12 |
| compromised_card | 0.857 | 0.750 | 0.800 | 8 |
| declined_transfer | 0.833 | 0.769 | 0.800 | 13 |
| get_physical_card | 0.889 | 0.727 | 0.800 | 11 |
| lost_or_stolen_card | 0.667 | 1.000 | 0.800 | 8 |
| pin_blocked | 0.889 | 0.727 | 0.800 | 11 |

Most frequent confusions (true intent predicted as another, counts summed over seeds):

| true | predicted | count |
| --- | --- | --- |
| transfer_not_received_by_recipient | balance_not_updated_after_bank_transfer | 5 |
| balance_not_updated_after_bank_transfer | transfer_not_received_by_recipient | 2 |
| balance_not_updated_after_cheque_or_cash_deposit | wrong_amount_of_cash_received | 2 |
| card_arrival | card_delivery_estimate | 2 |
| card_payment_wrong_exchange_rate | wrong_exchange_rate_for_cash_withdrawal | 2 |
| cash_withdrawal_not_recognised | lost_or_stolen_card | 2 |
| compromised_card | lost_or_stolen_card | 2 |
| declined_transfer | failed_transfer | 2 |
| get_physical_card | change_pin | 2 |
| pending_top_up | top_up_failed | 2 |

## s2_all-mpnet-base-v2

10 intents with the lowest F1 (averaged over seeds):

| intent | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| transfer_not_received_by_recipient | 0.750 | 0.529 | 0.621 | 17 |
| compromised_card | 0.667 | 0.750 | 0.706 | 8 |
| pending_top_up | 0.900 | 0.600 | 0.720 | 15 |
| balance_not_updated_after_bank_transfer | 0.684 | 0.765 | 0.722 | 17 |
| pin_blocked | 0.875 | 0.636 | 0.737 | 11 |
| topping_up_by_card | 0.778 | 0.700 | 0.737 | 10 |
| card_payment_not_recognised | 0.923 | 0.706 | 0.800 | 17 |
| lost_or_stolen_card | 0.667 | 1.000 | 0.800 | 8 |
| pending_transfer | 0.812 | 0.867 | 0.839 | 15 |
| transfer_timing | 0.846 | 0.846 | 0.846 | 13 |

Most frequent confusions (true intent predicted as another, counts summed over seeds):

| true | predicted | count |
| --- | --- | --- |
| transfer_not_received_by_recipient | balance_not_updated_after_bank_transfer | 4 |
| balance_not_updated_after_bank_transfer | pending_transfer | 3 |
| pending_top_up | top_up_failed | 3 |
| wrong_exchange_rate_for_cash_withdrawal | card_payment_wrong_exchange_rate | 3 |
| card_arrival | card_delivery_estimate | 2 |
| card_payment_not_recognised | compromised_card | 2 |
| card_payment_not_recognised | direct_debit_payment_not_recognised | 2 |
| compromised_card | lost_or_stolen_card | 2 |
| declined_transfer | failed_transfer | 2 |
| pin_blocked | change_pin | 2 |

## s3_quora-distilroberta-base

10 intents with the lowest F1 (averaged over seeds):

| intent | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| transfer_not_received_by_recipient | 0.805 | 0.529 | 0.638 | 17 |
| balance_not_updated_after_bank_transfer | 0.631 | 0.824 | 0.714 | 17 |
| compromised_card | 0.814 | 0.725 | 0.765 | 8 |
| topping_up_by_card | 0.674 | 0.900 | 0.770 | 10 |
| why_verify_identity | 0.752 | 0.850 | 0.797 | 12 |
| order_physical_card | 0.814 | 0.800 | 0.806 | 12 |
| lost_or_stolen_card | 0.704 | 0.950 | 0.808 | 8 |
| pin_blocked | 0.911 | 0.727 | 0.808 | 11 |
| declined_transfer | 0.810 | 0.846 | 0.827 | 13 |
| failed_transfer | 0.800 | 0.857 | 0.828 | 14 |

Most frequent confusions (true intent predicted as another, counts summed over seeds):

| true | predicted | count |
| --- | --- | --- |
| transfer_not_received_by_recipient | balance_not_updated_after_bank_transfer | 22 |
| balance_not_updated_after_bank_transfer | pending_transfer | 14 |
| compromised_card | lost_or_stolen_card | 11 |
| pending_top_up | topping_up_by_card | 11 |
| transfer_not_received_by_recipient | transfer_timing | 11 |
| card_payment_wrong_exchange_rate | wrong_exchange_rate_for_cash_withdrawal | 10 |
| declined_transfer | failed_transfer | 10 |
| card_arrival | card_delivery_estimate | 9 |
| pin_blocked | change_pin | 9 |
| unable_to_verify_identity | why_verify_identity | 9 |
