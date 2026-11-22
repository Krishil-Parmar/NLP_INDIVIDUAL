# E1 results on val

Written by `scripts/evaluate.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-22T02:22:26+00:00
- Git commit: `fcc372772ea990a2b76b60fe8a274494c0215739`
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
| s2_all-MiniLM-L6-v2 | 1 | 91.80 ± 0.00 | 92.66 ± 0.00 | 92.24 ± 0.00 | 92.11 ± 0.00 | 92.44 ± 0.00 | 91.80 ± 0.00 | 91.78 ± 0.00 |
| s2_all-mpnet-base-v2 | 1 | 91.30 ± 0.00 | 92.22 ± 0.00 | 91.66 ± 0.00 | 91.49 ± 0.00 | 92.01 ± 0.00 | 91.30 ± 0.00 | 91.21 ± 0.00 |
| s3_quora-distilroberta-base | 5 | 11.76 ± 3.69 | 18.15 ± 4.32 | 12.59 ± 4.13 | 13.18 ± 4.13 | 17.43 ± 3.84 | 11.76 ± 3.69 | 12.47 ± 3.74 |

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

## s2_all-MiniLM-L6-v2

10 intents with the lowest F1 (averaged over seeds):

| intent | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| transfer_not_received_by_recipient | 0.714 | 0.588 | 0.645 | 17 |
| balance_not_updated_after_bank_transfer | 0.636 | 0.824 | 0.718 | 17 |
| topping_up_by_card | 0.727 | 0.800 | 0.762 | 10 |
| order_physical_card | 0.818 | 0.750 | 0.783 | 12 |
| compromised_card | 0.857 | 0.750 | 0.800 | 8 |
| get_physical_card | 0.889 | 0.727 | 0.800 | 11 |
| lost_or_stolen_card | 0.667 | 1.000 | 0.800 | 8 |
| pending_top_up | 1.000 | 0.667 | 0.800 | 15 |
| pin_blocked | 0.889 | 0.727 | 0.800 | 11 |
| card_payment_not_recognised | 0.929 | 0.765 | 0.839 | 17 |

Most frequent confusions (true intent predicted as another, counts summed over seeds):

| true | predicted | count |
| --- | --- | --- |
| transfer_not_received_by_recipient | balance_not_updated_after_bank_transfer | 5 |
| balance_not_updated_after_bank_transfer | transfer_not_received_by_recipient | 2 |
| balance_not_updated_after_cheque_or_cash_deposit | wrong_amount_of_cash_received | 2 |
| cash_withdrawal_not_recognised | lost_or_stolen_card | 2 |
| compromised_card | lost_or_stolen_card | 2 |
| declined_transfer | failed_transfer | 2 |
| get_physical_card | change_pin | 2 |
| pending_top_up | top_up_failed | 2 |
| pending_top_up | topping_up_by_card | 2 |
| pin_blocked | change_pin | 2 |

## s2_all-mpnet-base-v2

10 intents with the lowest F1 (averaged over seeds):

| intent | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| transfer_not_received_by_recipient | 0.727 | 0.471 | 0.571 | 17 |
| compromised_card | 0.625 | 0.625 | 0.625 | 8 |
| balance_not_updated_after_bank_transfer | 0.650 | 0.765 | 0.703 | 17 |
| pending_top_up | 1.000 | 0.600 | 0.750 | 15 |
| lost_or_stolen_card | 0.615 | 1.000 | 0.762 | 8 |
| topping_up_by_card | 0.727 | 0.800 | 0.762 | 10 |
| card_payment_not_recognised | 0.923 | 0.706 | 0.800 | 17 |
| pending_transfer | 0.765 | 0.867 | 0.812 | 15 |
| failed_transfer | 0.800 | 0.857 | 0.828 | 14 |
| declined_transfer | 0.909 | 0.769 | 0.833 | 13 |

Most frequent confusions (true intent predicted as another, counts summed over seeds):

| true | predicted | count |
| --- | --- | --- |
| transfer_not_received_by_recipient | balance_not_updated_after_bank_transfer | 5 |
| balance_not_updated_after_bank_transfer | pending_transfer | 3 |
| compromised_card | lost_or_stolen_card | 3 |
| card_arrival | card_delivery_estimate | 2 |
| card_payment_not_recognised | compromised_card | 2 |
| card_payment_not_recognised | direct_debit_payment_not_recognised | 2 |
| declined_transfer | failed_transfer | 2 |
| fiat_currency_support | exchange_via_app | 2 |
| pending_top_up | top_up_failed | 2 |
| pending_top_up | topping_up_by_card | 2 |

## s3_quora-distilroberta-base

10 intents with the lowest F1 (averaged over seeds):

| intent | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| balance_not_updated_after_bank_transfer | 0.000 | 0.000 | 0.000 | 17 |
| balance_not_updated_after_cheque_or_cash_deposit | 0.000 | 0.000 | 0.000 | 18 |
| beneficiary_not_allowed | 0.000 | 0.000 | 0.000 | 16 |
| card_acceptance | 0.000 | 0.000 | 0.000 | 6 |
| card_arrival | 0.000 | 0.000 | 0.000 | 15 |
| card_not_working | 0.000 | 0.000 | 0.000 | 11 |
| card_payment_wrong_exchange_rate | 0.000 | 0.000 | 0.000 | 17 |
| card_swallowed | 0.000 | 0.000 | 0.000 | 6 |
| country_support | 0.000 | 0.000 | 0.000 | 13 |
| declined_transfer | 0.000 | 0.000 | 0.000 | 13 |

Most frequent confusions (true intent predicted as another, counts summed over seeds):

| true | predicted | count |
| --- | --- | --- |
| card_payment_wrong_exchange_rate | receiving_money | 43 |
| Refund_not_showing_up | wrong_amount_of_cash_received | 39 |
| wrong_exchange_rate_for_cash_withdrawal | receiving_money | 37 |
| change_pin | card_swallowed | 31 |
| request_refund | wrong_amount_of_cash_received | 28 |
| pin_blocked | card_swallowed | 26 |
| top_up_reverted | verify_source_of_funds | 26 |
| transfer_fee_charged | card_payment_wrong_exchange_rate | 26 |
| card_payment_fee_charged | card_payment_wrong_exchange_rate | 25 |
| exchange_charge | top_up_by_card_charge | 23 |
