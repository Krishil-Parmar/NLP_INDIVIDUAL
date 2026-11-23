# E1 results on test

Written by `scripts/evaluate.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-23T06:23:56+00:00
- Git commit: `703a0d9f933a537917d214f5c56dd4c9701beccb`
- Seed: None
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Dataset `banking77` revision: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`
- Dataset `clinc150` revision: `155b9c710419136e17307b80d0a13e68cd46b4ec`
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5, cleanlab 2.9.0

## In-scope classification (test), percent, mean ± std over seeds

Macro averages are primary (each of the 77 intents counts equally). Std is the sample standard deviation across seeds.

| system | seeds | accuracy | macro_precision | macro_recall | macro_f1 | weighted_precision | weighted_recall | weighted_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| s0_tfidf | 5 | 89.09 ± 0.00 | 89.61 ± 0.00 | 89.09 ± 0.00 | 89.11 ± 0.00 | 89.61 ± 0.00 | 89.09 ± 0.00 | 89.11 ± 0.00 |
| s1_distilroberta-base | 5 | 93.14 ± 0.16 | 93.41 ± 0.16 | 93.14 ± 0.16 | 93.13 ± 0.17 | 93.40 ± 0.16 | 93.14 ± 0.16 | 93.13 ± 0.17 |
| s2_all-MiniLM-L6-v2 | 1 | 91.98 ± 0.00 | 92.13 ± 0.00 | 91.98 ± 0.00 | 91.96 ± 0.00 | 92.13 ± 0.00 | 91.98 ± 0.00 | 91.96 ± 0.00 |
| s3_quora-distilroberta-base | 5 | 90.93 ± 0.22 | 91.08 ± 0.21 | 90.94 ± 0.22 | 90.89 ± 0.22 | 91.08 ± 0.21 | 90.93 ± 0.22 | 90.89 ± 0.22 |

## s0_tfidf

10 intents with the lowest F1 (averaged over seeds):

| intent | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| balance_not_updated_after_bank_transfer | 0.646 | 0.775 | 0.705 | 40 |
| card_not_working | 0.643 | 0.900 | 0.750 | 40 |
| pending_transfer | 0.844 | 0.675 | 0.750 | 40 |
| wrong_exchange_rate_for_cash_withdrawal | 0.806 | 0.725 | 0.763 | 40 |
| verify_my_identity | 0.811 | 0.750 | 0.779 | 40 |
| top_up_failed | 0.780 | 0.800 | 0.790 | 40 |
| failed_transfer | 0.739 | 0.850 | 0.791 | 40 |
| card_acceptance | 0.879 | 0.725 | 0.795 | 40 |
| why_verify_identity | 0.786 | 0.825 | 0.805 | 40 |
| topping_up_by_card | 0.906 | 0.725 | 0.806 | 40 |

Most frequent confusions (true intent predicted as another, counts summed over seeds):

| true | predicted | count |
| --- | --- | --- |
| card_acceptance | card_not_working | 35 |
| verify_my_identity | why_verify_identity | 35 |
| virtual_card_not_working | getting_virtual_card | 25 |
| card_payment_not_recognised | extra_charge_on_statement | 20 |
| fiat_currency_support | exchange_via_app | 20 |
| pending_transfer | balance_not_updated_after_bank_transfer | 20 |
| pin_blocked | get_physical_card | 20 |
| top_up_reverted | top_up_failed | 20 |
| transfer_not_received_by_recipient | balance_not_updated_after_bank_transfer | 20 |
| unable_to_verify_identity | verify_my_identity | 20 |

## s1_distilroberta-base

10 intents with the lowest F1 (averaged over seeds):

| intent | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| balance_not_updated_after_bank_transfer | 0.800 | 0.790 | 0.794 | 40 |
| topping_up_by_card | 0.863 | 0.785 | 0.821 | 40 |
| pending_transfer | 0.858 | 0.820 | 0.838 | 40 |
| transfer_into_account | 0.878 | 0.825 | 0.851 | 40 |
| declined_transfer | 0.981 | 0.755 | 0.853 | 40 |
| failed_transfer | 0.816 | 0.900 | 0.855 | 40 |
| why_verify_identity | 0.969 | 0.780 | 0.864 | 40 |
| declined_card_payment | 0.804 | 0.945 | 0.869 | 40 |
| transfer_not_received_by_recipient | 0.884 | 0.865 | 0.873 | 40 |
| direct_debit_payment_not_recognised | 0.919 | 0.840 | 0.877 | 40 |

Most frequent confusions (true intent predicted as another, counts summed over seeds):

| true | predicted | count |
| --- | --- | --- |
| why_verify_identity | verify_my_identity | 36 |
| balance_not_updated_after_bank_transfer | transfer_timing | 19 |
| card_arrival | card_delivery_estimate | 19 |
| card_delivery_estimate | card_arrival | 15 |
| reverted_card_payment? | declined_card_payment | 15 |
| top_up_reverted | top_up_failed | 15 |
| declined_transfer | declined_card_payment | 14 |
| get_disposable_virtual_card | getting_virtual_card | 14 |
| pending_transfer | balance_not_updated_after_bank_transfer | 14 |
| wrong_amount_of_cash_received | declined_cash_withdrawal | 14 |

## s2_all-MiniLM-L6-v2

10 intents with the lowest F1 (averaged over seeds):

| intent | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| pending_transfer | 0.811 | 0.750 | 0.779 | 40 |
| card_payment_not_recognised | 0.853 | 0.725 | 0.784 | 40 |
| balance_not_updated_after_bank_transfer | 0.750 | 0.825 | 0.786 | 40 |
| transfer_not_received_by_recipient | 0.833 | 0.750 | 0.789 | 40 |
| top_up_failed | 0.857 | 0.750 | 0.800 | 40 |
| topping_up_by_card | 0.800 | 0.800 | 0.800 | 40 |
| compromised_card | 0.838 | 0.775 | 0.805 | 40 |
| top_up_reverted | 0.810 | 0.850 | 0.829 | 40 |
| transfer_fee_charged | 0.833 | 0.875 | 0.854 | 40 |
| transfer_into_account | 0.892 | 0.825 | 0.857 | 40 |

Most frequent confusions (true intent predicted as another, counts summed over seeds):

| true | predicted | count |
| --- | --- | --- |
| top_up_failed | top_up_reverted | 6 |
| card_arrival | card_delivery_estimate | 5 |
| card_delivery_estimate | card_arrival | 4 |
| card_payment_not_recognised | compromised_card | 4 |
| declined_transfer | declined_card_payment | 4 |
| fiat_currency_support | exchange_via_app | 4 |
| pending_transfer | balance_not_updated_after_bank_transfer | 4 |
| why_verify_identity | verify_my_identity | 4 |
| Refund_not_showing_up | request_refund | 3 |
| compromised_card | lost_or_stolen_card | 3 |

## s3_quora-distilroberta-base

10 intents with the lowest F1 (averaged over seeds):

| intent | precision | recall | f1 | support |
| --- | --- | --- | --- | --- |
| balance_not_updated_after_bank_transfer | 0.695 | 0.740 | 0.717 | 40 |
| compromised_card | 0.793 | 0.710 | 0.749 | 40 |
| card_payment_not_recognised | 0.847 | 0.675 | 0.750 | 40 |
| pending_transfer | 0.801 | 0.720 | 0.758 | 40 |
| transfer_not_received_by_recipient | 0.820 | 0.750 | 0.783 | 40 |
| top_up_reverted | 0.852 | 0.735 | 0.788 | 40 |
| card_arrival | 0.857 | 0.750 | 0.800 | 40 |
| topping_up_by_card | 0.829 | 0.795 | 0.811 | 40 |
| verify_my_identity | 0.812 | 0.840 | 0.825 | 40 |
| top_up_failed | 0.837 | 0.815 | 0.825 | 40 |

Most frequent confusions (true intent predicted as another, counts summed over seeds):

| true | predicted | count |
| --- | --- | --- |
| card_arrival | card_delivery_estimate | 39 |
| pending_transfer | balance_not_updated_after_bank_transfer | 28 |
| why_verify_identity | verify_my_identity | 25 |
| top_up_reverted | top_up_failed | 24 |
| compromised_card | lost_or_stolen_card | 23 |
| card_payment_not_recognised | compromised_card | 21 |
| fiat_currency_support | exchange_via_app | 18 |
| verify_my_identity | why_verify_identity | 18 |
| balance_not_updated_after_bank_transfer | pending_transfer | 17 |
| exchange_via_app | fiat_currency_support | 17 |
