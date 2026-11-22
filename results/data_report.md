# Data report

Written by `scripts/make_splits.py`. Do not edit by hand.

## Provenance

- Generated (UTC): 2026-11-22T00:31:06+00:00
- Git commit: `5690964590dc980a83f3305694a98f11d9e4e608`
- Seed: 42
- Python 3.12.4 on macOS-27.0-arm64-arm-64bit
- Dataset `banking77` revision: `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`
- Dataset `clinc150` revision: `155b9c710419136e17307b80d0a13e68cd46b4ec`
- Packages: torch 2.15.0, transformers 5.24.0, datasets 5.2.0, sentence-transformers 6.3.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.5

## Final split sizes

| split | rows |
| --- | --- |
| train | 8992 |
| val | 1000 |
| test | 3079 |
| oos_val | 350 |
| oos_test | 1000 |

Validation is a stratified 10% split of the cleaned Banking77 train set (seed 42).

Out-of-scope share when mixed (step 8): validation 25.9% (350 OOS of 1350), test 24.5% (1000 OOS of 4079). Validation OOS comes from CLINC150 OOS train + validation, test OOS from CLINC150 OOS test.

## Rows removed, by step

Changed by normalisation (step 2, not removed): train 467, test 103.

| source | raw rows | empty | duplicate | conflict | leakage | in_scope_conflict |
| --- | --- | --- | --- | --- | --- | --- |
| banking77_train | 10003 | 0 | 4 | 0 | 7 | 0 |
| banking77_test | 3080 | 0 | 1 | 0 | 0 | 0 |
| clinc_oos_train | 250 | 0 | 0 | 0 | 0 | 0 |
| clinc_oos_validation | 100 | 0 | 0 | 0 | 0 | 0 |
| clinc_oos_test | 1000 | 0 | 0 | 0 | 0 | 0 |

Train/test leakage (step 4): 7 distinct texts appeared in both train and test after normalisation. They were removed from train only. Every removed row is listed in `data/processed/removed_rows.csv` with its reason.

## Token lengths (step 6, measured on train)

`max_length` is the 99th percentile of train lengths rounded up to a multiple of 8. "Pair" is a query plus a random other train query, which is what the S3 cross-encoder sees. Full numbers are in `data/processed/token_lengths.json`.

| tokeniser | mean | p50 | p95 | p99 | max | max_length | truncated train / val / test | pair max_length (S3) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `distilbert/distilroberta-base` | 15.8 | 13 | 36 | 50 | 93 | 56 | 0.56% / 0.40% / 0.29% | 80 |
| `microsoft/deberta-v3-small` | 16.0 | 13 | 36 | 52 | 98 | 56 | 0.66% / 0.50% / 0.39% | 80 |
| `sentence-transformers/all-MiniLM-L6-v2` | 16.2 | 13 | 37 | 53 | 98 | 56 | 0.67% / 0.60% / 0.39% | 80 |

## Class balance (step 7)

Queries per intent: train min 32, median 114, max 168; val min 3, median 13, max 19; test min 39, median 40, max 40.

The intents are uneven in train, so macro averages (each intent counts equally) are the primary metric (README section 8.1).

<details><summary>Full table (77 intents, fewest train queries first)</summary>

| intent | train | val | test |
| --- | --- | --- | --- |
| contactless_not_working | 32 | 3 | 40 |
| virtual_card_not_working | 37 | 4 | 40 |
| card_acceptance | 51 | 6 | 40 |
| card_swallowed | 55 | 6 | 40 |
| lost_or_stolen_card | 74 | 8 | 40 |
| atm_support | 77 | 8 | 39 |
| compromised_card | 77 | 8 | 40 |
| receiving_money | 86 | 9 | 40 |
| get_disposable_virtual_card | 87 | 10 | 40 |
| top_up_limits | 87 | 10 | 40 |
| getting_virtual_card | 88 | 10 | 40 |
| unable_to_verify_identity | 92 | 10 | 40 |
| topping_up_by_card | 93 | 10 | 40 |
| verify_my_identity | 94 | 10 | 40 |
| passcode_forgotten | 94 | 11 | 40 |
| get_physical_card | 95 | 11 | 40 |
| terminate_account | 97 | 11 | 40 |
| age_limit | 99 | 11 | 40 |
| top_up_by_bank_transfer_charge | 100 | 11 | 40 |
| card_not_working | 100 | 11 | 40 |
| card_delivery_estimate | 101 | 11 | 40 |
| exchange_rate | 101 | 11 | 40 |
| transfer_into_account | 102 | 11 | 40 |
| verify_source_of_funds | 102 | 11 | 40 |
| pin_blocked | 102 | 11 | 40 |
| top_up_by_card_charge | 103 | 11 | 40 |
| top_up_by_cash_or_cheque | 103 | 11 | 40 |
| exchange_via_app | 106 | 12 | 40 |
| order_physical_card | 108 | 12 | 40 |
| change_pin | 108 | 12 | 40 |
| lost_or_stolen_phone | 109 | 12 | 40 |
| exchange_charge | 109 | 12 | 40 |
| disposable_card_limits | 109 | 12 | 40 |
| why_verify_identity | 109 | 12 | 40 |
| edit_personal_details | 109 | 12 | 40 |
| fiat_currency_support | 113 | 13 | 40 |
| apple_pay_or_google_pay | 113 | 13 | 40 |
| verify_top_up | 113 | 13 | 40 |
| automatic_top_up | 114 | 13 | 40 |
| transfer_timing | 115 | 13 | 40 |
| country_support | 115 | 13 | 40 |
| card_about_to_expire | 116 | 13 | 40 |
| getting_spare_card | 116 | 13 | 40 |
| supported_cards_and_currencies | 116 | 13 | 40 |
| declined_transfer | 120 | 13 | 40 |
| visa_or_mastercard | 121 | 14 | 40 |
| failed_transfer | 123 | 14 | 40 |
| card_linking | 125 | 14 | 40 |
| pending_cash_withdrawal | 129 | 14 | 40 |
| top_up_failed | 130 | 15 | 40 |
| top_up_reverted | 131 | 15 | 40 |
| pending_transfer | 133 | 15 | 40 |
| pending_top_up | 134 | 15 | 40 |
| declined_card_payment | 138 | 15 | 40 |
| card_arrival | 138 | 15 | 40 |
| beneficiary_not_allowed | 140 | 16 | 40 |
| cancel_transfer | 141 | 16 | 40 |
| activate_my_card | 143 | 16 | 40 |
| pending_card_payment | 143 | 16 | 40 |
| cash_withdrawal_not_recognised | 144 | 16 | 40 |
| reverted_card_payment? | 145 | 16 | 40 |
| Refund_not_showing_up | 146 | 16 | 40 |
| wrong_exchange_rate_for_cash_withdrawal | 147 | 16 | 40 |
| extra_charge_on_statement | 149 | 17 | 40 |
| card_payment_wrong_exchange_rate | 150 | 17 | 40 |
| card_payment_not_recognised | 151 | 17 | 40 |
| request_refund | 152 | 17 | 40 |
| transfer_not_received_by_recipient | 154 | 17 | 40 |
| balance_not_updated_after_bank_transfer | 154 | 17 | 40 |
| transfer_fee_charged | 155 | 17 | 40 |
| declined_cash_withdrawal | 156 | 17 | 40 |
| transaction_charged_twice | 157 | 18 | 40 |
| cash_withdrawal_charge | 159 | 18 | 40 |
| wrong_amount_of_cash_received | 162 | 18 | 40 |
| balance_not_updated_after_cheque_or_cash_deposit | 163 | 18 | 40 |
| direct_debit_payment_not_recognised | 164 | 18 | 40 |
| card_payment_fee_charged | 168 | 19 | 40 |

</details>
