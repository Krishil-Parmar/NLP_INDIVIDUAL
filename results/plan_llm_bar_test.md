# Plan: the stronger LLM bar on Banking77 test (post-hoc, run once)

Written 24 Nov 2026, before any test call of the retrieval S4 variant. `scripts/run_s4.py` (test split, retrieval config) and `scripts/evaluate_llm_bar_test.py` both refuse to run without this file.

**Status: post-hoc.** This plan is not part of `results/preregistration.md`, and no pre-registered verdict changes.

## Why

The second independent review found that the frozen LLM bar (few-shot S4: 77 intent names plus 2 examples per intent) is weakly prompted, while the routers learn from about 9,000 labelled queries. On validation (`results/llm_bar_val.md`), a retrieval-prompted S4 roughly halved the error rate (12.22% to 6.96%). Router coverage at that bar was 62% to 68%, and router plus LLM was about equal to the LLM alone. This run measures the same on test, once.

## Fixed before the run

- **The bar:** `configs/s4_llm_retrieval.yaml`, unchanged from its validation run.
  - Prompt: `prompts/s4_intent_retrieval_v1.txt`.
  - Examples: each query's 10 most similar train queries by S2's encoder (train split only), each with its label.
  - Settings: temperature 0, the same model.
- **The queries:** Banking77 `test_fmt` (4,079 format-controlled queries: 3,079 in scope, 1,000 CLINC150 out of scope), the same as E6. One call per query.
- **The frozen thresholds:** `results/policy_llm_bar.json`, written by `scripts/evaluate_llm_bar.py` on validation and committed with this plan.
  - Rule: per router and seed, the highest-coverage threshold whose selective risk is at or below the retrieval S4's validation error (6.96%).
  - Temperatures: those of policy v1.
- **The routers:** S0, S1 and S3 with seeds 0 to 4, and S2 with seed 0. These are the frozen systems whose test predictions already exist; no router is re-run.

## What is reported (`results/llm_bar_test.md`)

1. **The bar itself:** the retrieval S4's test error rate, in-scope accuracy, macro precision, recall and F1, OOS recall, invalid replies, and cost.
2. **Coverage, recomputed:** each router's coverage at the retrieval S4's test error rate. It is split into in-scope and out-of-scope handling, with a bootstrap interval in which the bar is recomputed on every resample (as for RQ1).
3. **End to end, at the frozen thresholds:** each router's coverage and selective risk with the frozen validation thresholds. Then the end-to-end error (the router answers what it handles, the retrieval S4 the rest) against the retrieval S4 alone, as a paired bootstrap of the difference (1,000 resamples, stratified by in scope and out of scope, seed 0).
   - **Read** as "fewer errors" if the whole 95% interval is below 0, "more errors" if it is above 0, and "no clear difference" otherwise.
4. **RQ1-style yardstick:** the policy v1 best router's (S2) recomputed coverage, against N = 60%. Reported for information only.

## Spend and stop rules

- **Budget:** about US$6.65 expected (4,079 calls at about US$0.00163). The retrieval config's hard cap is US$19.20 cumulative over all S4 caches: the old US$13.20 plus the US$6 the owner allowed on 24 Nov 2026. US$12.24 was spent before this run.
- **Stop rules:** stop and report if replies are more than 1% invalid, or if the cap stops the run. Never re-run because of a result.

## Run once

`scripts/evaluate_llm_bar_test.py` refuses to run again once `results/llm_bar_test.json` exists. Any bug found after that is logged below, with a note on what changed.

## Deviations

None so far.
