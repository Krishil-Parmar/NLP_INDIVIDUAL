"""Unit tests for the E3 out-of-scope metrics and the paired bootstrap, on hand-computed toy data."""

import numpy as np
import pytest

from router.metrics import fpr_at_tpr, in_scope_threshold, oos_auroc, paired_bootstrap_diff, rates_at_threshold

IS_OOS = np.array([False, False, True, True])


def test_auroc_perfect_reversed_and_tied():
    assert oos_auroc(IS_OOS, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0  # every OOS scores above every in-scope
    assert oos_auroc(IS_OOS, np.array([0.8, 0.9, 0.1, 0.2])) == 0.0  # every OOS scores below
    assert oos_auroc(IS_OOS, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5  # all ties count half


def test_auroc_counts_pairs():
    # (OOS, in-scope) pairs: 0.5>0.1 yes, 0.5>0.6 no, 0.9>0.1 yes, 0.9>0.6 yes -> 3 of 4
    assert oos_auroc(IS_OOS, np.array([0.1, 0.6, 0.5, 0.9])) == pytest.approx(0.75)


# 20 in-scope queries scoring 0.01, 0.02, ..., 0.20 and 5 OOS queries.
IN_SCORES = np.arange(1, 21) / 100
OOS_SCORES = np.array([0.05, 0.15, 0.19, 0.30, 0.50])
TOY_IS_OOS = np.array([False] * 20 + [True] * 5)
TOY_SCORES = np.concatenate([IN_SCORES, OOS_SCORES])


def test_threshold_keeps_at_least_the_requested_share():
    # Keeping 95% of 20 means 19 queries, so the threshold is the 19th lowest score.
    assert in_scope_threshold(IN_SCORES, keep=0.95) == pytest.approx(0.19)
    assert in_scope_threshold(IN_SCORES, keep=0.50) == pytest.approx(0.10)
    # 55% of 100 is exactly 55, even though 0.55 * 100 is 55.00000000000001 in floating point.
    assert in_scope_threshold(np.arange(1, 101) / 100, keep=0.55) == pytest.approx(0.55)


def test_rates_at_threshold():
    rates = rates_at_threshold(TOY_IS_OOS, TOY_SCORES, threshold=0.19)
    assert rates["in_scope_kept"] == pytest.approx(19 / 20)
    assert rates["oos_recall"] == pytest.approx(2 / 5)  # 0.30 and 0.50 are above 0.19, so rejected


def test_fpr_at_tpr():
    result = fpr_at_tpr(TOY_IS_OOS, TOY_SCORES, keep=0.95)
    assert result["threshold"] == pytest.approx(0.19)
    assert result["fpr"] == pytest.approx(3 / 5)  # 0.05, 0.15 and 0.19 are kept although out of scope
    assert result["in_scope_kept"] == pytest.approx(0.95)


def test_bootstrap_identical_systems_give_zero():
    strata = TOY_IS_OOS
    metric = lambda rows: oos_auroc(TOY_IS_OOS[rows], TOY_SCORES[rows])  # noqa: E731
    result = paired_bootstrap_diff(metric, metric, strata, n_resamples=200, seed=0)
    assert result["diff"] == 0.0 and result["ci_low"] == 0.0 and result["ci_high"] == 0.0


def test_bootstrap_is_repeatable_with_a_seed():
    noisy = np.concatenate([IN_SCORES[::-1], OOS_SCORES])  # a worse system to compare against
    metric_a = lambda rows: oos_auroc(TOY_IS_OOS[rows], TOY_SCORES[rows])  # noqa: E731
    metric_b = lambda rows: oos_auroc(TOY_IS_OOS[rows], noisy[rows])  # noqa: E731
    first = paired_bootstrap_diff(metric_a, metric_b, TOY_IS_OOS, n_resamples=200, seed=7)
    second = paired_bootstrap_diff(metric_a, metric_b, TOY_IS_OOS, n_resamples=200, seed=7)
    assert first == second
    assert first["ci_low"] <= first["diff"] <= first["ci_high"]


def test_bootstrap_resamples_keep_class_counts():
    # The number of OOS rows in a resample is the same every time when sampling is stratified,
    # so this "metric" has no spread at all.
    count_oos = lambda rows: float(TOY_IS_OOS[rows].sum())  # noqa: E731
    zero = lambda rows: 0.0  # noqa: E731
    result = paired_bootstrap_diff(count_oos, zero, TOY_IS_OOS, n_resamples=200, seed=0)
    assert result["ci_low"] == result["ci_high"] == 5.0
