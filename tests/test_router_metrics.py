"""Unit tests for the E6 router metrics, calibration and cost model, on hand-computed toy data."""

import numpy as np
import pytest

from router.calibration import calibrated_confidence, fit_temperature, negative_log_likelihood
from router.cost import cost_per_query
from router.metrics import (
    aurc,
    coverage_at_risk,
    expected_calibration_error,
    handle_precision_recall,
    risk_coverage_curve,
)

# Five queries, most confident first: right, right, wrong, right, wrong.
CONF = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
CORRECT = np.array([True, True, False, True, False])


def test_risk_coverage_curve_by_hand():
    curve = risk_coverage_curve(CORRECT, CONF)
    assert curve["threshold"].tolist() == [0.9, 0.8, 0.7, 0.6, 0.5]
    assert curve["coverage"].tolist() == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    # errors among the handled: 0/1, 0/2, 1/3, 1/4, 2/5
    assert curve["risk"].tolist() == pytest.approx([0.0, 0.0, 1 / 3, 1 / 4, 2 / 5])


def test_ties_share_one_threshold():
    curve = risk_coverage_curve(np.array([True, False, True]), np.array([0.9, 0.9, 0.5]))
    assert curve["coverage"].tolist() == pytest.approx([2 / 3, 1.0])  # both 0.9s come in together
    assert curve["risk"].tolist() == pytest.approx([0.5, 1 / 3])


def test_aurc_is_the_step_area():
    curve = risk_coverage_curve(CORRECT, CONF)
    # each point adds 0.2 coverage: 0.2 * (0 + 0 + 1/3 + 1/4 + 2/5)
    assert aurc(curve) == pytest.approx(0.2 * (1 / 3 + 1 / 4 + 2 / 5))


def test_coverage_at_risk():
    assert coverage_at_risk(CORRECT, CONF, max_risk=0.0) == {"coverage": pytest.approx(0.4), "threshold": 0.8, "risk": 0.0}
    # risk 0.3: coverage 0.8 qualifies (1/4), coverage 0.6 does not (1/3) - the highest qualifying point wins
    assert coverage_at_risk(CORRECT, CONF, max_risk=0.3)["coverage"] == pytest.approx(0.8)
    assert coverage_at_risk(CORRECT, CONF, max_risk=0.3)["threshold"] == 0.6
    nothing = coverage_at_risk(np.array([False, False]), np.array([0.9, 0.8]), max_risk=0.1)
    assert nothing["coverage"] == 0.0 and nothing["threshold"] == float("inf")


def test_handle_precision_recall():
    handled = CONF >= 0.6  # the first four
    result = handle_precision_recall(CORRECT, handled)
    assert result["handle_precision"] == pytest.approx(3 / 4)  # 3 right out of 4 handled
    assert result["handle_recall"] == pytest.approx(1.0)  # all 3 right answers were handled
    assert np.isnan(handle_precision_recall(CORRECT, np.zeros(5, dtype=bool))["handle_precision"])


def test_ece():
    # Perfect calibration: confidence 0.75 everywhere and 3 of 4 right.
    assert expected_calibration_error(np.full(4, 0.75), np.array([1, 1, 1, 0])) == pytest.approx(0.0)
    # Always 0.9 sure, always wrong: the gap is 0.9.
    assert expected_calibration_error(np.full(4, 0.9), np.zeros(4)) == pytest.approx(0.9)


def test_temperature_recovers_a_known_overconfidence():
    # True model: logits z. Labels are sampled from softmax(z). The "system" reports 2 * z,
    # so it is over-confident by a factor of 2 and the fitted temperature should be close to 2.
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, size=(4000, 5))
    probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(5, p=p) for p in probs])
    t = fit_temperature(2 * z, labels)
    assert 1.8 < t < 2.2
    assert negative_log_likelihood(2 * z / t, labels) < negative_log_likelihood(2 * z, labels)


def test_calibrated_confidence_never_changes_the_argmax_order_within_a_row():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    assert np.all(calibrated_confidence(logits, 2.0) < calibrated_confidence(logits, 1.0))  # softer, never sharper


def test_cost_per_query():
    assert cost_per_query(coverage=0.0, llm_cost=0.002) == pytest.approx(0.002)  # everything escalated
    assert cost_per_query(coverage=0.75, llm_cost=0.002) == pytest.approx(0.0005)
    assert cost_per_query(coverage=1.0, llm_cost=0.002, router_cost=0.0001) == pytest.approx(0.0001)
