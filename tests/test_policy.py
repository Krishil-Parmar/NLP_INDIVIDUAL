"""Unit tests for src/router/policy.py."""

import numpy as np

from router import policy


def test_handle_at_or_above_threshold_escalate_below():
    assert policy.decide(0.80, t_high=0.80) == policy.HANDLE  # "at or above" includes equal
    assert policy.decide(0.95, t_high=0.80) == policy.HANDLE
    assert policy.decide(0.79, t_high=0.80) == policy.ESCALATE


def test_decide_all_matches_decide():
    confidence = np.array([0.1, 0.5, 0.8, 0.99])
    expected = [policy.decide(c, 0.5) for c in confidence]
    assert policy.decide_all(confidence, 0.5).tolist() == expected


def test_threshold_extremes():
    confidence = np.array([0.0, 0.5, 1.0])
    assert (policy.decide_all(confidence, 0.0) == policy.HANDLE).all()  # handle everything
    assert (policy.decide_all(confidence, 1.01) == policy.ESCALATE).all()  # escalate everything
