"""Unit tests for src/router/metrics.py. Expected values are worked out by hand."""

import math

import pytest

from router import metrics

# Toy example used below:
#   index   0  1  2  3  4  5
#   true    a  a  b  b  c  c
#   pred    a  b  b  b  c  a
# a: predicted at 0,5 -> 1 correct -> precision 1/2; true at 0,1 -> recall 1/2; F1 0.5
# b: predicted at 1,2,3 -> 2 correct -> precision 2/3; recall 2/2 = 1; F1 0.8
# c: predicted at 4 -> precision 1; recall 1/2; F1 2/3
Y_TRUE = ["a", "a", "b", "b", "c", "c"]
Y_PRED = ["a", "b", "b", "b", "c", "a"]
LABELS = ["a", "b", "c"]


def test_accuracy_and_macro_scores():
    m = metrics.classification_metrics(Y_TRUE, Y_PRED, labels=LABELS)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["macro_precision"] == pytest.approx((1 / 2 + 2 / 3 + 1) / 3)
    assert m["macro_recall"] == pytest.approx((1 / 2 + 1 + 1 / 2) / 3)
    assert m["macro_f1"] == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_weighted_differs_from_macro_when_support_is_uneven():
    # a has 3 true queries, b has 1.
    # a: precision 2/2, recall 2/3, F1 0.8.  b: precision 1/2, recall 1, F1 2/3.
    y_true, y_pred = ["a", "a", "a", "b"], ["a", "a", "b", "b"]
    m = metrics.classification_metrics(y_true, y_pred, labels=["a", "b"])
    assert m["macro_f1"] == pytest.approx((0.8 + 2 / 3) / 2)
    assert m["weighted_f1"] == pytest.approx((3 * 0.8 + 1 * 2 / 3) / 4)


def test_per_intent_and_worst_intents():
    per_intent = metrics.per_intent_scores(Y_TRUE, Y_PRED, LABELS)
    assert per_intent.loc["b", "f1"] == pytest.approx(0.8)
    assert per_intent.loc["a", "support"] == 2
    worst = metrics.worst_intents(per_intent, k=2)
    assert list(worst.index) == ["a", "c"]  # F1 0.5, then 2/3


def test_top_confused_pairs():
    matrix = metrics.confusion_counts(Y_TRUE, Y_PRED, LABELS)
    pairs = metrics.top_confused_pairs(matrix, LABELS, k=10)
    # Two mistakes: a predicted as b, and c predicted as a. Equal counts, so sorted by name.
    assert pairs.to_dict(orient="records") == [
        {"true": "a", "predicted": "b", "count": 1},
        {"true": "c", "predicted": "a", "count": 1},
    ]


def test_no_mistakes_gives_empty_confusion_table():
    matrix = metrics.confusion_counts(["a", "b"], ["a", "b"], ["a", "b"])
    assert len(metrics.top_confused_pairs(matrix, ["a", "b"])) == 0


def test_mean_std_over_seeds():
    summary = metrics.mean_std_over_seeds([{"acc": 1.0}, {"acc": 3.0}])
    assert summary["acc"]["mean"] == pytest.approx(2.0)
    assert summary["acc"]["std"] == pytest.approx(math.sqrt(2))  # sample std of [1, 3]
    assert metrics.mean_std_over_seeds([{"acc": 0.5}])["acc"]["std"] == 0.0
