"""Unit tests for the shared prediction format and S0, on small hand-made data."""

import numpy as np
import pandas as pd
import pytest

from router.systems import base, s0_tfidf

# ---- softmax and the shared Predictions format --------------------------------


def test_softmax_rows_sum_to_one():
    probs = base.softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.isfinite(probs).all()  # large logits must not overflow


def test_temperature_changes_confidence_not_prediction():
    logits = np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0]])
    p1, p2 = base.softmax(logits, 1.0), base.softmax(logits, 2.0)
    assert (p1.argmax(axis=1) == p2.argmax(axis=1)).all()
    assert (p2.max(axis=1) < p1.max(axis=1)).all()  # higher temperature = less confident


def test_predictions_from_logits():
    logits = np.array([[1.0, 3.0, 2.0], [5.0, 0.0, 0.0]])
    preds = base.predictions_from_logits(logits, ["x", "y", "z"])
    assert preds.pred_label == ["y", "x"]
    assert np.allclose(preds.confidence, base.softmax(logits).max(axis=1))
    assert np.allclose(preds.oos_score, 1.0 - preds.confidence)


def test_save_and_load_predictions_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "PREDICTIONS_DIR", tmp_path)
    eval_df = pd.DataFrame({"text": ["q1", "q2"], "label": ["x", "oos"], "is_oos": [False, True]})
    logits = np.array([[2.0, 0.0], [0.1, 0.0]])
    preds = base.predictions_from_logits(logits, ["x", "y"])
    base.save_predictions("toy", 3, "val", eval_df, preds)

    df, loaded_logits = base.load_predictions("toy", 3, "val")
    assert df["pred_label"].tolist() == ["x", "x"]
    assert df["is_oos"].tolist() == [False, True]
    assert np.allclose(loaded_logits, logits)
    assert base.list_runs("val") == {"toy": [3]}


# ---- S0 -------------------------------------------------------------------------

S0_CFG = {"vectoriser": {"lowercase": True, "sublinear_tf": True}, "max_iter": 1000}
LABEL_NAMES = ["refund", "card_lost"]


def _toy_train() -> pd.DataFrame:
    texts = [
        "I want a refund", "refund my money please", "how do I get a refund", "money back refund",
        "I lost my card", "my card is lost", "card lost help", "lost card what now",
    ]
    return pd.DataFrame({"text": texts, "label_id": [0, 0, 0, 0, 1, 1, 1, 1]})


def test_s0_learns_a_separable_toy_set():
    model = s0_tfidf.train(_toy_train(), {"ngram_range": [1, 1], "C": 10}, S0_CFG, seed=0)
    preds = s0_tfidf.predict(model, ["please refund me", "I LOST my card"], LABEL_NAMES)
    assert preds.pred_label == ["refund", "card_lost"]  # also checks lowercasing works


def test_s0_confidence_matches_predict_proba():
    model = s0_tfidf.train(_toy_train(), {"ngram_range": [1, 2], "C": 10}, S0_CFG, seed=0)
    texts = ["refund", "card"]
    preds = s0_tfidf.predict(model, texts, LABEL_NAMES)
    assert np.allclose(preds.confidence, model.predict_proba(texts).max(axis=1))


def test_s0_is_deterministic():
    params = {"ngram_range": [1, 2], "C": 10}
    a = s0_tfidf.predict(s0_tfidf.train(_toy_train(), params, S0_CFG, seed=0), ["refund card"], LABEL_NAMES)
    b = s0_tfidf.predict(s0_tfidf.train(_toy_train(), params, S0_CFG, seed=1), ["refund card"], LABEL_NAMES)
    assert np.array_equal(a.logits, b.logits)
