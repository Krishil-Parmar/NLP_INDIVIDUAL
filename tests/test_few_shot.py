"""Tests for router.few_shot (E8): the pieces that do not need a model."""

import numpy as np
import pandas as pd

from router import few_shot
from router.few_shot import Variant


def test_variant_tiers():
    assert Variant("frozen", "examples", "nearest").tier == "no training"
    assert Variant("frozen", "descriptions", "margin").tier == "no training"
    assert Variant("frozen", "examples", "head").tier == "head only"
    assert Variant("tuned", "examples", "nearest").tier == "encoder fine-tune"
    assert len(few_shot.all_variants(["examples", "descriptions"])) == 12
    assert few_shot.all_variants(["examples"])[0].name == "frozen/examples/nearest"  # S2 itself comes first


def test_per_intent_sample_takes_k_or_all():
    df = pd.DataFrame({"text": [f"q{i}" for i in range(7)], "label_id": [0, 0, 0, 0, 1, 1, 2]})
    sample = few_shot.per_intent_sample(df, k=2, seed=0)
    assert sample["label_id"].value_counts().to_dict() == {0: 2, 1: 2, 2: 1}  # intent 2 has only one row
    assert sample.equals(few_shot.per_intent_sample(df, k=2, seed=0))  # same seed, same draw


def test_add_descriptions_appends_one_row_per_intent():
    support = pd.DataFrame({"text": ["a", "b"], "label_id": [0, 1]})
    out = few_shot.add_descriptions(support, {"x": "about x", "y": "about y"}, ["x", "y"])
    assert out["text"].tolist() == ["a", "b", "about x", "about y"]
    assert out["label_id"].tolist() == [0, 1, 0, 1]


def test_margin_is_best_minus_runner_up():
    scores = np.array([[0.9, 0.1, 0.5], [0.4, 0.4, 0.2]])
    assert np.allclose(few_shot.margin_scores(scores), [0.4, 0.0])


def test_setfit_pairs_have_the_right_targets():
    texts = ["a1", "a2", "b1", "b2"]
    label_ids = np.array([0, 0, 1, 1])
    a, b, targets = few_shot.setfit_pairs(texts, label_ids, iterations=3, rng=np.random.default_rng(0))
    assert len(a) == len(b) == len(targets) == 3 * 4 * 2  # one positive and one negative per example per iteration
    label = dict(zip(texts, label_ids))
    for x, y, t in zip(a, b, targets):
        assert (label[x] == label[y]) == (t == 1.0)
        assert x != y  # a positive pair never pairs an example with itself


def test_run_metrics_counts_handled_oos_as_errors():
    names = ["x", "y"]
    pred = np.array([0, 1, 0, 1])
    true = np.array([0, 1, 1, -1])  # third is wrong, fourth is out of scope
    is_oos = np.array([False, False, False, True])
    conf = np.array([0.9, 0.8, 0.3, 0.95])  # the out-of-scope query is the most confident
    row = few_shot.run_metrics(pred, conf, true, is_oos, names, r_star=0.0, fixed_risks=[0.4])
    assert row["in_scope_accuracy"] == 2 / 3
    assert row["coverage_at_matched_risk"] == 0.0  # the most confident answer is already an error
    assert row["coverage_at_0.4"] == 0.75  # top three: one error in three (33%); all four would be 50%
    assert row["confidence_auroc"] == 1.0  # both right answers are more confident than the wrong one


def test_select_per_tier_prefers_simpler_when_close():
    aurc = {
        Variant("frozen", "examples", "nearest"): 0.300,
        Variant("frozen", "examples", "margin"): 0.204,
        Variant("frozen", "descriptions", "margin"): 0.200,
        Variant("frozen", "examples", "head"): 0.250,
        Variant("tuned", "examples", "nearest"): 0.100,
        Variant("tuned", "descriptions", "head"): 0.050,
    }
    chosen = few_shot.select_per_tier(aurc, simpler_within=0.005)
    assert chosen["no training"].name == "frozen/examples/margin"  # within 0.005 of the best and simpler
    assert chosen["head only"].name == "frozen/examples/head"
    assert chosen["encoder fine-tune"].name == "tuned/descriptions/head"  # 0.05 better: no tie
