"""Unit tests for E2 (label noise) on toy data: flags, dropping by key, the blind audit and its scoring."""

import numpy as np
import pandas as pd
import pytest

from router import label_noise

LABELS = ["card_arrival", "top_up_failed", "pin_blocked"]


def toy_clusters(n_per_class: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated 2-D clusters, one per class."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    features = np.concatenate([c + rng.normal(size=(n_per_class, 2)) for c in centres])
    labels = np.repeat(np.arange(3), n_per_class)
    return features, labels


def test_a_planted_wrong_label_is_flagged():
    features, labels = toy_clusters()
    noisy = labels.copy()
    noisy[5] = 2  # a class-0 point labelled as class 2
    probs, chosen_C, grid = label_noise.cross_val_pred_probs(features, noisy, 3, [0.1, 1.0], folds=3, seed=0, max_iter=1000)
    assert probs.shape == (90, 3) and chosen_C in (0.1, 1.0) and len(grid) == 2
    flagged = label_noise.flag_issues(noisy, probs, "prune_by_noise_rate", "self_confidence", n_jobs=1)
    assert 5 in flagged
    assert flagged[0] == 5  # the planted error is the most severe


def test_suggestion_never_equals_the_given_label():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    given = np.array([0, 1])
    assert label_noise.suggested_label_ids(given, probs).tolist() == [1, 2]


def make_df(keys: list[tuple[str, int]], labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"text": [f"q{i}" for i in range(len(keys))], "label": labels,
                         "source": [s for s, _ in keys], "orig_index": [i for _, i in keys]})


def test_drop_flagged_matches_by_key_not_position():
    df = make_df([("banking77_train", 7), ("banking77_train", 3), ("clinc_oos_validation", 3)], LABELS)
    flagged = pd.DataFrame({"source": ["banking77_train"], "orig_index": [3]})
    kept = label_noise.drop_flagged(df, flagged)
    # orig_index 3 from Banking77 goes; the CLINC row with the same number stays.
    assert kept["orig_index"].tolist() == [7, 3]
    assert kept["source"].tolist() == ["banking77_train", "clinc_oos_validation"]


def test_stop_checks():
    train = make_df([("t", i) for i in range(6)], ["card_arrival"] * 3 + ["top_up_failed"] * 3)
    limits = {"max_train_share_flagged": 0.10, "max_intent_share_removed": 0.3333, "min_cleaner_cv_accuracy": 0.85}
    nothing = pd.DataFrame({"source": [], "orig_index": []})
    assert label_noise.stop_checks(nothing, train, 0.90, limits) == []
    two_of_one_intent = pd.DataFrame({"source": ["t", "t"], "orig_index": [0, 1]})
    problems = label_noise.stop_checks(two_of_one_intent, train, 0.80, limits)
    assert len(problems) == 3  # share of train, card_arrival loses 2 of 3, weak cleaner


def toy_flagged() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame({
        "text": [f"{name} example {i}" for name in LABELS for i in range(5)],
        "label": [name for name in LABELS for _ in range(5)],
        "source": "banking77_train", "orig_index": range(15),
    })
    flagged = pd.DataFrame({
        "split": "train", "source": "banking77_train", "orig_index": [0, 5, 10, 11],
        "text": ["card_arrival example 0", "top_up_failed example 0", "pin_blocked example 0", "pin_blocked example 1"],
        "given_label": ["card_arrival", "top_up_failed", "pin_blocked", "pin_blocked"],
        "suggested_label": ["top_up_failed", "pin_blocked", "card_arrival", "top_up_failed"],
    })
    return train, flagged


def test_blind_audit_hides_the_given_label_and_maps_back():
    train, flagged = toy_flagged()
    sheet, key = label_noise.make_blind_audit(flagged, train, n_items=4, seed=0, examples_per_option=2)
    assert list(sheet.columns) == ["item", "text", "option_a", "a_examples", "option_b", "b_examples", "verdict", "notes"]
    assert "given" not in " ".join(sheet.columns)  # nothing on the sheet says which label the dataset chose
    assert (sheet["verdict"] == "").all()
    merged = sheet.merge(key, on="item")
    for _, row in merged.iterrows():
        given = row["option_a"] if row["given_option"] == "a" else row["option_b"]
        other = row["option_b"] if row["given_option"] == "a" else row["option_a"]
        assert (given, other) == (row["given_label"], row["suggested_label"])
        assert row["text"] not in row["a_examples"] + row["b_examples"]  # the item is never its own example
    # Examples come only from unflagged rows.
    assert "pin_blocked example 1" not in " ".join(sheet["a_examples"].tolist() + sheet["b_examples"].tolist())
    again, _ = label_noise.make_blind_audit(flagged, train, n_items=4, seed=0, examples_per_option=2)
    pd.testing.assert_frame_equal(sheet, again)  # same seed, same sheet


def test_score_audit_follows_the_fixed_rules():
    key = pd.DataFrame({"item": [1, 2, 3, 4, 5], "given_option": ["a", "b", "a", "b", "a"]})
    # item 1 picks the suggestion (b), 2 picks the suggestion (a), 3 picks the given label,
    # 4 says both, 5 says neither.
    sheet = pd.DataFrame({"item": [1, 2, 3, 4, 5], "verdict": ["b", "A ", "a", "both", "neither"]})
    scored = label_noise.score_audit(sheet, key)
    assert scored["complete"]
    assert (scored["real_error"], scored["picked_suggestion"], scored["neither"]) == (3, 2, 1)
    assert (scored["ambiguous"], scored["false_flag"]) == (1, 1)
    assert scored["real_error_share"] == pytest.approx(0.6)


def test_score_audit_waits_for_every_verdict():
    key = pd.DataFrame({"item": [1, 2], "given_option": ["a", "b"]})
    scored = label_noise.score_audit(pd.DataFrame({"item": [1, 2], "verdict": ["a", ""]}), key)
    assert scored == {"complete": False, "n_done": 1, "n_items": 2, "unreadable_items": []}
    typo = label_noise.score_audit(pd.DataFrame({"item": [1, 2], "verdict": ["a", "bb"]}), key)
    assert typo["unreadable_items"] == [2]


def test_wilson_interval_known_values():
    low, high = label_noise.wilson_interval(50, 100)
    assert (low, high) == pytest.approx((0.4038, 0.5962), abs=1e-4)
    low, high = label_noise.wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12) and high == pytest.approx(0.2775, abs=1e-4)


def test_list_runs_hides_cleaned_runs_by_default(tmp_path, monkeypatch):
    from router.systems import base

    for tag in ["s1_distilroberta-base", "s1_distilroberta-base_cleaned"]:
        (tmp_path / tag).mkdir()
        (tmp_path / tag / "seed0_val.csv").write_text("")
    monkeypatch.setattr(base, "PREDICTIONS_DIR", tmp_path)
    assert list(base.list_runs("val")) == ["s1_distilroberta-base"]
    assert list(base.list_runs("val", include_cleaned=True)) == ["s1_distilroberta-base", "s1_distilroberta-base_cleaned"]


def test_cleaned_runs_get_their_own_tags_and_checkpoints():
    from router.systems import registry

    assert registry.run_tag("s3_quora-distilroberta-base", "cleaned") == "s3_quora-distilroberta-base_cleaned"
    assert registry.run_tag("s0_tfidf") == "s0_tfidf"
    assert registry.s0_checkpoint(2).parts[-2:] == ("s0", "seed2.joblib")  # unchanged for the original runs
    assert registry.s0_checkpoint(2, "cleaned").parts[-2:] == ("s0_cleaned", "seed2.joblib")


def test_cleaned_train_is_train_minus_the_flagged_train_rows():
    from router import data

    flagged = label_noise.load_flagged()
    original, cleaned = data.load_train(), data.load_train("cleaned")
    assert len(cleaned) == len(original) - int((flagged["split"] == "train").sum())
    assert not label_noise.is_flagged(cleaned, flagged).any()
    with pytest.raises(ValueError):
        data.load_train("relabelled")
