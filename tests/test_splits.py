"""Checks on the saved splits in data/processed/ (the standing leak check).

Skipped if scripts/make_splits.py has not been run yet. If any of these fail
after the splits exist, stop: a later result could be inflated by a leak.
"""

import itertools

import pandas as pd
import pytest

from router import data
from router.preprocess import overlap_count

SPLITS = list(data.SPLIT_FILES)

pytestmark = pytest.mark.skipif(
    not all((data.PROCESSED_DIR / f).exists() for f in data.SPLIT_FILES.values()),
    reason="splits not created yet: run scripts/make_splits.py",
)


@pytest.fixture(scope="module")
def splits():
    return {name: data.load_split(name) for name in SPLITS}


def test_no_text_appears_in_two_splits(splits):
    for a, b in itertools.combinations(SPLITS, 2):
        assert overlap_count(splits[a], splits[b]) == 0, f"{a} and {b} share texts"


def test_no_duplicates_or_empty_text_within_a_split(splits):
    for name, df in splits.items():
        assert not df["text"].duplicated().any(), f"duplicate text in {name}"
        assert (df["text"].str.len() > 0).all(), f"empty text in {name}"


def test_every_in_scope_split_has_all_77_intents(splits):
    label_map = data.load_label_map()
    assert len(label_map) == 77
    for name in ["train", "val", "test"]:
        assert set(splits[name]["label"]) == set(label_map), f"{name} intents differ from label map"


def test_label_ids_match_label_map(splits):
    label_map = data.load_label_map()
    for name in ["train", "val", "test"]:
        df = splits[name]
        assert (df["label"].map(label_map) == df["label_id"]).all()


def test_oos_pools_are_only_oos(splits):
    for name in ["oos_val", "oos_test"]:
        assert (splits[name]["label"] == data.OOS_LABEL).all()
        assert (splits[name]["label_id"] == data.OOS_LABEL_ID).all()


def test_row_counts_add_up_to_the_hub_sizes(splits):
    cfg = data.load_config("data")
    removed = pd.read_csv(data.PROCESSED_DIR / data.REMOVED_ROWS_FILE, keep_default_na=False)
    n_removed_train = (removed["source"] == "banking77_train").sum()
    n_removed_test = (removed["source"] == "banking77_test").sum()
    expected = cfg["banking77"]["expected_sizes"]
    assert len(splits["train"]) + len(splits["val"]) + n_removed_train == expected["train"]
    assert len(splits["test"]) + n_removed_test == expected["test"]


def test_eval_set_mixes_in_oos(splits):
    mixed = data.load_eval_set("val", with_oos=True)
    assert len(mixed) == len(splits["val"]) + len(splits["oos_val"])
    assert mixed["is_oos"].sum() == len(splits["oos_val"])
