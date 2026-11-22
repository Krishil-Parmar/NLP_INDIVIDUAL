"""Unit tests for src/router/preprocess.py, on small hand-made data."""

import numpy as np
import pandas as pd

from router import preprocess as pp


def make_df(rows: list[tuple[str, str]], source: str = "toy") -> pd.DataFrame:
    """Build a DataFrame in the project's column format from (text, label) pairs."""
    return pd.DataFrame(
        {
            "text": [t for t, _ in rows],
            "label": [lab for _, lab in rows],
            "label_id": 0,
            "orig_index": range(len(rows)),
            "source": source,
        }
    )


# ---- Step 2: normalise_text -------------------------------------------------

def test_normalise_collapses_whitespace_and_strips():
    assert pp.normalise_text("  where is\tmy \n\n card  ") == "where is my card"


def test_normalise_nfkc_full_width_and_non_breaking_space():
    # Full-width letters and digits become plain ASCII; a non-breaking space becomes a space.
    assert pp.normalise_text("Ｃａｒｄ １２３") == "Card 123"


def test_normalise_keeps_case_punctuation_digits_and_domain_terms():
    text = "Why was my PayID transfer of $5.00 to BSB 062-000 declined?"
    assert pp.normalise_text(text) == text


def test_normalise_missing_value_becomes_empty():
    assert pp.normalise_text(None) == ""
    assert pp.normalise_text(float("nan")) == ""


# ---- Step 3: drop_empty_and_duplicates ---------------------------------------

def test_same_label_duplicate_keeps_first_copy():
    df = make_df([("hello", "a"), ("hello", "a"), ("bye", "b")])
    kept, removed = pp.drop_empty_and_duplicates(df)
    assert kept["text"].tolist() == ["hello", "bye"]
    assert kept["orig_index"].tolist() == [0, 2]  # the first copy is the one kept
    assert removed["reason"].tolist() == ["duplicate"]


def test_conflicting_labels_drop_every_copy():
    df = make_df([("top up", "a"), ("top up", "b"), ("bye", "b")])
    kept, removed = pp.drop_empty_and_duplicates(df)
    assert kept["text"].tolist() == ["bye"]
    assert sorted(removed["reason"].tolist()) == ["conflict", "conflict"]


def test_empty_rows_are_removed():
    df = make_df([("", "a"), ("ok", "a")])
    kept, removed = pp.drop_empty_and_duplicates(df)
    assert kept["text"].tolist() == ["ok"]
    assert removed["reason"].tolist() == ["empty"]


def test_kept_plus_removed_equals_input():
    df = make_df([("x", "a"), ("x", "a"), ("y", "a"), ("y", "b"), ("", "c"), ("z", "c")])
    kept, removed = pp.drop_empty_and_duplicates(df)
    assert len(kept) + len(removed) == len(df)


# ---- Step 4: remove_overlap (leakage) ----------------------------------------

def test_leakage_removed_from_train_only():
    train = make_df([("a query", "x"), ("only in train", "y")], source="train")
    test = make_df([("a query", "x"), ("only in test", "y")], source="test")
    kept, removed = pp.remove_overlap(train, test, reason="leakage")
    assert kept["text"].tolist() == ["only in train"]
    assert removed["reason"].tolist() == ["leakage"]
    assert len(test) == 2  # the protected split is not touched
    assert pp.overlap_count(kept, test) == 0


# ---- Step 5: stratified_val_split ---------------------------------------------

def _balanced_toy(n_per_label: int = 20, labels: str = "abcde") -> pd.DataFrame:
    rows = [(f"{lab} query {i}", lab) for lab in labels for i in range(n_per_label)]
    return make_df(rows)


def test_split_is_deterministic_for_a_seed():
    df = _balanced_toy()
    train1, val1 = pp.stratified_val_split(df, 0.1, seed=42)
    train2, val2 = pp.stratified_val_split(df, 0.1, seed=42)
    assert val1["text"].tolist() == val2["text"].tolist()
    assert train1["text"].tolist() == train2["text"].tolist()


def test_split_is_disjoint_complete_and_stratified():
    df = _balanced_toy()
    train, val = pp.stratified_val_split(df, 0.1, seed=42)
    assert pp.overlap_count(train, val) == 0
    assert len(train) + len(val) == len(df)
    # 10% of 20 per label = 2 per label in validation, for every label
    assert val["label"].value_counts().to_dict() == {lab: 2 for lab in "abcde"}


# ---- Step 6: lengths -----------------------------------------------------------

def test_choose_max_length_rounds_up_to_multiple_of_8():
    lengths = np.array([5, 10, 17, 20])
    assert pp.choose_max_length(lengths, 100) == 24  # max is 20, rounded up to 24
    assert pp.choose_max_length(np.array([16, 16]), 100) == 16  # already a multiple of 8


def test_truncation_share():
    assert pp.truncation_share(np.array([4, 8, 9, 20]), 8) == 0.5


class FakeTokenizer:
    """Stands in for a Hugging Face tokenizer: one token per word plus 2 special tokens."""

    def __call__(self, texts, pair_texts=None, add_special_tokens=True):
        if pair_texts is None:
            return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}
        return {"input_ids": [[0] * (len(a.split()) + len(b.split()) + 3) for a, b in zip(texts, pair_texts)]}


def test_token_lengths_single_and_pair():
    tok = FakeTokenizer()
    assert pp.token_lengths(["one two", "three"], tok).tolist() == [4, 3]
    assert pp.token_lengths(["one two"], tok, pair_texts=["three"]).tolist() == [6]
