"""Text preprocessing pipeline (README section 6.2).

Every system gets its data through these functions, so they are kept as small
plain functions that each do one step and can be unit-tested on toy data.

All DataFrames here use the same columns:
    text        the query, after normalisation
    label       intent name, e.g. "card_arrival" (or "oos" for out-of-scope)
    label_id    integer id from data/processed/label_map.json (-1 for oos)
    orig_index  row number in the original Hub split, so any row can be traced back
    source      where the row came from, e.g. "banking77_train" or "clinc_oos_test"
"""

import math
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Matches any run of whitespace: spaces, tabs, newlines, and unicode spaces.
_WHITESPACE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Step 2: normalise text
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """Normalise one query.

    NFKC turns look-alike unicode characters into their plain form
    (for example full-width "Ｃａｒｄ" becomes "Card"). Then repeated
    whitespace becomes one space and the ends are stripped.

    Case, punctuation and digits are kept on purpose: amounts like "$5.00"
    and terms like "PayID" carry meaning, and transformer tokenisers
    handle case themselves.
    """
    if not isinstance(text, str):  # missing values (None or NaN) become empty text
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE.sub(" ", text)
    return text.strip()


def normalise_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with normalise_text applied to the text column."""
    df = df.copy()
    df["text"] = df["text"].map(normalise_text)
    return df


# ---------------------------------------------------------------------------
# Step 3: drop empty rows and duplicates (within one split)
# ---------------------------------------------------------------------------

def drop_empty_and_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove empty rows and duplicate texts from one split.

    Expects df["text"] to be normalised already.

    Rules:
      - empty text: removed (reason "empty")
      - same text appears with DIFFERENT labels: every copy removed
        (reason "conflict"), because no system can get all copies right
      - same text appears more than once with the SAME label: the first copy
        is kept and the rest removed (reason "duplicate")

    Returns (kept_rows, removed_rows). removed_rows has an extra "reason" column.
    """
    is_empty = df["text"].str.len() == 0
    empty_rows = df[is_empty].assign(reason="empty")
    df = df[~is_empty]

    # For each row, count how many different labels its text has across the split.
    labels_per_text = df.groupby("text")["label"].transform("nunique")
    is_conflict = labels_per_text > 1
    conflict_rows = df[is_conflict].assign(reason="conflict")
    df = df[~is_conflict]

    # keep="first" marks every copy after the first one as a duplicate.
    is_duplicate = df.duplicated(subset="text", keep="first")
    duplicate_rows = df[is_duplicate].assign(reason="duplicate")
    kept = df[~is_duplicate]

    removed = pd.concat([empty_rows, conflict_rows, duplicate_rows])
    return kept.reset_index(drop=True), removed.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 4: leakage check (text that appears in two splits)
# ---------------------------------------------------------------------------

def remove_overlap(
    df: pd.DataFrame, protected: pd.DataFrame, reason: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows from df whose text also appears in the protected split.

    The protected split is never changed. For the train/test leakage check,
    df is train and protected is test, so leaked rows leave train, not test.

    Returns (kept_rows, removed_rows). removed_rows gets reason=<reason>.
    """
    is_overlap = df["text"].isin(set(protected["text"]))
    removed = df[is_overlap].assign(reason=reason)
    kept = df[~is_overlap]
    return kept.reset_index(drop=True), removed.reset_index(drop=True)


def overlap_count(a: pd.DataFrame, b: pd.DataFrame) -> int:
    """Number of distinct texts that appear in both a and b. Should be 0 after cleaning."""
    return len(set(a["text"]) & set(b["text"]))


# ---------------------------------------------------------------------------
# Step 5: stratified validation split
# ---------------------------------------------------------------------------

def stratified_val_split(
    df: pd.DataFrame, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (train, validation), keeping each intent's share the same in both.

    stratify= makes sklearn sample each label separately, so every intent ends up
    with about `fraction` of its rows in validation. The same seed always gives
    the same split. Rows are sorted by orig_index so the saved files are stable.
    """
    train_df, val_df = train_test_split(
        df, test_size=fraction, stratify=df["label"], random_state=seed
    )
    train_df = train_df.sort_values("orig_index").reset_index(drop=True)
    val_df = val_df.sort_values("orig_index").reset_index(drop=True)
    return train_df, val_df


# ---------------------------------------------------------------------------
# Step 6: token length check
# ---------------------------------------------------------------------------

def token_lengths(
    texts: list[str], tokenizer, pair_texts: list[str] | None = None
) -> np.ndarray:
    """Number of tokens per input for one tokeniser, including special tokens like [CLS].

    With pair_texts, each input is the pair (texts[i], pair_texts[i]) encoded together,
    which is what the S3 cross-encoder sees.
    """
    if pair_texts is None:
        encoded = tokenizer(texts, add_special_tokens=True)["input_ids"]
    else:
        encoded = tokenizer(texts, pair_texts, add_special_tokens=True)["input_ids"]
    return np.array([len(ids) for ids in encoded])


def length_stats(lengths: np.ndarray) -> dict[str, float]:
    """Summary of a token-length array: mean, median, 95th and 99th percentile, max."""
    return {
        "mean": float(np.mean(lengths)),
        "p50": float(np.percentile(lengths, 50)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
        "max": int(np.max(lengths)),
    }


def choose_max_length(lengths: np.ndarray, percentile: float) -> int:
    """Pick max_length as the given percentile of lengths, rounded up to a multiple of 8.

    Multiples of 8 are slightly faster on GPUs and make the number easy to quote.
    """
    value = np.percentile(lengths, percentile)
    return int(math.ceil(value / 8) * 8)


def truncation_share(lengths: np.ndarray, max_length: int) -> float:
    """Share of texts longer than max_length, i.e. the ones that would be cut off."""
    return float(np.mean(lengths > max_length))


# ---------------------------------------------------------------------------
# Step 7: class balance
# ---------------------------------------------------------------------------

def class_balance(df: pd.DataFrame) -> pd.Series:
    """Number of queries per intent, largest first."""
    return df["label"].value_counts()
