"""Label noise (E2): find likely label errors, drop them, and build the hand audit.

The steps, one small function each:
  1. cross_val_pred_probs: out-of-sample class probabilities for every row. A
     logistic regression on frozen sentence embeddings is fitted in K folds, so
     each row is scored by a model that never saw its label.
  2. flag_issues: cleanlab's confident learning (Northcutt et al., 2021) flags
     rows whose given label disagrees with those probabilities by more than the
     estimated noise rates allow.
  3. drop_flagged: removes flagged rows from any split by their
     (source, orig_index) key, never by row position.
  4. make_blind_audit, score_audit and wilson_interval: the 100-item hand check.
     score_audit_by_annotator and audit_agreement split it by who judged each
     item (the owner or a blind Claude judge) and check how often they agree.

"Cleaned" in this project means flagged rows removed, never relabelled.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from router import data
from router.systems.base import CLEANED_SUFFIX

FLAGGED_FILE = "banking77_flagged.csv"
PRED_PROBS_FILE = "banking77_cleaner_pred_probs.npz"
AUDIT_FILE = "e2_audit.csv"
AUDIT_KEY_FILE = "e2_audit_key.csv"
AUDIT_JUDGE_FILE = "e2_audit_claude.csv"  # the Claude judge's blind verdicts on every audit item
AUDIT_VERDICTS = ("a", "b", "both", "neither")
OWNER, JUDGE = "owner", "claude"  # values of the audit sheet's "annotator" column
KEY = ["source", "orig_index"]  # identifies a row across every split file


def cleaned_tag(tag: str) -> str:
    """Run tag of a system trained on cleaned labels, e.g. "s1_distilroberta-base_cleaned"."""
    return tag + CLEANED_SUFFIX


# ---------------------------------------------------------------------------
# 1 and 2: probabilities and flags
# ---------------------------------------------------------------------------

def cross_val_pred_probs(
    features: np.ndarray, label_ids: np.ndarray, n_classes: int,
    C_grid: list[float], folds: int, seed: int, max_iter: int,
) -> tuple[np.ndarray, float, list[dict]]:
    """Out-of-fold class probabilities, keeping the C with the lowest log loss.

    Returns (pred_probs of shape (rows, n_classes), chosen C, one summary row per C).
    The same folds are used for every C, so the C values are compared fairly.
    """
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    best_probs, best_C, best_loss, grid = None, None, np.inf, []
    for C in C_grid:
        model = LogisticRegression(C=C, max_iter=max_iter)
        probs = cross_val_predict(model, features, label_ids, cv=splitter, method="predict_proba")
        loss = float(log_loss(label_ids, probs, labels=list(range(n_classes))))
        accuracy = float(np.mean(probs.argmax(axis=1) == label_ids))
        grid.append({"C": C, "cv_log_loss": loss, "cv_accuracy": accuracy})
        if loss < best_loss:
            best_probs, best_C, best_loss = probs, C, loss
    return best_probs, best_C, grid


def flag_issues(label_ids: np.ndarray, pred_probs: np.ndarray, filter_by: str, rank_by: str, n_jobs: int) -> np.ndarray:
    """Row positions cleanlab flags as likely label errors, most severe first."""
    from cleanlab.filter import find_label_issues  # imported here: only E2 needs cleanlab

    return find_label_issues(
        labels=label_ids, pred_probs=pred_probs, filter_by=filter_by,
        return_indices_ranked_by=rank_by, n_jobs=n_jobs,
    )


def suggested_label_ids(label_ids: np.ndarray, pred_probs: np.ndarray) -> np.ndarray:
    """The most probable label other than the given one, per row.

    Excluding the given label means the suggestion always differs from it, so the
    audit never shows the same label twice.
    """
    probs = pred_probs.copy()
    probs[np.arange(len(label_ids)), label_ids] = -1.0
    return probs.argmax(axis=1)


def flagged_table(rows: pd.DataFrame, pred_probs: np.ndarray, ranked: np.ndarray, label_names: list[str]) -> pd.DataFrame:
    """One row per flagged query, most severe first: key, text, given and suggested label."""
    picked = rows.iloc[ranked]
    given = picked["label_id"].to_numpy()
    probs = pred_probs[ranked]
    suggested = suggested_label_ids(given, probs)
    return pd.DataFrame({
        "rank": np.arange(1, len(ranked) + 1),
        "split": picked["split"].to_numpy(),
        "source": picked["source"].to_numpy(),
        "orig_index": picked["orig_index"].to_numpy(),
        "text": picked["text"].to_numpy(),
        "given_label": [label_names[i] for i in given],
        "suggested_label": [label_names[i] for i in suggested],
        "given_label_prob": probs[np.arange(len(ranked)), given].round(4),  # cleanlab's self-confidence
        "suggested_label_prob": probs[np.arange(len(ranked)), suggested].round(4),
    })


# ---------------------------------------------------------------------------
# 3: dropping flagged rows
# ---------------------------------------------------------------------------

def load_flagged() -> pd.DataFrame:
    """The saved flagged list (data/processed/banking77_flagged.csv)."""
    path = data.PROCESSED_DIR / FLAGGED_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: uv run python scripts/find_label_issues.py")
    return pd.read_csv(path, keep_default_na=False)


def is_flagged(df: pd.DataFrame, flagged: pd.DataFrame) -> np.ndarray:
    """True for each row of df whose (source, orig_index) is in the flagged list."""
    keys = set(zip(flagged["source"], flagged["orig_index"].astype(int)))
    return np.array([(s, int(i)) in keys for s, i in zip(df["source"], df["orig_index"])], dtype=bool)


def drop_flagged(df: pd.DataFrame, flagged: pd.DataFrame) -> pd.DataFrame:
    """df without its flagged rows. Rows are matched by key, so order does not matter."""
    return df[~is_flagged(df, flagged)].reset_index(drop=True)


def stop_checks(flagged: pd.DataFrame, train_df: pd.DataFrame, cv_accuracy: float, limits: dict) -> list[str]:
    """Reasons to stop before retraining (empty list: carry on)."""
    problems = []
    removed = is_flagged(train_df, flagged)
    share = float(removed.mean())
    if share > limits["max_train_share_flagged"]:
        problems.append(f"{share:.1%} of train flagged (limit {limits['max_train_share_flagged']:.0%})")
    per_intent = pd.Series(removed).groupby(train_df["label"].to_numpy()).mean()
    for intent, intent_share in per_intent[per_intent > limits["max_intent_share_removed"]].items():
        problems.append(f"intent {intent} loses {intent_share:.1%} of its train rows")
    if cv_accuracy < limits["min_cleaner_cv_accuracy"]:
        problems.append(f"cleaner CV accuracy {cv_accuracy:.1%} is below {limits['min_cleaner_cv_accuracy']:.0%}")
    return problems


# ---------------------------------------------------------------------------
# 4: the blind audit
# ---------------------------------------------------------------------------

def make_blind_audit(
    flagged: pd.DataFrame, train_df: pd.DataFrame, n_items: int, seed: int, examples_per_option: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sheet for the owner, answer key) for a random sample of flagged rows.

    The sheet shows each query with two candidate labels, A and B: the given label
    and cleanlab's suggestion, in a random order, so the owner cannot tell which
    one the dataset chose. Each option comes with a few unflagged train queries of
    that intent. The key (kept in a separate file) records which option was given.
    """
    rng = np.random.default_rng(seed)
    clean_train = drop_flagged(train_df, flagged)
    picks = rng.choice(len(flagged), size=min(n_items, len(flagged)), replace=False)

    def examples(intent: str, exclude: str) -> str:
        texts = clean_train.loc[(clean_train["label"] == intent) & (clean_train["text"] != exclude), "text"].tolist()
        chosen = rng.choice(len(texts), size=min(examples_per_option, len(texts)), replace=False)
        return " | ".join(texts[i] for i in sorted(chosen))

    sheet, key = [], []
    for item, row_pos in enumerate(picks, start=1):
        row = flagged.iloc[row_pos]
        given_is_a = bool(rng.random() < 0.5)
        option_a, option_b = (row["given_label"], row["suggested_label"]) if given_is_a else (row["suggested_label"], row["given_label"])
        sheet.append({
            "item": item, "text": row["text"],
            "option_a": option_a, "a_examples": examples(option_a, row["text"]),
            "option_b": option_b, "b_examples": examples(option_b, row["text"]),
            "verdict": "", "notes": "",
        })
        key.append({
            "item": item, "split": row["split"], "source": row["source"], "orig_index": int(row["orig_index"]),
            "given_option": "a" if given_is_a else "b",
            "given_label": row["given_label"], "suggested_label": row["suggested_label"],
        })
    return pd.DataFrame(sheet), pd.DataFrame(key)


def wilson_interval(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n (better than p ± 1.96 SE when n is small)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (float(centre - half), float(centre + half))


def clean_verdicts(sheet: pd.DataFrame) -> pd.Series:
    """The sheet's verdicts indexed by item, trimmed and lowercased ("A " counts as "a")."""
    return sheet.set_index("item")["verdict"].astype(str).str.strip().str.lower()


def audit_categories(verdicts: pd.Series, key: pd.DataFrame) -> pd.Series:
    """Scored category of each verdict, by the rules fixed before the audit:

    real error: the suggested label was picked, or neither label fits
    ambiguous:  both were picked
    false flag: the given label was picked
    """
    given = key.set_index("item")["given_option"].reindex(verdicts.index)
    if given.isna().any():
        raise ValueError(f"items missing from the key: {given[given.isna()].index.tolist()}")
    suggested = given.map({"a": "b", "b": "a"})
    category = pd.Series("unreadable", index=verdicts.index)
    category[(verdicts == suggested) | (verdicts == "neither")] = "real error"
    category[verdicts == "both"] = "ambiguous"
    category[verdicts == given] = "false flag"
    return category


def score_audit(sheet: pd.DataFrame, key: pd.DataFrame) -> dict:
    """Score the verdicts on the sheet against the key (rules in audit_categories).

    Scores only the items on the sheet, so a part of the sheet (one annotator's
    items) can be scored on its own. Returns {"complete": False, ...} until every
    item has a valid verdict.
    """
    verdicts = clean_verdicts(sheet)
    done = verdicts.isin(AUDIT_VERDICTS)
    if not done.all():
        return {"complete": False, "n_done": int(done.sum()), "n_items": len(verdicts),
                "unreadable_items": [int(i) for i, v in verdicts[~done].items() if v not in ("", "nan")]}
    category = audit_categories(verdicts, key)
    suggested = key.set_index("item")["given_option"].map({"a": "b", "b": "a"}).reindex(verdicts.index)
    counts = {
        "real_error": int((category == "real error").sum()),
        "picked_suggestion": int((verdicts == suggested).sum()),
        "neither": int((verdicts == "neither").sum()),
        "ambiguous": int((category == "ambiguous").sum()),
        "false_flag": int((category == "false flag").sum()),
    }
    n = len(verdicts)
    low, high = wilson_interval(counts["real_error"], n)
    return {"complete": True, "n_items": n, **counts,
            "real_error_share": counts["real_error"] / n, "real_error_ci95": [low, high]}


def score_audit_by_annotator(sheet: pd.DataFrame, key: pd.DataFrame) -> dict[str, dict]:
    """score_audit for each annotator on the items they judged (the sheet's "annotator" column)."""
    if "annotator" not in sheet.columns:
        return {}
    return {str(name): score_audit(part, key) for name, part in sheet.groupby("annotator", sort=True)}


def audit_agreement(first: pd.DataFrame, second: pd.DataFrame, key: pd.DataFrame) -> dict:
    """How often two annotators agree on the items both judged.

    Counts agreement on the raw verdict (a, b, both, neither) and on the scored
    category (real error, ambiguous, false flag), and lists every disagreement.
    """
    a, b = clean_verdicts(first), clean_verdicts(second)
    shared = a[a.isin(AUDIT_VERDICTS)].index.intersection(b[b.isin(AUDIT_VERDICTS)].index)
    a, b = a[shared], b[shared]
    cat_a, cat_b = audit_categories(a, key), audit_categories(b, key)
    return {
        "n_items": len(shared),
        "same_verdict": int((a == b).sum()),
        "same_category": int((cat_a == cat_b).sum()),
        "disagreements": [
            {"item": int(i), "first_verdict": a[i], "second_verdict": b[i],
             "first_category": cat_a[i], "second_category": cat_b[i]}
            for i in shared if a[i] != b[i]
        ],
    }
