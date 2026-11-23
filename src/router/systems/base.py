"""The shared output every system produces (README section 7).

For each query, every system gives:
    pred_label  the predicted intent name
    confidence  in-scope confidence: how sure it is about pred_label (higher = surer)
    oos_score   out-of-scope score (higher = more likely out of scope)

Systems with a softmax (S0, S1) also keep their raw logits, so temperature
scaling (M4) can be fitted later from saved files without retraining.

Prediction files are saved per system, seed and split:
    results/predictions/<system_tag>/seed<n>_<split>.csv
    results/predictions/<system_tag>/seed<n>_<split>_logits.npy
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from router.data import RESULTS_DIR, load_label_map

PREDICTIONS_DIR = RESULTS_DIR / "predictions"
# E2: a system trained on cleaned labels saves its runs under "<tag>_cleaned".
CLEANED_SUFFIX = "_cleaned"


@dataclass
class Predictions:
    """One system's output for a list of queries (all arrays have one entry per query)."""

    pred_label: list[str]
    confidence: np.ndarray
    oos_score: np.ndarray
    logits: np.ndarray | None = None  # raw per-intent scores, shape (n_queries, n_intents), if the system has them
    matched_example: list[str] | None = None  # S2/S3: the labelled example the query was matched to (shown in the demo)


def label_names_in_id_order() -> list[str]:
    """Intent names ordered by label_id, so label_names[i] is the name of class i."""
    label_map = load_label_map()
    return sorted(label_map, key=label_map.get)


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Turn raw scores into probabilities that sum to 1 in each row.

    Dividing by a temperature above 1 makes the probabilities less extreme
    (used for calibration in M4). It never changes which class is highest.
    """
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)  # subtract each row's max so exp() cannot overflow
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predictions_from_logits(
    logits: np.ndarray, label_names: list[str], temperature: float = 1.0
) -> Predictions:
    """Build Predictions from raw scores.

    confidence is the maximum softmax probability. oos_score is 1 - confidence:
    the simple baseline of Hendrycks and Gimpel (2017), where low confidence
    is taken as a sign the query is out of scope.
    """
    probs = softmax(logits, temperature)
    best = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    return Predictions(
        pred_label=[label_names[i] for i in best],
        confidence=confidence,
        oos_score=1.0 - confidence,
        logits=logits,
    )


# ---------------------------------------------------------------------------
# Saving and loading prediction files
# ---------------------------------------------------------------------------

def _paths(system_tag: str, seed: int, split: str) -> tuple[Path, Path]:
    folder = PREDICTIONS_DIR / system_tag
    return folder / f"seed{seed}_{split}.csv", folder / f"seed{seed}_{split}_logits.npy"


def save_predictions(
    system_tag: str, seed: int, split: str, eval_df: pd.DataFrame, preds: Predictions
) -> None:
    """Save one run's predictions next to the true labels, row for row."""
    if len(eval_df) != len(preds.pred_label):
        raise ValueError("eval_df and predictions have different lengths")
    csv_path, logits_path = _paths(system_tag, seed, split)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "text": eval_df["text"].values,
            "label": eval_df["label"].values,
            "is_oos": eval_df["is_oos"].values,
            "pred_label": preds.pred_label,
            "confidence": preds.confidence,
            "oos_score": preds.oos_score,
        }
    )
    if preds.matched_example is not None:
        out["matched_example"] = preds.matched_example
    out.to_csv(csv_path, index=False)
    if preds.logits is not None:
        np.save(logits_path, preds.logits.astype(np.float32))


def load_predictions(system_tag: str, seed: int, split: str) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Load (predictions table, logits or None) for one run."""
    csv_path, logits_path = _paths(system_tag, seed, split)
    df = pd.read_csv(csv_path, keep_default_na=False)
    logits = np.load(logits_path) if logits_path.exists() else None
    return df, logits


def list_runs(split: str, include_cleaned: bool = False) -> dict[str, list[int]]:
    """Every saved run for a split, as {system_tag: [seeds]}.

    Runs trained on cleaned labels (E2) are left out unless include_cleaned is
    True, so E1, E3, E6 and checkpoint selection only ever see the original runs.
    """
    runs: dict[str, list[int]] = {}
    if not PREDICTIONS_DIR.exists():
        return runs
    # The split must match exactly: a plain glob "seed*_test.csv" would also match "seed0_jeff_test.csv".
    pattern = re.compile(rf"seed(\d+)_{re.escape(split)}\.csv")
    for folder in sorted(PREDICTIONS_DIR.iterdir()):
        if folder.name.endswith(CLEANED_SUFFIX) and not include_cleaned:
            continue
        seeds = sorted(int(match.group(1)) for p in folder.glob("seed*.csv") if (match := pattern.fullmatch(p.name)))
        if seeds:
            runs[folder.name] = seeds
    return runs


def save_run_info(system_tag: str, seed: int, info: dict) -> None:
    """Save provenance and settings for one training run as JSON next to its predictions."""
    path = PREDICTIONS_DIR / system_tag / f"seed{seed}_run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)
