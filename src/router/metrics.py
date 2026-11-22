"""Evaluation metrics.

This file currently holds the E1 classification metrics (README section 8.1):
accuracy, precision, recall and F1. Macro averages are primary (each intent
counts equally); weighted averages are reported next to them.
Risk-coverage, AURC, out-of-scope metrics and bootstrap intervals come in M4.

Everything is computed with scikit-learn, not by hand.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str] | None = None
) -> dict[str, float]:
    """Accuracy plus macro and weighted precision, recall and F1.

    labels fixes which intents are averaged over (pass all 77 so every
    system is averaged over the same set). zero_division=0 means an intent
    that is never predicted gets precision 0 instead of an error.
    """
    result = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for average in ["macro", "weighted"]:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        )
        result[f"{average}_precision"] = float(p)
        result[f"{average}_recall"] = float(r)
        result[f"{average}_f1"] = float(f)
    return result


def per_intent_scores(y_true: list[str], y_pred: list[str], labels: list[str]) -> pd.DataFrame:
    """Precision, recall, F1 and support (number of true queries) for each intent."""
    p, r, f, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": support}, index=pd.Index(labels, name="intent")
    )


def worst_intents(per_intent: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """The k intents with the lowest F1 (ties broken by intent name, so the order is stable)."""
    return per_intent.reset_index().sort_values(["f1", "intent"]).head(k).set_index("intent")


def confusion_counts(y_true: list[str], y_pred: list[str], labels: list[str]) -> np.ndarray:
    """Confusion matrix: entry [i, j] counts queries of intent i predicted as intent j."""
    return confusion_matrix(y_true, y_pred, labels=labels)


def top_confused_pairs(matrix: np.ndarray, labels: list[str], k: int = 10) -> pd.DataFrame:
    """The k most frequent (true intent, predicted intent) mistakes from a confusion matrix."""
    rows = []
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            if i != j and matrix[i, j] > 0:  # off-diagonal cells are mistakes
                rows.append({"true": true_label, "predicted": pred_label, "count": int(matrix[i, j])})
    if not rows:
        return pd.DataFrame(columns=["true", "predicted", "count"])
    pairs = pd.DataFrame(rows)
    # Most frequent first; names break ties so the order never changes between runs.
    return pairs.sort_values(["count", "true", "predicted"], ascending=[False, True, True]).head(k).reset_index(drop=True)


def mean_std_over_seeds(per_seed: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean and standard deviation of each metric across seeds.

    Uses the sample standard deviation (ddof=1), the usual choice when the
    seeds are a sample of all possible seeds. With one seed the std is 0.
    """
    summary = {}
    for key in per_seed[0]:
        values = np.array([run[key] for run in per_seed])
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[key] = {"mean": float(values.mean()), "std": std}
    return summary
