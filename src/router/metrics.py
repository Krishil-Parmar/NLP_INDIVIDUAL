"""Evaluation metrics (README section 8.1).

- E1 classification: accuracy, precision, recall and F1. Macro averages are
  primary (each intent counts equally); weighted averages are reported next to them.
- E3 out-of-scope detection: AUROC, FPR at 95% TPR and rates at a fixed threshold.
- A paired, stratified bootstrap for comparing two systems.
- E6 router metrics: risk-coverage curve, AURC, coverage at a risk level,
  HANDLE precision and recall, and expected calibration error.

Standard metrics come from scikit-learn, not hand-written code.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


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


# ---------------------------------------------------------------------------
# E3: out-of-scope (OOS) detection.
# is_oos is a True/False array (True = out of scope). oos_score is the system's
# score where higher means "more likely out of scope" (1 - confidence).
# A query is KEPT (treated as in scope) when its oos_score is at or below a threshold.
# ---------------------------------------------------------------------------

def oos_auroc(is_oos: np.ndarray, oos_score: np.ndarray) -> float:
    """Area under the ROC curve for telling OOS queries from in-scope ones.

    It is the chance that a random OOS query gets a higher oos_score than a
    random in-scope query (ties count half). 1.0 = perfect, 0.5 = coin flip.
    """
    return float(roc_auc_score(np.asarray(is_oos, dtype=bool), oos_score))


def in_scope_threshold(in_scope_scores: np.ndarray, keep: float = 0.95) -> float:
    """The lowest threshold that keeps at least `keep` of the in-scope queries.

    Sort the in-scope scores; if 95% of n queries must be kept, the threshold
    is the score of the ceil(0.95 * n)-th lowest query.
    """
    sorted_scores = np.sort(np.asarray(in_scope_scores, dtype=float))
    # round() guards against floating-point error: in Python 0.55 * 100 = 55.00000000000001,
    # which ceil() would turn into 56 instead of 55.
    n_keep = int(np.ceil(round(keep * len(sorted_scores), 9)))
    return float(sorted_scores[n_keep - 1])  # n_keep-th lowest (lists count from 0)


def rates_at_threshold(is_oos: np.ndarray, oos_score: np.ndarray, threshold: float) -> dict[str, float]:
    """At a fixed threshold: the share of in-scope queries kept, and the share of
    OOS queries rejected (OOS recall)."""
    is_oos = np.asarray(is_oos, dtype=bool)
    kept = np.asarray(oos_score) <= threshold
    return {
        "in_scope_kept": float(kept[~is_oos].mean()),  # ~ flips True/False: ~is_oos = in scope
        "oos_recall": float((~kept[is_oos]).mean()),
    }


def fpr_at_tpr(is_oos: np.ndarray, oos_score: np.ndarray, keep: float = 0.95) -> dict[str, float]:
    """FPR at 95% TPR, with in-scope as the positive class (Hendrycks and Gimpel, 2017).

    Choose the threshold that keeps 95% of in-scope queries (the TPR); the FPR
    is the share of OOS queries that are also kept. Lower is better. The
    threshold is returned too, so it can be frozen on validation and reused on test.
    """
    is_oos = np.asarray(is_oos, dtype=bool)
    oos_score = np.asarray(oos_score, dtype=float)
    threshold = in_scope_threshold(oos_score[~is_oos], keep)
    rates = rates_at_threshold(is_oos, oos_score, threshold)
    return {"threshold": threshold, "fpr": 1.0 - rates["oos_recall"], "in_scope_kept": rates["in_scope_kept"]}


# ---------------------------------------------------------------------------
# E6: router metrics (README section 8.1).
# correct is a True/False array: the system's predicted intent is right. An
# out-of-scope query is never right for S0-S3, which cannot answer "oos".
# confidence: higher = more sure. Policy v1 HANDLEs a query when its
# confidence is at or above the threshold, and escalates it otherwise.
# ---------------------------------------------------------------------------

def risk_coverage_curve(correct: np.ndarray, confidence: np.ndarray) -> pd.DataFrame:
    """One row per distinct threshold, from the strictest to the loosest.

    At threshold t (a confidence value that occurs in the data) the router
    handles every query with confidence >= t. coverage = share of queries
    handled; risk = share of the handled queries that are wrong (selective risk).
    """
    correct = np.asarray(correct, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    order = np.argsort(-confidence, kind="stable")  # most confident first
    sorted_conf = confidence[order]
    errors_so_far = np.cumsum(~correct[order])  # running count of wrong answers
    # The last position of each run of equal confidence: every query up to it is handled at that threshold.
    last = np.flatnonzero(np.r_[sorted_conf[1:] != sorted_conf[:-1], True])
    n_handled = last + 1
    return pd.DataFrame({
        "threshold": sorted_conf[last],
        "coverage": n_handled / len(sorted_conf),
        "risk": errors_so_far[last] / n_handled,
        "n_handled": n_handled,
    })


def aurc(curve: pd.DataFrame) -> float:
    """Area under the risk-coverage curve, as a step function: each point's risk
    times the extra coverage it adds. Lower is better (Geifman and El-Yaniv, 2017)."""
    coverage_added = np.diff(np.r_[0.0, curve["coverage"].to_numpy()])
    return float(np.sum(curve["risk"].to_numpy() * coverage_added))


def coverage_at_risk(correct: np.ndarray, confidence: np.ndarray, max_risk: float) -> dict[str, float]:
    """The highest coverage whose selective risk is at or below max_risk, with its threshold.

    If no threshold qualifies, the router handles nothing: coverage 0 and an
    infinite threshold (no confidence can reach it).
    """
    curve = risk_coverage_curve(correct, confidence)
    allowed = curve[curve["risk"] <= max_risk + 1e-12]  # tiny tolerance for floating-point ties
    if allowed.empty:
        return {"coverage": 0.0, "threshold": float("inf"), "risk": 0.0}
    best = allowed.loc[allowed["coverage"].idxmax()]
    return {"coverage": float(best["coverage"]), "threshold": float(best["threshold"]), "risk": float(best["risk"])}


def handle_precision_recall(correct: np.ndarray, handled: np.ndarray) -> dict[str, float]:
    """HANDLE precision: of the queries handled, the share handled correctly (1 - selective risk).
    HANDLE recall: of the queries the system got right, the share it handled instead of escalating.
    NaN when there is nothing to divide by."""
    correct = np.asarray(correct, dtype=bool)
    handled = np.asarray(handled, dtype=bool)
    precision = float(correct[handled].mean()) if handled.any() else float("nan")
    recall = float(handled[correct].mean()) if correct.any() else float("nan")
    return {"handle_precision": precision, "handle_recall": recall}


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error (Guo et al., 2017).

    Split queries into equal-width confidence bins; in each bin compare the mean
    confidence with the actual accuracy; average the gaps, weighted by how many
    queries fall in each bin. 0 = perfectly calibrated.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_of = np.digitize(confidence, edges[1:-1], right=True)  # bin index 0 .. n_bins-1
    ece = 0.0
    for b in range(n_bins):
        in_bin = bin_of == b
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Comparing two systems: paired, stratified bootstrap (rigour rule 3)
# ---------------------------------------------------------------------------

def paired_bootstrap_diff(
    metric_a: Callable[[np.ndarray], float],
    metric_b: Callable[[np.ndarray], float],
    strata: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict[str, float]:
    """metric_a minus metric_b, with a bootstrap confidence interval.

    metric_a and metric_b each take an array of row indices and return one
    number for that subset of queries (for example, system A's AUROC on those rows).

    Each resample draws rows with replacement, separately inside each stratum
    (for E3: in-scope and OOS), so every resample keeps the same class counts.
    Both systems are scored on the same rows (paired), so query difficulty they
    share cancels out of the difference. The interval is the middle `ci` share
    of the resampled differences (percentile interval).
    """
    rng = np.random.default_rng(seed)
    strata = np.asarray(strata)
    groups = [np.flatnonzero(strata == value) for value in np.unique(strata)]  # row indices of each stratum
    diffs = np.empty(n_resamples)
    for r in range(n_resamples):
        rows = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
        diffs[r] = metric_a(rows) - metric_b(rows)
    all_rows = np.arange(len(strata))
    tail = (1.0 - ci) / 2  # e.g. 0.025 on each side for a 95% interval
    low, high = np.quantile(diffs, [tail, 1.0 - tail])
    return {
        "diff": float(metric_a(all_rows) - metric_b(all_rows)),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_resamples": n_resamples,
    }
