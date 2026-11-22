"""S0: TF-IDF features + logistic regression. The classical baseline.

TF-IDF turns each query into a sparse vector of word (and word-pair) weights.
Logistic regression learns one weight per feature per intent. Confidence is
the maximum predicted probability. This shows how much of the task can be
solved by keywords alone (README section 7.2).
"""

from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from router.systems.base import Predictions, predictions_from_logits


def build_model(ngram_range: tuple[int, int], C: float, cfg: dict, seed: int) -> Pipeline:
    """An untrained TF-IDF + logistic regression pipeline."""
    vectoriser = TfidfVectorizer(
        lowercase=cfg["vectoriser"]["lowercase"],
        sublinear_tf=cfg["vectoriser"]["sublinear_tf"],
        ngram_range=tuple(ngram_range),
    )
    classifier = LogisticRegression(C=C, max_iter=cfg["max_iter"], random_state=seed)
    # A Pipeline runs the steps in order: text -> TF-IDF vector -> classifier.
    return Pipeline([("tfidf", vectoriser), ("lr", classifier)])


def train(train_df: pd.DataFrame, params: dict, cfg: dict, seed: int) -> Pipeline:
    """Fit S0 on the train split. params holds the chosen ngram_range and C."""
    model = build_model(params["ngram_range"], params["C"], cfg, seed)
    model.fit(train_df["text"], train_df["label_id"])
    return model


def predict(model: Pipeline, texts: list[str], label_names: list[str]) -> Predictions:
    """Predict intents for a list of queries.

    decision_function gives one raw score per intent (the logits). For this
    multinomial model, softmax of these scores equals predict_proba.
    """
    # classes_ must be 0..76 in order so that column i is intent label_names[i].
    assert list(model.classes_) == list(range(len(label_names))), "class order mismatch"
    logits = model.decision_function(list(texts))
    if logits.ndim == 1:
        # With exactly two classes sklearn returns one score s per query (for class 1).
        # softmax([0, s]) gives the same probabilities as predict_proba, so use that.
        logits = np.column_stack([np.zeros_like(logits), logits])
    return predictions_from_logits(logits, label_names)


def grid_search(
    train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: dict, label_names: list[str]
) -> tuple[dict, list[dict]]:
    """Try every (ngram_range, C) in the config grid; pick the best validation macro-F1.

    val_df must contain in-scope queries only. Ties keep the earlier grid entry.
    Returns (best_params, one result row per grid point).
    """
    seed = cfg["seeds"][0]
    rows = []
    best_params, best_f1 = None, -1.0
    for ngram_range, C in product(cfg["grid"]["ngram_range"], cfg["grid"]["C"]):
        params = {"ngram_range": list(ngram_range), "C": C}
        model = train(train_df, params, cfg, seed)
        preds = predict(model, val_df["text"].tolist(), label_names)
        macro_f1 = f1_score(val_df["label"], preds.pred_label, labels=label_names, average="macro")
        rows.append({**params, "val_macro_f1": float(macro_f1)})
        if macro_f1 > best_f1:
            best_params, best_f1 = params, macro_f1
    return best_params, rows


def save(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load(path: Path) -> Pipeline:
    return joblib.load(path)
