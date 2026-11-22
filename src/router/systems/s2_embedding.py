"""S2: embedding nearest neighbour. No training.

A frozen sentence-embedding model turns every query into a vector. The labelled
train queries (the "support set") are embedded once. A new query is compared to
every support example by cosine similarity; each intent scores the similarity of
its closest example, and the best intent wins.

Confidence is that best similarity (cosine, between -1 and 1). A query far from
every labelled example gets low confidence, which is also why S2 works as the
out-of-scope signal and as the retriever for S3 (README section 7.2).
A new intent is added by adding its examples to the support set, no retraining.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from router.systems.base import Predictions


@dataclass
class SupportSet:
    """The labelled examples a query is compared against."""

    texts: list[str]
    label_ids: np.ndarray  # intent id of each example
    embeddings: np.ndarray  # one unit-length row per example


def system_tag(model_name: str) -> str:
    """Short name used for result folders, e.g. "s2_all-MiniLM-L6-v2"."""
    return "s2_" + model_name.split("/")[-1]


def load_encoder(model_name: str, device) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=str(device))


def embed(encoder: SentenceTransformer, texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Unit-length embeddings, one row per text. For unit vectors, a dot product is the cosine similarity."""
    return encoder.encode(
        list(texts), batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )


def build_support(encoder: SentenceTransformer, train_df: pd.DataFrame, batch_size: int = 64) -> SupportSet:
    """Embed the labelled train queries once."""
    return SupportSet(
        texts=train_df["text"].tolist(),
        label_ids=train_df["label_id"].to_numpy(),
        embeddings=embed(encoder, train_df["text"].tolist(), batch_size),
    )


def class_scores(similarities: np.ndarray, support_label_ids: np.ndarray, n_classes: int) -> np.ndarray:
    """For each query and each intent: the similarity of that intent's closest support example.

    similarities has shape (n_queries, n_support). Returns (n_queries, n_classes).
    An intent with no support examples scores -inf, so it can never win.
    """
    scores = np.full((similarities.shape[0], n_classes), -np.inf)
    for c in range(n_classes):
        columns = np.flatnonzero(support_label_ids == c)  # support examples of intent c
        if len(columns):
            scores[:, c] = similarities[:, columns].max(axis=1)
    return scores


def predict(
    encoder: SentenceTransformer, support: SupportSet, texts: list[str], label_names: list[str], batch_size: int = 64
) -> Predictions:
    """Predictions in the shared format, plus the nearest support example for each query."""
    similarities = embed(encoder, texts, batch_size) @ support.embeddings.T
    scores = class_scores(similarities, support.label_ids, len(label_names))
    best_intent = scores.argmax(axis=1)
    confidence = scores.max(axis=1)
    nearest = similarities.argmax(axis=1)  # the single closest example (it belongs to best_intent)
    return Predictions(
        pred_label=[label_names[i] for i in best_intent],
        confidence=confidence,
        oos_score=1.0 - confidence,
        logits=scores,
        matched_example=[support.texts[i] for i in nearest],
    )
