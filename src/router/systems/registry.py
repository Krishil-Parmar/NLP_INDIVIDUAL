"""The shared scoring interface: any trained system as one function, texts in, Predictions out.

    predict = load_predictor("s3")
    preds = predict(["where is my card?"])

The demo, the notebook and the latency measurement all use this, so they call
every system in exactly the same way. Training is done by scripts/train.py.
"""

import hashlib
import json
from collections.abc import Callable

import numpy as np
import pandas as pd

from router import data
from router.seed import pick_device
from router.systems import s0_tfidf, s1_encoder, s2_embedding, s3_pairwise
from router.systems.base import CLEANED_SUFFIX, Predictions, label_names_in_id_order

CHECKPOINT_DIR = data.PROJECT_ROOT / "checkpoints"
SYSTEMS = ["s0", "s1", "s2", "s3"]
CONFIG_FOR_SYSTEM = {"s1": "s1_encoder", "s2": "s2_embedding", "s3": "s3_pairwise"}


def system_and_checkpoint(tag: str) -> tuple[str, str | None]:
    """Turn a result folder name back into (system, model ID), e.g.
    "s2_all-mpnet-base-v2" -> ("s2", "sentence-transformers/all-mpnet-base-v2")."""
    system = tag.split("_")[0]  # the part before the first underscore: "s0", "s1", ...
    if system == "s0":
        return "s0", None
    cfg = data.load_config(CONFIG_FOR_SYSTEM[system])
    for name in cfg.get("candidates", []) + [cfg["checkpoint"]]:
        if f"{system}_{name.split('/')[-1]}" == tag:
            return system, name
    raise ValueError(f"No model ID in the configs matches result folder {tag!r}")


def run_tag(tag: str, labels: str = "original") -> str:
    """The run tag for a system trained on "original" or "cleaned" labels (E2)."""
    return tag + CLEANED_SUFFIX if labels == "cleaned" else tag


def s0_checkpoint(seed: int, labels: str = "original"):
    """Where an S0 model is saved: checkpoints/s0/seed<n>.joblib, or checkpoints/s0_cleaned/... for E2."""
    return CHECKPOINT_DIR / run_tag("s0", labels) / f"seed{seed}.joblib"


def s2_support_set(
    encoder, model_name: str, batch_size: int = 64, labels: str = "original", support_df: pd.DataFrame | None = None,
) -> s2_embedding.SupportSet:
    """The S2 support set (train split), with its embeddings cached on disk.

    labels="cleaned" uses the train split without the rows cleanlab flagged (E2).
    support_df replaces the train split with other support examples (columns
    text and label_id), as E7 does with the Jeff support split.
    The cache file name includes a hash of the texts, so a changed support set
    can never reuse stale embeddings.
    """
    train_df = support_df if support_df is not None else data.load_train(labels)
    digest = hashlib.sha1("\n".join(train_df["text"]).encode()).hexdigest()[:10]
    cache = CHECKPOINT_DIR / "s2" / s2_embedding.system_tag(model_name) / f"support_{digest}.npy"
    if cache.exists():
        return s2_embedding.SupportSet(train_df["text"].tolist(), train_df["label_id"].to_numpy(), np.load(cache))
    support = s2_embedding.build_support(encoder, train_df, batch_size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, support.embeddings)
    return support


def load_predictor(
    system: str, seed: int = 0, device=None, checkpoint: str | None = None, labels: str = "original",
    support: tuple[pd.DataFrame, list[str]] | None = None,
) -> Callable[[list[str]], Predictions]:
    """Load a trained system and return its predict function.

    checkpoint overrides the chosen model in the system's config (used when
    comparing candidates). labels="cleaned" loads the E2 model trained without the
    flagged rows. support = (support DataFrame with text and label_id, label names)
    gives S2 or S3 a different support set and label set, with no retraining (E7).
    Raises FileNotFoundError if the system was not trained.
    """
    device = device or pick_device()
    if support is not None and system not in ("s2", "s3"):
        raise ValueError(f"{system} is a softmax over Banking77 labels; only s2 and s3 can take a new support set")
    support_df, label_names = support if support is not None else (None, label_names_in_id_order())

    if system == "s0":
        model = s0_tfidf.load(s0_checkpoint(seed, labels))
        return lambda texts: s0_tfidf.predict(model, list(texts), label_names)

    if system == "s1":
        name = checkpoint or data.load_config("s1_encoder")["checkpoint"]
        folder = CHECKPOINT_DIR / "s1" / run_tag(s1_encoder.system_tag(name), labels) / f"seed{seed}"
        if not folder.exists():
            raise FileNotFoundError(folder)
        model, tokenizer = s1_encoder.load(folder, device)
        max_length = s1_encoder.max_length_for(name)
        return lambda texts: s1_encoder.predict(model, tokenizer, list(texts), label_names, device, max_length)

    if system in ("s2", "s3"):
        s2_cfg = data.load_config("s2_embedding")
        s2_name = (checkpoint if system == "s2" and checkpoint else None) or s2_cfg["checkpoint"]
        if not s2_name:
            raise ValueError("No S2 model chosen yet in configs/s2_embedding.yaml")
        encoder = s2_embedding.load_encoder(s2_name, device)
        support_set = s2_support_set(encoder, s2_name, s2_cfg["batch_size"], labels, support_df)
        if system == "s2":
            return lambda texts: s2_embedding.predict(encoder, support_set, list(texts), label_names, s2_cfg["batch_size"])

        s3_cfg = data.load_config("s3_pairwise")
        s3_name = checkpoint or s3_cfg["checkpoint"]
        folder = CHECKPOINT_DIR / "s3" / run_tag(s3_pairwise.system_tag(s3_name), labels) / f"seed{seed}"
        if not folder.exists():
            raise FileNotFoundError(folder)
        model, tokenizer = s3_pairwise.load(folder, device)
        max_length = s3_max_length(s3_cfg)
        return lambda texts: s3_pairwise.predict(model, tokenizer, encoder, support_set, list(texts), label_names, s3_cfg, device, max_length)

    raise ValueError(f"Unknown system {system!r}. Choose from {SYSTEMS}")


def s3_max_length(s3_cfg: dict) -> int:
    """Pair max_length for S3, from the M0 token length statistics of its tokeniser."""
    with open(data.PROCESSED_DIR / "token_lengths.json") as f:
        return json.load(f)[s3_cfg["length_reference"]]["pair"]["max_length"]
