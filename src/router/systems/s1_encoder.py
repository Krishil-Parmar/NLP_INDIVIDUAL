"""S1: a small pretrained encoder fine-tuned with a softmax classification head.

This is the standard approach for intent classification and the reference point
a marker expects (README section 7.2). The encoder reads the query, a linear
layer on top produces one score (logit) per intent, and the whole network is
trained with cross-entropy loss. Confidence is the maximum softmax probability.

Training is a plain PyTorch loop rather than the Hugging Face Trainer, so every
step is visible and no extra dependency is needed.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from router.data import PROCESSED_DIR
from router.seed import set_seed
from router.systems.base import Predictions, predictions_from_logits


def system_tag(checkpoint: str) -> str:
    """Short name used for result folders, e.g. "s1_distilroberta-base"."""
    return "s1_" + checkpoint.split("/")[-1]


def max_length_for(checkpoint: str) -> int:
    """max_length chosen for this tokeniser in M0 (data/processed/token_lengths.json)."""
    with open(PROCESSED_DIR / "token_lengths.json") as f:
        lengths = json.load(f)
    if checkpoint not in lengths:
        raise KeyError(f"No token length statistics for {checkpoint}. Add it to configs/data.yaml tokenisers.")
    return lengths[checkpoint]["single"]["max_length"]


def load_base(checkpoint: str, label_names: list[str]):
    """Download the pretrained encoder and attach a fresh classification head (one output per intent)."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={name: i for i, name in enumerate(label_names)},
    )
    return model, tokenizer


def _encode(tokenizer, texts: list[str], max_length: int, device: torch.device) -> dict:
    """Tokenise a batch, padding to the longest query in the batch and cutting at max_length."""
    batch = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()  # no gradients needed for prediction: faster and uses less memory
def predict_logits(model, tokenizer, texts: list[str], device: torch.device, max_length: int, batch_size: int = 64) -> np.ndarray:
    """Raw scores (n_queries, n_intents) for a list of queries."""
    model.eval()  # switches off dropout
    chunks = []
    for start in range(0, len(texts), batch_size):
        batch = _encode(tokenizer, texts[start : start + batch_size], max_length, device)
        chunks.append(model(**batch).logits.float().cpu().numpy())
    return np.concatenate(chunks)


def predict(model, tokenizer, texts: list[str], label_names: list[str], device: torch.device, max_length: int, batch_size: int = 64) -> Predictions:
    """Predictions in the shared format (README section 7)."""
    logits = predict_logits(model, tokenizer, list(texts), device, max_length, batch_size)
    return predictions_from_logits(logits, label_names)


def train(
    train_df: pd.DataFrame,
    cfg: dict,
    checkpoint: str,
    seed: int,
    device: torch.device,
    label_names: list[str],
    val_df: pd.DataFrame | None = None,
    max_steps: int | None = None,
):
    """Fine-tune `checkpoint` on the train split.

    val_df (in-scope validation queries) is only used to log accuracy after each
    epoch; it does not change training. max_steps stops early (notebook fast mode
    and timing probes). Returns (model, tokenizer, log) where log has one row per epoch.
    """
    hp = cfg["training"]
    max_length = max_length_for(checkpoint)
    set_seed(seed)  # same seed -> same classifier head initialisation and batch order
    model, tokenizer = load_base(checkpoint, label_names)
    model.to(device)

    texts = train_df["text"].tolist()
    labels = torch.tensor(train_df["label_id"].values)
    # The loader yields shuffled row indices; a seeded generator fixes the order.
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(range(len(texts)), batch_size=hp["batch_size"], shuffle=True, generator=generator)

    total_steps = hp["epochs"] * len(loader)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    # Learning rate rises linearly from 0 during warmup, then falls linearly to 0.
    scheduler = get_linear_schedule_with_warmup(optimizer, int(hp["warmup_fraction"] * total_steps), total_steps)

    log = []
    step = 0
    start_time = time.perf_counter()
    for epoch in range(1, hp["epochs"] + 1):
        model.train()  # switches dropout on
        epoch_losses = []
        for batch_idx in loader:
            idx = batch_idx.tolist()
            batch = _encode(tokenizer, [texts[i] for i in idx], max_length, device)
            output = model(**batch, labels=labels[idx].to(device))  # passing labels makes the model compute cross-entropy loss
            output.loss.backward()  # gradients of the loss for every weight
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["max_grad_norm"])  # cap very large updates
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            epoch_losses.append(output.loss.item())
            step += 1
            if step >= total_steps:
                break
        row = {"epoch": epoch, "steps": step, "train_loss": float(np.mean(epoch_losses)), "seconds": time.perf_counter() - start_time}
        if val_df is not None:
            preds = predict(model, tokenizer, val_df["text"].tolist(), label_names, device, max_length, hp["eval_batch_size"])
            row["val_accuracy"] = float(np.mean(np.array(preds.pred_label) == val_df["label"].values))
        log.append(row)
        print(f"  epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
        if step >= total_steps:
            break
    return model, tokenizer, log


def save(model, tokenizer, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)


def load(folder: Path, device: torch.device):
    """Load a fine-tuned checkpoint saved by save(). Returns (model, tokenizer)."""
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder).to(device)
    model.eval()
    return model, tokenizer
