"""S3: pairwise paraphrase matcher. The main system (topic 2 used as the method).

Instead of choosing among 77 fixed outputs, S3 asks one question about a pair of
queries: "do these two mean the same thing?". A cross-encoder reads both texts
together and outputs one score. To classify a query:
  1. S2 (embeddings) shortlists candidate labelled examples: the `top_intents`
     most similar intents, with their `examples_per_intent` closest examples each.
  2. The cross-encoder scores every (query, candidate example) pair.
  3. The intent of the best-scoring example wins. Confidence is that pair's
     probability of "same meaning", and that example is shown as the evidence.

The cross-encoder starts from a checkpoint already trained on paraphrase
detection (Quora question pairs) and is fine-tuned on pairs built from the
Banking77 train split. Approach after Zhang et al. (2020).
"""

import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from router.seed import set_seed
from router.systems.base import Predictions
from router.systems.s2_embedding import SupportSet, embed


def system_tag(checkpoint: str) -> str:
    """Short name used for result folders, e.g. "s3_quora-distilroberta-base"."""
    return "s3_" + checkpoint.split("/")[-1]


# ---------------------------------------------------------------------------
# Candidate selection (uses S2 similarities)
# ---------------------------------------------------------------------------

def candidate_examples(
    similarity_row: np.ndarray, support_label_ids: np.ndarray, top_intents: int, per_intent: int,
    exclude: int | None = None,
) -> list[int]:
    """Indices of support examples to pair with one query.

    Walks the support examples from most to least similar. Intents are taken in
    the order they first appear (that is, by their closest example), keeping the
    first `top_intents` intents and up to `per_intent` examples of each.
    `exclude` skips one index (the query itself, when building training pairs).
    """
    chosen: list[int] = []
    taken: dict[int, int] = {}  # intent id -> examples taken so far
    for idx in np.argsort(-similarity_row):  # most similar first
        if idx == exclude:
            continue
        intent = int(support_label_ids[idx])
        if intent not in taken:
            if len(taken) == top_intents:
                continue  # already have enough intents; only fill the ones we have
            taken[intent] = 0
        if taken[intent] < per_intent:
            chosen.append(int(idx))
            taken[intent] += 1
        if len(taken) == top_intents and all(n == per_intent for n in taken.values()):
            break
    return chosen


def similarities_in_chunks(query_embeddings: np.ndarray, support_embeddings: np.ndarray, chunk: int = 1024):
    """Yield (start_row, similarity block) so a large query set never needs one huge matrix."""
    for start in range(0, len(query_embeddings), chunk):
        yield start, query_embeddings[start : start + chunk] @ support_embeddings.T


def make_training_pairs(support: SupportSet, n_negatives: int, rng: np.random.Generator) -> pd.DataFrame:
    """Labelled pairs built from the train split only.

    For each train query:
      - 1 positive: a random OTHER train query with the same intent (label 1)
      - n_negatives hard negatives: the closest example of each of the
        n_negatives most similar OTHER intents, found with S2 (label 0)
    Hard negatives teach the model the near misses it will meet at prediction time.
    """
    rows = []
    label_ids = support.label_ids
    for start, block in similarities_in_chunks(support.embeddings, support.embeddings):
        for offset, sim_row in enumerate(block):
            i = start + offset
            same = np.flatnonzero(label_ids == label_ids[i])
            same = same[same != i]  # never pair a query with itself
            if len(same):
                rows.append((support.texts[i], support.texts[int(rng.choice(same))], 1.0))
            # Closest example of each of the n_negatives + 1 most similar intents. One extra
            # is requested because the query's own intent is usually among them and is dropped.
            nearest_per_intent = candidate_examples(sim_row, label_ids, n_negatives + 1, 1, exclude=i)
            negatives = [j for j in nearest_per_intent if label_ids[j] != label_ids[i]][:n_negatives]
            for j in negatives:
                rows.append((support.texts[i], support.texts[j], 0.0))
    return pd.DataFrame(rows, columns=["text_a", "text_b", "label"])


def shortlist_recall(encoder, support: SupportSet, texts: list[str], label_ids: np.ndarray, cfg: dict) -> float:
    """Share of queries whose true intent is in S2's shortlist. S3 can never be right above this."""
    query_embeddings = embed(encoder, list(texts))
    hits = []
    for start, block in similarities_in_chunks(query_embeddings, support.embeddings):
        for offset, sim_row in enumerate(block):
            cands = candidate_examples(sim_row, support.label_ids, cfg["retrieval"]["top_intents"], cfg["retrieval"]["examples_per_intent"])
            hits.append(label_ids[start + offset] in set(support.label_ids[cands]))
    return float(np.mean(hits))


# ---------------------------------------------------------------------------
# Cross-encoder
# ---------------------------------------------------------------------------

def load_base(checkpoint: str):
    """The pretrained paraphrase cross-encoder (one output: same meaning or not), in float32."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=1, dtype=torch.float32)
    return model, tokenizer


def _encode_pairs(tokenizer, a: list[str], b: list[str], max_length: int, device) -> dict:
    """Tokenise pairs together, the way a cross-encoder reads them: [query] [SEP] [example]."""
    batch = tokenizer(a, b, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def score_pairs(model, tokenizer, a: list[str], b: list[str], device, max_length: int, batch_size: int = 128) -> np.ndarray:
    """Raw score (logit) for each pair. sigmoid(score) is the probability of "same meaning"."""
    model.eval()
    out = []
    for start in range(0, len(a), batch_size):
        batch = _encode_pairs(tokenizer, a[start : start + batch_size], b[start : start + batch_size], max_length, device)
        out.append(model(**batch).logits.float().squeeze(-1).cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def train(pairs: pd.DataFrame, cfg: dict, checkpoint: str, seed: int, device, max_length: int, max_steps: int | None = None):
    """Fine-tune the cross-encoder on labelled pairs with binary cross-entropy.

    Returns (model, tokenizer, log) where log has one row per epoch.
    """
    hp = cfg["training"]
    set_seed(seed)
    model, tokenizer = load_base(checkpoint)
    model.to(device)
    a, b = pairs["text_a"].tolist(), pairs["text_b"].tolist()
    labels = torch.tensor(pairs["label"].to_numpy(), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(range(len(a)), batch_size=hp["batch_size"], shuffle=True, generator=generator)

    total_steps = hp["epochs"] * len(loader)
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(hp["warmup_fraction"] * total_steps), total_steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()  # sigmoid + binary cross-entropy in one stable step

    log, step, start_time = [], 0, time.perf_counter()
    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        losses = []
        for batch_idx in loader:
            idx = batch_idx.tolist()
            batch = _encode_pairs(tokenizer, [a[i] for i in idx], [b[i] for i in idx], max_length, device)
            logits = model(**batch).logits.squeeze(-1)
            loss = loss_fn(logits, labels[idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
            step += 1
            if step >= total_steps:
                break
        row = {"epoch": epoch, "steps": step, "train_loss": float(np.mean(losses)), "seconds": time.perf_counter() - start_time}
        log.append(row)
        print(f"  epoch {epoch}: steps={step}, train_loss={row['train_loss']:.4f}, seconds={row['seconds']:.1f}")
        if step >= total_steps:
            break
    return model, tokenizer, log


def predict(
    model, tokenizer, encoder, support: SupportSet, texts: list[str], label_names: list[str],
    cfg: dict, device, max_length: int,
) -> Predictions:
    """Shortlist with S2, score the pairs, and keep the best pair per query."""
    texts = list(texts)
    query_embeddings = embed(encoder, texts)
    candidates: list[list[int]] = []
    for _, block in similarities_in_chunks(query_embeddings, support.embeddings):
        for sim_row in block:
            candidates.append(candidate_examples(sim_row, support.label_ids, cfg["retrieval"]["top_intents"], cfg["retrieval"]["examples_per_intent"]))

    # Flatten all pairs, score them in batches, then cut the scores back per query.
    a = [texts[q] for q, cands in enumerate(candidates) for _ in cands]
    b = [support.texts[j] for cands in candidates for j in cands]
    scores = score_pairs(model, tokenizer, a, b, device, max_length, cfg["score_batch_size"])

    pred_label, confidence, matched = [], [], []
    position = 0
    for cands in candidates:
        query_scores = scores[position : position + len(cands)]
        position += len(cands)
        best = int(np.argmax(query_scores))
        pred_label.append(label_names[support.label_ids[cands[best]]])
        confidence.append(1.0 / (1.0 + np.exp(-query_scores[best])))  # sigmoid: logit -> probability
        matched.append(support.texts[cands[best]])
    confidence = np.array(confidence)
    return Predictions(pred_label=pred_label, confidence=confidence, oos_score=1.0 - confidence, matched_example=matched)


def save(model, tokenizer, folder) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(folder)
    tokenizer.save_pretrained(folder)


def load(folder, device):
    tokenizer = AutoTokenizer.from_pretrained(folder)
    model = AutoModelForSequenceClassification.from_pretrained(folder, dtype=torch.float32).to(device)
    model.eval()
    return model, tokenizer
