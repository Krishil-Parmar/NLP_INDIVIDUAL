"""E8: fixing S2 for a small new intent set (README section 8, E8). Post-hoc, not pre-registered.

The diagnosis of the Jeff failure (scripts/diagnose_jeff.py, results/e7_diagnosis.md;
wording revised after two reviews on 24 Nov 2026) found:
  1. S2's confidence (the cosine to the single closest support example) no longer
     separates its right answers from its wrong ones, so no threshold can keep one and drop the other.
  2. Jeff queries sit far from every labelled example: few have a support example at
     cosine 0.7 or more, while most Banking77 queries do, even with 10 examples per intent.
     At equal distance S2 is no less accurate on Jeff. Why Jeff queries are far (more varied
     wording, or intents 10 examples cannot cover) is not known.
  3. Only 10 support examples per intent (Banking77 has about 120), and a nearest-example
     rule that uses them in the simplest way.

Each candidate fix targets one cause. A variant is three choices:
  encoder: "frozen" (MiniLM as S2 uses it) or "tuned" (fine-tuned on the support set,
           SetFit-style; Tunstall et al., 2022)                          -> cause 2
  support: "examples" or "descriptions" (the examples plus one extra support row per
           intent: its definition from configs/jeff.yaml)                 -> causes 2 and 3
  scoring: "nearest" (S2's rule: the cosine of the closest example), "margin" (the best
           intent's score minus the runner-up's) or "head" (logistic regression
           trained on the support embeddings; confidence = its top probability) -> cause 1

Variants fall into three tiers by what adding a new intent then costs:
  no training (frozen encoder, nearest or margin): add its examples, nothing else.
  head only (frozen encoder, head): refit a small classifier, about a second.
  encoder fine-tune (tuned): retrain the encoder, about a minute on a laptop.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from router.metrics import aurc, classification_metrics, coverage_at_risk, oos_auroc, risk_coverage_curve
from router.seed import set_seed
from router.systems.s2_embedding import class_scores, embed

TIERS = ["no training", "head only", "encoder fine-tune"]


@dataclass(frozen=True)
class Variant:
    encoder: str  # "frozen" or "tuned"
    support: str  # "examples" or "descriptions"
    scoring: str  # "nearest", "margin" or "head"

    @property
    def name(self) -> str:
        return f"{self.encoder}/{self.support}/{self.scoring}"

    @property
    def tier(self) -> str:
        """What adding a new intent costs with this variant (see the module docstring)."""
        if self.encoder == "tuned":
            return "encoder fine-tune"
        return "head only" if self.scoring == "head" else "no training"

    @property
    def simplicity(self) -> int:
        """Lower is simpler: used to break near-ties inside a tier (examples before descriptions, nearest first)."""
        return ["examples", "descriptions"].index(self.support) * 3 + ["nearest", "margin", "head"].index(self.scoring)


def all_variants(supports: list[str]) -> list[Variant]:
    """Every combination of encoder, support and scoring, S2 itself (frozen/examples/nearest) first."""
    return [Variant(e, s, c) for e, s, c in product(["frozen", "tuned"], supports, ["nearest", "margin", "head"])]


def per_intent_sample(df: pd.DataFrame, k: int, seed: int) -> pd.DataFrame:
    """k random rows of each label_id (all of them when an intent has fewer), in their original order."""
    rng = np.random.default_rng(seed)
    picked = []
    for label_id in sorted(df["label_id"].unique()):
        rows = np.flatnonzero(df["label_id"].to_numpy() == label_id)
        picked.append(rng.choice(rows, size=min(k, len(rows)), replace=False))
    return df.iloc[np.sort(np.concatenate(picked))].reset_index(drop=True)


def add_descriptions(support: pd.DataFrame, descriptions: dict[str, str], label_names: list[str]) -> pd.DataFrame:
    """The support set plus one row per intent whose text is that intent's definition."""
    extra = pd.DataFrame({"text": [descriptions[name] for name in label_names], "label_id": range(len(label_names))})
    return pd.concat([support[["text", "label_id"]], extra], ignore_index=True)


def margin_scores(scores: np.ndarray) -> np.ndarray:
    """For each query: the best intent's score minus the second best's. A small margin means
    two intents are nearly tied, which is exactly when a neighbouring-intent error is likely."""
    top_two = np.sort(scores, axis=1)[:, -2:]  # the two highest scores, ascending
    return top_two[:, 1] - top_two[:, 0]


def setfit_pairs(texts: list[str], label_ids: np.ndarray, iterations: int, rng: np.random.Generator):
    """SetFit training pairs: for every support example, `iterations` times, one positive
    pair (another example of the same intent, target 1) and one negative pair (an example
    of another intent, target 0). Returns (texts_a, texts_b, targets)."""
    a, b, targets = [], [], []
    positions = np.arange(len(label_ids))
    for _ in range(iterations):
        for i, label in enumerate(label_ids):
            same = np.flatnonzero((label_ids == label) & (positions != i))
            other = np.flatnonzero(label_ids != label)
            if len(same):
                a.append(texts[i]); b.append(texts[rng.choice(same)]); targets.append(1.0)
            a.append(texts[i]); b.append(texts[rng.choice(other)]); targets.append(0.0)
    return a, b, np.array(targets, dtype=np.float32)


def _embed_with_grad(encoder: SentenceTransformer, texts: list[str], device) -> torch.Tensor:
    """Sentence embeddings with gradients on (encoder.encode switches them off)."""
    features = encoder.preprocess(texts)
    features = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in features.items()}
    return encoder(features)["sentence_embedding"]


def fine_tune(model_name: str, texts: list[str], label_ids: np.ndarray, cfg: dict, seed: int, device) -> SentenceTransformer:
    """SetFit-style contrastive fine-tuning of the sentence encoder on the support set.

    The encoder learns to put same-intent pairs close together (cosine towards 1) and
    different-intent pairs apart (cosine towards 0), with a mean squared error loss on
    the cosine (CosineSimilarityLoss in SetFit). Settings are SetFit's defaults.
    """
    hp = cfg["fine_tune"]
    set_seed(seed)
    a, b, targets = setfit_pairs(texts, label_ids, hp["pair_iterations"], np.random.default_rng(seed))
    encoder = SentenceTransformer(model_name, device=str(device))
    encoder.train()  # dropout on while training
    targets_t = torch.tensor(targets)
    loader = DataLoader(range(len(a)), batch_size=hp["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    total_steps = hp["epochs"] * len(loader)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=hp["learning_rate"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(hp["warmup_fraction"] * total_steps), total_steps)
    for _ in range(hp["epochs"]):
        for batch in loader:
            idx = batch.tolist()
            emb_a = _embed_with_grad(encoder, [a[i] for i in idx], device)
            emb_b = _embed_with_grad(encoder, [b[i] for i in idx], device)
            cosine = torch.nn.functional.cosine_similarity(emb_a, emb_b)
            loss = torch.nn.functional.mse_loss(cosine, targets_t[idx].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), hp["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    encoder.eval()  # dropout off for prediction
    return encoder


def predict_variant(scoring: str, encoder: SentenceTransformer, support: pd.DataFrame, texts: list[str],
                    n_classes: int, cfg: dict, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(predicted intent id, confidence) for each text, with one scoring rule."""
    support_emb = embed(encoder, support["text"].tolist())
    query_emb = embed(encoder, list(texts))
    label_ids = support["label_id"].to_numpy()
    if scoring == "head":
        head = LogisticRegression(C=cfg["head"]["C"], max_iter=cfg["head"]["max_iter"], random_state=seed)
        head.fit(support_emb, label_ids)
        probabilities = np.zeros((len(texts), n_classes))
        probabilities[:, head.classes_] = head.predict_proba(query_emb)  # columns in label-id order
        return probabilities.argmax(axis=1), probabilities.max(axis=1)
    scores = class_scores(query_emb @ support_emb.T, label_ids, n_classes)
    confidence = scores.max(axis=1) if scoring == "nearest" else margin_scores(scores)
    return scores.argmax(axis=1), confidence


def run_metrics(pred_ids: np.ndarray, confidence: np.ndarray, true_ids: np.ndarray, is_oos: np.ndarray,
                label_names: list[str], r_star: float, fixed_risks: list[float]) -> dict[str, float]:
    """Classification and routing metrics for one run. Out-of-scope rows have true id -1,
    so handling one is always an error, as in E6 and E7."""
    correct = (pred_ids == true_ids) & ~is_oos
    in_scope = ~is_oos
    cls = classification_metrics([label_names[i] for i in true_ids[in_scope]],
                                 [label_names[i] for i in pred_ids[in_scope]], label_names)
    right, wrong = correct[in_scope], ~correct[in_scope]
    row = {
        "in_scope_accuracy": cls["accuracy"],
        "macro_f1": cls["macro_f1"],
        "oos_auroc": oos_auroc(is_oos, -confidence),  # lower confidence should mean out of scope
        # How well confidence separates right from wrong in-scope answers (0.5 = no better than chance).
        "confidence_auroc": float(roc_auc_score(right, confidence[in_scope])) if right.any() and wrong.any() else float("nan"),
        "aurc": aurc(risk_coverage_curve(correct, confidence)),
        "coverage_at_matched_risk": coverage_at_risk(correct, confidence, r_star)["coverage"],
    }
    for risk in fixed_risks:
        row[f"coverage_at_{risk:g}"] = coverage_at_risk(correct, confidence, risk)["coverage"]
    return row


def select_per_tier(mean_aurc: dict[Variant, float], simpler_within: float) -> dict[str, Variant]:
    """The chosen variant of each tier: the lowest mean AURC, except that a simpler variant
    of the same tier within `simpler_within` of the best is taken instead (the simplest such one)."""
    chosen = {}
    for tier in TIERS:
        in_tier = {v: a for v, a in mean_aurc.items() if v.tier == tier}
        if not in_tier:
            continue
        best = min(in_tier.values())
        close = [v for v, a in in_tier.items() if a <= best + simpler_within]
        chosen[tier] = min(close, key=lambda v: v.simplicity)
    return chosen
