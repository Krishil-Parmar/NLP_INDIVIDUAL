"""Diagnosis of the E7 failure (RQ6 not met): why does S2 collapse on the Jeff set?

Run from the project root:
    uv run python scripts/diagnose_jeff.py

Nothing here is tuned and no test prediction is made. It reads:
  - the frozen S2 predictions already saved for the Jeff test split (E7) and for
    Banking77 test_fmt (E6), to compare how S2's confidence behaves on each;
  - Banking77 VALIDATION, where S2 is re-run with its support set cut to 10, 20 and
    50 examples per intent (five random draws each). This separates the two
    explanations: "Jeff is a new domain" versus "Jeff has only 10 examples per intent".

Writes results/e7_diagnosis.json and results/e7_diagnosis.md.
"""

import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import roc_auc_score

from router import data, few_shot, jeff
from router.metrics import confusion_counts, top_confused_pairs
from router.preprocess import format_control
from router.provenance import metadata_markdown, run_metadata
from router.seed import pick_device
from router.systems.base import load_predictions
from router.systems.s2_embedding import load_encoder
from router.tables import md_table

S2_TAG = "s2_all-MiniLM-L6-v2"
SUPPORT_SIZES = [10, 20, 50]
DRAWS = [0, 1, 2, 3, 4]
FIXED_RISKS = [0.01, 0.05]


def content_words(text: str) -> set[str]:
    """Lower-cased words that are not English stop words (scikit-learn's list)."""
    return {w for w in text.lower().split() if w not in ENGLISH_STOP_WORDS}


def confidence_quality(df: pd.DataFrame, logits: np.ndarray) -> dict[str, float]:
    """How well S2's confidence separates its right in-scope answers from its wrong ones."""
    in_scope = ~df["is_oos"].to_numpy(dtype=bool)
    correct = (df["pred_label"] == df["label"]).to_numpy()[in_scope]
    nearest = df["confidence"].to_numpy(dtype=float)[in_scope]
    margin = few_shot.margin_scores(logits[in_scope])
    return {
        "n_in_scope": int(in_scope.sum()),
        "in_scope_accuracy": float(correct.mean()),
        "auroc_nearest": float(roc_auc_score(correct, nearest)),  # 0.5 = no better than chance
        "auroc_margin": float(roc_auc_score(correct, margin)),
        "median_nearest_right": float(np.median(nearest[correct])),
        "median_nearest_wrong": float(np.median(nearest[~correct])),
    }


def jeff_error_anatomy(names: list[str]) -> dict:
    """Where S2 goes wrong on the Jeff test split: shortlist rank, intents, confusions, word overlap."""
    df, logits = load_predictions(S2_TAG, 0, "jeff_test_fmt")
    in_scope = ~df["is_oos"].to_numpy(dtype=bool)
    ins, scores = df[in_scope].reset_index(drop=True), logits[in_scope]
    true_ids = ins["label"].map({n: i for i, n in enumerate(names)}).to_numpy()
    true_score = scores[np.arange(len(scores)), true_ids]
    rank = (scores > true_score[:, None]).sum(axis=1)  # 0 = the true intent scored highest
    correct = (ins["pred_label"] == ins["label"]).to_numpy()
    overlap = np.array([len(content_words(q) & content_words(m)) for q, m in zip(ins["text"], ins["matched_example"])])
    per_intent = ins.assign(correct=correct).groupby("label")["correct"].mean().sort_values()
    pairs = top_confused_pairs(confusion_counts(ins["label"].tolist(), ins["pred_label"].tolist(), names), names, k=8)
    return {
        "top_k_accuracy": {str(k): float((rank < k).mean()) for k in (1, 2, 3)},
        "per_intent_accuracy": {k: float(v) for k, v in per_intent.items()},
        "top_confused_pairs": pairs.to_dict("records"),
        "overlap_right": float(overlap[correct].mean()),
        "overlap_wrong": float(overlap[~correct].mean()),
        "oos_median_nearest": float(df.loc[~in_scope, "confidence"].median()),
    }


def support_size_check(r_star: float) -> list[dict]:
    """S2 on Banking77 validation (format-controlled, with OOS) with smaller support sets."""
    label_map = data.load_label_map()
    names = sorted(label_map, key=label_map.get)
    train = data.load_train()
    eval_df = data.load_eval_set("val", with_oos=True)
    texts = eval_df["text"].map(format_control).tolist()
    is_oos = eval_df["is_oos"].to_numpy(dtype=bool)
    true_ids = np.where(is_oos, -1, eval_df["label"].map(label_map).fillna(-1).to_numpy()).astype(int)
    encoder = load_encoder(few_shot_config()["encoder_model"], pick_device())
    cfg = {"head": {}}  # nearest scoring only: S2 as it is

    def one(support: pd.DataFrame) -> dict[str, float]:
        pred, conf = few_shot.predict_variant("nearest", encoder, support, texts, len(names), cfg, seed=0)
        return few_shot.run_metrics(pred, conf, true_ids, is_oos, names, r_star, FIXED_RISKS)

    rows = [{"support": "full train", "per_intent": "all (about 120)", **one(train)}]
    for k in SUPPORT_SIZES:
        runs = [one(few_shot.per_intent_sample(train, k, seed)) for seed in DRAWS]
        mean = {m: float(np.mean([r[m] for r in runs])) for m in runs[0]}
        std = {f"{m}_std": float(np.std([r[m] for r in runs])) for m in runs[0]}
        rows.append({"support": f"{len(DRAWS)} random draws", "per_intent": str(k), **mean, **std})
    return rows


def few_shot_config() -> dict:
    return data.load_config("e8_few_shot")


def pct(x: float) -> str:
    return f"{100 * x:.1f}"


def report(result: dict) -> str:
    q, a = result["confidence_quality"], result["jeff"]
    quality_rows = [{"data": name, "in-scope queries": v["n_in_scope"], "in-scope accuracy (%)": pct(v["in_scope_accuracy"]),
                     "AUROC right vs wrong, nearest cosine (%)": pct(v["auroc_nearest"]),
                     "AUROC right vs wrong, margin (%)": pct(v["auroc_margin"]),
                     "median cosine, right": f"{v['median_nearest_right']:.3f}",
                     "median cosine, wrong": f"{v['median_nearest_wrong']:.3f}"} for name, v in q.items()]
    size_rows = [{"support": r["support"], "examples per intent": r["per_intent"],
                  "in-scope accuracy (%)": pct(r["in_scope_accuracy"]),
                  "confidence AUROC (%)": pct(r["confidence_auroc"]), "OOS AUROC (%)": pct(r["oos_auroc"]),
                  f"coverage at {pct(result['r_star_val'])}% risk (%)": pct(r["coverage_at_matched_risk"]),
                  "coverage at 5% risk (%)": pct(r["coverage_at_0.05"]), "coverage at 1% risk (%)": pct(r["coverage_at_0.01"])}
                 for r in result["support_size_check"]]
    intent_rows = [{"intent": k, "accuracy (%)": pct(v)} for k, v in a["per_intent_accuracy"].items()]
    pair_rows = [{k: v for k, v in p.items()} for p in a["top_confused_pairs"]]
    return "\n\n".join([
        "# E7 diagnosis: why S2 fails on the Jeff set",
        "Written by `scripts/diagnose_jeff.py`. Do not edit by hand. Reads saved predictions and Banking77 validation only; "
        "nothing is tuned and no test prediction is made.",
        "## Provenance", metadata_markdown(result["metadata"]),
        "## 1. The right intent is usually on the shortlist",
        f"S2 on the Jeff test split (in-scope queries): the true intent scores highest in {pct(a['top_k_accuracy']['1'])}% "
        f"of queries, is in the top two in {pct(a['top_k_accuracy']['2'])}% and in the top three in {pct(a['top_k_accuracy']['3'])}%. "
        "S2 finds the right neighbourhood but picks the wrong neighbour.",
        "## 2. The confidence stops separating right from wrong answers",
        "AUROC here is the chance that a random right answer gets a higher confidence than a random wrong one "
        "(50% = no better than a coin flip). Coverage at low risk needs this to be high: the router keeps the most confident answers.",
        md_table(quality_rows),
        "## 3. Support size explains part of the gap, not all of it (Banking77 validation)",
        "S2 re-run on Banking77 validation (format-controlled, with the OOS pool) with its support set cut to k random "
        f"examples per intent. The first coverage column is at few-shot S4's validation error ({pct(result['r_star_val'])}%), the bar policy v1 was set on. "
        "Mean over five draws (standard deviations in the JSON file).",
        md_table(size_rows),
        f"For comparison, on the Jeff test split (10 support examples per intent) S2 covers {pct(result['jeff_s2_e7']['coverage_at_0.05'])}% "
        f"at 5% risk and {pct(result['jeff_s2_e7']['coverage_at_0.01'])}% at 1% risk (`results/e7_jeff.json`). Cutting Banking77 "
        "to 10 examples per intent halves the coverage at 5% risk. **Correction (24 Nov 2026):** the first version of this report said "
        "the rest of the Jeff gap comes from the Jeff intents themselves. That is not supported. At the same cosine distance from the "
        "nearest labelled example, S2 is about as accurate on Jeff as on Banking77 (`results/headline_reanalysis.md`, section C). "
        "Banking77 looks easy because most of its test queries have a close paraphrase in train, and Jeff's do not.",
        "## 4. Where the Jeff errors are",
        "Accuracy per intent (S2, Jeff test split):", md_table(intent_rows),
        "Most frequent confusions:", md_table(pair_rows),
        f"Content words shared with the matched support example (scikit-learn English stop words removed): "
        f"{a['overlap_right']:.2f} on average for right answers, {a['overlap_wrong']:.2f} for wrong ones. "
        "So the number of shared words cannot tell a right answer from a wrong one. Reading the errors shows what goes wrong: "
        "many wrong matches share the topic or one salient word with the query but answer a different request "
        "(`results/error_analysis.md`). Median S2 confidence on the out-of-scope queries: "
        f"{a['oos_median_nearest']:.3f}.",
    ]) + "\n"


def main() -> None:
    names = jeff.intent_names(data.load_config("jeff"))
    policy = json.loads((data.RESULTS_DIR / "policy_v1.json").read_text())
    quality = {}
    for name, split in [("Banking77 test_fmt (E6)", "test_fmt"), ("Jeff test (E7)", "jeff_test_fmt")]:
        df, logits = load_predictions(S2_TAG, 0, split)
        quality[name] = confidence_quality(df, logits)
    result = {
        "metadata": run_metadata(few_shot_config()),
        "r_star_val": policy["matched_risk"],
        "confidence_quality": quality,
        "jeff": jeff_error_anatomy(names),
        "support_size_check": support_size_check(policy["matched_risk"]),
        "jeff_s2_e7": json.loads((data.RESULTS_DIR / "e7_jeff.json").read_text())["systems"][S2_TAG]["per_seed"][0],
    }
    (data.RESULTS_DIR / "e7_diagnosis.json").write_text(json.dumps(result, indent=2))
    (data.RESULTS_DIR / "e7_diagnosis.md").write_text(report(result))
    print("Wrote results/e7_diagnosis.json and results/e7_diagnosis.md")


if __name__ == "__main__":
    main()
