"""E1: accuracy, precision, recall and F1 for every system with saved predictions.

Run from the project root:
    uv run python scripts/evaluate.py --split val

Scores in-scope queries only (E1 is in-scope accuracy; OOS is E3 and E6).
Per system: mean and standard deviation over seeds, the 10 intents with the
lowest F1 (F1 averaged over seeds) and the 10 most frequent confusions
(counts summed over seeds).

Writes results/e1_<split>.json and results/e1_<split>.md.

Splits ending in "_fmt" hold predictions on format-controlled text
(scripts/predict_format_control.py): lowercased, punctuation removed. E1 on
them shows how much in-scope accuracy that control costs each system.

Test splits are refused until results/preregistration.md exists (rigour rule 1).
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from router import data
from router.metrics import (
    classification_metrics,
    confusion_counts,
    mean_std_over_seeds,
    per_intent_scores,
    top_confused_pairs,
    worst_intents,
)
from router.provenance import metadata_markdown, run_metadata
from router.systems.base import label_names_in_id_order, list_runs, load_predictions
from router.tables import md_table

# Metrics in the order the rubric names them: accuracy, precision, recall, F1.
METRIC_ORDER = [
    "accuracy",
    "macro_precision", "macro_recall", "macro_f1",
    "weighted_precision", "weighted_recall", "weighted_f1",
]


def evaluate_system(tag: str, seeds: list[int], split: str, label_names: list[str]) -> dict:
    per_seed, intent_tables = [], []
    confusion_total = np.zeros((len(label_names), len(label_names)), dtype=int)
    for seed in seeds:
        preds, _ = load_predictions(tag, seed, split)
        in_scope = preds[~preds["is_oos"]]
        y_true, y_pred = in_scope["label"].tolist(), in_scope["pred_label"].tolist()
        per_seed.append(classification_metrics(y_true, y_pred, labels=label_names))
        intent_tables.append(per_intent_scores(y_true, y_pred, label_names))
        confusion_total += confusion_counts(y_true, y_pred, label_names)
    # Average each intent's precision, recall and F1 over seeds.
    mean_intents = sum(intent_tables) / len(intent_tables)
    return {
        "seeds": seeds,
        "n_queries": len(in_scope),
        "per_seed": per_seed,
        "summary": mean_std_over_seeds(per_seed),
        "worst_intents": worst_intents(mean_intents).reset_index().to_dict(orient="records"),
        "top_confused_pairs": top_confused_pairs(confusion_total, label_names).to_dict(orient="records"),
    }


def pct(summary: dict, key: str) -> str:
    return f"{100 * summary[key]['mean']:.2f} ± {100 * summary[key]['std']:.2f}"


def write_markdown(results: dict, split: str, meta: dict) -> str:
    main_rows = [
        {"system": tag, "seeds": len(r["seeds"]), **{k: pct(r["summary"], k) for k in METRIC_ORDER}}
        for tag, r in results.items()
    ]
    parts = [
        f"# E1 results on {split}",
        "Written by `scripts/evaluate.py`. Do not edit by hand.",
        "## Provenance",
        metadata_markdown(meta),
        f"## In-scope classification ({split}), percent, mean ± std over seeds",
        *(["**Format-controlled text:** queries were lowercased and stripped of punctuation before prediction "
           "(the E3 control, `router.preprocess.format_control`). The models were trained on the raw text."]
          if split.endswith("_fmt") else []),
        "Macro averages are primary (each of the 77 intents counts equally). Std is the sample standard deviation across seeds.",
        md_table(main_rows),
    ]
    for tag, r in results.items():
        worst = [
            {"intent": w["intent"], "precision": f"{w['precision']:.3f}", "recall": f"{w['recall']:.3f}",
             "f1": f"{w['f1']:.3f}", "support": int(w["support"])}
            for w in r["worst_intents"]
        ]
        parts += [
            f"## {tag}",
            "10 intents with the lowest F1 (averaged over seeds):",
            md_table(worst),
            "Most frequent confusions (true intent predicted as another, counts summed over seeds):",
            md_table(r["top_confused_pairs"]),
        ]
    return "\n\n".join(parts) + "\n"


def main(split: str) -> None:
    if split.startswith("test") and not (data.RESULTS_DIR / "preregistration.md").exists():
        sys.exit("Refusing to score a test split: results/preregistration.md does not exist yet (rigour rule 1).")
    label_names = label_names_in_id_order()
    runs = list_runs(split)
    if not runs:
        sys.exit(f"No saved predictions for split {split!r}. Run scripts/train.py first.")

    results = {tag: evaluate_system(tag, seeds, split, label_names) for tag, seeds in runs.items()}
    meta = run_metadata(data.load_config("data"))
    with open(data.RESULTS_DIR / f"e1_{split}.json", "w") as f:
        json.dump({"metadata": meta, "split": split, "systems": results}, f, indent=2, default=str)
    (data.RESULTS_DIR / f"e1_{split}.md").write_text(write_markdown(results, split, meta))

    print(pd.DataFrame({tag: {k: pct(r["summary"], k) for k in METRIC_ORDER} for tag, r in results.items()}).T.to_string())
    print(f"\nWrote results/e1_{split}.json and results/e1_{split}.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val", "test", "val_fmt", "test_fmt"], default="val")
    main(parser.parse_args().split)
