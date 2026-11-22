"""E2 step 1: flag likely label errors in Banking77 train and validation with cleanlab.

Run from the project root:
    uv run python scripts/find_label_issues.py

Settings are in configs/e2_label_noise.yaml. The test split is never read.

Writes:
    data/processed/banking77_flagged.csv            flagged rows, most severe first
    data/processed/banking77_cleaner_pred_probs.npz the out-of-fold probabilities, so
                                                    the flags can be re-derived in a second
    results/e2_cleaner.json, results/e2_cleaner.md  cleaner quality and flag counts
    results/e2_audit.csv, results/e2_audit_key.csv  the blind 100-item audit (only if
                                                    the sheet does not exist yet: a sheet
                                                    the owner has started is never replaced)

Ends by printing the stop checks. If any fails, do not retrain on cleaned labels.
"""

import json
import sys
import time

import numpy as np
import pandas as pd

from router import data, label_noise
from router.provenance import metadata_markdown, run_metadata
from router.seed import pick_device
from router.systems import s2_embedding
from router.systems.base import label_names_in_id_order
from router.tables import md_table


def load_rows(splits: list[str]) -> pd.DataFrame:
    """The checked rows, in a fixed order: each split file as saved, one after another."""
    return pd.concat([data.load_split(name).assign(split=name) for name in splits]).reset_index(drop=True)


def write_report(summary: dict, flagged: pd.DataFrame, meta: dict) -> str:
    by_split = [{"split": s, "rows": n, "flagged": summary["flagged_per_split"][s],
                 "share": f"{100 * summary['flagged_per_split'][s] / n:.2f}%"}
                for s, n in summary["rows_per_split"].items()]
    grid = [{"C": g["C"], "CV log loss": f"{g['cv_log_loss']:.4f}", "CV accuracy": f"{100 * g['cv_accuracy']:.2f}%"}
            for g in summary["grid"]]
    intents = (flagged[flagged["split"] == "train"].groupby("given_label").size()
               .div(pd.Series(summary["train_rows_per_intent"])).dropna().sort_values(ascending=False).head(15))
    intent_rows = [{"intent": k, "train rows flagged": f"{100 * v:.1f}%"} for k, v in intents.items()]
    pairs = flagged.groupby(["given_label", "suggested_label"]).size().sort_values(ascending=False).head(15)
    pair_rows = [{"given label": g, "suggested label": s, "flagged": int(n)} for (g, s), n in pairs.items()]
    parts = [
        "# E2 step 1: label issues flagged by cleanlab",
        "Written by `scripts/find_label_issues.py`. Do not edit by hand.",
        "## Provenance", metadata_markdown(meta),
        "## Cleaner",
        f"Frozen `{summary['embedding_model']}` embeddings, then logistic regression with "
        f"{summary['folds']}-fold stratified cross-validation (seed {summary['seed']}). "
        f"Every row's probabilities come from the fold that did not see it. C kept: **{summary['chosen_C']}** "
        f"(lowest CV log loss). Embedding took {summary['embed_seconds']:.0f} s on {summary['device']}.",
        md_table(grid),
        "## Flagged rows",
        f"cleanlab `find_label_issues`, `filter_by={summary['filter_by']}`, ranked by `{summary['rank_by']}`. "
        "The suggested label is the most probable label other than the given one.",
        md_table(by_split),
        "## Intents with the largest share of train rows flagged (top 15)",
        md_table(intent_rows),
        "## Most frequent (given, suggested) label pairs (top 15)",
        md_table(pair_rows),
        "## Stop checks before retraining",
        "\n".join(f"- {p}" for p in summary["stop_problems"]) or "All passed.",
    ]
    return "\n\n".join(parts) + "\n"


def main() -> None:
    cfg = data.load_config("e2_label_noise")
    meta = run_metadata({**data.load_config("data"), "e2_label_noise": cfg}, seed=cfg["cleaner"]["seed"])
    label_names = label_names_in_id_order()
    rows = load_rows(cfg["splits"])
    label_ids = rows["label_id"].to_numpy()
    print(f"Checking {len(rows)} rows from {cfg['splits']}")

    # 1. Embeddings (frozen model, no training).
    device = pick_device()
    start = time.perf_counter()
    encoder = s2_embedding.load_encoder(cfg["cleaner"]["embedding_model"], device)
    features = s2_embedding.embed(encoder, rows["text"].tolist(), cfg["cleaner"]["batch_size"])
    embed_seconds = time.perf_counter() - start
    print(f"Embedded with {cfg['cleaner']['embedding_model']} in {embed_seconds:.0f}s on {device}")

    # 2. Out-of-fold probabilities, C chosen by log loss.
    c = cfg["cleaner"]
    pred_probs, chosen_C, grid = label_noise.cross_val_pred_probs(
        features, label_ids, len(label_names), c["C_grid"], c["folds"], c["seed"], c["max_iter"])
    for g in grid:
        print(f"  C={g['C']}: CV log loss {g['cv_log_loss']:.4f}, CV accuracy {g['cv_accuracy']:.4f}")
    # Saved as float32, and the flags are computed from exactly what is saved,
    # so re-deriving them from the file gives the same list.
    pred_probs = pred_probs.astype(np.float32)
    np.savez_compressed(
        data.PROCESSED_DIR / label_noise.PRED_PROBS_FILE, pred_probs=pred_probs, label_id=label_ids,
        source=rows["source"].to_numpy(dtype=str), orig_index=rows["orig_index"].to_numpy(), split=rows["split"].to_numpy(dtype=str),
    )

    # 3. Flags.
    ranked = label_noise.flag_issues(label_ids, pred_probs, cfg["cleanlab"]["filter_by"], cfg["cleanlab"]["rank_by"], cfg["cleanlab"]["n_jobs"])
    flagged = label_noise.flagged_table(rows, pred_probs, ranked, label_names)
    flagged.to_csv(data.PROCESSED_DIR / label_noise.FLAGGED_FILE, index=False)

    train_df = rows[rows["split"] == "train"]
    cv_accuracy = next(g["cv_accuracy"] for g in grid if g["C"] == chosen_C)
    problems = label_noise.stop_checks(flagged, train_df, cv_accuracy, cfg["stop_if"])
    summary = {
        "embedding_model": c["embedding_model"], "folds": c["folds"], "seed": c["seed"], "device": str(device),
        "embed_seconds": embed_seconds, "grid": grid, "chosen_C": chosen_C, "cv_accuracy": cv_accuracy,
        "filter_by": cfg["cleanlab"]["filter_by"], "rank_by": cfg["cleanlab"]["rank_by"],
        "rows_per_split": rows["split"].value_counts().reindex(cfg["splits"]).astype(int).to_dict(),
        "flagged_per_split": flagged["split"].value_counts().reindex(cfg["splits"], fill_value=0).astype(int).to_dict(),
        "n_flagged": len(flagged),
        "train_rows_per_intent": train_df["label"].value_counts().astype(int).to_dict(),
        "stop_problems": problems,
    }
    with open(data.RESULTS_DIR / "e2_cleaner.json", "w") as f:
        json.dump({"metadata": meta, **summary}, f, indent=2, default=str)
    (data.RESULTS_DIR / "e2_cleaner.md").write_text(write_report(summary, flagged, meta))
    print(f"Flagged {len(flagged)} rows: {summary['flagged_per_split']}")

    # 4. The audit sheet, created once.
    sheet_path, key_path = data.RESULTS_DIR / label_noise.AUDIT_FILE, data.RESULTS_DIR / label_noise.AUDIT_KEY_FILE
    if sheet_path.exists():
        print(f"{sheet_path.name} already exists, so it was left alone (the owner may have started it).")
    else:
        a = cfg["audit"]
        sheet, key = label_noise.make_blind_audit(flagged, train_df, a["n_items"], a["seed"], a["examples_per_option"])
        sheet.to_csv(sheet_path, index=False)
        key.to_csv(key_path, index=False)
        print(f"Wrote the blind audit sheet ({len(sheet)} items) and its key")

    print("Wrote data/processed/banking77_flagged.csv, results/e2_cleaner.json and results/e2_cleaner.md")
    if problems:
        print("STOP before retraining:\n  " + "\n  ".join(problems))
        sys.exit(1)
    print("Stop checks: all passed")


if __name__ == "__main__":
    main()
