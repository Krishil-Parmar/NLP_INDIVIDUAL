"""Train S0 or S1 and save validation predictions for every seed.

Run from the project root:
    uv run python scripts/train.py --system s0
    uv run python scripts/train.py --system s1 --checkpoint distilbert/distilroberta-base --seeds 0
    uv run python scripts/train.py --system s1 --checkpoint X --probe 30          # timing only, writes nothing
    uv run python scripts/train.py --system s1 --checkpoint X --probe 30 --determinism

Only the train and validation splits are used. The test split is never loaded here.

Writes, per seed:
    results/predictions/<tag>/seed<n>_val.csv, seed<n>_val_logits.npy, seed<n>_run.json
    checkpoints/<s0|s1>/...   (gitignored)
    results/training_logs/<tag>_seed<n>.csv   (S1 only)
    results/s0_grid.json      (S0 only)
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from router import data
from router.provenance import run_metadata
from router.seed import pick_device
from router.systems import s0_tfidf, s1_encoder
from router.systems.base import label_names_in_id_order, save_predictions, save_run_info

CHECKPOINT_DIR = data.PROJECT_ROOT / "checkpoints"


def load_train_and_val() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, validation in-scope only, validation mixed with OOS)."""
    train_df = data.load_split("train")
    val_mixed = data.load_eval_set("val", with_oos=True)
    val_in_scope = val_mixed[~val_mixed["is_oos"]].reset_index(drop=True)
    return train_df, val_in_scope, val_mixed


def run_s0(seeds: list[int]) -> None:
    cfg = data.load_config("s0_tfidf")
    label_names = label_names_in_id_order()
    train_df, val_in, val_mixed = load_train_and_val()

    print("S0 grid search on validation macro-F1")
    best, grid_rows = s0_tfidf.grid_search(train_df, val_in, cfg, label_names)
    for row in grid_rows:
        print(f"  ngram={row['ngram_range']} C={row['C']}: val macro-F1 {row['val_macro_f1']:.4f}")
    print(f"  chosen: {best}")
    with open(data.RESULTS_DIR / "s0_grid.json", "w") as f:
        json.dump({"metadata": run_metadata(cfg), "chosen": best, "grid": grid_rows}, f, indent=2, default=str)

    for seed in seeds:
        start = time.perf_counter()
        model = s0_tfidf.train(train_df, best, cfg, seed)
        train_seconds = time.perf_counter() - start
        preds = s0_tfidf.predict(model, val_mixed["text"].tolist(), label_names)
        save_predictions(cfg["system_tag"], seed, "val", val_mixed, preds)
        s0_tfidf.save(model, CHECKPOINT_DIR / "s0" / f"seed{seed}.joblib")
        save_run_info(cfg["system_tag"], seed, {
            "metadata": run_metadata(cfg, seed), "params": best, "train_seconds": train_seconds,
        })
        print(f"  seed {seed}: trained in {train_seconds:.1f}s, predictions saved")


def run_s1(checkpoint: str, seeds: list[int]) -> None:
    cfg = data.load_config("s1_encoder")
    label_names = label_names_in_id_order()
    train_df, val_in, val_mixed = load_train_and_val()
    device = pick_device()
    tag = s1_encoder.system_tag(checkpoint)
    max_length = s1_encoder.max_length_for(checkpoint)
    logs_dir = data.RESULTS_DIR / "training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"S1 {checkpoint} seed {seed} on {device}")
        meta = run_metadata(cfg, seed)  # recorded before training starts
        start = time.perf_counter()
        model, tokenizer, log = s1_encoder.train(train_df, cfg, checkpoint, seed, device, label_names, val_df=val_in)
        train_seconds = time.perf_counter() - start
        preds = s1_encoder.predict(model, tokenizer, val_mixed["text"].tolist(), label_names, device, max_length, cfg["training"]["eval_batch_size"])
        save_predictions(tag, seed, "val", val_mixed, preds)
        s1_encoder.save(model, tokenizer, CHECKPOINT_DIR / "s1" / tag / f"seed{seed}")
        pd.DataFrame(log).to_csv(logs_dir / f"{tag}_seed{seed}.csv", index=False)
        save_run_info(tag, seed, {
            "metadata": meta, "checkpoint": checkpoint, "device": str(device),
            "max_length": max_length, "train_seconds": train_seconds, "epoch_log": log,
        })
        print(f"  seed {seed}: trained in {train_seconds / 60:.1f} min, predictions and checkpoint saved")


def probe_s1(checkpoint: str, steps: int, determinism: bool) -> None:
    """Train a few steps to time them. With determinism, do it twice and compare outputs."""
    cfg = data.load_config("s1_encoder")
    label_names = label_names_in_id_order()
    train_df, val_in, _ = load_train_and_val()
    device = pick_device()
    max_length = s1_encoder.max_length_for(checkpoint)
    sample = val_in["text"].tolist()[:64]

    outputs = []
    for attempt in range(2 if determinism else 1):
        model, tokenizer, log = s1_encoder.train(train_df, cfg, checkpoint, 0, device, label_names, max_steps=steps)
        seconds = log[-1]["seconds"]  # training loop time only, model loading excluded
        logits = s1_encoder.predict_logits(model, tokenizer, sample, device, max_length)
        outputs.append(logits)
        steps_per_epoch = int(np.ceil(len(train_df) / cfg["training"]["batch_size"]))
        full_run_min = seconds / steps * steps_per_epoch * cfg["training"]["epochs"] / 60
        print(f"probe {attempt + 1}: {steps} steps in {seconds:.1f}s on {device} ({seconds / steps:.3f}s/step), "
              f"final train loss {log[-1]['train_loss']:.4f} -> one full run is about {full_run_min:.1f} min")
    if determinism:
        diff = float(np.abs(outputs[0] - outputs[1]).max())
        same_argmax = bool((outputs[0].argmax(1) == outputs[1].argmax(1)).all())
        print(f"determinism: max |logit difference| = {diff:.3e}, same predictions on 64 queries: {same_argmax}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=["s0", "s1"], required=True)
    parser.add_argument("--checkpoint", help="S1 only: Hub model ID (default: the chosen one in configs/s1_encoder.yaml)")
    parser.add_argument("--seeds", type=int, nargs="+", help="default: the seeds in the system's config")
    parser.add_argument("--probe", type=int, metavar="N", help="S1 only: train N steps, print timing, save nothing")
    parser.add_argument("--determinism", action="store_true", help="with --probe: run twice and compare outputs")
    args = parser.parse_args()

    if args.system == "s0":
        run_s0(args.seeds or data.load_config("s0_tfidf")["seeds"])
    else:
        s1_cfg = data.load_config("s1_encoder")
        checkpoint = args.checkpoint or s1_cfg["checkpoint"]
        if not checkpoint:
            sys.exit("No S1 checkpoint chosen yet: pass --checkpoint or fill in configs/s1_encoder.yaml")
        if args.probe:
            probe_s1(checkpoint, args.probe, args.determinism)
        else:
            run_s1(checkpoint, args.seeds or s1_cfg["seeds"])
