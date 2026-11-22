"""Train S0, S1, S2 or S3 and save validation predictions for every seed.

Run from the project root:
    uv run python scripts/train.py --system s0
    uv run python scripts/train.py --system s1 --checkpoint distilbert/distilroberta-base --seeds 0
    uv run python scripts/train.py --system s1 --checkpoint X --probe 30          # timing only, writes nothing
    uv run python scripts/train.py --system s1 --checkpoint X --probe 30 --determinism
    uv run python scripts/train.py --system s2 --checkpoint sentence-transformers/all-MiniLM-L6-v2
    uv run python scripts/train.py --system s3 [--probe 50]
    uv run python scripts/train.py --system s1 --labels cleaned   # E2: train without the flagged rows

With --labels cleaned (E2), every choice stays frozen (same model, hyperparameters
and, for S0, the grid parameters already chosen); only the training rows change.
Runs are saved under "<tag>_cleaned".

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
from router.systems import registry, s0_tfidf, s1_encoder, s2_embedding, s3_pairwise
from router.systems.base import label_names_in_id_order, save_predictions, save_run_info

CHECKPOINT_DIR = data.PROJECT_ROOT / "checkpoints"


def load_train_and_val(labels: str = "original") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(train, validation in-scope only, validation mixed with OOS).

    labels="cleaned" drops the train rows cleanlab flagged (E2). Validation is
    always the full original split, so every run is scored on the same rows.
    """
    train_df = data.load_train(labels)
    val_mixed = data.load_eval_set("val", with_oos=True)
    val_in_scope = val_mixed[~val_mixed["is_oos"]].reset_index(drop=True)
    return train_df, val_in_scope, val_mixed


def run_s0(seeds: list[int], labels: str = "original") -> None:
    cfg = data.load_config("s0_tfidf")
    label_names = label_names_in_id_order()
    train_df, val_in, val_mixed = load_train_and_val(labels)
    tag = registry.run_tag(cfg["system_tag"], labels)

    if labels == "cleaned":
        # E2 keeps every choice frozen: reuse the grid parameters chosen on the original labels.
        with open(data.RESULTS_DIR / "s0_grid.json") as f:
            best = json.load(f)["chosen"]
        print(f"S0 on cleaned labels ({len(train_df)} train rows) with the frozen parameters {best}")
    else:
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
        save_predictions(tag, seed, "val", val_mixed, preds)
        s0_tfidf.save(model, registry.s0_checkpoint(seed, labels))
        save_run_info(tag, seed, {
            "metadata": run_metadata(cfg, seed), "params": best, "train_seconds": train_seconds,
            "labels": labels, "n_train_rows": len(train_df),
        })
        print(f"  seed {seed}: trained in {train_seconds:.1f}s, predictions saved")


def run_s1(checkpoint: str, seeds: list[int], labels: str = "original") -> None:
    cfg = data.load_config("s1_encoder")
    label_names = label_names_in_id_order()
    train_df, val_in, val_mixed = load_train_and_val(labels)
    device = pick_device()
    tag = registry.run_tag(s1_encoder.system_tag(checkpoint), labels)
    max_length = s1_encoder.max_length_for(checkpoint)
    logs_dir = data.RESULTS_DIR / "training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    # Git state is recorded once, before any training, so editing files while a
    # long multi-seed run is going cannot make later seeds look "dirty".
    meta_at_start = run_metadata(cfg)

    for seed in seeds:
        print(f"S1 {checkpoint} seed {seed} on {device}, {labels} labels ({len(train_df)} train rows)")
        meta = {**meta_at_start, "seed": seed}
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
            "labels": labels, "n_train_rows": len(train_df),
        })
        print(f"  seed {seed}: trained in {train_seconds / 60:.1f} min, predictions and checkpoint saved")


def run_s2(checkpoint: str, labels: str = "original") -> None:
    """S2 has no training: embed the train split once and predict validation (deterministic, seed 0 only)."""
    cfg = data.load_config("s2_embedding")
    label_names = label_names_in_id_order()
    _, _, val_mixed = load_train_and_val(labels)
    device = pick_device()
    tag = registry.run_tag(s2_embedding.system_tag(checkpoint), labels)
    meta = run_metadata(cfg, seed=0)
    start = time.perf_counter()
    encoder = s2_embedding.load_encoder(checkpoint, device)
    support = registry.s2_support_set(encoder, checkpoint, cfg["batch_size"], labels)
    preds = s2_embedding.predict(encoder, support, val_mixed["text"].tolist(), label_names, cfg["batch_size"])
    seconds = time.perf_counter() - start
    save_predictions(tag, 0, "val", val_mixed, preds)
    save_run_info(tag, 0, {"metadata": meta, "checkpoint": checkpoint, "device": str(device), "seconds": seconds,
                           "labels": labels, "n_train_rows": len(support.texts)})
    print(f"S2 {checkpoint}: embedded {len(support.texts)} support examples and predicted validation in {seconds:.1f}s on {device}")


def _s3_setup(labels: str = "original"):
    """Shared S3 inputs: config, labels, data, device, S2 retriever and support set, pair max_length.

    labels="cleaned" builds the support set (and so the training pairs) without the flagged rows (E2).
    """
    cfg = data.load_config("s3_pairwise")
    s2_cfg = data.load_config("s2_embedding")
    if not s2_cfg["checkpoint"]:
        sys.exit("S3 needs its retriever: choose the S2 model first (configs/s2_embedding.yaml).")
    label_names = label_names_in_id_order()
    _, val_in, val_mixed = load_train_and_val(labels)
    device = pick_device()
    encoder = s2_embedding.load_encoder(s2_cfg["checkpoint"], device)
    support = registry.s2_support_set(encoder, s2_cfg["checkpoint"], s2_cfg["batch_size"], labels)
    return cfg, s2_cfg, label_names, val_in, val_mixed, device, encoder, support, registry.s3_max_length(cfg)


def run_s3(seeds: list[int], labels: str = "original") -> None:
    cfg, s2_cfg, label_names, val_in, val_mixed, device, encoder, support, max_length = _s3_setup(labels)
    checkpoint = cfg["checkpoint"]
    tag = registry.run_tag(s3_pairwise.system_tag(checkpoint), labels)
    logs_dir = data.RESULTS_DIR / "training_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    meta_at_start = run_metadata(cfg)  # git state recorded once, before training

    recall = s3_pairwise.shortlist_recall(encoder, support, val_in["text"].tolist(), val_in["label_id"].to_numpy(), cfg)
    print(f"S2 shortlist recall on validation (true intent among the {cfg['retrieval']['top_intents']} shortlisted): {recall:.4f}")

    for seed in seeds:
        print(f"S3 {checkpoint} seed {seed} on {device}, {labels} labels ({len(support.texts)} support rows)")
        start = time.perf_counter()
        pairs = s3_pairwise.make_training_pairs(support, cfg, np.random.default_rng(seed))
        model, tokenizer, log = s3_pairwise.train(pairs, cfg, checkpoint, seed, device, max_length)
        train_seconds = time.perf_counter() - start
        preds = s3_pairwise.predict(model, tokenizer, encoder, support, val_mixed["text"].tolist(), label_names, cfg, device, max_length)
        save_predictions(tag, seed, "val", val_mixed, preds)
        s3_pairwise.save(model, tokenizer, CHECKPOINT_DIR / "s3" / tag / f"seed{seed}")
        pd.DataFrame(log).to_csv(logs_dir / f"{tag}_seed{seed}.csv", index=False)
        save_run_info(tag, seed, {
            "metadata": {**meta_at_start, "seed": seed}, "checkpoint": checkpoint, "retriever": s2_cfg["checkpoint"],
            "device": str(device), "max_length": max_length, "n_training_pairs": len(pairs),
            "positive_share": float(pairs["label"].mean()), "val_shortlist_recall": recall,
            "train_seconds": train_seconds, "epoch_log": log,
            "labels": labels, "n_train_rows": len(support.texts),
        })
        print(f"  seed {seed}: {len(pairs)} pairs, trained in {train_seconds / 60:.1f} min, predictions and checkpoint saved")


def probe_s3(steps: int) -> None:
    """Time S3: pair building, a few training steps, and prediction on 200 validation queries."""
    cfg, _, label_names, val_in, _, device, encoder, support, max_length = _s3_setup()
    start = time.perf_counter()
    pairs = s3_pairwise.make_training_pairs(support, cfg, np.random.default_rng(0))
    pair_seconds = time.perf_counter() - start
    _, _, log = s3_pairwise.train(pairs, cfg, cfg["checkpoint"], 0, device, max_length, max_steps=steps)
    step_seconds = log[-1]["seconds"] / steps
    model, tokenizer = s3_pairwise.load_base(cfg["checkpoint"])  # untrained copy: only prediction speed is measured here
    model.to(device)
    start = time.perf_counter()
    s3_pairwise.predict(model, tokenizer, encoder, support, val_in["text"].tolist()[:200], label_names, cfg, device, max_length)
    predict_seconds_per_query = (time.perf_counter() - start) / 200
    steps_per_run = int(np.ceil(len(pairs) / cfg["training"]["batch_size"])) * cfg["training"]["epochs"]
    per_seed_min = (pair_seconds + steps_per_run * step_seconds + 1350 * predict_seconds_per_query) / 60
    print(f"probe: {len(pairs)} pairs built in {pair_seconds:.1f}s; {step_seconds:.3f}s/step x {steps_per_run} steps; "
          f"prediction {1000 * predict_seconds_per_query:.1f} ms/query on {device}")
    print(f"estimate: {per_seed_min:.1f} min per seed, {per_seed_min * len(cfg['seeds']):.1f} min for {len(cfg['seeds'])} seeds")


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
    parser.add_argument("--system", choices=["s0", "s1", "s2", "s3"], required=True)
    parser.add_argument("--checkpoint", help="S1/S2: Hub model ID (default: the chosen one in the system's config)")
    parser.add_argument("--seeds", type=int, nargs="+", help="default: the seeds in the system's config")
    parser.add_argument("--probe", type=int, metavar="N", help="S1/S3: train N steps, print timing, save nothing")
    parser.add_argument("--determinism", action="store_true", help="with --probe: run twice and compare outputs")
    parser.add_argument("--labels", choices=["original", "cleaned"], default="original",
                        help="cleaned: train without the rows cleanlab flagged (E2)")
    args = parser.parse_args()
    if args.probe and args.labels != "original":
        sys.exit("--probe only times training; use it with the original labels")

    if args.system == "s0":
        run_s0(args.seeds or data.load_config("s0_tfidf")["seeds"], args.labels)
    elif args.system == "s2":
        s2_checkpoint = args.checkpoint or data.load_config("s2_embedding")["checkpoint"]
        if not s2_checkpoint:
            sys.exit("Pass --checkpoint or fill in configs/s2_embedding.yaml")
        run_s2(s2_checkpoint, args.labels)
    elif args.system == "s3":
        if args.probe:
            probe_s3(args.probe)
        else:
            run_s3(args.seeds or data.load_config("s3_pairwise")["seeds"], args.labels)
    else:
        s1_cfg = data.load_config("s1_encoder")
        checkpoint = args.checkpoint or s1_cfg["checkpoint"]
        if not checkpoint:
            sys.exit("No S1 checkpoint chosen yet: pass --checkpoint or fill in configs/s1_encoder.yaml")
        if args.probe:
            probe_s1(checkpoint, args.probe, args.determinism)
        else:
            run_s1(checkpoint, args.seeds or s1_cfg["seeds"], args.labels)
