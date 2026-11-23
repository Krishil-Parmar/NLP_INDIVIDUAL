"""Predict an evaluation split with the saved models. No training.

Run from the project root:
    uv run python scripts/predict.py --split val --check                   # dry run, see below
    uv run python scripts/predict.py --split val --check --labels cleaned
    uv run python scripts/predict.py --split test                          # M5, after pre-registration
    uv run python scripts/predict.py --split test --labels cleaned         # E2 on test

The systems and seeds are the frozen ones in results/policy_v1.json: S0, S1 and
S3 with seeds 0 to 4, S2 with seed 0. labels="cleaned" loads the E2 models
trained without the flagged rows (run tags ending "_cleaned").

The queries are the mixed set (in-scope plus out-of-scope), raw text, exactly
as scripts/train.py saved them for validation. The format-controlled copy of a
split is made afterwards by scripts/predict_format_control.py.

--check predicts validation again and compares it with the saved validation
files instead of writing anything: the same predicted labels, and confidences
equal to 4 decimal places. It proves this script loads the right checkpoint for
every system and seed before it is pointed at test.

--split test is refused until results/preregistration.md exists (rigour rule 1).
"""

import argparse
import gc
import json
import sys
import time

import numpy as np
import torch

from router import data
from router.provenance import run_metadata
from router.seed import pick_device
from router.systems.base import PREDICTIONS_DIR, load_predictions, save_predictions
from router.systems.registry import load_predictor, run_tag, system_and_checkpoint

POLICY_FILE = data.RESULTS_DIR / "policy_v1.json"


def frozen_runs() -> dict[str, list[int]]:
    """{system tag: seeds} frozen in policy_v1.json."""
    policy = json.loads(POLICY_FILE.read_text())
    return {tag: sorted(int(seed) for seed in v["thresholds"]) for tag, v in policy["systems"].items()}


def same_as_saved(tag: str, seed: int, split: str, preds) -> list[str]:
    """Differences between new predictions and the saved file for the same run (empty list: identical)."""
    saved, _ = load_predictions(tag, seed, split)
    problems = []
    if list(saved["pred_label"]) != list(preds.pred_label):
        n = sum(a != b for a, b in zip(saved["pred_label"], preds.pred_label))
        problems.append(f"{n} predicted labels differ")
    if not np.allclose(saved["confidence"].to_numpy(dtype=float), preds.confidence, atol=1e-4):
        problems.append("confidences differ by more than 1e-4")
    return problems


def main(split: str, labels: str, check: bool) -> None:
    if split == "test" and not (data.RESULTS_DIR / "preregistration.md").exists():
        sys.exit("Refusing to predict on the test split: results/preregistration.md does not exist yet (rigour rule 1).")
    if check and split != "val":
        sys.exit("--check compares with the saved validation files, so it needs --split val.")
    eval_df = data.load_eval_set(split, with_oos=True)
    device = pick_device()
    meta = run_metadata(data.load_config("data"))
    print(f"{split}: {len(eval_df)} queries ({int(eval_df['is_oos'].sum())} OOS), labels {labels}, device {device}"
          + (" | CHECK: nothing is written" if check else ""))

    failures = []
    for base_tag, seeds in frozen_runs().items():
        system, checkpoint = system_and_checkpoint(base_tag)
        tag = run_tag(base_tag, labels)
        for seed in seeds:
            start = time.perf_counter()
            predict = load_predictor(system, seed=seed, device=device, checkpoint=checkpoint, labels=labels)
            preds = predict(eval_df["text"].tolist())
            if check:
                problems = same_as_saved(tag, seed, split, preds)
                failures += [f"{tag} seed {seed}: {p}" for p in problems]
                status = "same as saved" if not problems else "; ".join(problems)
            else:
                save_predictions(tag, seed, split, eval_df, preds)
                info = {"metadata": meta, "split": split, "labels": labels, "system": system,
                        "checkpoint": checkpoint, "seed": seed, "device": str(device)}
                with open(PREDICTIONS_DIR / tag / f"seed{seed}_{split}_info.json", "w") as f:
                    json.dump(info, f, indent=2, default=str)
                status = "saved"
            print(f"  {tag} seed {seed}: {status} ({time.perf_counter() - start:.1f}s)", flush=True)
            del predict, preds  # free the model before loading the next one
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()

    if failures:
        sys.exit("CHECK FAILED:\n  " + "\n  ".join(failures))
    print("Check passed: every run reproduces its saved validation predictions." if check else f"Wrote split {split!r}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--labels", choices=["original", "cleaned"], default="original")
    parser.add_argument("--check", action="store_true", help="compare with the saved validation files, write nothing")
    args = parser.parse_args()
    main(args.split, args.labels, args.check)
