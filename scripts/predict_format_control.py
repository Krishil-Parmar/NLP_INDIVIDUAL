"""Re-predict an evaluation set with format-controlled text (the E3 control).

Run from the project root:
    uv run python scripts/predict_format_control.py --split val

Why: CLINC150's out-of-scope queries are all lowercase with no punctuation, and
most Banking77 queries are not, so formatting alone separates them (see
results/e3_val.md, "surface cues"). This script lowercases and strips
punctuation from EVERY query in the mixed set (in-scope and OOS alike, with
router.preprocess.format_control) and predicts again with the saved models.

No training happens. The models and the S2/S3 support examples are unchanged;
only the evaluated queries change. Every system and seed that has saved
predictions for the split is re-run. Output goes to the split name "<split>_fmt"
(results/predictions/<system>/seed<n>_<split>_fmt.csv), which
scripts/evaluate_oos.py and scripts/evaluate.py read like any other split.

--split test is refused until results/preregistration.md exists (rigour rule 1).
"""

import argparse
import gc
import json
import sys
import time

import torch

from router import data
from router.preprocess import format_control
from router.provenance import run_metadata
from router.seed import pick_device
from router.systems.base import PREDICTIONS_DIR, list_runs, save_predictions
from router.systems.registry import load_predictor

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


def main(split: str) -> None:
    if split == "test" and not (data.RESULTS_DIR / "preregistration.md").exists():
        sys.exit("Refusing to predict on the test split: results/preregistration.md does not exist yet (rigour rule 1).")
    runs = list_runs(split)
    if not runs:
        sys.exit(f"No saved predictions for split {split!r}, so there is nothing to re-run.")

    eval_df = data.load_eval_set(split, with_oos=True)
    controlled = eval_df.assign(text=eval_df["text"].map(format_control))
    out_split = f"{split}_fmt"
    device = pick_device()
    meta = run_metadata(data.load_config("data"))
    print(f"{len(controlled)} queries ({int(controlled['is_oos'].sum())} OOS), device {device}")

    for tag, seeds in runs.items():
        system, checkpoint = system_and_checkpoint(tag)
        for seed in seeds:
            start = time.perf_counter()
            predict = load_predictor(system, seed=seed, device=device, checkpoint=checkpoint)
            preds = predict(controlled["text"].tolist())
            save_predictions(tag, seed, out_split, controlled, preds)
            info = {"metadata": meta, "source_split": split, "transform": "router.preprocess.format_control",
                    "system": system, "checkpoint": checkpoint, "seed": seed, "device": str(device)}
            with open(PREDICTIONS_DIR / tag / f"seed{seed}_{out_split}_info.json", "w") as f:
                json.dump(info, f, indent=2, default=str)
            print(f"  {tag} seed {seed}: {time.perf_counter() - start:.1f}s")
            del predict, preds  # free the model before loading the next one
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
    print(f"Wrote split {out_split!r} for {len(runs)} systems.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    main(parser.parse_args().split)
