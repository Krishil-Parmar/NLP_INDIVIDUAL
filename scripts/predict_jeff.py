"""E7: predict the Jeff test split with S2 and S3, using the Jeff support split as their support set.

Run from the project root:
    uv run python scripts/predict_jeff.py --smoke 5    # plumbing check on 5 support queries, nothing saved
    uv run python scripts/predict_jeff.py              # the Jeff test split (after pre-registration)

No training and no new settings. S2 (seed 0) and S3 (seeds 0 to 4) are the
frozen models from results/policy_v1.json; only their support set changes, from
Banking77 train to the 140 Jeff support queries (10 per intent), and so does the
label set (the 14 Jeff intents). S0 and S1 are softmax classifiers over the 77
Banking77 intents and cannot output a Jeff intent, so they are not run here.

Queries are format-controlled like E6 (router.preprocess.format_control); the
support examples are not, as in E6. Predictions go to split "jeff_test_fmt".

The test split is refused until results/preregistration.md exists (rigour rule 1).
"""

import argparse
import gc
import json
import sys
import time

import torch

from router import data, jeff
from router.preprocess import format_control
from router.provenance import run_metadata
from router.seed import pick_device
from router.systems.base import PREDICTIONS_DIR, save_predictions
from router.systems.registry import load_predictor, system_and_checkpoint

POLICY_FILE = data.RESULTS_DIR / "policy_v1.json"
OUT_SPLIT = "jeff_test_fmt"


def matcher_runs() -> dict[str, list[int]]:
    """{tag: seeds} of the frozen S2 and S3 runs in policy_v1.json."""
    policy = json.loads(POLICY_FILE.read_text())
    return {tag: sorted(int(s) for s in v["thresholds"]) for tag, v in policy["systems"].items()
            if system_and_checkpoint(tag)[0] in ("s2", "s3")}


def main(smoke: int | None) -> None:
    if smoke is None and not (data.RESULTS_DIR / "preregistration.md").exists():
        sys.exit("Refusing to predict the Jeff test split: results/preregistration.md does not exist yet (rigour rule 1).")
    support, names = jeff.jeff_support()
    if smoke:  # a few support queries against the full support set: checks the plumbing only
        eval_df = jeff.load_jeff("support").head(smoke)
    else:
        eval_df = jeff.load_jeff("test")
    controlled = eval_df.assign(text=eval_df["text"].map(format_control))
    device = pick_device()
    meta = run_metadata({"jeff": data.load_config("jeff")})
    print(f"{len(controlled)} queries ({int(controlled['is_oos'].sum())} OOS), {len(support)} support examples, "
          f"{len(names)} intents, device {device}" + (" | SMOKE: nothing is saved" if smoke else ""))

    for tag, seeds in matcher_runs().items():
        system, checkpoint = system_and_checkpoint(tag)
        for seed in seeds:
            start = time.perf_counter()
            predict = load_predictor(system, seed=seed, device=device, checkpoint=checkpoint, support=(support, names))
            preds = predict(controlled["text"].tolist())
            if smoke:
                print(f"  {tag} seed {seed}: {list(zip(controlled['label'], preds.pred_label))}")
            else:
                save_predictions(tag, seed, OUT_SPLIT, controlled, preds)
                info = {"metadata": meta, "source": "data/jeff/jeff_queries.csv (split test)",
                        "support": "data/jeff/jeff_queries.csv (split support)", "transform": "router.preprocess.format_control",
                        "system": system, "checkpoint": checkpoint, "seed": seed, "device": str(device)}
                with open(PREDICTIONS_DIR / tag / f"seed{seed}_{OUT_SPLIT}_info.json", "w") as f:
                    json.dump(info, f, indent=2, default=str)
                print(f"  {tag} seed {seed}: saved ({time.perf_counter() - start:.1f}s)", flush=True)
            del predict, preds
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
    if not smoke:
        print(f"Wrote split {OUT_SPLIT!r}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", type=int, default=None, help="predict this many SUPPORT queries and print them; nothing saved")
    main(parser.parse_args().smoke)
