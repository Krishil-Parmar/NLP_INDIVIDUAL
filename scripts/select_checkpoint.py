"""Choose a system's model with the rule fixed in its config (S1 or S2).

Needs seed 0 of every candidate run first (scripts/train.py ... --seeds 0).
Rule: higher validation macro-F1 wins if the gap is more than min_f1_gap points;
otherwise the candidate with lower CPU p95 latency (batch size 1) wins.
A candidate with no run counts as failed.

Run from the project root:
    uv run python scripts/select_checkpoint.py --system s2
    uv run python scripts/select_checkpoint.py --system s1 --note "why a candidate has no run"

Writes results/<system>_selection.json. Filling in `checkpoint:` in the config
is done by hand afterwards, so the choice is a visible, recorded decision.
"""

import argparse
import json
import platform

import torch

from router import data
from router.cost import measure_latency
from router.metrics import classification_metrics
from router.provenance import run_metadata
from router.systems import s1_encoder, s2_embedding
from router.systems.base import label_names_in_id_order, list_runs, load_predictions
from router.systems.registry import load_predictor

CONFIGS = {"s1": "s1_encoder", "s2": "s2_embedding"}
TAGS = {"s1": s1_encoder.system_tag, "s2": s2_embedding.system_tag}
N_LATENCY_QUERIES = 200


def main(system: str, notes: list[str]) -> None:
    cfg = data.load_config(CONFIGS[system])
    label_names = label_names_in_id_order()
    latency_texts = data.load_split("val")["text"].tolist()[:N_LATENCY_QUERIES]
    cpu = torch.device("cpu")
    runs = list_runs("val")

    results = {}
    for checkpoint in cfg["candidates"]:
        tag = TAGS[system](checkpoint)
        if 0 not in runs.get(tag, []):
            print(f"{checkpoint}: no seed-0 run, treated as failed")
            results[checkpoint] = {"status": "missing"}
            continue
        preds, _ = load_predictions(tag, 0, "val")
        in_scope = preds[~preds["is_oos"]]
        metrics = classification_metrics(in_scope["label"], in_scope["pred_label"], labels=label_names)
        predict = load_predictor(system, seed=0, device=cpu, checkpoint=checkpoint)
        latency = measure_latency(lambda text: predict([text]), latency_texts)
        results[checkpoint] = {"status": "ok", "val_metrics": metrics, "cpu_latency": latency}
        print(f"{checkpoint}: val macro-F1 {metrics['macro_f1']:.4f}, accuracy {metrics['accuracy']:.4f}, "
              f"CPU p50 {latency['p50_ms']:.1f} ms, p95 {latency['p95_ms']:.1f} ms")

    ok = {c: r for c, r in results.items() if r["status"] == "ok"}
    if not ok:
        raise SystemExit("No candidate has a seed-0 run.")
    if len(ok) == 1:
        winner, reason = next(iter(ok)), "only candidate with a completed run"
    else:
        by_f1 = sorted(ok, key=lambda c: ok[c]["val_metrics"]["macro_f1"], reverse=True)
        gap_points = 100 * (ok[by_f1[0]]["val_metrics"]["macro_f1"] - ok[by_f1[1]]["val_metrics"]["macro_f1"])
        min_gap = cfg["selection"]["min_f1_gap"]
        if gap_points > min_gap:
            winner, reason = by_f1[0], f"macro-F1 higher by {gap_points:.2f} points (> {min_gap})"
        else:
            winner = min(ok, key=lambda c: ok[c]["cpu_latency"]["p95_ms"])
            reason = f"macro-F1 gap {gap_points:.2f} points (<= {min_gap}), so lower CPU p95 latency decides"

    out = {
        "metadata": run_metadata(cfg, seed=0),
        "latency_machine": f"{platform.machine()} {platform.platform()}, torch threads {torch.get_num_threads()}",
        "rule": cfg["selection"],
        "candidates": results,
        "winner": winner,
        "reason": reason,
        "notes": notes,
    }
    with open(data.RESULTS_DIR / f"{system}_selection.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWinner: {winner} ({reason})")
    print(f"Next: set `checkpoint:` in configs/{CONFIGS[system]}.yaml and log the decision in the README.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=list(CONFIGS), required=True)
    parser.add_argument("--note", action="append", default=[], help="free-text note saved in the result file (repeatable)")
    args = parser.parse_args()
    main(args.system, args.note)
