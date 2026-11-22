"""Choose the S1 checkpoint with the rule fixed in configs/s1_encoder.yaml.

Needs seed 0 of every candidate trained first (scripts/train.py --seeds 0).
Rule: higher validation macro-F1 wins if the gap is more than min_f1_gap points;
otherwise the candidate with lower CPU p95 latency (batch size 1) wins.

Run from the project root:
    uv run python scripts/select_s1.py

Writes results/s1_selection.json. Filling in `checkpoint:` in the config is
done by hand afterwards, so the choice is a visible, recorded decision.
"""

import json
import platform

import torch

from router import data
from router.cost import measure_latency
from router.metrics import classification_metrics
from router.provenance import run_metadata
from router.systems import s1_encoder
from router.systems.base import label_names_in_id_order, load_predictions

CHECKPOINT_DIR = data.PROJECT_ROOT / "checkpoints"
N_LATENCY_QUERIES = 200


def main() -> None:
    cfg = data.load_config("s1_encoder")
    label_names = label_names_in_id_order()
    val_in = data.load_split("val")
    latency_texts = val_in["text"].tolist()[:N_LATENCY_QUERIES]
    cpu = torch.device("cpu")

    results = {}
    for checkpoint in cfg["candidates"]:
        tag = s1_encoder.system_tag(checkpoint)
        folder = CHECKPOINT_DIR / "s1" / tag / "seed0"
        if not folder.exists():
            print(f"{checkpoint}: no seed-0 checkpoint, treated as failed")
            results[checkpoint] = {"status": "missing"}
            continue
        preds, _ = load_predictions(tag, 0, "val")
        in_scope = preds[~preds["is_oos"]]
        metrics = classification_metrics(in_scope["label"], in_scope["pred_label"], labels=label_names)

        model, tokenizer = s1_encoder.load(folder, cpu)
        max_length = s1_encoder.max_length_for(checkpoint)
        latency = measure_latency(
            lambda text: s1_encoder.predict_logits(model, tokenizer, [text], cpu, max_length),
            latency_texts,
        )
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
        if gap_points > cfg["selection"]["min_f1_gap"]:
            winner, reason = by_f1[0], f"macro-F1 higher by {gap_points:.2f} points (> {cfg['selection']['min_f1_gap']})"
        else:
            winner = min(ok, key=lambda c: ok[c]["cpu_latency"]["p95_ms"])
            reason = f"macro-F1 gap {gap_points:.2f} points (<= {cfg['selection']['min_f1_gap']}), so lower CPU p95 latency decides"

    out = {
        "metadata": run_metadata(cfg, seed=0),
        "latency_machine": f"{platform.machine()} {platform.processor()} {platform.platform()}, torch threads {torch.get_num_threads()}",
        "rule": cfg["selection"],
        "candidates": results,
        "winner": winner,
        "reason": reason,
    }
    with open(data.RESULTS_DIR / "s1_selection.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWinner: {winner} ({reason})")
    print("Next: set `checkpoint:` in configs/s1_encoder.yaml and log the decision in the README.")


if __name__ == "__main__":
    main()
