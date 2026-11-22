"""Run S4 (Claude Sonnet 4.6 on Amazon Bedrock) on a format-controlled evaluation set.

Run from the project root (the llm dependency group holds the Anthropic SDK):
    uv run --group llm python scripts/run_s4.py --split val --limit 5    # smoke test: 5 queries
    uv run --group llm python scripts/run_s4.py --split val              # the whole validation set

Every query is lowercased and stripped of punctuation first (format_control),
the same input as the primary E3 and E6 numbers. Replies are cached in
results/s4_cache/, so a query is never paid for twice, and the run stops
before total spend could cross spend_cap_usd in configs/s4_llm.yaml.

When every query in the split is cached, the predictions are saved as split
"<split>_fmt" (results/predictions/s4_claude-sonnet-4-6/seed0_<split>_fmt.csv),
where scripts/evaluate.py, evaluate_oos.py and evaluate_router.py read them.

--split test is refused until results/preregistration.md exists (rigour rule 1).
"""

import argparse
import json
import sys
from collections import Counter

import numpy as np

from router import data
from router.preprocess import format_control
from router.provenance import run_metadata
from router.systems import s4_llm
from router.systems.base import PREDICTIONS_DIR, label_names_in_id_order, save_predictions


def main(split: str, limit: int | None) -> None:
    if split == "test" and not (data.RESULTS_DIR / "preregistration.md").exists():
        sys.exit("Refusing to run S4 on the test split: results/preregistration.md does not exist yet (rigour rule 1).")
    cfg = data.load_config("s4_llm")
    label_names = label_names_in_id_order()
    template = (data.PROJECT_ROOT / cfg["prompt_file"]).read_text(encoding="utf-8")
    system_prompt = s4_llm.build_system_prompt(template, label_names, cfg["oos_answer"])
    prices = cfg["prices_usd_per_million_tokens"]

    eval_df = data.load_eval_set(split, with_oos=True)
    controlled = eval_df.assign(text=eval_df["text"].map(format_control))
    texts = controlled["text"].tolist()
    to_run = texts[:limit] if limit else texts

    cache = s4_llm.ResponseCache(data.PROJECT_ROOT / cfg["cache_file"])
    tracker = s4_llm.SpendTracker(
        cfg["spend_cap_usd"], cache.total_cost(), s4_llm.upper_bound_call_cost(system_prompt, cfg["max_tokens"], prices)
    )
    meta = run_metadata({**data.load_config("data"), "s4_llm": cfg})  # recorded before any call
    print(f"{cfg['model_id']} | {len(to_run)} of {len(texts)} queries requested | "
          f"{len(cache.records)} replies already cached | spend so far ${tracker.spent:.4f} of ${cfg['spend_cap_usd']:.2f}")

    def key(text: str) -> str:
        return s4_llm.cache_key(cfg["model_id"], system_prompt, text)

    if any(cache.get(key(t)) is None for t in to_run):
        client = s4_llm.make_client(cfg)
        counts = s4_llm.fill_cache(to_run, cfg, label_names, system_prompt, cache, tracker, client)
    else:
        counts = {"calls": 0, "stopped_by_cap": False}

    # Summary over every cached reply for the requested queries.
    records = [r for r in (cache.get(key(t)) for t in to_run) if r is not None]
    labels = [s4_llm.parse_answer(r["raw"], label_names, cfg["oos_answer"]) for r in records]
    latency = np.array([r["latency_ms"] for r in records])
    cost = np.array([r["cost_usd"] for r in records])
    uncached_in_split = sum(cache.get(key(t)) is None for t in texts)
    print(f"API calls made this run: {counts['calls']}" + (" (stopped by the spend cap)" if counts["stopped_by_cap"] else ""))
    if records:
        mean_usage = {k: float(np.mean([r["usage"][k] for r in records])) for k in records[0]["usage"]}
        print(f"replies: {len(records)} | invalid: {labels.count(s4_llm.INVALID)} | out_of_scope: {labels.count(data.OOS_LABEL)} "
              f"| stop reasons: {dict(Counter(r['stop_reason'] for r in records))}")
        print(f"mean tokens per call: { {k: round(v, 1) for k, v in mean_usage.items()} }")
        print(f"mean cost per call ${cost.mean():.5f} | API latency p50 {np.percentile(latency, 50):.0f} ms, p95 {np.percentile(latency, 95):.0f} ms")
        print(f"total spend (all cached calls) ${cache.total_cost():.4f} | projected cost of the {uncached_in_split} "
              f"uncached queries in this split: ${uncached_in_split * cost.mean():.2f}")
        for record, label in list(zip(records, labels))[:5]:
            print(f"  {label:<32} <- {record['text']!r}  raw={record['raw']!r}")

    if uncached_in_split == 0:
        preds = s4_llm.predictions_from_cache(texts, cfg, label_names, system_prompt, cache)
        out_split = f"{split}_fmt"
        save_predictions(cfg["system_tag"], 0, out_split, controlled, preds)
        all_records = [cache.get(key(t)) for t in texts]
        info = {
            "metadata": meta,
            "model_id": cfg["model_id"],
            "prompt_version": cfg["prompt_version"],
            "n_queries": len(texts),
            "spend_usd_this_split": float(sum(r["cost_usd"] for r in all_records)),
            "mean_cost_per_call_usd": float(np.mean([r["cost_usd"] for r in all_records])),
            "api_latency_ms": {"p50": float(np.percentile([r["latency_ms"] for r in all_records], 50)),
                               "p95": float(np.percentile([r["latency_ms"] for r in all_records], 95))},
            "first_call_utc": min(r["timestamp_utc"] for r in all_records),
            "last_call_utc": max(r["timestamp_utc"] for r in all_records),
        }
        with open(PREDICTIONS_DIR / cfg["system_tag"] / f"seed0_{out_split}_info.json", "w") as f:
            json.dump(info, f, indent=2, default=str)
        print(f"Saved S4 predictions for split {out_split!r}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--limit", type=int, default=None, help="only the first N queries (smoke test)")
    args = parser.parse_args()
    main(args.split, args.limit)
