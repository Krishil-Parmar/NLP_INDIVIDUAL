"""Run S4 (Claude Sonnet 4.6 on Amazon Bedrock) on a format-controlled evaluation set.

Run from the project root (the llm dependency group holds the Anthropic SDK):
    uv run --group llm python scripts/run_s4.py --split val --limit 5    # smoke test: 5 queries
    uv run --group llm python scripts/run_s4.py --split val              # the whole validation set
    uv run --group llm python scripts/run_s4.py --split val --config s4_llm_fewshot   # few-shot variant
    uv run --group llm python scripts/run_s4.py --split test --config s4_jeff          # the Jeff set (E7)
    uv run --group llm python scripts/run_s4.py --split test_v2 --config s4_jeff_fewshot   # Jeff test v2 (E8)
    uv run --group llm python scripts/run_s4.py --split val --config s4_llm_retrieval      # retrieval variant, validation only

A config with "retrieval_k" (the retrieval variant) sends each query with its k most similar
train queries and their labels, found with S2's encoder. It is a validation-only experiment.

A config with "dataset: jeff" runs S4 on the Jeff set instead of Banking77: the
14 Jeff intents with their definitions from configs/jeff.yaml, few-shot examples
from the Jeff support split, and the Jeff test split as the queries. Its
predictions are saved as split "jeff_<split>_fmt".

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

from router import data, jeff
from router.preprocess import format_control
from router.provenance import run_metadata
from router.systems import s4_llm
from router.systems.base import PREDICTIONS_DIR, label_names_in_id_order, save_predictions


def retrieval_messages(queries: list[str], cfg: dict) -> list[str]:
    """For each (format-controlled) query: a message with its cfg["retrieval_k"] most similar train
    queries and their labels, by cosine with S2's encoder. Train split only, so validation stays unseen."""
    from router.seed import pick_device
    from router.systems import registry, s2_embedding

    encoder = s2_embedding.load_encoder(cfg["retrieval_encoder"], pick_device())
    train = data.load_train()
    support = registry.s2_support_set(encoder, cfg["retrieval_encoder"])  # cached embeddings of the same train rows, same order
    similarities = s2_embedding.embed(encoder, queries) @ support.embeddings.T
    nearest = np.argsort(-similarities, axis=1, kind="stable")[:, : cfg["retrieval_k"]]
    texts, labels = train["text"].map(format_control).tolist(), train["label"].tolist()
    return [s4_llm.retrieval_message(q, [(texts[j], labels[j]) for j in row]) for q, row in zip(queries, nearest)]


def dataset_inputs(cfg: dict, split: str) -> dict:
    """What S4 is given and asked about for this config's dataset (Banking77 unless "dataset: jeff")."""
    if cfg.get("dataset", "banking77") == "jeff":
        if split not in ("test", "test_v2"):
            sys.exit("The Jeff set has no validation split: use --split test (E7) or --split test_v2 (E8).")
        jeff_cfg = data.load_config("jeff")
        return {"label_names": jeff.intent_names(jeff_cfg), "example_pool": jeff.load_jeff("support"),
                "descriptions": jeff.intent_descriptions(jeff_cfg), "oos_description": jeff.oos_description(jeff_cfg),
                "eval_df": jeff.load_jeff(split), "out_split": f"jeff_{split}_fmt"}
    return {"label_names": label_names_in_id_order(), "example_pool": data.load_split("train"),
            "descriptions": None, "oos_description": "",
            "eval_df": data.load_eval_set(split, with_oos=True), "out_split": f"{split}_fmt"}


def main(split: str, limit: int | None, config: str, show_prompt: bool = False) -> None:
    if split == "test" and not show_prompt and not (data.RESULTS_DIR / "preregistration.md").exists():
        sys.exit("Refusing to run S4 on the test split: results/preregistration.md does not exist yet (rigour rule 1).")
    if split == "test_v2" and not show_prompt and not (data.RESULTS_DIR / "preregistration_e8.md").exists():
        sys.exit("Refusing to run S4 on Jeff test v2: results/preregistration_e8.md does not exist yet.")
    cfg = data.load_config(config)
    if cfg.get("retrieval_k") and split != "val":  # checked before any query is read
        sys.exit("The retrieval variant is a validation-only experiment: use --split val.")
    inputs = dataset_inputs(cfg, split)
    label_names = inputs["label_names"]
    template = (data.PROJECT_ROOT / cfg["prompt_file"]).read_text(encoding="utf-8")
    examples = None
    if "examples_per_intent" in cfg:  # few-shot variant: examples come from train (Banking77) or support (Jeff) only
        examples = s4_llm.pick_examples(inputs["example_pool"], label_names, cfg["examples_per_intent"], cfg["example_seed"])
    system_prompt = s4_llm.build_system_prompt(template, label_names, cfg["oos_answer"], examples,
                                               inputs["descriptions"], inputs["oos_description"])
    if show_prompt:  # print the prompt and stop: no query is read and no call is made
        print(system_prompt)
        return
    prices = cfg["prices_usd_per_million_tokens"]

    eval_df = inputs["eval_df"]
    controlled = eval_df.assign(text=eval_df["text"].map(format_control))
    texts = controlled["text"].tolist()
    if cfg.get("retrieval_k"):  # retrieval variant: each API input is the query plus its nearest labelled train queries
        texts = retrieval_messages(texts, cfg)
    to_run = texts[:limit] if limit else texts

    cache = s4_llm.ResponseCache(data.PROJECT_ROOT / cfg["cache_file"])
    # Spend so far counts every S4 variant's cache, so all of them share the one cap.
    spent = sum(s4_llm.ResponseCache(data.PROJECT_ROOT / path).total_cost() for path in cfg["budget_cache_files"])
    tracker = s4_llm.SpendTracker(
        # The bound includes the longest query, which matters when queries carry retrieved examples.
        cfg["spend_cap_usd"], spent, s4_llm.upper_bound_call_cost(system_prompt + max(to_run, key=len), cfg["max_tokens"], prices)
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
        print(f"spend in this variant's cache ${cache.total_cost():.4f}, all S4 variants ${tracker.spent:.4f} | projected cost of the {uncached_in_split} "
              f"uncached queries in this split: ${uncached_in_split * cost.mean():.2f}")
        for record, label in list(zip(records, labels))[:5]:
            print(f"  {label:<32} <- {record['text']!r}  raw={record['raw']!r}")

    if uncached_in_split == 0:
        preds = s4_llm.predictions_from_cache(texts, cfg, label_names, system_prompt, cache)
        out_split = inputs["out_split"]
        save_predictions(cfg["system_tag"], 0, out_split, controlled, preds)
        all_records = [cache.get(key(t)) for t in texts]
        info = {
            "metadata": meta,
            "model_id": cfg["model_id"],
            "prompt_version": cfg["prompt_version"],
            "examples_per_intent": cfg.get("examples_per_intent", 0),
            "retrieval_k": cfg.get("retrieval_k", 0),
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
    parser.add_argument("--split", choices=["val", "test", "test_v2"], default="val")  # test_v2: Jeff configs only (E8)
    parser.add_argument("--limit", type=int, default=None, help="only the first N queries (smoke test)")
    parser.add_argument("--config", choices=["s4_llm", "s4_llm_fewshot", "s4_llm_retrieval", "s4_jeff", "s4_jeff_fewshot"], default="s4_llm",
                        help="which S4 variant (and dataset)")
    parser.add_argument("--show-prompt", action="store_true", help="print the system prompt and stop (no API call)")
    args = parser.parse_args()
    main(args.split, args.limit, args.config, args.show_prompt)
