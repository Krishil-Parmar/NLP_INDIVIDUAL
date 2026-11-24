"""How strict is the LLM bar? Post-hoc, VALIDATION ONLY (owner approval 24 Nov 2026).

Run from the project root, after
    uv run --group llm python scripts/run_s4.py --split val --config s4_llm_retrieval
then
    uv run python scripts/evaluate_llm_bar.py

The second independent review asked whether few-shot S4 (77 intent names plus 2 examples
per intent) is a weak bar, since the routers learn from about 9,000 labelled queries. The
retrieval variant gives S4 each query's 10 most similar labelled train queries instead.
This script compares the three S4 variants on format-controlled validation (with the
out-of-scope pool) and asks what each router covers at each variant's error rate.

Everything is on validation, where the routers' thresholds were also chosen, so the
coverage numbers are in-sample. Nothing here touches a test split or changes a verdict.
Writes results/llm_bar_val.json and results/llm_bar_val.md.
"""

import json
import sys

import numpy as np
import pandas as pd

from router import data
from router.calibration import calibrated_confidence
from router.metrics import classification_metrics, coverage_at_risk
from router.provenance import metadata_markdown, run_metadata
from router.systems.base import PREDICTIONS_DIR, load_predictions
from router.tables import md_table

SPLIT = "val_fmt"
S4_VARIANTS = {"zero-shot": "s4_llm", "few-shot (frozen bar)": "s4_llm_fewshot", "retrieval (10 nearest train queries)": "s4_llm_retrieval"}


def s4_rows(names: list[str]) -> tuple[list[dict], dict[str, np.ndarray], pd.DataFrame]:
    """Each S4 variant's validation metrics, its wrong-answer mask, and the shared query table."""
    rows, wrong, reference = [], {}, None
    for variant, config in S4_VARIANTS.items():
        tag = data.load_config(config)["system_tag"]
        path = PREDICTIONS_DIR / tag / f"seed0_{SPLIT}.csv"
        if not path.exists():
            sys.exit(f"No {SPLIT} predictions for {tag}. Run scripts/run_s4.py --split val --config {config} first.")
        df = pd.read_csv(path, keep_default_na=False)
        reference = df if reference is None else reference
        if df["text"].tolist() != reference["text"].tolist():
            sys.exit(f"{tag} did not score the same validation queries in the same order.")
        is_oos = df["is_oos"].to_numpy(dtype=bool)
        w = (df["pred_label"] != df["label"]).to_numpy()
        info = json.loads((PREDICTIONS_DIR / tag / f"seed0_{SPLIT}_info.json").read_text())
        cls = classification_metrics(df["label"][~is_oos].tolist(), df["pred_label"][~is_oos].tolist(), names)
        rows.append({"variant": variant, "system_tag": tag, "error_rate": float(w.mean()), "in_scope_accuracy": cls["accuracy"],
                     "macro_f1": cls["macro_f1"], "oos_recall": float(1 - w[is_oos].mean()),
                     "invalid": int((df["pred_label"] == "invalid").sum()), "cost_per_call_usd": info["mean_cost_per_call_usd"]})
        wrong[variant] = w
    return rows, wrong, reference


def router_runs(policy: dict, reference: pd.DataFrame) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """{router: [(correct, confidence) per seed]} on validation, with E6's confidence (temperature-scaled for S0, S1)."""
    runs = {}
    for tag, frozen in policy["systems"].items():
        runs[tag] = []
        for seed in sorted(frozen["thresholds"], key=int):
            df, logits = load_predictions(tag, int(seed), SPLIT)
            if df["text"].tolist() != reference["text"].tolist():
                sys.exit(f"{tag} seed {seed} did not score the same validation queries in the same order.")
            temperature = frozen["temperatures"].get(seed)
            confidence = calibrated_confidence(logits, temperature) if temperature else df["confidence"].to_numpy(dtype=float)
            runs[tag].append(((df["pred_label"] == df["label"]).to_numpy(), confidence))
    return runs


def main() -> None:
    policy = json.loads((data.RESULTS_DIR / "policy_v1.json").read_text())
    names = sorted(data.load_label_map(), key=data.load_label_map().get)
    s4, wrong, reference = s4_rows(names)
    is_oos = reference["is_oos"].to_numpy(dtype=bool)
    runs = router_runs(policy, reference)
    coverage_rows, pair_rows = [], []
    for variant, w in wrong.items():
        bar = float(w.mean())
        for tag, seeds in runs.items():
            per_seed = []
            for correct, confidence in seeds:
                threshold = coverage_at_risk(correct, confidence, bar)["threshold"]
                handled = confidence >= threshold
                combined = np.where(handled, ~correct, w)  # router answers what it handles, this S4 variant the rest
                per_seed.append({"coverage": handled.mean(), "in_scope_handled": (handled & ~is_oos).sum() / (~is_oos).sum(),
                                 "oos_handled": (handled & is_oos).sum(), "pair_error": combined.mean(),
                                 "pair_oos_answered": combined[is_oos].sum()})
            mean = {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}
            coverage_rows.append({"bar": variant, "bar_risk": bar, "system": tag, **mean, "s4_error": bar,
                                  "s4_oos_answered": int(w[is_oos].sum())})
    s2_correct = runs["s2_all-MiniLM-L6-v2"][0][0]
    in_scope_errors = {variant: {"s4_in_scope_wrong": int((w & ~is_oos).sum()), "s2_right": int((w & ~is_oos & s2_correct).sum())}
                       for variant, w in wrong.items()}
    out = {"metadata": run_metadata({"s4_llm_retrieval": data.load_config("s4_llm_retrieval")}), "split": SPLIT,
           "n_queries": len(reference), "n_oos": int(is_oos.sum()), "s4": s4, "coverage": coverage_rows,
           "s2_on_s4_in_scope_errors": in_scope_errors}
    (data.RESULTS_DIR / "llm_bar_val.json").write_text(json.dumps(out, indent=2, default=float))
    (data.RESULTS_DIR / "llm_bar_val.md").write_text(report(out))
    print("Wrote results/llm_bar_val.json and results/llm_bar_val.md")


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def report(out: dict) -> str:
    s4_table = [{"S4 variant": r["variant"], "error rate on the mix (%)": pct(r["error_rate"]), "in-scope accuracy (%)": pct(r["in_scope_accuracy"]),
                 "macro F1 (%)": pct(r["macro_f1"]), "OOS recall (%)": pct(r["oos_recall"]), "invalid": r["invalid"],
                 "US$ per call": f"{r['cost_per_call_usd']:.5f}",
                 "S2 right on its in-scope errors": f"{out['s2_on_s4_in_scope_errors'][r['variant']]['s2_right']} of "
                                                     f"{out['s2_on_s4_in_scope_errors'][r['variant']]['s4_in_scope_wrong']}"} for r in out["s4"]]
    cov_table = [{"bar (S4 variant)": r["bar"], "bar (%)": pct(r["bar_risk"]), "router": r["system"], "coverage (%)": pct(r["coverage"]),
                  "in-scope handled (% of in-scope)": pct(r["in_scope_handled"]), "OOS handled": f"{r['oos_handled']:.1f}",
                  "router + S4 wrong (%)": pct(r["pair_error"]), "S4 alone wrong (%)": pct(r["s4_error"]),
                  "OOS answered: pair / S4 alone": f"{r['pair_oos_answered']:.1f} / {r['s4_oos_answered']}"} for r in out["coverage"]]
    return "\n\n".join([
        "# How strict is the LLM bar? (validation only, post-hoc)",
        "Written by `scripts/evaluate_llm_bar.py`. Do not edit by hand. Validation only: no test split is touched and no "
        "verdict changes. The routers' thresholds below are chosen on these same queries, so the coverage numbers are in-sample.",
        "## Provenance", metadata_markdown(out["metadata"]),
        f"## The three S4 variants ({out['n_queries']} format-controlled validation queries, {out['n_oos']} out of scope)",
        md_table(s4_table),
        "## Router coverage at each variant's error rate",
        "For each S4 variant as the bar: each router's threshold is the one with the highest coverage whose selective risk is at "
        "or below that variant's error rate; then the router answers what it handles and that S4 variant answers the rest. "
        "Means over seeds.",
        md_table(cov_table),
    ]) + "\n"


if __name__ == "__main__":
    main()
