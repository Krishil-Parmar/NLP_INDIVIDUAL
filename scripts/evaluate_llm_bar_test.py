"""The stronger LLM bar on Banking77 test: run once, post-hoc (plan: results/plan_llm_bar_test.md).

Run from the project root, after
    uv run --group llm python scripts/run_s4.py --split test --config s4_llm_retrieval
then
    uv run python scripts/evaluate_llm_bar_test.py

The bar is S4 given each query's 10 most similar labelled train queries (the retrieval
variant). Two views, as for policy v1 in E6:
  - Recomputed: each router's coverage at the retrieval S4's test error rate (the RQ1-style
    number), with a bootstrap interval in which that error rate is recomputed on every resample.
  - Frozen: the thresholds chosen on validation at the retrieval S4's validation error
    (results/policy_llm_bar.json) applied unchanged. For these the end-to-end error is reported
    (the router answers what it handles, the retrieval S4 the rest) against the retrieval S4
    alone, with a paired bootstrap interval. The plan's reading: "fewer errors" if the whole
    interval is below 0, "more errors" if it is above 0, otherwise "no clear difference".

Nothing here changes a pre-registered verdict. Writes results/llm_bar_test.json and .md.
Refused without the plan, and refused once results/llm_bar_test.json exists (run once).
"""

import json
import sys

import numpy as np
import pandas as pd

from router import data
from router.calibration import calibrated_confidence
from router.metrics import bootstrap_ci, classification_metrics, coverage_at_risk, paired_bootstrap_diff
from router.provenance import metadata_markdown, run_metadata
from router.systems.base import PREDICTIONS_DIR, load_predictions
from router.tables import md_table

SPLIT = "test_fmt"


def reading(diff: dict) -> str:
    """The plan's reading of a paired interval on (pair error minus LLM-alone error)."""
    if diff["ci_high"] < 0:
        return "fewer errors"
    if diff["ci_low"] > 0:
        return "more errors"
    return "no clear difference"


def main() -> None:
    plan, out_file = data.RESULTS_DIR / "plan_llm_bar_test.md", data.RESULTS_DIR / "llm_bar_test.json"
    if not plan.exists():
        sys.exit("Refusing: results/plan_llm_bar_test.md does not exist.")
    if out_file.exists():
        sys.exit("Refusing: results/llm_bar_test.json exists. This test evaluation runs once.")
    bar_policy = json.loads((data.RESULTS_DIR / "policy_llm_bar.json").read_text())
    policy_v1 = json.loads((data.RESULTS_DIR / "policy_v1.json").read_text())
    e6 = json.loads((data.RESULTS_DIR / "e6_test_fmt.json").read_text())
    boot = data.load_config("e6_router")["bootstrap"]
    names = sorted(data.load_label_map(), key=data.load_label_map().get)

    tag = bar_policy["llm_system"]
    s4 = pd.read_csv(PREDICTIONS_DIR / tag / f"seed0_{SPLIT}.csv", keep_default_na=False)
    info = json.loads((PREDICTIONS_DIR / tag / f"seed0_{SPLIT}_info.json").read_text())
    is_oos = s4["is_oos"].to_numpy(dtype=bool)
    wrong = (s4["pred_label"] != s4["label"]).to_numpy()
    r_star = float(wrong.mean())
    cls = classification_metrics(s4["label"][~is_oos].tolist(), s4["pred_label"][~is_oos].tolist(), names)
    s4_row = {"system_tag": tag, "error_rate": r_star, "in_scope_accuracy": cls["accuracy"], "macro_precision": cls["macro_precision"],
              "macro_recall": cls["macro_recall"], "macro_f1": cls["macro_f1"], "oos_recall": float(1 - wrong[is_oos].mean()),
              "invalid": int((s4["pred_label"] == "invalid").sum()), "cost_per_call_usd": info["mean_cost_per_call_usd"],
              "spend_usd": info["spend_usd_this_split"], "oos_answered": int(wrong[is_oos].sum()),
              "fewshot_error_rate": e6["s4"]["error_rate"]}

    rows = {}
    for router, frozen in bar_policy["systems"].items():
        recomputed, deployed, arrays, pair_wrong = [], [], [], []
        for seed in sorted(frozen["thresholds"], key=int):
            df, logits = load_predictions(router, int(seed), SPLIT)
            if df["text"].tolist() != s4["text"].tolist():
                sys.exit(f"{router} seed {seed} did not score the same test queries in the same order as {tag}.")
            temperature = frozen["temperatures"].get(seed)
            conf = calibrated_confidence(logits, temperature) if temperature else df["confidence"].to_numpy(dtype=float)
            correct = (df["pred_label"] == df["label"]).to_numpy()
            arrays.append((correct, conf))
            at_r = coverage_at_risk(correct, conf, r_star)
            h = conf >= at_r["threshold"]
            recomputed.append({"coverage": at_r["coverage"], "in_scope_handled": (h & ~is_oos).sum() / (~is_oos).sum(),
                               "oos_handled": (h & is_oos).sum()})
            hf = conf >= frozen["thresholds"][seed]
            combined = np.where(hf, ~correct, wrong)
            pair_wrong.append(combined)
            deployed.append({"coverage": hf.mean(), "selective_risk": (~correct[hf]).mean() if hf.any() else 0.0,
                             "oos_handled": (hf & is_oos).sum(), "pair_error": combined.mean(),
                             "pair_oos_answered": combined[is_oos].sum(), "llm_calls": (~hf).sum()})

        def coverage_metric(r: np.ndarray, arrays=arrays) -> float:
            risk = wrong[r].mean()  # the bar is recomputed on every resample
            return float(np.mean([coverage_at_risk(c[r], k[r], risk)["coverage"] for c, k in arrays]))

        diff = paired_bootstrap_diff(lambda r, w=pair_wrong: float(np.mean([x[r].mean() for x in w])),
                                     lambda r: float(wrong[r].mean()), is_oos, boot["resamples"], boot["seed"], boot["ci"])
        mean = lambda rows_: {k: float(np.mean([x[k] for x in rows_])) for k in rows_[0]}  # noqa: E731
        rows[router] = {"seeds": len(arrays), "recomputed": mean(recomputed),
                        "recomputed_ci": bootstrap_ci(coverage_metric, is_oos, boot["resamples"], boot["seed"], boot["ci"]),
                        "frozen": mean(deployed), "pair_minus_llm": diff, "reading": reading(diff)}

    best = policy_v1["best_router"]
    out = {"metadata": run_metadata({"s4_llm_retrieval": data.load_config("s4_llm_retrieval")}, seed=boot["seed"]),
           "split": SPLIT, "n_queries": len(s4), "n_oos": int(is_oos.sum()), "s4": s4_row, "routers": rows,
           "best_router_policy_v1": best, "rq1_yardstick_N": data.load_config("preregistration")["N"]}
    out_file.write_text(json.dumps(out, indent=2, default=float))
    (data.RESULTS_DIR / "llm_bar_test.md").write_text(report(out))
    print(f"Retrieval S4 test error {100 * r_star:.2f}%. Wrote results/llm_bar_test.json and results/llm_bar_test.md")


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def report(out: dict) -> str:
    s4 = out["s4"]
    rec = [{"router": r, "seeds": v["seeds"], "coverage (%)": pct(v["recomputed"]["coverage"]),
            "95% CI": f"[{pct(v['recomputed_ci']['ci_low'])}, {pct(v['recomputed_ci']['ci_high'])}]",
            "in-scope handled (% of in-scope)": pct(v["recomputed"]["in_scope_handled"]),
            "OOS handled": f"{v['recomputed']['oos_handled']:.1f}"} for r, v in out["routers"].items()]
    dep = [{"router": r, "coverage (%)": pct(v["frozen"]["coverage"]), "LLM calls": f"{v['frozen']['llm_calls']:.0f}",
            "selective risk (%)": pct(v["frozen"]["selective_risk"]), "router + LLM wrong (%)": pct(v["frozen"]["pair_error"]),
            "LLM alone wrong (%)": pct(s4["error_rate"]),
            "difference (points, 95% CI)": f"{pct(v['pair_minus_llm']['diff'])} [{pct(v['pair_minus_llm']['ci_low'])}, {pct(v['pair_minus_llm']['ci_high'])}]",
            "reading": v["reading"], "OOS answered: pair / LLM alone": f"{v['frozen']['pair_oos_answered']:.1f} / {s4['oos_answered']}"}
           for r, v in out["routers"].items()]
    best = out["routers"][out["best_router_policy_v1"]]["recomputed"]["coverage"]
    return "\n\n".join([
        "# The stronger LLM bar on Banking77 test (post-hoc, run once)",
        "Written by `scripts/evaluate_llm_bar_test.py`. Do not edit by hand. Plan: `results/plan_llm_bar_test.md`, committed with the "
        "frozen thresholds (`results/policy_llm_bar.json`) before the test run. No pre-registered verdict changes.",
        "## Provenance", metadata_markdown(out["metadata"]),
        f"## The bar: S4 with the 10 most similar labelled train queries ({out['n_queries']} format-controlled test queries, {out['n_oos']} out of scope)",
        f"Error rate {pct(s4['error_rate'])}% (the frozen few-shot S4: {pct(s4['fewshot_error_rate'])}%). In-scope accuracy "
        f"{pct(s4['in_scope_accuracy'])}%, macro precision {pct(s4['macro_precision'])}%, macro recall {pct(s4['macro_recall'])}%, "
        f"macro F1 {pct(s4['macro_f1'])}%. OOS recall {pct(s4['oos_recall'])}%. Invalid replies {s4['invalid']}. "
        f"US${s4['cost_per_call_usd']:.5f} per call, US${s4['spend_usd']:.2f} for this split.",
        "## Coverage at this bar, recomputed on test",
        "Means over seeds; the interval is a bootstrap of the seed mean with the bar recomputed on every resample.",
        md_table(rec),
        f"RQ1-style yardstick, post-hoc: the policy v1 best router ({out['best_router_policy_v1']}) covers {pct(best)}% at this bar, "
        f"against N = {pct(out['rq1_yardstick_N'])}%. The pre-registered RQ1 verdict is unchanged.",
        "## Thresholds frozen on validation at this bar, applied unchanged (end to end)",
        "The router answers what it handles and the retrieval S4 answers the rest. Paired bootstrap of the error difference "
        "(pair minus LLM alone), 1,000 resamples stratified by in-scope and out-of-scope.",
        md_table(dep),
    ]) + "\n"


if __name__ == "__main__":
    main()
