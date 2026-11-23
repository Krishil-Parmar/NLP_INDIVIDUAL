"""E7: domain shift. Router metrics on the Jeff test split (README 6.1, RQ6).

Run from the project root, after scripts/predict_jeff.py and
scripts/run_s4.py --split test --config s4_jeff / s4_jeff_fewshot:
    uv run python scripts/evaluate_jeff.py

Definitions (fixed in results/preregistration.md before any Jeff test run):
  - Queries: the 180 Jeff test queries (140 in scope, 40 out of scope), format-controlled.
  - correct: the predicted intent equals the true label. S2 and S3 cannot answer
    "out of scope", so handling an out-of-scope query is always an error.
  - Matched risk r*: few-shot S4's error rate on the same queries (the S4 variant
    frozen in results/policy_v1.json on Banking77 validation).
  - Coverage at matched risk: recomputed on these queries per seed, mean over
    seeds, with a bootstrap interval (r* recomputed on every resample, rows drawn
    inside the in-scope and out-of-scope strata, 1,000 resamples). The RQ6 number.
  - Frozen thresholds: policy v1's Banking77 thresholds applied unchanged.
  - In-scope accuracy and macro-F1 over the 14 Jeff intents, and OOS AUROC.
S0 and S1 are not applicable: they cannot output a Jeff intent.

Writes results/e7_jeff.json and results/e7_jeff.md. Never touches results/policy_v1.json.
Refused until results/preregistration.md exists (rigour rule 1).
"""

import json
import sys

import numpy as np
import pandas as pd

from router import data, jeff
from router.metrics import bootstrap_ci, classification_metrics, coverage_at_risk, mean_std_over_seeds, oos_auroc
from router.preprocess import format_control
from router.provenance import metadata_markdown, run_metadata
from router.systems.base import PREDICTIONS_DIR, list_runs, load_predictions
from router.tables import md_table

SPLIT = "jeff_test_fmt"
POLICY_FILE = data.RESULTS_DIR / "policy_v1.json"


def load_checked(tag: str, seeds: list[int], expected: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """{seed: predictions} for one system, checked to be the Jeff test queries in the same order."""
    runs = {}
    for seed in seeds:
        df, _ = load_predictions(tag, seed, SPLIT)
        same = (df["text"].tolist() == expected["text"].tolist() and df["label"].tolist() == expected["label"].tolist()
                and df["is_oos"].astype(str).tolist() == expected["is_oos"].astype(str).tolist())
        if not same:
            sys.exit(f"{tag} seed {seed} did not score the Jeff test queries in order; the comparison would be unfair.")
        runs[seed] = df
    return runs


def in_scope_row(df: pd.DataFrame, names: list[str]) -> dict[str, float]:
    """In-scope accuracy and macro-F1 over the 14 intents, plus OOS AUROC, for one run."""
    in_scope = ~df["is_oos"].to_numpy(dtype=bool)
    scores = classification_metrics(df["label"][in_scope].tolist(), df["pred_label"][in_scope].tolist(), names)
    return {"in_scope_accuracy": scores["accuracy"], "macro_f1": scores["macro_f1"],
            "oos_auroc": oos_auroc(df["is_oos"].to_numpy(dtype=bool), df["oos_score"].to_numpy(dtype=float))}


def main() -> None:
    if not (data.RESULTS_DIR / "preregistration.md").exists():
        sys.exit("Refusing to score the Jeff test split: results/preregistration.md does not exist yet (rigour rule 1).")
    cfg_e6 = data.load_config("e6_router")
    policy = json.loads(POLICY_FILE.read_text())
    names = jeff.intent_names(data.load_config("jeff"))
    queries = jeff.load_jeff("test")
    expected = queries.assign(text=queries["text"].map(format_control))
    is_oos = expected["is_oos"].to_numpy(dtype=bool)
    available = list_runs(SPLIT)
    boot = cfg_e6["bootstrap"]

    # S4: every Jeff variant is reported; the variant frozen on Banking77 validation sets the matched risk.
    s4_tags = [data.load_config(c)["system_tag"] for c in ["s4_jeff", "s4_jeff_fewshot"]]
    llm_tag = policy["llm_system"]
    if llm_tag not in s4_tags or llm_tag not in available:
        sys.exit(f"No {SPLIT!r} predictions for the frozen S4 variant {llm_tag}. Run scripts/run_s4.py first.")
    s4 = {}
    for tag in [t for t in s4_tags if t in available]:
        df = load_checked(tag, [0], expected)[0]
        wrong = (df["pred_label"] != df["label"]).to_numpy()
        info = json.loads((PREDICTIONS_DIR / tag / f"seed0_{SPLIT}_info.json").read_text())
        s4[tag] = {"error_rate": float(wrong.mean()), "in_scope_accuracy": float(1 - wrong[~is_oos].mean()),
                   "oos_recall": float(1 - wrong[is_oos].mean()), "invalid": int((df["pred_label"] == "invalid").sum()),
                   "cost_per_call_usd": info["mean_cost_per_call_usd"], "spend_usd": info["spend_usd_this_split"],
                   **in_scope_row(df, names)}
    s4_wrong = (load_checked(llm_tag, [0], expected)[0]["pred_label"] != expected["label"]).to_numpy()
    r_star = float(s4_wrong.mean())

    # Routers: the frozen S2 and S3 runs, with the Jeff support set.
    routers = [tag for tag in policy["systems"] if tag.split("_")[0] in ("s2", "s3")]
    results = {}
    for tag in routers:
        seeds = sorted(int(s) for s in policy["systems"][tag]["thresholds"])
        if tag not in available or available[tag] != seeds:
            sys.exit(f"{tag}: expected seeds {seeds} on {SPLIT!r}, found {available.get(tag)}. Run scripts/predict_jeff.py.")
        runs = load_checked(tag, seeds, expected)
        per_seed = []
        arrays = []
        for seed, df in runs.items():
            correct = (df["pred_label"] == df["label"]).to_numpy()
            confidence = df["confidence"].to_numpy(dtype=float)
            arrays.append((correct, confidence))
            frozen = float(policy["systems"][tag]["thresholds"][str(seed)])
            per_seed.append({**in_scope_row(df, names),
                             **jeff.routing_row(correct, confidence, is_oos, r_star, frozen, cfg_e6["fixed_risks"])})

        def coverage_metric(rows: np.ndarray, arrays=arrays) -> float:
            risk = s4_wrong[rows].mean()  # the matched risk is recomputed on every resample
            return float(np.mean([coverage_at_risk(c[rows], conf[rows], risk)["coverage"] for c, conf in arrays]))

        results[tag] = {"seeds": seeds, "per_seed": per_seed, "summary": mean_std_over_seeds(per_seed),
                        "coverage_at_matched_risk_ci": bootstrap_ci(coverage_metric, is_oos, boot["resamples"], boot["seed"], boot["ci"])}

    meta = run_metadata({"jeff": data.load_config("jeff"), "e6_router": cfg_e6}, seed=boot["seed"])
    out = {"metadata": meta, "split": SPLIT, "n_queries": len(expected), "n_oos": int(is_oos.sum()),
           "llm_system": llm_tag, "matched_risk": r_star, "s4_variants": s4, "systems": results,
           "best_router": policy["best_router"], "not_applicable": ["s0_tfidf", "s1_distilroberta-base"]}
    with open(data.RESULTS_DIR / "e7_jeff.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    (data.RESULTS_DIR / "e7_jeff.md").write_text(write_markdown(out, meta))
    best = results[policy["best_router"]]
    ci = best["coverage_at_matched_risk_ci"]
    print(f"matched risk (few-shot S4 error on Jeff test): {100 * r_star:.2f}%")
    print(f"best router {policy['best_router']}: coverage at matched risk {100 * ci['value']:.2f}% "
          f"[{100 * ci['ci_low']:.2f}, {100 * ci['ci_high']:.2f}]. Wrote results/e7_jeff.json and .md")


def pct(summary: dict, key: str) -> str:
    return f"{100 * summary[key]['mean']:.2f} ± {100 * summary[key]['std']:.2f}"


def write_markdown(out: dict, meta: dict) -> str:
    s4_rows = [{"variant": tag, "error rate (%)": f"{100 * v['error_rate']:.2f}", "in-scope accuracy (%)": f"{100 * v['in_scope_accuracy']:.2f}",
                "macro-F1 (%)": f"{100 * v['macro_f1']:.2f}", "OOS recall (%)": f"{100 * v['oos_recall']:.2f}", "invalid": v["invalid"],
                "US$ per call": f"{v['cost_per_call_usd']:.5f}", "sets matched risk": "yes" if tag == out["llm_system"] else ""}
               for tag, v in out["s4_variants"].items()]
    rows, frozen_rows, fixed_rows = [], [], []
    for tag, r in out["systems"].items():
        s, ci = r["summary"], r["coverage_at_matched_risk_ci"]
        rows.append({"system": tag, "seeds": len(r["seeds"]), "in-scope accuracy (%)": pct(s, "in_scope_accuracy"),
                     "macro-F1 (%)": pct(s, "macro_f1"), "OOS AUROC (%)": pct(s, "oos_auroc"),
                     "coverage at matched risk (%)": pct(s, "coverage_at_matched_risk"),
                     "95% CI": f"[{100 * ci['ci_low']:.2f}, {100 * ci['ci_high']:.2f}]",
                     "selective risk there (%)": pct(s, "selective_risk_at_matched_risk"), "AURC (%)": pct(s, "aurc")})
        frozen_rows.append({"system": tag, "coverage (%)": pct(s, "coverage_frozen_threshold"),
                            "selective risk (%)": pct(s, "selective_risk_frozen_threshold"),
                            f"OOS handled (of {out['n_oos']})": f"{s['oos_handled_frozen_threshold']['mean']:.1f}"})
        fixed_rows.append({"system": tag, **{f"coverage at risk {100 * float(k.split('_')[-1]):g}% (%)": pct(s, k)
                                             for k in s if k.startswith("coverage_at_0")}})
    parts = [
        "# E7 results: the Jeff set (domain shift)",
        "Written by `scripts/evaluate_jeff.py`. Do not edit by hand.",
        "## Provenance", metadata_markdown(meta),
        "## Setup",
        f"{out['n_queries']} Jeff test queries ({out['n_oos']} out of scope), format-controlled. S2 and S3 are the frozen models with the "
        "140 Jeff support queries as their support set: no retraining, no new settings. S0 and S1 are not applicable: they are "
        "softmax classifiers over the 77 Banking77 intents and cannot output a Jeff intent. The Jeff set was written by a Claude agent "
        "(README 6.1); S4 is a Claude model too.",
        "## S4 on the Jeff set",
        md_table(s4_rows),
        f"Matched risk (the frozen few-shot variant's error rate on these queries): **{100 * out['matched_risk']:.2f}%**.",
        "## Routers at matched risk (threshold recomputed on these queries), mean ± std over seeds",
        "The interval is a bootstrap of the seed-mean coverage, with the matched risk recomputed on every resample.",
        md_table(rows),
        "## Routers at the frozen Banking77 thresholds (policy v1 applied unchanged)",
        md_table(frozen_rows),
        "## Coverage at fixed risk levels",
        md_table(fixed_rows),
    ]
    return "\n\n".join(parts) + "\n"


if __name__ == "__main__":
    main()
