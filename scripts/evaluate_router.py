"""E6: router policy v1, risk-coverage and cost. The headline experiment.

Run from the project root (after scripts/run_s4.py has saved S4's predictions):
    uv run python scripts/evaluate_router.py --split val_fmt

Policy v1 HANDLEs a query when the router's confidence is at or above a
threshold and escalates the rest to S4 (the LLM). Definitions (README 8.1):
  - correct: the predicted intent equals the true label. An out-of-scope query
    is never correct for S0-S3; handling one is the dangerous error.
  - matched risk r*: S4's error rate on the same queries.
  - threshold: per system and seed, the one giving the highest coverage whose
    selective risk is at or below r* (fitted on validation, frozen for test).
  - cost per query = router_cost + (1 - coverage) * llm_cost, with llm_cost
    S4's measured mean cost per call.
S0 and S1 are temperature-scaled first (fitted on validation in-scope rows).

On validation this writes the frozen choices to results/policy_v1.json (per
seed temperatures and thresholds, and the best router: highest mean coverage
at matched risk). A test split applies them unchanged and is refused until
results/preregistration.md exists (rigour rule 1).

Writes results/e6_<split>.json and .md, results/figures/e6_<split>_risk_coverage.png,
and on validation results/policy_v1.json and results/calibration_<split>.json.
"""

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")  # draw straight to a file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from router import data
from router.calibration import calibrated_confidence, fit_temperature
from router.cost import cost_per_query
from router.metrics import (
    aurc,
    coverage_at_risk,
    expected_calibration_error,
    handle_precision_recall,
    mean_std_over_seeds,
    paired_bootstrap_diff,
    risk_coverage_curve,
)
from router.provenance import metadata_markdown, run_metadata
from router.systems.base import PREDICTIONS_DIR, label_names_in_id_order, list_runs, load_predictions
from router.tables import md_table

FIGURES_DIR = data.RESULTS_DIR / "figures"
POLICY_FILE = data.RESULTS_DIR / "policy_v1.json"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"  # chart ink colours (dataviz reference palette)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_runs(split: str, tags: list[str]) -> dict[str, dict[int, tuple[pd.DataFrame, np.ndarray | None]]]:
    """{tag: {seed: (predictions, logits)}} for the given systems, checked to cover the same rows."""
    available = list_runs(split)
    missing = [tag for tag in tags if tag not in available]
    if missing:
        sys.exit(f"No saved {split!r} predictions for {missing}. Run the earlier steps first.")
    runs = {tag: {seed: load_predictions(tag, seed, split) for seed in available[tag]} for tag in tags}
    reference = runs[tags[0]][min(runs[tags[0]])][0]
    for tag, by_seed in runs.items():
        for seed, (df, _) in by_seed.items():
            same = all(df[col].equals(reference[col]) for col in ["text", "label", "is_oos"])
            if not same:
                sys.exit(f"{tag} seed {seed} scored different rows from {tags[0]}; the comparison would be unfair.")
    return runs


# ---------------------------------------------------------------------------
# Calibration (validation only)
# ---------------------------------------------------------------------------

def fit_calibration(runs: dict, cfg: dict, label_names: list[str]) -> dict:
    """Temperature per seed for the softmax systems, and ECE before/after on in-scope rows."""
    label_id = {name: i for i, name in enumerate(label_names)}
    out = {}
    for tag in cfg["ece_systems"]:
        out[tag] = {}
        for seed, (df, logits) in runs[tag].items():
            in_scope = ~df["is_oos"].to_numpy(dtype=bool)
            correct = (df["pred_label"] == df["label"]).to_numpy()[in_scope]
            before = df["confidence"].to_numpy(dtype=float)[in_scope]
            row = {"ece_before": expected_calibration_error(before, correct, cfg["ece_bins"])}
            if tag in cfg["calibrate"]:
                # Check the saved logits match the saved confidences before trusting them.
                if not np.allclose(calibrated_confidence(logits, 1.0), df["confidence"], atol=1e-4):
                    sys.exit(f"{tag} seed {seed}: saved logits do not reproduce the saved confidences.")
                true_ids = df["label"][in_scope].map(label_id).to_numpy()
                temperature = fit_temperature(logits[in_scope], true_ids)
                after = calibrated_confidence(logits[in_scope], temperature)
                row |= {"temperature": temperature, "ece_after": expected_calibration_error(after, correct, cfg["ece_bins"])}
            out[tag][seed] = row
    return out


# ---------------------------------------------------------------------------
# Routing metrics
# ---------------------------------------------------------------------------

def correct_and_confidence(df: pd.DataFrame, logits, temperature: float | None) -> tuple[np.ndarray, np.ndarray]:
    correct = (df["pred_label"] == df["label"]).to_numpy()
    if temperature is None:
        return correct, df["confidence"].to_numpy(dtype=float)
    return correct, calibrated_confidence(logits, temperature)


def evaluate_router(by_seed: dict, temperatures: dict | None, frozen: dict | None,
                    r_star: float, llm_cost: float, cfg: dict) -> dict:
    """Per-seed routing results, then mean and std over seeds.

    frozen is None on validation (thresholds are chosen here) and holds the
    validation thresholds on test (applied unchanged).
    """
    per_seed, thresholds = [], {}
    for seed, (df, logits) in by_seed.items():
        correct, confidence = correct_and_confidence(df, logits, temperatures[seed] if temperatures else None)
        is_oos = df["is_oos"].to_numpy(dtype=bool)
        threshold = coverage_at_risk(correct, confidence, r_star)["threshold"] if frozen is None else frozen[seed]
        thresholds[seed] = threshold
        handled = confidence >= threshold
        coverage = float(handled.mean())
        query_cost = cost_per_query(coverage, llm_cost, cfg["router_cost_usd"])
        row = {
            "aurc": aurc(risk_coverage_curve(correct, confidence)),
            "coverage": coverage,
            "selective_risk": float((~correct[handled]).mean()) if handled.any() else 0.0,
            **handle_precision_recall(correct, handled),
            "oos_handled": float((handled & is_oos).sum()),
            "cost_per_1000_usd": 1000 * query_cost,
            "saving_vs_llm": 1.0 - query_cost / llm_cost,
        }
        for risk in cfg["fixed_risks"]:
            row[f"coverage_at_{risk:g}"] = coverage_at_risk(correct, confidence, risk)["coverage"]
        per_seed.append(row)
    return {"seeds": list(by_seed), "per_seed": per_seed, "thresholds": thresholds, "summary": mean_std_over_seeds(per_seed)}


def s4_reference(df: pd.DataFrame, info: dict) -> dict:
    """S4's error rate (the matched risk), its accuracy on each part, and its measured cost and latency."""
    wrong = (df["pred_label"] != df["label"]).to_numpy()
    is_oos = df["is_oos"].to_numpy(dtype=bool)
    return {
        "error_rate": float(wrong.mean()),
        "in_scope_accuracy": float(1 - wrong[~is_oos].mean()),
        "oos_recall": float(1 - wrong[is_oos].mean()),
        "invalid": int((df["pred_label"] == "invalid").sum()),
        "cost_per_call_usd": info["mean_cost_per_call_usd"],
        "api_latency_ms": info["api_latency_ms"],
        "model_id": info["model_id"],
    }


def compare(runs: dict, temperatures: dict, s4_wrong: np.ndarray, cfg: dict) -> list[dict]:
    """Paired bootstrap of coverage at matched risk, A - B. r* is recomputed on every resample."""
    boot = cfg["bootstrap"]

    def coverage_metric(tag: str):
        arrays = [correct_and_confidence(df, logits, temperatures.get(tag, {}).get(seed))
                  for seed, (df, logits) in runs[tag].items()]

        def metric(rows: np.ndarray) -> float:
            r_star = s4_wrong[rows].mean()
            return float(np.mean([coverage_at_risk(c[rows], conf[rows], r_star)["coverage"] for c, conf in arrays]))

        return metric

    strata = next(iter(runs[cfg["llm_system"]].values()))[0]["is_oos"].to_numpy(dtype=bool)
    out = []
    for a, b in cfg["comparisons"]:
        result = paired_bootstrap_diff(coverage_metric(a), coverage_metric(b), strata,
                                       n_resamples=boot["resamples"], seed=boot["seed"], ci=boot["ci"])
        out.append({"a": a, "b": b, **result})
    return out


# ---------------------------------------------------------------------------
# Figure and report
# ---------------------------------------------------------------------------

def draw(runs: dict, temperatures: dict, results: dict, r_star: float, cfg: dict, split: str) -> str:
    """Risk-coverage curves (first seed of each system) with S4's error rate as the matched-risk line."""
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=200)
    for s in cfg["figure"]:
        by_seed = runs[s["tag"]]
        seed = min(by_seed)
        df, logits = by_seed[seed]
        correct, confidence = correct_and_confidence(df, logits, temperatures.get(s["tag"], {}).get(seed))
        curve = risk_coverage_curve(correct, confidence)
        ax.plot(curve["coverage"], 100 * curve["risk"], color=s["color"], linewidth=2, linestyle=s["style"],
                label=f"{s['name']} (AURC {100 * aurc(curve):.1f})")
        point = results[s["tag"]]["per_seed"][0]  # this seed's matched-risk operating point
        ax.plot(point["coverage"], 100 * point["selective_risk"], marker="o", markersize=7, color=s["color"],
                markeredgecolor="#ffffff", markeredgewidth=1.5, linestyle="none")
    ax.axhline(100 * r_star, color=INK, linewidth=1, linestyle=":")
    ax.text(0.02, 100 * r_star + 0.6, f"S4 error rate {100 * r_star:.1f}% (matched risk)", color=INK, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 40)
    ax.set_xlabel("Coverage: share of queries the router handles itself", color=INK, fontsize=9)
    ax.set_ylabel("Selective risk: % of handled queries wrong", color=INK, fontsize=9)
    split_name = {"val_fmt": "validation", "test_fmt": "test"}.get(split, split)
    ax.set_title(f"Risk-coverage ({split_name}, format-controlled text, seed 0)", color=INK, fontsize=10, loc="left")
    ax.grid(color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c3c2b7")
    legend = ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color(INK)  # legend text stays in ink; the line beside it carries the colour
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"e6_{split}_risk_coverage.png"
    fig.tight_layout()
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    return str(path.relative_to(data.PROJECT_ROOT))


def pct(summary: dict, key: str) -> str:
    return f"{100 * summary[key]['mean']:.2f} ± {100 * summary[key]['std']:.2f}"


def write_markdown(results, s4, comparisons, calibration, best, latency, figure, split, cfg, meta) -> str:
    risks = [f"coverage_at_{r:g}" for r in cfg["fixed_risks"]]
    main_rows = [
        {"system": tag, "seeds": len(r["seeds"]),
         "coverage at matched risk (%)": pct(r["summary"], "coverage"),
         "selective risk (%)": pct(r["summary"], "selective_risk"),
         "HANDLE precision (%)": pct(r["summary"], "handle_precision"),
         "HANDLE recall (%)": pct(r["summary"], "handle_recall"),
         "OOS handled (of n)": f"{r['summary']['oos_handled']['mean']:.1f}",
         "AURC (%)": pct(r["summary"], "aurc"),
         "cost per 1,000 queries (US$)": f"{r['summary']['cost_per_1000_usd']['mean']:.3f}",
         "saving vs LLM only (%)": pct(r["summary"], "saving_vs_llm")}
        for tag, r in results.items()
    ]
    fixed_rows = [{"system": tag, **{f"coverage at risk {float(k.split('_')[-1]) * 100:g}% (%)": pct(r["summary"], k) for k in risks}}
                  for tag, r in results.items()]
    comparison_rows = [
        {"A": c["a"], "B": c["b"], "coverage A - B (points)": f"{100 * c['diff']:+.2f}",
         f"{round(100 * cfg['bootstrap']['ci'])}% CI": f"[{100 * c['ci_low']:+.2f}, {100 * c['ci_high']:+.2f}]",
         "resamples": c["n_resamples"]}
        for c in comparisons
    ]
    n_oos = int(next(iter(results.values()))["n_oos"])
    parts = [
        f"# E6 results on {split}",
        "Written by `scripts/evaluate_router.py`. Do not edit by hand.",
        "## Provenance",
        metadata_markdown(meta),
        "## S4, the LLM being replaced",
        f"`{s4['model_id']}` on Amazon Bedrock, zero-shot, temperature 0, format-controlled text. "
        f"Error rate on all queries (the matched risk): **{100 * s4['error_rate']:.2f}%**. "
        f"In-scope accuracy {100 * s4['in_scope_accuracy']:.2f}%, OOS recall {100 * s4['oos_recall']:.2f}%, invalid replies {s4['invalid']}. "
        f"Measured cost US${s4['cost_per_call_usd']:.5f} per call, so US${1000 * s4['cost_per_call_usd']:.3f} per 1,000 queries if every query goes to the LLM. "
        f"API latency p50 {s4['api_latency_ms']['p50']:.0f} ms, p95 {s4['api_latency_ms']['p95']:.0f} ms (from Sydney).",
        f"## Policy v1 at matched risk, mean ± std over seeds",
        f"Each router handles the queries at or above its threshold and escalates the rest to S4. The threshold gives the "
        f"highest coverage whose selective risk is at or below S4's error rate. {n_oos} of the queries are out of scope; "
        f"\"OOS handled\" counts those the router answered itself, the dangerous error. Router cost is taken as US$0 (local model); "
        f"the saving is against sending every query to S4.",
        md_table(main_rows),
        f"**Best router** (highest mean coverage at matched risk, the rule fixed in README 7.2): `{best}`.",
        "## Coverage at fixed risk levels",
        md_table(fixed_rows),
    ]
    if comparison_rows:
        parts += ["## Comparisons (paired, stratified bootstrap of coverage at matched risk)",
                  "The matched risk is recomputed on every resample. An interval that spans 0 means no clear difference.",
                  md_table(comparison_rows)]
    if calibration:
        cal_rows = []
        for tag, by_seed in calibration.items():
            rows = list(by_seed.values())
            cal_rows.append({
                "system": tag,
                "temperature (mean)": f"{np.mean([r['temperature'] for r in rows]):.3f}" if "temperature" in rows[0] else "not scaled",
                "ECE before (%)": f"{100 * np.mean([r['ece_before'] for r in rows]):.2f}",
                "ECE after (%)": f"{100 * np.mean([r['ece_after'] for r in rows]):.2f}" if "ece_after" in rows[0] else "-",
            })
        parts += ["## Calibration (in-scope rows, fitted and measured on the same validation set)",
                  "Temperature scaling for the softmax systems; ECE with 15 bins. In-sample, so the after-scaling ECE is optimistic; the honest number comes from test at M5.",
                  md_table(cal_rows)]
    if latency:
        lat_rows = [{"system": tag, "p50 (ms)": f"{v['p50_ms']:.1f}", "p95 (ms)": f"{v['p95_ms']:.1f}"} for tag, v in latency["systems"].items()]
        parts += [f"## Latency (batch size 1, warm model, {latency['machine']}, {latency['n_queries']} queries)", md_table(lat_rows)]
    parts += ["## Figure", f"Risk-coverage curves, first seed of each system, with the matched-risk operating points: `{figure}`"]
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(split: str) -> None:
    is_test = split.startswith("test")
    if is_test and not (data.RESULTS_DIR / "preregistration.md").exists():
        sys.exit("Refusing to score a test split: results/preregistration.md does not exist yet (rigour rule 1).")
    cfg = data.load_config("e6_router")
    label_names = label_names_in_id_order()
    runs = load_runs(split, [cfg["llm_system"]] + cfg["routers"])

    with open(PREDICTIONS_DIR / cfg["llm_system"] / f"seed0_{split}_info.json") as f:
        s4 = s4_reference(runs[cfg["llm_system"]][0][0], json.load(f))
    s4_wrong = (runs[cfg["llm_system"]][0][0]["pred_label"] != runs[cfg["llm_system"]][0][0]["label"]).to_numpy()

    calibration = {}
    if is_test:  # apply the choices frozen on validation
        with open(POLICY_FILE) as f:
            policy = json.load(f)
        temperatures = {tag: {int(s): t for s, t in v["temperatures"].items()} for tag, v in policy["systems"].items() if v["temperatures"]}
        frozen = {tag: {int(s): t for s, t in v["thresholds"].items()} for tag, v in policy["systems"].items()}
    else:
        calibration = fit_calibration(runs, cfg, label_names)
        temperatures = {tag: {seed: row["temperature"] for seed, row in by_seed.items()}
                        for tag, by_seed in calibration.items() if tag in cfg["calibrate"]}
        frozen = {}

    results = {}
    for tag in cfg["routers"]:
        results[tag] = evaluate_router(runs[tag], temperatures.get(tag), frozen.get(tag) if is_test else None,
                                       s4["error_rate"], s4["cost_per_call_usd"], cfg)
        results[tag]["n_oos"] = int(next(iter(runs[tag].values()))[0]["is_oos"].sum())
    best = max(cfg["routers"], key=lambda tag: results[tag]["summary"]["coverage"]["mean"])
    comparisons = compare(runs, temperatures, s4_wrong, cfg)
    latency_file = data.RESULTS_DIR / "latency.json"
    latency = json.loads(latency_file.read_text()) if latency_file.exists() else None
    figure = draw(runs, temperatures, results, s4["error_rate"], cfg, split)

    meta = run_metadata({**data.load_config("data"), "e6_router": cfg}, seed=cfg["bootstrap"]["seed"])
    payload = {"metadata": meta, "split": split, "s4": s4, "systems": results, "best_router": best,
               "comparisons": comparisons, "calibration": calibration, "latency": latency, "figure": figure}
    with open(data.RESULTS_DIR / f"e6_{split}.json", "w") as f:
        json.dump(payload, f, indent=2, default=str)
    (data.RESULTS_DIR / f"e6_{split}.md").write_text(
        write_markdown(results, s4, comparisons, calibration, best, latency, figure, split, cfg, meta)
    )
    if not is_test:
        with open(data.RESULTS_DIR / f"calibration_{split}.json", "w") as f:
            json.dump({"metadata": meta, "split": split, "systems": calibration}, f, indent=2, default=str)
        policy = {
            "metadata": meta,
            "fitted_on": split,
            "rule": "per seed: threshold with the highest coverage whose selective risk <= S4's error rate; best router = highest mean coverage",
            "matched_risk": s4["error_rate"],
            "llm_cost_per_call_usd": s4["cost_per_call_usd"],
            "best_router": best,
            "systems": {tag: {"temperatures": temperatures.get(tag, {}), "thresholds": r["thresholds"]} for tag, r in results.items()},
        }
        with open(POLICY_FILE, "w") as f:
            json.dump(policy, f, indent=2, default=str)

    print(f"S4 error rate (matched risk): {100 * s4['error_rate']:.2f}%  | cost per call ${s4['cost_per_call_usd']:.5f}")
    print(pd.DataFrame({tag: {k: pct(r["summary"], k) for k in ["coverage", "selective_risk", "aurc", "saving_vs_llm"]}
                        | {"oos_handled": f"{r['summary']['oos_handled']['mean']:.1f}"}
                        for tag, r in results.items()}).T.to_string())
    for c in comparisons:
        print(f"{c['a']} - {c['b']}: {100 * c['diff']:+.2f} points, CI [{100 * c['ci_low']:+.2f}, {100 * c['ci_high']:+.2f}]")
    print(f"best router: {best}\nWrote results/e6_{split}.json, .md and {figure}" + ("" if is_test else ", results/policy_v1.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val_fmt", "test_fmt"], default="val_fmt")
    main(parser.parse_args().split)
