"""E8 final test: the chosen fixes and the unfixed S2, run once on the fresh Jeff test v2.

Run from the project root, in this order:
    uv run python scripts/develop_few_shot.py                     # chooses the fixes (development data)
    # results/preregistration_e8.md written and committed
    uv run python scripts/make_jeff_set.py --v2                   # the fresh test split
    uv run --group llm python scripts/run_s4.py --split test_v2 --config s4_jeff_fewshot
    uv run python scripts/evaluate_few_shot.py

Post-hoc experiment, not in the main pre-registration. Its own plan, fixed before this
script first ran: results/preregistration_e8.md.

Systems:
  - control: S2 unchanged (frozen/examples/nearest). Shows whether v2 is as hard as the old test.
  - the variant chosen for each tier in results/e8_dev.json, with the same fine-tuned
    encoders (checkpoints/e8/) and the same scoring.
Queries: the 180 Jeff test v2 queries, format-controlled as in E7.
Matched risk r*: few-shot S4's error on v2 (the S4 variant frozen in policy v1).
Per run: the development metrics (router.few_shot.run_metrics) and the development
threshold applied unchanged (coverage, selective risk, out-of-scope queries handled).
Intervals (1,000 stratified resamples, as E7):
  - coverage at matched risk, seed mean, r* recomputed on every resample;
  - AURC difference from the control, paired (the primary E8 number).

Writes results/e8_test_v2.json and .md, and the predictions to results/e8_predictions/.
Refused until results/preregistration_e8.md exists, and refused once results/e8_test_v2.json
exists: the fresh test is used once.
"""

import json
import sys

import numpy as np
import pandas as pd

from router import data, few_shot, jeff
from router.metrics import aurc, bootstrap_ci, coverage_at_risk, paired_bootstrap_diff, risk_coverage_curve
from router.preprocess import format_control
from router.provenance import metadata_markdown, run_metadata
from router.seed import pick_device
from router.systems.base import PREDICTIONS_DIR
from router.systems.s2_embedding import load_encoder
from router.tables import md_table
from sentence_transformers import SentenceTransformer

CHECKPOINTS = data.PROJECT_ROOT / "checkpoints" / "e8" / "jeff"
SPLIT = "jeff_test_v2_fmt"
CONTROL = "frozen/examples/nearest"


def s4_wrong(expected: pd.DataFrame, llm_tag: str) -> np.ndarray:
    """Few-shot S4's wrong answers on v2, checked to be the same queries in the same order."""
    df = pd.read_csv(PREDICTIONS_DIR / llm_tag / f"seed0_{SPLIT}.csv", keep_default_na=False)
    if df["text"].tolist() != expected["text"].tolist() or df["label"].tolist() != expected["label"].tolist():
        sys.exit(f"{llm_tag} predictions on {SPLIT} are not the v2 queries in order.")
    return (df["pred_label"] != df["label"]).to_numpy()


def main() -> None:
    # Paths are built here, not at import, so a test can point RESULTS_DIR at an empty folder.
    plan, out_file, pred_dir = (data.RESULTS_DIR / "preregistration_e8.md", data.RESULTS_DIR / "e8_test_v2.json",
                                data.RESULTS_DIR / "e8_predictions")
    if not plan.exists():
        sys.exit("Refusing: results/preregistration_e8.md does not exist. Fix the E8 test plan first.")
    if out_file.exists():
        sys.exit("Refusing: results/e8_test_v2.json exists. The fresh test is used once.")
    cfg = data.load_config("e8_few_shot")
    dev = json.loads((data.RESULTS_DIR / "e8_dev.json").read_text())
    policy = json.loads((data.RESULTS_DIR / "policy_v1.json").read_text())
    boot = data.load_config("e6_router")["bootstrap"]

    # The fresh test queries and the support set, exactly as in development.
    jeff_cfg = data.load_config("jeff")
    support, names = jeff.jeff_support()
    definitions = {i["name"]: i["description"] for i in jeff_cfg["intents"]}
    supports = {"examples": support[["text", "label_id"]], "descriptions": few_shot.add_descriptions(support, definitions, names)}
    v2 = jeff.load_jeff(jeff_cfg["v2"]["split"])
    v2 = v2.assign(text=v2["text"].map(format_control))
    is_oos = v2["is_oos"].to_numpy(dtype=bool)
    true_ids = np.array([names.index(l) if not o else -1 for l, o in zip(v2["label"], is_oos)])
    wrong_llm = s4_wrong(v2, policy["llm_system"])
    r_star = float(wrong_llm.mean())

    systems = [CONTROL] + [v for v in dev["chosen"].values() if v != CONTROL]
    dev_thresholds = {(r["variant"], r["seed"]): r["dev_threshold"] for r in dev["jeff"]["per_run"]}
    device = pick_device()
    frozen = load_encoder(cfg["encoder_model"], device)
    results, arrays = {}, {}
    for name in systems:
        variant = few_shot.Variant(*name.split("/"))
        seeds = cfg["seeds"] if variant.encoder == "tuned" else [0]
        per_seed, arrays[name] = [], []
        for seed in seeds:
            encoder = frozen if variant.encoder == "frozen" else \
                SentenceTransformer(str(CHECKPOINTS / variant.support / f"seed{seed}"), device=str(device))
            pred, conf = few_shot.predict_variant(variant.scoring, encoder, supports[variant.support], v2["text"].tolist(),
                                                  len(names), cfg, seed)
            correct = (pred == true_ids) & ~is_oos
            arrays[name].append((correct, conf))
            threshold = dev_thresholds[(name, seed)]
            handled = conf >= threshold
            per_seed.append({
                "seed": seed, **few_shot.run_metrics(pred, conf, true_ids, is_oos, names, r_star, cfg["fixed_risks"]),
                "dev_threshold": threshold, "coverage_dev_threshold": float(handled.mean()),
                "selective_risk_dev_threshold": float((~correct[handled]).mean()) if handled.any() else 0.0,
                "oos_handled_dev_threshold": int((handled & is_oos).sum()),
            })
            folder = pred_dir / name.replace("/", "_")
            folder.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"text": v2["text"], "label": v2["label"], "is_oos": is_oos,
                          "pred_label": [names[i] for i in pred], "confidence": conf}).to_csv(folder / f"seed{seed}_{SPLIT}.csv", index=False)

        def coverage_metric(rows: np.ndarray, runs=arrays[name]) -> float:
            risk = wrong_llm[rows].mean()  # the matched risk is recomputed on every resample
            return float(np.mean([coverage_at_risk(c[rows], k[rows], risk)["coverage"] for c, k in runs]))

        frame = pd.DataFrame(per_seed).drop(columns="seed")
        results[name] = {"tier": variant.tier, "seeds": seeds, "per_seed": per_seed,
                         "mean": frame.mean().to_dict(), "std": frame.std(ddof=0).to_dict(),
                         "coverage_at_matched_risk_ci": bootstrap_ci(coverage_metric, is_oos, boot["resamples"], boot["seed"], boot["ci"])}

    def mean_aurc(name: str):
        return lambda rows: float(np.mean([aurc(risk_coverage_curve(c[rows], k[rows])) for c, k in arrays[name]]))

    for name in systems[1:]:  # control minus fix: positive = the fix lowers AURC (better)
        results[name]["aurc_gain_vs_control"] = paired_bootstrap_diff(mean_aurc(CONTROL), mean_aurc(name), is_oos,
                                                                      boot["resamples"], boot["seed"], boot["ci"])
    e6 = json.loads((data.RESULTS_DIR / "e6_test_fmt.json").read_text())
    rq1 = e6["systems"][e6["best_router"]]["coverage_at_matched_risk_ci"]  # the RQ1 number, Banking77 test
    for name in systems:
        results[name]["drop_from_rq1"] = rq1["value"] - results[name]["mean"]["coverage_at_matched_risk"]
    out = {"metadata": run_metadata({"e8_few_shot": cfg, "jeff": jeff_cfg}, seed=boot["seed"]), "split": SPLIT,
           "n_queries": len(v2), "n_oos": int(is_oos.sum()), "llm_system": policy["llm_system"], "matched_risk": r_star,
           "s4_in_scope_accuracy": float(1 - wrong_llm[~is_oos].mean()), "control": CONTROL, "chosen": dev["chosen"],
           "systems": results, "rq1_coverage": rq1, "d_threshold": data.load_config("preregistration")["D"]}
    out_file.write_text(json.dumps(out, indent=2, default=float))
    (data.RESULTS_DIR / "e8_test_v2.md").write_text(report(out))
    print(f"matched risk on v2: {100 * r_star:.2f}%. Wrote results/e8_test_v2.json and results/e8_test_v2.md")


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def report(out: dict) -> str:
    rows, gains = [], []
    for name, r in out["systems"].items():
        m, s, ci = r["mean"], r["std"], r["coverage_at_matched_risk_ci"]
        rows.append({"system": name + (" (control: S2 unchanged)" if name == out["control"] else ""), "tier": r["tier"],
                     "seeds": len(r["seeds"]),
                     "in-scope accuracy (%)": f"{pct(m['in_scope_accuracy'])} ± {pct(s['in_scope_accuracy'])}",
                     "OOS AUROC (%)": pct(m["oos_auroc"]), "AURC (%)": f"{pct(m['aurc'])} ± {pct(s['aurc'])}",
                     "coverage at matched risk (%)": pct(m["coverage_at_matched_risk"]),
                     "95% CI": f"[{pct(ci['ci_low'])}, {pct(ci['ci_high'])}]",
                     "coverage at 1% (%)": pct(m["coverage_at_0.01"]), "coverage at 5% (%)": pct(m["coverage_at_0.05"])})
        if "aurc_gain_vs_control" in r:
            g = r["aurc_gain_vs_control"]
            gains.append({"system": name, "AURC gain vs control (points)": pct(g["diff"]),
                          "95% CI": f"[{pct(g['ci_low'])}, {pct(g['ci_high'])}]"})
    drops = [{"system": name, "coverage at matched risk (%)": pct(r["mean"]["coverage_at_matched_risk"]),
              "drop from Banking77 test (points)": pct(r["drop_from_rq1"]),
              f"within D = {pct(out['d_threshold'])}?": "yes" if r["drop_from_rq1"] <= out["d_threshold"] else "no"}
             for name, r in out["systems"].items()]
    deploy = [{"system": name, "coverage (%)": pct(r["mean"]["coverage_dev_threshold"]),
               "selective risk (%)": pct(r["mean"]["selective_risk_dev_threshold"]),
               "OOS handled (of 40)": f"{r['mean']['oos_handled_dev_threshold']:.1f}"} for name, r in out["systems"].items()]
    return "\n\n".join([
        "# E8 final test: the fixes on the fresh Jeff test v2",
        "Written by `scripts/evaluate_few_shot.py`. Do not edit by hand. **Post-hoc experiment, not in the main "
        "pre-registration**; its plan is `results/preregistration_e8.md`. The pre-registered RQ6 verdict (not met) is unchanged.",
        "## Provenance", metadata_markdown(out["metadata"]),
        "## Setup",
        f"{out['n_queries']} fresh Jeff queries ({out['n_oos']} out of scope), format-controlled, run once. "
        f"Matched risk: few-shot S4's error on them, {pct(out['matched_risk'])}% (in-scope accuracy {pct(out['s4_in_scope_accuracy'])}%).",
        "## Results (mean ± std over seeds; interval: bootstrap of the seed mean, matched risk recomputed per resample)",
        md_table(rows),
        "## Primary number: AURC gain over the control (paired bootstrap; positive = the fix is better)",
        md_table(gains),
        "## The RQ6 yardstick, post-hoc",
        f"Banking77 test coverage at matched risk (the RQ1 number): {pct(out['rq1_coverage']['value'])}% "
        f"[{pct(out['rq1_coverage']['ci_low'])}, {pct(out['rq1_coverage']['ci_high'])}]. RQ6's rule, applied to each system on v2. "
        "This does not change the pre-registered RQ6 verdict.",
        md_table(drops),
        "## The development thresholds applied unchanged (what a deployed router would do)",
        md_table(deploy),
    ]) + "\n"


if __name__ == "__main__":
    main()
