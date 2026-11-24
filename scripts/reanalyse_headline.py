"""Post-hoc re-analysis of the headline result, after an independent critical review (24 Nov 2026).

Run from the project root:
    uv run python scripts/reanalyse_headline.py

Reads saved predictions only. It makes no new prediction, tunes nothing and changes no
pre-registered verdict. It asks five questions the pre-registered tables do not answer:

  A. What does "coverage at matched risk" consist of? Banking77 test_fmt is 75.5% in-scope
     queries and 24.5% CLINC150 out-of-scope queries. Each router's coverage is split into
     in-scope queries handled and out-of-scope queries handled, at fixed risk bars, at the
     matched risk, and at the frozen policy v1 thresholds.
  B. Is the router plus the LLM worse than the LLM alone? End-to-end wrong answers when the
     router handles what it is confident about and S4 answers the rest (policy v1 as frozen),
     with a paired bootstrap interval on the difference.
  C. Why does S2 fail on Jeff? S2's in-scope accuracy by the cosine to its nearest labelled
     example, on Banking77 test, the Jeff test split and Jeff test v2 (S2 unchanged).
  D. RQ6 at one common bar: S2's coverage on Banking77 and on Jeff at the same fixed risk.
  E. A pipeline bug: out-of-scope queries that become word for word a Banking77 query once
     format control is applied (make_splits.py compared raw text). A sensitivity check
     without them, labelled as such.
  F. Why S2 covers so little at strict risk on Banking77: its most confident errors.

Checks: section A must reproduce the saved E6 coverage numbers exactly, or the script stops.
Writes results/headline_reanalysis.json and results/headline_reanalysis.md.
"""

import json
import sys

import numpy as np
import pandas as pd

from router import data
from router.calibration import calibrated_confidence
from router.label_noise import wilson_interval
from router.metrics import coverage_at_risk, paired_bootstrap_diff
from router.preprocess import format_control
from router.provenance import metadata_markdown, run_metadata
from router.systems.base import load_predictions
from router.tables import md_table

SPLIT = "test_fmt"
S2 = "s2_all-MiniLM-L6-v2"
BARS = [0.01, 0.02, 0.05, 0.10]
CONTROL_V2 = data.RESULTS_DIR / "e8_predictions" / "frozen_examples_nearest" / "seed0_jeff_test_v2_fmt.csv"
COSINE_BANDS = [(-1.0, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def router_runs(policy: dict) -> dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """{router: [(correct, confidence, is_oos) per seed]} on test_fmt, with E6's confidence
    (temperature-scaled for S0 and S1, as frozen in policy v1)."""
    runs = {}
    for tag, frozen in policy["systems"].items():
        runs[tag] = []
        for seed in sorted(frozen["thresholds"], key=int):
            df, logits = load_predictions(tag, int(seed), SPLIT)
            temperature = frozen["temperatures"].get(seed)
            confidence = calibrated_confidence(logits, temperature) if temperature else df["confidence"].to_numpy(dtype=float)
            correct = (df["pred_label"] == df["label"]).to_numpy()
            runs[tag].append((correct, confidence, df["is_oos"].to_numpy(dtype=bool)))
    return runs


def split_coverage(correct, confidence, is_oos, threshold) -> dict[str, float]:
    """Coverage at a threshold, split into in-scope queries handled and out-of-scope queries handled."""
    handled = confidence >= threshold
    return {"coverage": float(handled.mean()), "in_scope_handled": float((handled & ~is_oos).sum() / (~is_oos).sum()),
            "oos_handled": float((handled & is_oos).sum())}


def section_a(runs: dict, policy: dict, r_star: float) -> list[dict]:
    rows = []
    for tag, seeds in runs.items():
        for bar_name, bar in [*[(f"{100 * b:g}% risk", b) for b in BARS], (f"matched risk ({100 * r_star:.2f}%)", r_star), ("frozen policy v1", None)]:
            per_seed = []
            for (correct, confidence, is_oos), seed in zip(seeds, sorted(policy["systems"][tag]["thresholds"], key=int)):
                threshold = policy["systems"][tag]["thresholds"][seed] if bar is None else coverage_at_risk(correct & ~is_oos, confidence, bar)["threshold"]
                per_seed.append(split_coverage(correct & ~is_oos, confidence, is_oos, threshold))
            rows.append({"system": tag, "bar": bar_name, **{k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0]}})
    return rows


def check_against_e6(rows: list[dict], e6: dict, r_star: float) -> None:
    """Section A must reproduce E6's saved coverage numbers, or this analysis is not of the same runs."""
    for tag, result in e6["systems"].items():
        saved = {f"{100 * b:g}% risk": result["summary"][f"coverage_at_{b:g}"]["mean"] for b in [0.01, 0.02, 0.05]}
        saved[f"matched risk ({100 * r_star:.2f}%)"] = result["summary"]["coverage_at_matched_risk"]["mean"]
        saved["frozen policy v1"] = result["summary"]["coverage"]["mean"]
        for bar, value in saved.items():
            ours = next(r["coverage"] for r in rows if r["system"] == tag and r["bar"] == bar)
            if abs(ours - value) > 1e-9:
                sys.exit(f"{tag} at {bar}: {ours} here, {value} in e6_test_fmt.json. Not the same runs.")
    print("Check passed: section A reproduces the saved E6 coverage numbers.")


def section_b(runs: dict, policy: dict, s4_wrong: np.ndarray, boot: dict) -> list[dict]:
    """End to end at the frozen thresholds: the router answers what it handles, S4 the rest."""
    is_oos = runs[S2][0][2]
    rows = [{"system": "S4 alone", "llm_calls": len(s4_wrong), "wrong": int(s4_wrong.sum()), "error_rate": float(s4_wrong.mean()),
             "in_scope_wrong": int(s4_wrong[~is_oos].sum()), "oos_answered": int(s4_wrong[is_oos].sum()), "diff_vs_s4": None}]
    for tag, seeds in runs.items():
        thresholds = [policy["systems"][tag]["thresholds"][s] for s in sorted(policy["systems"][tag]["thresholds"], key=int)]
        wrong_runs = [np.where(conf >= t, ~(c & ~o), s4_wrong) for (c, conf, o), t in zip(seeds, thresholds)]
        diff = paired_bootstrap_diff(lambda r, w=wrong_runs: float(np.mean([x[r].mean() for x in w])),
                                     lambda r: float(s4_wrong[r].mean()), is_oos, boot["resamples"], boot["seed"], boot["ci"])
        calls = np.mean([(conf < t).sum() for (_, conf, _), t in zip(seeds, thresholds)])
        rows.append({"system": f"{tag} + S4", "llm_calls": float(calls), "wrong": float(np.mean([w.sum() for w in wrong_runs])),
                     "error_rate": float(np.mean([w.mean() for w in wrong_runs])),
                     "in_scope_wrong": float(np.mean([w[~is_oos].sum() for w in wrong_runs])),
                     "oos_answered": float(np.mean([w[is_oos].sum() for w in wrong_runs])), "diff_vs_s4": diff})
    return rows


def section_c() -> list[dict]:
    """S2's in-scope accuracy by the cosine to its nearest labelled example (S2's confidence)."""
    sets = {"Banking77 test": load_predictions(S2, 0, SPLIT)[0], "Jeff test (E7)": load_predictions(S2, 0, "jeff_test_fmt")[0],
            "Jeff test v2 (E8 control)": pd.read_csv(CONTROL_V2, keep_default_na=False)}
    rows = []
    for name, df in sets.items():
        df = df[~df["is_oos"].to_numpy(dtype=bool)]  # in-scope queries only
        correct = (df["pred_label"] == df["label"]).to_numpy()
        cosine = df["confidence"].to_numpy(dtype=float)
        for low, high in COSINE_BANDS:
            band = (cosine >= low) & (cosine < high)
            k, n = int(correct[band].sum()), int(band.sum())
            ci = wilson_interval(k, n) if n else (float("nan"), float("nan"))
            rows.append({"data": name, "band": f"{max(low, 0):.1f} to {min(high, 1.0):.1f}", "n": n,
                         "accuracy": k / n if n else float("nan"), "ci_low": ci[0], "ci_high": ci[1]})
    return rows


def section_d(runs: dict) -> list[dict]:
    """S2's coverage on Banking77 test and on the Jeff test split at the same fixed risk bars."""
    jeff, _ = load_predictions(S2, 0, "jeff_test_fmt")
    j_correct = ((jeff["pred_label"] == jeff["label"]) & ~jeff["is_oos"]).to_numpy()
    j_conf = jeff["confidence"].to_numpy(dtype=float)
    b_correct, b_conf, b_oos = runs[S2][0]
    rows = []
    for bar in [*BARS, 0.15]:
        b = coverage_at_risk(b_correct & ~b_oos, b_conf, bar)["coverage"]
        j = coverage_at_risk(j_correct, j_conf, bar)["coverage"]
        rows.append({"bar": bar, "banking77": b, "jeff": j, "drop": b - j})
    return rows


def section_e(runs: dict) -> dict:
    """Out-of-scope queries identical to a Banking77 query once format control is applied."""
    banking = set(pd.concat([data.load_split(s) for s in ["train", "val", "test"]])["text"].map(format_control))
    found = {}
    for pool in ["oos_val", "oos_test"]:
        texts = data.load_split(pool)["text"]
        found[pool] = sorted(t for t in texts if format_control(t) in banking)
    df, _ = load_predictions(S2, 0, SPLIT)
    duplicate = df["is_oos"].to_numpy(dtype=bool) & df["text"].isin({format_control(t) for t in found["oos_test"]}).to_numpy()
    order = np.argsort(-df["confidence"].to_numpy(dtype=float), kind="stable")
    ranks = [int(np.flatnonzero(order == i)[0]) + 1 for i in np.flatnonzero(duplicate)]
    sensitivity = []
    for tag, seeds in runs.items():
        for bar in [0.01, 0.02, 0.05]:
            with_rows = np.mean([coverage_at_risk(c & ~o, k, bar)["coverage"] for c, k, o in seeds])
            without = np.mean([coverage_at_risk((c & ~o)[~duplicate], k[~duplicate], bar)["coverage"] for c, k, o in seeds])
            sensitivity.append({"system": tag, "bar": bar, "with": float(with_rows), "without": float(without)})
    return {"found": found, "s2_confidence_rank_of_test_duplicates": ranks, "sensitivity": sensitivity}


def section_f(n_top: int = 300, n_show: int = 10) -> dict:
    """S2's errors among its n_top most confident Banking77 test queries, and the first n_show of them."""
    df, _ = load_predictions(S2, 0, SPLIT)
    df = df.sort_values("confidence", ascending=False, kind="stable").reset_index(drop=True)
    top = df.head(n_top)
    wrong = top[top["pred_label"] != top["label"]]
    shown = [{"rank": int(i) + 1, "cosine": round(float(r["confidence"]), 3), "query": r["text"], "true": r["label"],
              "predicted": r["pred_label"], "nearest train example": r["matched_example"]} for i, r in wrong.head(n_show).iterrows()]
    return {"n_top": n_top, "errors": int(len(wrong)), "oos_errors": int(wrong["is_oos"].sum()), "shown": shown}


def pct(x) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.2f}"


def report(out: dict) -> str:
    a_rows = [{"system": r["system"], "bar": r["bar"], "coverage (%)": pct(r["coverage"]),
               "in-scope queries handled (% of in-scope)": pct(r["in_scope_handled"]),
               "out-of-scope handled (of 1,000)": f"{r['oos_handled']:.1f}"} for r in out["a"]]
    b_rows = [{"system": r["system"], "LLM calls": f"{r['llm_calls']:.0f}", "wrong answers": f"{r['wrong']:.1f}",
               "error rate (%)": pct(r["error_rate"]), "in-scope wrong": f"{r['in_scope_wrong']:.1f}",
               "out-of-scope answered": f"{r['oos_answered']:.1f}",
               "error minus S4 alone (points, 95% CI)": "" if r["diff_vs_s4"] is None else
               f"{pct(r['diff_vs_s4']['diff'])} [{pct(r['diff_vs_s4']['ci_low'])}, {pct(r['diff_vs_s4']['ci_high'])}]"} for r in out["b"]]
    c_rows = [{"data": r["data"], "cosine to nearest example": r["band"], "queries": r["n"], "accuracy (%)": pct(r["accuracy"]),
               "95% Wilson CI": f"[{pct(r['ci_low'])}, {pct(r['ci_high'])}]" if r["n"] else ""} for r in out["c"]]
    d_rows = [{"risk bar (%)": f"{100 * r['bar']:g}", "S2 Banking77 test (%)": pct(r["banking77"]), "S2 Jeff test (%)": pct(r["jeff"]),
               "drop (points)": pct(r["drop"])} for r in out["d"]]
    e = out["e"]
    e_rows = [{"system": r["system"], "risk bar (%)": f"{100 * r['bar']:g}", "coverage (%)": pct(r["with"]),
               "without the duplicated rows (%)": pct(r["without"])} for r in e["sensitivity"]]
    s4 = out["s4"]
    return "\n\n".join([
        "# Headline re-analysis (post-hoc)",
        "Written by `scripts/reanalyse_headline.py`. Do not edit by hand. Reads saved predictions only: no new prediction, "
        "no tuning, and no pre-registered verdict changes. Prompted by an independent critical review on 24 Nov 2026.",
        "## Provenance", metadata_markdown(out["metadata"]),
        "## A. What coverage at matched risk is made of (Banking77 test_fmt)",
        f"{out['n_in_scope']} in-scope and {out['n_oos']} out-of-scope queries: in-scope share {pct(out['in_scope_share'])}%. "
        f"S4's in-scope accuracy is {pct(s4['in_scope_accuracy'])}%, below every router's (`results/e1_test.md`), so at S4's overall "
        "error rate a router can handle almost every in-scope query. Coverage above the in-scope share can only come from handling "
        "out-of-scope queries, which are always errors under policy v1 (it has no REJECT). Means over seeds.",
        md_table(a_rows),
        "Check: the coverage column reproduces `results/e6_test_fmt.json` exactly.",
        "## B. Router plus LLM end to end, at the frozen policy v1 thresholds",
        "The router answers what it handles and S4 answers the rest. Paired bootstrap of the error difference, 1,000 resamples "
        "stratified by in-scope and out-of-scope. Negative = fewer errors than S4 alone. Means over seeds.",
        md_table(b_rows),
        "## C. S2's in-scope accuracy by distance to its nearest labelled example",
        "S2's confidence is the cosine to the nearest support example. Banking77's support is its train split; the Jeff sets' "
        "support is the 140 Jeff support examples.",
        md_table(c_rows),
        "## D. RQ6 at one common risk bar",
        "The pre-registered RQ6 drop (75.80 points) compares coverage at two different bars (S4's error: 10.27% on Banking77, "
        "0.56% on Jeff). Here both sides use the same bar. The RQ6 verdict (not met) is unchanged.",
        md_table(d_rows),
        "## E. Pipeline bug: out-of-scope queries that duplicate a Banking77 query after format control",
        "`scripts/make_splits.py` removed out-of-scope queries that also appear in Banking77 by comparing normalised raw text. "
        "CLINC150 is lowercase and unpunctuated, so a case or punctuation difference hid the match until format control. "
        f"Validation pool: {len(e['found']['oos_val'])} {e['found']['oos_val']}. Test pool: {len(e['found']['oos_test'])} "
        f"{e['found']['oos_test']}. S2's confidence rank of the test duplicates (1 = most confident of 4,079): "
        f"{e['s2_confidence_rank_of_test_duplicates']}. The splits stay as they are (test sets are not edited after the run); "
        "this sensitivity check drops the test duplicates for information only.",
        md_table(e_rows),
        "## F. S2's most confident errors on Banking77 test",
        f"Errors among S2's {out['f']['n_top']} most confident test queries: {out['f']['errors']} "
        f"({out['f']['oos_errors']} out of scope). The first {len(out['f']['shown'])}, by confidence rank. The predicted "
        "intent is always the label of the nearest train example, so a near-copy with a different label is a confident error.",
        md_table(out["f"]["shown"]),
    ]) + "\n"


def main() -> None:
    policy = json.loads((data.RESULTS_DIR / "policy_v1.json").read_text())
    e6 = json.loads((data.RESULTS_DIR / "e6_test_fmt.json").read_text())
    boot = data.load_config("e6_router")["bootstrap"]
    s4_df, _ = load_predictions(policy["llm_system"], 0, SPLIT)
    s4_wrong = (s4_df["pred_label"] != s4_df["label"]).to_numpy()
    r_star = float(s4_wrong.mean())
    runs = router_runs(policy)
    a = section_a(runs, policy, r_star)
    check_against_e6(a, e6, r_star)
    is_oos = runs[S2][0][2]
    out = {"metadata": run_metadata({"e6_router": data.load_config("e6_router")}, seed=boot["seed"]),
           "n_in_scope": int((~is_oos).sum()), "n_oos": int(is_oos.sum()), "in_scope_share": float((~is_oos).mean()),
           "s4": {"error_rate": r_star, "in_scope_accuracy": float(1 - s4_wrong[~is_oos].mean())},
           "a": a, "b": section_b(runs, policy, s4_wrong, boot), "c": section_c(), "d": section_d(runs), "e": section_e(runs), "f": section_f()}
    (data.RESULTS_DIR / "headline_reanalysis.json").write_text(json.dumps(out, indent=2, default=float))
    (data.RESULTS_DIR / "headline_reanalysis.md").write_text(report(out))
    print("Wrote results/headline_reanalysis.json and results/headline_reanalysis.md")


if __name__ == "__main__":
    main()
