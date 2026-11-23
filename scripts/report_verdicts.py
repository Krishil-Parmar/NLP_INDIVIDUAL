"""The pre-registered verdicts, read from the test result files (README section 4).

Run from the project root, after every M5 evaluation script:
    uv run python scripts/report_verdicts.py

Each hypothesis is judged by the rule fixed in results/preregistration.md, with
the thresholds in configs/preregistration.yaml: "met" when the point estimate
reaches the threshold, otherwise "not met". The 95% bootstrap interval is shown
next to it, and when the interval crosses the threshold the verdict is labelled
"not clear-cut". This script computes nothing new: every number comes from a
result file that another script wrote.

Writes results/verdicts.md and results/verdicts.json.
"""

import json
import sys

from router import data
from router.provenance import metadata_markdown, run_metadata
from router.tables import md_table


def load(name: str) -> dict | None:
    path = data.RESULTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def judge(value: float, low: float, high: float, threshold: float, higher_is_better: bool = True) -> str:
    """The verdict for one point estimate and its interval against one threshold."""
    met = value >= threshold if higher_is_better else value <= threshold
    crosses = low < threshold < high
    return ("met" if met else "not met") + (" (not clear-cut: the interval crosses the threshold)" if crosses else "")


def main() -> None:
    if not (data.RESULTS_DIR / "preregistration.md").exists():
        sys.exit("No results/preregistration.md: there are no pre-registered verdicts to report yet.")
    t = data.load_config("preregistration")
    policy = load("policy_v1.json")
    best = policy["best_router"]
    e6, e3, e2, e7, latency = (load(f) for f in ["e6_test_fmt.json", "e3_test_fmt.json", "e2_test.json", "e7_jeff.json", "latency.json"])
    rows = []

    rq1 = None
    if e6:
        ci = e6["systems"][best]["coverage_at_matched_risk_ci"]
        rq1 = ci
        rows.append({"id": "RQ1", "hypothesis": f"best router ({best}) coverage at matched risk on Banking77 test >= {100 * t['N']:.0f}%",
                     "result": f"{100 * ci['value']:.2f}% [{100 * ci['ci_low']:.2f}, {100 * ci['ci_high']:.2f}] "
                               f"(matched risk {100 * e6['s4']['error_rate']:.2f}%)",
                     "verdict": judge(ci["value"], ci["ci_low"], ci["ci_high"], t["N"])})
    if e3:
        c = next(c for c in e3["comparisons"] if c["a"].startswith("s3_") and c["b"].startswith("s1_"))
        rows.append({"id": "RQ2", "hypothesis": f"S3 minus S1 OOS AUROC on test >= {100 * t['M']:.0f} point",
                     "result": f"{100 * c['diff']:+.2f} points [{100 * c['ci_low']:+.2f}, {100 * c['ci_high']:+.2f}]",
                     "verdict": judge(c["diff"], c["ci_low"], c["ci_high"], t["M"])})
    rows.append({"id": "RQ3", "hypothesis": f"S3 within {100 * t['K']:.0f} accuracy points of a retrained softmax (E4)",
                 "result": "E4 not run", "verdict": "not tested"})
    if e2:
        effects = "; ".join(f"{name} {100 * r['training_effect']['diff']:+.2f} [{100 * r['training_effect']['ci_low']:+.2f}, "
                            f"{100 * r['training_effect']['ci_high']:+.2f}]" for name, r in e2["systems"].items())
        order = " > ".join(next(iter(e2["rankings"].values())))
        rows.append({"id": "RQ4", "hypothesis": "system ranking stable between original and cleaned training labels",
                     "result": f"order {order}; training effect, macro-F1 points: {effects}",
                     "verdict": "met" if e2["ranking_stable"] else "not met"})
    rows.append({"id": "RQ5", "hypothesis": f"best router flip rate under paraphrase < {100 * t['F']:.0f}% (E5)",
                 "result": "E5 not run", "verdict": "not tested"})
    if e7 and rq1:
        jeff_ci = e7["systems"][best]["coverage_at_matched_risk_ci"]
        drop = rq1["value"] - jeff_ci["value"]
        rows.append({"id": "RQ6", "hypothesis": f"coverage drop from Banking77 test to the Jeff set <= {100 * t['D']:.0f} points",
                     "result": f"{100 * drop:+.2f} points: Banking77 {100 * rq1['value']:.2f}% "
                               f"[{100 * rq1['ci_low']:.2f}, {100 * rq1['ci_high']:.2f}], Jeff {100 * jeff_ci['value']:.2f}% "
                               f"[{100 * jeff_ci['ci_low']:.2f}, {100 * jeff_ci['ci_high']:.2f}] (Jeff matched risk {100 * e7['matched_risk']:.2f}%)",
                     "verdict": judge(drop, rq1["ci_low"] - jeff_ci["ci_high"], rq1["ci_high"] - jeff_ci["ci_low"],
                                      t["D"], higher_is_better=False)})
    if latency and best in latency["systems"]:
        p95 = latency["systems"][best]["p95_ms"]
        rows.append({"id": "R4", "hypothesis": f"best router p95 latency <= {t['L_ms']} ms ({latency['machine']}, batch 1)",
                     "result": f"p95 {p95:.1f} ms over {latency['systems'][best]['n_queries']} queries",
                     "verdict": "met" if p95 <= t["L_ms"] else "not met"})

    meta = run_metadata({"preregistration": t})
    note = ("RQ6's interval is the widest pair of the two coverage intervals (low = Banking77 low minus Jeff high), "
            "since the two test sets are independent; it is used only to label a verdict not clear-cut.")
    parts = ["# Pre-registered verdicts", "Written by `scripts/report_verdicts.py` from the test result files. Do not edit by hand.",
             "## Provenance", metadata_markdown(meta), "## Verdicts", md_table(rows), note]
    (data.RESULTS_DIR / "verdicts.md").write_text("\n\n".join(parts) + "\n")
    with open(data.RESULTS_DIR / "verdicts.json", "w") as f:
        json.dump({"metadata": meta, "thresholds": t, "verdicts": rows}, f, indent=2, default=str)
    print(md_table(rows))


if __name__ == "__main__":
    main()
