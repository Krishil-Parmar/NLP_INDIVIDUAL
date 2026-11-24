"""Build report/summary.pdf from report/summary.md and check it fits in two pages.

Run from the project root:
    uv run python scripts/build_summary.py          # TODO notes shown as they are
    uv run python scripts/build_summary.py --fill   # TODOs replaced by filler text of the
                                                    # budgeted length, to test what fits

What it does:
  1. Fills the results table between the RESULTS markers: classification metrics from
     results/e1_test.json (S4's from results/headline_reanalysis.json, on format-controlled
     text), and coverage at matched risk, at 5% and at 1% risk from results/e6_test_fmt.json.
     Fills the requirements table between the REQUIREMENTS markers: the evidence column from
     the result files, and the "Met?" column from REQUIREMENT_VERDICTS below. Until a file
     exists its cells stay empty. No number is typed by hand.
  2. Uses the E6 test risk-coverage figure as the headline, or a grey placeholder of the
     same size until it exists.
  3. Runs pandoc (markdown -> LaTeX) and XeLaTeX (LaTeX -> PDF) inside report/build/.
  4. Reads the page count from the XeLaTeX log and fails if it is over 2 pages.

Output: report/summary.pdf (or report/build/summary_fill_test.pdf with --fill).
Needs pandoc and XeLaTeX installed on the machine (not Python packages).
"""

import argparse
import json
import re
import shutil
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")  # draw to files only, no window
import matplotlib.pyplot as plt  # noqa: E402

from router import data  # noqa: E402

REPORT_DIR = data.PROJECT_ROOT / "report"
BUILD_DIR = REPORT_DIR / "build"
SOURCE = REPORT_DIR / "summary.md"
HEADLINE = data.RESULTS_DIR / "figures" / "e6_test_fmt_risk_coverage.png"  # written by evaluate_router.py at M5
MAX_PAGES = 2

TODO_PATTERN = re.compile(r"\[TODO (\d+) words: [^\]]*\]")
RESULTS_BLOCK = re.compile(r"<!-- RESULTS:START -->.*?<!-- RESULTS:END -->", re.DOTALL)
REQUIREMENTS_BLOCK = re.compile(r"<!-- REQUIREMENTS:START -->.*?<!-- REQUIREMENTS:END -->", re.DOTALL)

# Proposed by Claude on 24 Nov 2026 from the evidence column; the owner confirms or changes them.
# Only R4 has a pre-registered bar (L = 100 ms); the others are judgements, stated as such.
REQUIREMENT_VERDICTS = {
    "R1": "Yes (no preset bar)",
    "R2": "No",
    "R3": "Yes (frozen bar)",
    "R4": "Yes",
    "R5": "Not tested",
    "R6": "Not tested",
    "R7": "No",
}

# Table rows: (row name, prefix of the system tags in results/predictions/ that fill it)
ROWS = [
    ("S0 TF-IDF + LR", "s0_"),
    ("S1 Fine-tuned encoder", "s1_"),
    ("S2 Embedding k-NN", "s2_"),
    ("S3 Pairwise matcher", "s3_"),
    ("S4 LLM few-shot", "s4_"),
]
METRICS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]


def fill_todos(text: str) -> str:
    """Replace every [TODO N words: ...] with N words of filler, to see how much fits."""
    filler = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
              "incididunt ut labore et dolore magna aliqua").split()

    def replace(match: re.Match) -> str:
        n = int(match.group(1))
        return " ".join(filler[i % len(filler)] for i in range(n))

    return TODO_PATTERN.sub(replace, text)


def load_result(name: str) -> dict | None:
    path = data.RESULTS_DIR / name
    return json.loads(path.read_text()) if path.exists() else None


def mean_std(summary: dict, key: str) -> str:
    return f"{100 * summary[key]['mean']:.1f} ± {100 * summary[key]['std']:.1f}"


def results_block() -> str:
    """The results table: E1 (raw test) and E6 coverage at matched risk (format-controlled test)."""
    e1 = load_result("e1_test.json")
    e6 = load_result("e6_test_fmt.json")
    systems = e1["systems"] if e1 else {}
    routers = e6["systems"] if e6 else {}
    reanalysis = load_result("headline_reanalysis.json")
    lines = [
        "<!-- RESULTS:START -->",
        "| System | Accuracy | Macro precision | Macro recall | Macro F1 | Coverage at S4 risk | Coverage at 5% risk | Coverage at 1% risk |",
        # In pandoc, the number of dashes sets each column's relative width.
        "| ---------------- | ---------- | ---------- | ---------- | ---------- | -------- | -------- | -------- |",
    ]
    for name, prefix in ROWS:
        tags = [t for t in systems if t.startswith(prefix)]
        cells = [""] * len(METRICS)
        if len(tags) == 1:  # exactly one run folder per system on test (the chosen model)
            cells = [mean_std(systems[tags[0]]["summary"], m) for m in METRICS]
        elif prefix == "s4_" and reanalysis:  # S4 ran on format-controlled test only; marked with a dagger
            cells = [f"{100 * reanalysis['s4']['classification'][m]:.1f}†" for m in METRICS]
        router_tags = [t for t in routers if t.startswith(prefix)]
        coverage = [mean_std(routers[router_tags[0]]["summary"], k) if len(router_tags) == 1 else ("-" if e6 else "")
                    for k in ["coverage_at_matched_risk", "coverage_at_0.05", "coverage_at_0.01"]]
        lines.append(f"| {name} | " + " | ".join(cells + coverage) + " |")
    caption = (": Banking77 test, percent, mean ± s.d. over seeds (five for S0, S1, S3; one for S2). Accuracy to F1: "
               "in-scope queries, raw text (E1); † S4, format-controlled text.")
    if e6 and reanalysis:
        caption += (f" Coverage: share of the format-controlled mix ({100 * reanalysis['in_scope_share']:.1f}% in scope, the rest "
                    f"CLINC150 out of scope; E6) handled at or below a risk bar: S4's error ({100 * e6['s4']['error_rate']:.1f}%), "
                    "5% or 1%. S4 is less accurate in scope than every router, so the matched-risk column sits near the in-scope share.")
    lines += ["", caption, "<!-- RESULTS:END -->"]
    return "\n".join(lines)


def requirements_block() -> str:
    """The requirements table: evidence read from the result files, verdicts from REQUIREMENT_VERDICTS."""
    e1, e6, e7 = load_result("e1_test.json"), load_result("e6_test_fmt.json"), load_result("e7_jeff.json")
    re_an, e8, latency = load_result("headline_reanalysis.json"), load_result("e8_test_v2.json"), load_result("latency.json")
    evidence = {k: "" for k in REQUIREMENT_VERDICTS}
    if e1:
        acc = {t: 100 * r["summary"]["accuracy"]["mean"] for t, r in e1["systems"].items()}
        best = max(acc, key=acc.get)
        evidence["R1"] = f"Best E1 accuracy {acc[best]:.1f}% ({best.split('_')[0].upper()})"
    if re_an:
        pair = next(r for r in re_an["b"] if r["system"].startswith("s2_") and "post-hoc" not in r["system"])
        s4 = re_an["b"][0]
        evidence["R2"] = f"S2 plus LLM answers {pair['oos_answered']:.0f} of {re_an['n_oos']} OOS queries; LLM alone {s4['oos_answered']:.0f}"
        evidence["R3"] = (f"{pair['llm_calls']:.0f} of {s4['llm_calls']:.0f} LLM calls; {100 * pair['error_rate']:.1f}% wrong "
                          f"against {100 * s4['error_rate']:.1f}%")
    if latency:
        evidence["R4"] = f"S2 p95 {latency['systems']['s2_all-MiniLM-L6-v2']['p95_ms']:.1f} ms against 100 ms"
    evidence["R5"] = "E4 not built" + (f"; Jeff needed a fine-tune (E8, post-hoc)" if e8 else "")
    evidence["R6"] = "E5 not built"
    if e7:
        cov = e7["systems"][e7["best_router"]]["summary"]["coverage_at_matched_risk"]["mean"]
        evidence["R7"] = f"S2 covers {100 * cov:.1f}% of the Jeff set at matched risk"
    names = {"R1": "Correct intent on in-scope queries", "R2": "OOS and uncertain queries not handled",
             "R3": "Fewer LLM calls, error no higher", "R4": "Router latency within limit", "R5": "New intent from about 10 examples",
             "R6": "Stable under rewording", "R7": "Works on the Jeff set (own intents)"}
    checks = {"R1": "E1", "R2": "E3, E6", "R3": "E6", "R4": "E6", "R5": "E4", "R6": "E5", "R7": "E7"}
    lines = ["<!-- REQUIREMENTS:START -->",
             "| Requirement | Check | Met? | Evidence |",
             "| ---------------------------------------------- | ---------- | ---------- | ------------------------------ |"]
    lines += [f"| {r} {names[r]} | {checks[r]} | {REQUIREMENT_VERDICTS[r]} | {evidence[r]} |" for r in REQUIREMENT_VERDICTS]
    lines += ["", ": Requirements from README section 2.5. Only R4 had a preset bar. (Rubric 1)", "<!-- REQUIREMENTS:END -->"]
    return "\n".join(lines)


def headline_figure() -> str:
    """Copy the headline figure into the build folder, or draw a placeholder. Returns its file name."""
    target = BUILD_DIR / "headline.png"
    if HEADLINE.exists():
        shutil.copy(HEADLINE, target)
        return target.name
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.text(0.5, 0.5, "headline figure placeholder (added in M4)", ha="center", va="center", fontsize=12)
    ax.set_facecolor("#e6e6e6")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return target.name


def run(command: list[str]) -> subprocess.CompletedProcess:
    """Run a command inside report/build and return its result."""
    return subprocess.run(command, cwd=BUILD_DIR, capture_output=True, text=True)


def main(fill: bool) -> None:
    for tool in ["pandoc", "xelatex"]:
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found. Install it to build the PDF summary.")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    text = SOURCE.read_text()
    text = RESULTS_BLOCK.sub(lambda _: results_block(), text)
    text = REQUIREMENTS_BLOCK.sub(lambda _: requirements_block(), text)
    text = text.replace("HEADLINE_FIGURE", headline_figure())
    if fill:
        text = fill_todos(text)
    (BUILD_DIR / "summary.md").write_text(text)

    result = run(["pandoc", "summary.md", "--standalone", "-o", "summary.tex"])
    if result.returncode != 0:
        sys.exit(f"pandoc failed:\n{result.stderr}")
    result = run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "summary.tex"])
    log = (BUILD_DIR / "summary.log").read_text(errors="replace")
    if result.returncode != 0:
        errors = [line for line in log.splitlines() if line.startswith("!")]
        sys.exit("XeLaTeX failed:\n" + "\n".join(errors[:10]) + "\nFull log: report/build/summary.log")

    match = re.search(r"Output written on \S+ \((\d+) pages?", log)
    pages = int(match.group(1)) if match else -1
    output = BUILD_DIR / "summary_fill_test.pdf" if fill else REPORT_DIR / "summary.pdf"
    shutil.copy(BUILD_DIR / "summary.pdf", output)

    todos_left = len(TODO_PATTERN.findall(SOURCE.read_text()))
    print(f"Wrote {output.relative_to(data.PROJECT_ROOT)}: {pages} page(s), limit {MAX_PAGES}. "
          f"TODO placeholders left in summary.md: {todos_left}.")
    if pages > MAX_PAGES or pages < 0:
        sys.exit(f"Summary is {pages} pages: over the {MAX_PAGES}-page limit (or page count unreadable).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fill", action="store_true", help="replace TODOs with filler text to test the fit")
    main(parser.parse_args().fill)
