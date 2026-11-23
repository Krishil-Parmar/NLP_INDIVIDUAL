"""Build report/summary.pdf from report/summary.md and check it fits in two pages.

Run from the project root:
    uv run python scripts/build_summary.py          # TODO notes shown as they are
    uv run python scripts/build_summary.py --fill   # TODOs replaced by filler text of the
                                                    # budgeted length, to test what fits

What it does:
  1. Fills the results table between the RESULTS markers: classification metrics from
     results/e1_test.json and coverage at matched risk from results/e6_test_fmt.json, once
     those files exist (M5). Until then the cells stay empty. No number is typed by hand.
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
    lines = [
        "<!-- RESULTS:START -->",
        "| System | Accuracy | Macro precision | Macro recall | Macro F1 | Coverage at matched risk |",
        # In pandoc, the number of dashes sets each column's relative width.
        "| ------------------------ | ------------ | ------------ | ------------ | ------------ | -------------- |",
    ]
    for name, prefix in ROWS:
        tags = [t for t in systems if t.startswith(prefix)]
        cells = [""] * len(METRICS)
        if len(tags) == 1:  # exactly one run folder per system on test (the chosen model)
            cells = [mean_std(systems[tags[0]]["summary"], m) for m in METRICS]
        elif e1 and prefix == "s4_":
            cells = ["-"] * len(METRICS)  # S4 ran on format-controlled test only; see the caption
        router_tags = [t for t in routers if t.startswith(prefix)]
        coverage = mean_std(routers[router_tags[0]]["summary"], "coverage_at_matched_risk") if len(router_tags) == 1 else ""
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {coverage} |")
    caption = (": Banking77 test, percent, mean ± standard deviation over seeds (five for S0, S1 and S3; "
               "S2 has no training, so one run). Classification columns: in-scope queries, raw text (E1).")
    if e6:
        s4 = e6["s4"]
        caption += (f" Coverage at matched risk: format-controlled test with out-of-scope queries mixed in (E6), "
                    f"at few-shot S4's error rate of {100 * s4['error_rate']:.1f}%. S4 ran on that text only: "
                    f"in-scope accuracy {100 * s4['in_scope_accuracy']:.1f}%.")
    lines += ["", caption + " Filled in by `scripts/build_summary.py` from `results/`.", "<!-- RESULTS:END -->"]
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
