"""E3: out-of-scope (OOS) detection for every system with saved predictions.

Run from the project root:
    uv run python scripts/evaluate_oos.py --split val

The saved predictions cover the mixed set: Banking77 in-scope queries plus the
CLINC150 OOS pool (README section 6). Each row has is_oos and the system's
oos_score (1 - confidence; higher = more likely out of scope).

Per system (mean and std over seeds):
  - AUROC, with OOS as the positive class
  - FPR at 95% TPR: the share of OOS queries still kept when the threshold keeps
    95% of in-scope queries, and that threshold
Plus paired bootstrap intervals for the comparisons in configs/e3_oos.yaml, the
OOS queries the inspected system accepted most confidently, and a ROC figure.

It also reports "surface cues": how well a trivial formatting rule (starts
lowercase, no final punctuation, no comma) separates OOS from in-scope on its
own. On the raw text it does very well, because CLINC150 queries are all
lowercase and unpunctuated. The splits ending in "_fmt" hold predictions on
format-controlled text (scripts/predict_format_control.py), where these cues
are gone; those are the primary E3 numbers.

On test, each seed's threshold is frozen from the matching validation file
(results/e3_val.json or e3_val_fmt.json) and applied unchanged, which gives an
honest OOS recall. Test splits are refused until results/preregistration.md
exists (rigour rule 1).

Writes results/e3_<split>.json, results/e3_<split>.md and results/figures/e3_<split>_roc.png.
"""

import argparse
import json
import sys

import matplotlib

matplotlib.use("Agg")  # draw straight to a file, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from router import data
from router.metrics import fpr_at_tpr, mean_std_over_seeds, oos_auroc, paired_bootstrap_diff, rates_at_threshold
from router.provenance import metadata_markdown, run_metadata
from router.systems.base import list_runs, load_predictions
from router.tables import md_table

FIGURES_DIR = data.RESULTS_DIR / "figures"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"  # chart ink colours (dataviz reference palette)
SPLITS = ["val", "test", "val_fmt", "test_fmt"]

# Trivial formatting rules. Each returns True when the query "looks like" CLINC150.
SURFACE_CUES = {
    "starts lowercase": lambda text: text[:1].islower(),
    "no final punctuation": lambda text: not text.rstrip().endswith(("?", ".", "!")),
    "no comma": lambda text: "," not in text,
}


def surface_cues(df: pd.DataFrame) -> list[dict]:
    """How well each formatting rule alone separates OOS from in-scope (AUROC 0.5 = not at all)."""
    is_oos = df["is_oos"].to_numpy(dtype=bool)
    rows = []
    for name, rule in SURFACE_CUES.items():
        cue = df["text"].map(rule).to_numpy(dtype=float)  # 1.0 where the rule fires
        rows.append({"cue": name, "in_scope_share": float(cue[~is_oos].mean()),
                     "oos_share": float(cue[is_oos].mean()), "auroc": oos_auroc(is_oos, cue)})
    return rows


def load_all_runs(split: str) -> dict[str, dict[int, pd.DataFrame]]:
    """Every saved prediction table, as {system_tag: {seed: table}}.

    Paired comparisons only make sense if every run scored the same queries in
    the same order, so that is checked here once.
    """
    tables = {
        tag: {seed: load_predictions(tag, seed, split)[0] for seed in seeds}
        for tag, seeds in list_runs(split).items()
    }
    if not tables:
        sys.exit(f"No saved predictions for split {split!r}. Run scripts/train.py first.")
    reference = next(iter(next(iter(tables.values())).values()))  # the first table of the first system
    for tag, runs in tables.items():
        for seed, df in runs.items():
            if not (df["text"].equals(reference["text"]) and df["is_oos"].equals(reference["is_oos"])):
                sys.exit(f"{tag} seed {seed} scored different rows from the other runs; paired comparison is impossible.")
    if not reference["is_oos"].any():
        sys.exit("These predictions contain no OOS rows, so E3 cannot be computed.")
    return tables


def labels_and_scores(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df["is_oos"].to_numpy(dtype=bool), df["oos_score"].to_numpy(dtype=float)


def evaluate_system(runs: dict[int, pd.DataFrame], keep: float, val_thresholds: dict[int, float] | None) -> dict:
    """Metrics for each seed run, then mean and std over seeds.

    val_thresholds is None on validation. On test it holds each seed's frozen
    validation threshold, so OOS recall on test is measured out of sample.
    """
    per_seed, thresholds = [], {}
    for seed, df in runs.items():
        is_oos, score = labels_and_scores(df)
        at_keep = fpr_at_tpr(is_oos, score, keep)
        row = {"auroc": oos_auroc(is_oos, score), "fpr_at_tpr": at_keep["fpr"]}
        thresholds[seed] = at_keep["threshold"]
        if val_thresholds is not None:
            frozen = rates_at_threshold(is_oos, score, val_thresholds[seed])
            row["oos_recall_at_val_threshold"] = frozen["oos_recall"]
            row["in_scope_kept_at_val_threshold"] = frozen["in_scope_kept"]
        per_seed.append(row)
    return {
        "seeds": list(runs),
        "n_in_scope": int((~is_oos).sum()),
        "n_oos": int(is_oos.sum()),
        "per_seed": per_seed,
        "thresholds": thresholds,  # per seed; on validation these are the ones frozen for test
        "summary": mean_std_over_seeds(per_seed),
    }


def mean_auroc_on_rows(runs: dict[int, pd.DataFrame]):
    """A function of row indices: the system's AUROC on those rows, averaged over its seeds."""
    arrays = [labels_and_scores(df) for df in runs.values()]

    def metric(rows: np.ndarray) -> float:
        return float(np.mean([oos_auroc(is_oos[rows], score[rows]) for is_oos, score in arrays]))

    return metric


def compare(tables: dict, cfg: dict) -> list[dict]:
    """Paired bootstrap of AUROC(A) - AUROC(B) for each configured pair."""
    boot = cfg["bootstrap"]
    rows = []
    for a, b in cfg["comparisons"]:
        if a not in tables or b not in tables:
            print(f"Skipping comparison {a} vs {b}: no saved predictions for one of them.")
            continue
        strata = next(iter(tables[a].values()))["is_oos"].to_numpy(dtype=bool)
        result = paired_bootstrap_diff(
            mean_auroc_on_rows(tables[a]), mean_auroc_on_rows(tables[b]), strata,
            n_resamples=boot["resamples"], seed=boot["seed"], ci=boot["ci"],
        )
        rows.append({"a": a, "b": b, **result})
    return rows


def most_confidently_accepted(runs: dict[int, pd.DataFrame], k: int = 10) -> list[dict]:
    """The k OOS queries with the lowest oos_score in the first seed run: the worst misses."""
    df = runs[min(runs)]
    oos = df[df["is_oos"]].sort_values("oos_score").head(k)
    columns = ["text", "pred_label", "confidence"] + (["matched_example"] if "matched_example" in df else [])
    return oos[columns].to_dict(orient="records")


def draw_roc(tables: dict, cfg: dict, split: str) -> str | None:
    """ROC curves (first seed of each system) for the systems listed in the config."""
    systems = [s for s in cfg["figure"] if s["tag"] in tables]
    if not systems:
        return None
    fig, ax = plt.subplots(figsize=(5.6, 4.4), dpi=200)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":")
    ax.text(0.62, 0.55, "chance", color=MUTED, fontsize=8, rotation=38)
    for s in systems:
        runs = tables[s["tag"]]
        is_oos, score = labels_and_scores(runs[min(runs)])
        fpr, tpr, _ = roc_curve(is_oos, score)
        auroc = oos_auroc(is_oos, score)
        ax.plot(fpr, tpr, color=s["color"], linewidth=2, linestyle=s["style"],
                label=f"{s['name']} (AUROC {100 * auroc:.1f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("In-scope queries wrongly flagged as out of scope", color=INK, fontsize=9)
    ax.set_ylabel("Out-of-scope queries caught", color=INK, fontsize=9)
    ax.set_title(f"Out-of-scope detection on {split} (seed {min(next(iter(tables.values())))})",
                 color=INK, fontsize=10, loc="left")
    ax.grid(color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#c3c2b7")
    legend = ax.legend(loc="lower right", fontsize=8, frameon=False)
    for text in legend.get_texts():
        text.set_color(INK)  # legend text stays in ink; the line beside it carries the colour
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"e3_{split}_roc.png"
    fig.tight_layout()
    fig.savefig(path, facecolor="#ffffff")
    plt.close(fig)
    return str(path.relative_to(data.PROJECT_ROOT))


def pct(summary: dict, key: str) -> str:
    return f"{100 * summary[key]['mean']:.2f} ± {100 * summary[key]['std']:.2f}"


def write_markdown(results: dict, comparisons: list[dict], accepted: list[dict], cues: list[dict],
                   figure: str | None, split: str, cfg: dict, meta: dict) -> str:
    keep_pct = round(100 * cfg["in_scope_keep"])
    metric_names = {
        "auroc": "AUROC",
        "fpr_at_tpr": f"FPR at {keep_pct}% TPR",
        "oos_recall_at_val_threshold": "OOS recall (val threshold)",
        "in_scope_kept_at_val_threshold": "in-scope kept (val threshold)",
    }
    first = next(iter(results.values()))
    main_rows = [
        {"system": tag, "seeds": len(r["seeds"]),
         **{metric_names[k]: pct(r["summary"], k) for k in r["summary"]},
         "threshold (mean)": f"{np.mean(list(r['thresholds'].values())):.4f}"}
        for tag, r in results.items()
    ]
    comparison_rows = [
        {"A": c["a"], "B": c["b"], "AUROC A - B (points)": f"{100 * c['diff']:+.2f}",
         f"{round(100 * cfg['bootstrap']['ci'])}% CI": f"[{100 * c['ci_low']:+.2f}, {100 * c['ci_high']:+.2f}]",
         "resamples": c["n_resamples"]}
        for c in comparisons
    ]
    accepted_rows = [
        {k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in row.items()} for row in accepted
    ]
    is_validation = split.startswith("val")
    text_note = (
        "**Format-controlled text:** every query (in-scope and OOS) was lowercased and stripped of punctuation "
        "before prediction (`router.preprocess.format_control`), so formatting cannot give the answer away. "
        "These are the primary E3 numbers."
        if split.endswith("_fmt") else
        "**Raw text.** Formatting separates the two sources on its own (surface cues below), so these numbers "
        f"overstate OOS detection. The primary numbers are in `results/e3_{split}_fmt.md`."
    )
    cue_rows = [
        {"rule (fires = looks like CLINC150)": c["cue"], "in-scope share": f"{100 * c['in_scope_share']:.1f}",
         "OOS share": f"{100 * c['oos_share']:.1f}", "AUROC of the rule alone": f"{100 * c['auroc']:.1f}"}
        for c in cues
    ]
    parts = [
        f"# E3 results on {split}",
        "Written by `scripts/evaluate_oos.py`. Do not edit by hand.",
        "## Provenance",
        metadata_markdown(meta),
        f"## Out-of-scope detection ({split}), percent, mean ± std over seeds",
        text_note,
        f"{first['n_in_scope']} in-scope Banking77 queries and {first['n_oos']} CLINC150 out-of-scope queries. "
        "AUROC: OOS is the positive class, higher is better. "
        f"FPR at {keep_pct}% TPR: the threshold keeps at least {keep_pct}% of in-scope queries; "
        "FPR is the share of OOS queries also kept, lower is better "
        + ("(so OOS recall at that threshold is 100 minus this number). " if is_validation else ". ")
        + "Std is the sample standard deviation across seeds.",
        "**Limitation:** the OOS queries come from a different dataset (CLINC150) than the in-scope ones, "
        "so rejection looks easier than in real use even with formatting controlled (topics and length still differ). "
        "The Jeff set's finance-adjacent OOS queries are the harder test.",
        md_table(main_rows),
        "## Surface cues (no model)",
        "AUROC of a trivial formatting rule on its own, on the text the systems saw. 50 means the rule carries no signal.",
        md_table(cue_rows),
    ]
    if comparison_rows:
        parts += [
            "## Comparisons (paired, stratified bootstrap)",
            "Each system's AUROC is averaged over its seeds; in-scope and OOS queries are resampled separately, "
            "with the same rows for both systems. An interval that spans 0 means no clear difference.",
            md_table(comparison_rows),
        ]
    if accepted_rows:
        parts += [
            f"## OOS queries {cfg['inspect_system']} accepted most confidently (first seed)",
            "The lowest OOS scores among out-of-scope queries: the misses to read first.",
            md_table(accepted_rows),
        ]
    if figure:
        parts += ["## Figure", f"ROC curves, first seed of each system: `{figure}`"]
    return "\n\n".join(parts) + "\n"


def main(split: str) -> None:
    is_test = split.startswith("test")
    if is_test and not (data.RESULTS_DIR / "preregistration.md").exists():
        sys.exit("Refusing to score a test split: results/preregistration.md does not exist yet (rigour rule 1).")
    cfg = data.load_config("e3_oos")
    tables = load_all_runs(split)

    val_thresholds: dict[str, dict[int, float]] = {}
    if is_test:  # freeze each seed's threshold from the matching validation run
        with open(data.RESULTS_DIR / f"e3_{split.replace('test', 'val')}.json") as f:
            val = json.load(f)["systems"]
        val_thresholds = {tag: {int(s): t for s, t in r["thresholds"].items()} for tag, r in val.items()}

    results = {
        tag: evaluate_system(runs, cfg["in_scope_keep"], val_thresholds.get(tag) if is_test else None)
        for tag, runs in tables.items()
    }
    comparisons = compare(tables, cfg)
    inspect = cfg["inspect_system"]
    accepted = most_confidently_accepted(tables[inspect]) if inspect in tables else []
    cues = surface_cues(next(iter(next(iter(tables.values())).values())))  # same texts in every run
    figure = draw_roc(tables, cfg, split)

    meta = run_metadata({**data.load_config("data"), "e3_oos": cfg}, seed=cfg["bootstrap"]["seed"])
    with open(data.RESULTS_DIR / f"e3_{split}.json", "w") as f:
        json.dump({"metadata": meta, "split": split, "systems": results, "comparisons": comparisons,
                   "most_confidently_accepted": accepted, "surface_cues": cues, "figure": figure},
                  f, indent=2, default=str)
    (data.RESULTS_DIR / f"e3_{split}.md").write_text(
        write_markdown(results, comparisons, accepted, cues, figure, split, cfg, meta)
    )

    print(pd.DataFrame({tag: {k: pct(r["summary"], k) for k in r["summary"]} for tag, r in results.items()}).T.to_string())
    for c in comparisons:
        print(f"{c['a']} - {c['b']}: {100 * c['diff']:+.2f} points, CI [{100 * c['ci_low']:+.2f}, {100 * c['ci_high']:+.2f}]")
    print(f"\nWrote results/e3_{split}.json, results/e3_{split}.md" + (f" and {figure}" if figure else ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, default="val")
    main(parser.parse_args().split)
