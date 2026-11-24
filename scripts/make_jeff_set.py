"""Build the Jeff set from the writer's file: check it, split it, report the blind label check.

Run from the project root:
    uv run python scripts/make_jeff_set.py         # the Jeff set (support + test)
    uv run python scripts/make_jeff_set.py --v2    # Jeff test v2 (E8), test only

Reads data/jeff/jeff_written.csv (columns text, intent, notes), written by a
Claude agent from the intent definitions in configs/jeff.yaml. Writes
data/jeff/jeff_queries.csv with columns id, text, intent, split, notes, where id
is the row number in the written file.

If data/jeff/jeff_label_check.csv exists (a second agent's blind labels, columns
id, label), it also writes results/jeff_label_check.md: how often the two agree.
The writer's label stays the label either way.

With --v2 it builds the fresh test split written for E8 instead (configs/jeff.yaml,
key v2): every row goes to split test_v2, the blind check is reported to
results/jeff_v2_label_check.md, and each v2 query's closest v1 query (difflib ratio,
lower-cased) is reported, so a v2 query that copies a v1 support example is caught.

Nothing here runs a model, and nothing here looks at how any system does on the set.
"""

import argparse
import difflib
import sys

import pandas as pd

from router import data, jeff
from router.provenance import metadata_markdown, run_metadata
from router.tables import md_table

CHECK_FILE = "data/jeff/jeff_label_check.csv"
CHECK_REPORT = "jeff_label_check.md"


def check_report(agreement: dict, meta: dict, title: str = "Jeff set: blind label check", extra: list[str] | None = None) -> str:
    """The label-check report as markdown."""
    n, agree = agreement["n_queries"], agreement["agree"]
    per_label = [{"label": k, "agree": f"{v['agree']} of {v['n']}"} for k, v in agreement["per_label"].items()]
    parts = [
        f"# {title}",
        "Written by `scripts/make_jeff_set.py`. Do not edit by hand.",
        "## Provenance", metadata_markdown(meta),
        "## Agreement",
        "A second Claude agent labelled every query from the text alone, given only the intent definitions in "
        "`configs/jeff.yaml`, without the writer's labels. The writer's label stays the label; this measures how clear the labels are.",
        f"**Same label on {agree} of {n} queries ({100 * agree / n:.1f}%).**",
        md_table(per_label),
        "## Disagreements (for the owner's review)",
        md_table(agreement["disagreements"]) if agreement["disagreements"] else "None.",
        "## Marked unsure by the checker (for the owner's review)",
        "The checker gave a label but thought a careful reader could pick another.",
        md_table(agreement["unsure"]) if agreement["unsure"] else "None.",
        "## Caveat",
        "Both agents are Claude models working from the same definitions, so agreement shows the labels are clear to a model reader. "
        "It does not show that real users write this clearly: the writer was asked for one clear label per message, "
        "so the set is likely cleaner than real traffic.",
    ] + (extra or [])
    return "\n\n".join(parts) + "\n"


def closest_v1(v2: pd.DataFrame, v1: pd.DataFrame) -> pd.DataFrame:
    """For each v2 query: the most similar v1 query (difflib ratio on lower-cased text), its split and intent."""
    rows = []
    for _, q in v2.iterrows():
        ratios = [difflib.SequenceMatcher(None, q["text"].lower(), t.lower()).ratio() for t in v1["text"]]
        best = int(max(range(len(ratios)), key=ratios.__getitem__))
        rows.append({"v2_id": q["id"], "v2_text": q["text"], "v2_intent": q["intent"], "ratio": round(ratios[best], 3),
                     "v1_text": v1["text"].iloc[best], "v1_split": v1["split"].iloc[best], "v1_intent": v1["intent"].iloc[best]})
    return pd.DataFrame(rows).sort_values("ratio", ascending=False)


def main_v2() -> None:
    """Build Jeff test v2: check the written file, put every row in the test_v2 split, report the checks."""
    cfg = data.load_config("jeff")
    v2 = cfg["v2"]
    written = pd.read_csv(data.PROJECT_ROOT / v2["written_file"], keep_default_na=False)
    written.insert(0, "id", range(1, len(written) + 1))
    problems = jeff.check_written(written, {**cfg, "queries_per_intent": v2["queries_per_intent"]})
    v1 = pd.read_csv(data.PROJECT_ROOT / cfg["file"], keep_default_na=False)
    normalised = lambda t: " ".join(t.lower().split())  # noqa: E731 (same rule as check_written)
    copied = set(written["text"].map(normalised)) & set(v1["text"].map(normalised))
    if copied:
        problems.append(f"texts also in the v1 set: {sorted(copied)}")
    if problems:
        sys.exit("Problems in the v2 written file:\n  " + "\n  ".join(problems))

    queries = written.assign(split=v2["split"])[["id", "text", "intent", "split", "notes"]]
    near = closest_v1(queries, v1)
    too_close = near[near["ratio"] > v2["max_ratio_to_v1"]]
    if len(too_close):
        sys.exit(f"{len(too_close)} v2 queries are closer to a v1 query than {v2['max_ratio_to_v1']}:\n"
                 + too_close[["v2_id", "ratio", "v2_text", "v1_text"]].to_string())
    queries.to_csv(data.PROJECT_ROOT / v2["file"], index=False)
    print(f"Wrote {v2['file']}: {len(queries)} queries, all in split {v2['split']}.")

    near_rows = near.head(5).to_dict("records")
    v1_test = v1[v1["split"] == "test"]
    v1_reference = closest_v1(v1_test, v1[v1["split"] == "support"].reset_index(drop=True))["ratio"].max()
    extra = [
        "## Closest v1 query for each v2 query",
        f"difflib ratio on lower-cased text, against all 320 v1 queries (support and old test). 1.0 = identical. "
        f"Cap: {v2['max_ratio_to_v1']}. For reference, the closest any v1 test query came to a v1 support example: {v1_reference:.3f}. "
        f"Highest: {near['ratio'].max():.3f}. Above 0.8: {int((near['ratio'] > 0.8).sum())}. "
        f"Median: {near['ratio'].median():.3f}. The five highest:",
        md_table(near_rows),
    ]
    agreement = jeff.label_agreement(queries, pd.read_csv(data.PROJECT_ROOT / v2["check_file"], keep_default_na=False))
    meta = run_metadata({"jeff": cfg})
    report = check_report(agreement, meta, "Jeff test v2: blind label check and overlap with v1", extra)
    (data.RESULTS_DIR / "jeff_v2_label_check.md").write_text(report)
    print(f"Label check: same label on {agreement['agree']} of {agreement['n_queries']}. "
          f"Highest ratio to a v1 query: {near['ratio'].max():.3f}. Wrote results/jeff_v2_label_check.md")


def main() -> None:
    cfg = data.load_config("jeff")
    written = pd.read_csv(data.PROJECT_ROOT / cfg["written_file"], keep_default_na=False)
    written.insert(0, "id", range(1, len(written) + 1))
    problems = jeff.check_written(written, cfg)
    if problems:
        sys.exit("Problems in the written file:\n  " + "\n  ".join(problems))

    queries = jeff.assign_splits(written, cfg["support_per_intent"], cfg["split_seed"])
    queries = queries[["id", "text", "intent", "split", "notes"]]
    queries.to_csv(data.PROJECT_ROOT / cfg["file"], index=False)
    counts = queries.groupby(["split"])["intent"].value_counts().unstack(0).fillna(0).astype(int)
    print(counts.to_string())
    print(f"Wrote {cfg['file']}: {len(queries)} queries, "
          f"{int((queries['split'] == 'support').sum())} support, {int((queries['split'] == 'test').sum())} test.")

    check_path = data.PROJECT_ROOT / CHECK_FILE
    if check_path.exists():
        agreement = jeff.label_agreement(queries, pd.read_csv(check_path, keep_default_na=False))
        meta = run_metadata({"jeff": cfg}, seed=cfg["split_seed"])
        (data.RESULTS_DIR / CHECK_REPORT).write_text(check_report(agreement, meta))
        print(f"Label check: same label on {agreement['agree']} of {agreement['n_queries']}. Wrote results/{CHECK_REPORT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v2", action="store_true", help="build Jeff test v2 (E8) instead")
    main_v2() if parser.parse_args().v2 else main()
