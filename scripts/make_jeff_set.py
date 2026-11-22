"""Build the Jeff set from the writer's file: check it, split it, report the blind label check.

Run from the project root:
    uv run python scripts/make_jeff_set.py

Reads data/jeff/jeff_written.csv (columns text, intent, notes), written by a
Claude agent from the intent definitions in configs/jeff.yaml. Writes
data/jeff/jeff_queries.csv with columns id, text, intent, split, notes, where id
is the row number in the written file.

If data/jeff/jeff_label_check.csv exists (a second agent's blind labels, columns
id, label), it also writes results/jeff_label_check.md: how often the two agree.
The writer's label stays the label either way.

Nothing here runs a model, and nothing here looks at how any system does on the set.
"""

import sys

import pandas as pd

from router import data, jeff
from router.provenance import metadata_markdown, run_metadata
from router.tables import md_table

CHECK_FILE = "data/jeff/jeff_label_check.csv"
CHECK_REPORT = "jeff_label_check.md"


def check_report(agreement: dict, meta: dict) -> str:
    """The label-check report as markdown."""
    n, agree = agreement["n_queries"], agreement["agree"]
    per_label = [{"label": k, "agree": f"{v['agree']} of {v['n']}"} for k, v in agreement["per_label"].items()]
    parts = [
        "# Jeff set: blind label check",
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
    ]
    return "\n\n".join(parts) + "\n"


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
    main()
