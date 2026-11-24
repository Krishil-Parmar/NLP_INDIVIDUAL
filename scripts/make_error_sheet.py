"""M6 error analysis (rigour rule 8): build the sheet of router errors, then summarise the categories.

Run from the project root:
    uv run python scripts/make_error_sheet.py              # writes the sheet and the owner's blind check file
    uv run python scripts/make_error_sheet.py --summarise  # after the categories are filled in

Which errors (fixed before anyone reads a row):
  - Banking77: 100 errors drawn at random (seed 0) from every query S2 handles wrongly on
    test_fmt at its frozen policy v1 threshold. That is what the deployed router gets wrong:
    a wrong intent, or an out-of-scope query answered as if in scope.
  - Jeff (old test split, now E8 development data): every in-scope query S2 gets wrong, plus
    every out-of-scope query S2 would handle at the frozen Banking77 threshold.
Rows are shuffled (seed 0) so the reader does not see them grouped by source or intent.

Categories (one per row; the definitions are also in results/error_analysis.md):
  ambiguous     the query fits both the true and the predicted intent; a careful person could pick either
  label_error   the true label looks wrong and the predicted intent fits better
  same_topic    the predicted intent shares the topic but the request is different (another action,
                time window or object); a careful person would pick the true label
  oos_accepted  an out-of-scope query was handled as if it were in scope
  other         none of these (the note says why)

Claude categorises every row (results/error_analysis/claude_categories.csv). The owner
categorises a random 20 blind (results/error_analysis/owner_check.csv, Claude's categories
not shown); agreement on those 20 is reported, as in the M4b label audit.
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd

from router import data
from router.systems.base import load_predictions
from router.tables import md_table

FOLDER = data.RESULTS_DIR / "error_analysis"
S2_TAG = "s2_all-MiniLM-L6-v2"
N_BANKING77 = 100
N_OWNER_CHECK = 20
CATEGORIES = {
    "ambiguous": "The query fits both the true and the predicted intent; a careful person could pick either.",
    "label_error": "The true label looks wrong and the predicted intent fits better.",
    "same_topic": "The predicted intent shares the topic, but the request is different (another action, time window "
                  "or object); a careful person would pick the true label.",
    "oos_accepted": "An out-of-scope query was handled as if it were in scope.",
    "other": "None of these; the note says why.",
}
COLUMNS = ["source", "text", "true_label", "predicted_label", "confidence", "matched_example"]


def handled_errors(split: str, threshold: float) -> pd.DataFrame:
    """Rows S2 handles (confidence at or above the threshold) with the wrong answer."""
    df, _ = load_predictions(S2_TAG, 0, split)
    wrong = (df["confidence"] >= threshold) & (df["pred_label"] != df["label"])
    return df[wrong]


def as_sheet(df: pd.DataFrame, source: str) -> pd.DataFrame:
    return pd.DataFrame({"source": source, "text": df["text"], "true_label": df["label"], "predicted_label": df["pred_label"],
                         "confidence": df["confidence"].round(3), "matched_example": df["matched_example"]})


def build() -> None:
    threshold = json.loads((data.RESULTS_DIR / "policy_v1.json").read_text())["systems"][S2_TAG]["thresholds"]["0"]
    b77 = handled_errors("test_fmt", threshold)
    b77 = b77.iloc[np.sort(np.random.default_rng(0).choice(len(b77), size=N_BANKING77, replace=False))]
    jeff, _ = load_predictions(S2_TAG, 0, "jeff_test_fmt")
    jeff_wrong = (~jeff["is_oos"] & (jeff["pred_label"] != jeff["label"])) | (jeff["is_oos"] & (jeff["confidence"] >= threshold))
    sheet = pd.concat([as_sheet(b77, "banking77_test"), as_sheet(jeff[jeff_wrong], "jeff_old_test")], ignore_index=True)
    sheet = sheet.iloc[np.random.default_rng(0).permutation(len(sheet))].reset_index(drop=True)
    sheet.insert(0, "item_id", range(1, len(sheet) + 1))
    FOLDER.mkdir(exist_ok=True)
    sheet.to_csv(FOLDER / "sheet.csv", index=False)
    owner = sheet.sample(n=N_OWNER_CHECK, random_state=0).sort_values("item_id")
    owner.drop(columns=["confidence"]).assign(category="", note="").to_csv(FOLDER / "owner_check.csv", index=False)
    print(f"Wrote {len(sheet)} rows ({len(b77)} Banking77 of {int(handled_errors('test_fmt', threshold).shape[0])} handled errors, "
          f"{int(jeff_wrong.sum())} Jeff) to results/error_analysis/sheet.csv, and {N_OWNER_CHECK} rows to owner_check.csv.")


def summarise() -> None:
    sheet = pd.read_csv(FOLDER / "sheet.csv", keep_default_na=False)
    claude = pd.read_csv(FOLDER / "claude_categories.csv", keep_default_na=False)
    if sorted(claude["item_id"]) != sorted(sheet["item_id"]) or not claude["category"].isin(list(CATEGORIES)).all():
        sys.exit("claude_categories.csv must have one valid category for every item_id in the sheet.")
    merged = sheet.merge(claude, on="item_id")
    counts = merged.groupby(["source", "category"]).size().unstack(0, fill_value=0).reindex(list(CATEGORIES), fill_value=0)
    rows = [{"category": c, "definition": CATEGORIES[c], **{s: f"{n} ({100 * n / counts[s].sum():.0f}%)" for s, n in counts.loc[c].items()}}
            for c in CATEGORIES]
    parts = ["# M6 error analysis: router errors read by hand",
             "Written by `scripts/make_error_sheet.py --summarise`. Do not edit by hand. Rows: `results/error_analysis/sheet.csv`; "
             "Claude's category and note per row: `results/error_analysis/claude_categories.csv`.",
             "## Which errors",
             "Banking77: 100 errors drawn at random (seed 0) from all queries S2 handles wrongly on format-controlled test at its "
             "frozen policy v1 threshold. Jeff: every in-scope query S2 gets wrong on the old Jeff test split, plus every out-of-scope "
             "query it would handle at the frozen Banking77 threshold.",
             "## Categories (Claude, every row)", md_table(rows)]
    owner_file = FOLDER / "owner_check.csv"
    owner = pd.read_csv(owner_file, keep_default_na=False)
    if (owner["category"] != "").all():
        both = owner[["item_id", "category"]].merge(claude[["item_id", "category"]], on="item_id", suffixes=("_owner", "_claude"))
        agree = int((both["category_owner"] == both["category_claude"]).sum())
        parts += ["## Owner's blind check", f"The owner categorised {len(both)} random rows without seeing Claude's categories. "
                  f"Same category on {agree} of {len(both)}.",
                  md_table(both[both["category_owner"] != both["category_claude"]].to_dict("records")) if agree < len(both) else "No disagreements."]
    else:
        parts += ["## Owner's blind check", "Not done yet: `results/error_analysis/owner_check.csv` has empty categories."]
    (data.RESULTS_DIR / "error_analysis.md").write_text("\n\n".join(parts) + "\n")
    print("Wrote results/error_analysis.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summarise", action="store_true", help="summarise the filled-in categories")
    summarise() if parser.parse_args().summarise else build()
