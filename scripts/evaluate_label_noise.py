"""E2 step 2: does label noise change the conclusions? (RQ4)

Run from the project root, after scripts/find_label_issues.py and the cleaned
training runs (scripts/train.py --labels cleaned):
    uv run python scripts/evaluate_label_noise.py --split val

Definitions (README section 8.1, fixed before any cleaned run):
  - Main table, S0 to S3, in-scope queries, raw text. Four cells:
      train labels {original, cleaned} x evaluation rows {all, flagged removed}.
    Macro-F1 (E1's primary metric) and accuracy, mean and std over seeds.
  - Ranking: systems ordered by mean macro-F1 in each cell. "Stable" means the
    same order in all four cells.
  - Training effect: cleaned minus original macro-F1 per system on all rows,
    paired bootstrap stratified by intent (router.metrics.paired_bootstrap_diff).
  - S4 evaluation effect: S4's in-scope accuracy and mixed-set error rate (the
    E6 matched risk) on format-controlled validation, all rows versus flagged
    rows removed. From cached replies only: no API calls.
  - Audit: scored once every verdict in results/e2_audit.csv is filled in, for
    all items together and for each annotator (owner, Claude judge) on their own
    items, plus the owner-vs-judge agreement on the items both judged.

Writes results/e2_<split>.json and results/e2_<split>.md.
Test splits are refused until results/preregistration.md exists (rigour rule 1).
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from router import data, label_noise
from router.metrics import classification_metrics, mean_std_over_seeds, paired_bootstrap_diff
from router.preprocess import format_control
from router.provenance import metadata_markdown, run_metadata
from router.systems import s4_llm
from router.systems.base import label_names_in_id_order, list_runs, load_predictions
from router.tables import md_table

CELLS = [("original", "all"), ("cleaned", "all"), ("original", "flagged removed"), ("cleaned", "flagged removed")]


def load_runs(tag: str, seeds: list[int], split: str, eval_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Each seed's predictions, after checking they line up row for row with eval_df."""
    runs = {}
    for seed in seeds:
        preds, _ = load_predictions(tag, seed, split)
        expected = eval_df["text"].map(format_control) if split.endswith("_fmt") else eval_df["text"]
        if len(preds) != len(eval_df) or not (preds["text"].to_numpy() == expected.to_numpy()).all():
            sys.exit(f"{tag} seed {seed}: predictions do not line up with the {split} rows")
        runs[seed] = preds
    return runs


def cell_metrics(runs: dict[int, pd.DataFrame], keep: np.ndarray, label_names: list[str]) -> dict:
    """Mean and std over seeds of E1's metrics on the in-scope rows where keep is True."""
    per_seed = [classification_metrics(df.loc[keep, "label"].tolist(), df.loc[keep, "pred_label"].tolist(), labels=label_names)
                for df in runs.values()]
    return mean_std_over_seeds(per_seed)


def training_effect(original: dict, cleaned: dict, in_scope: np.ndarray, label_names: list[str], boot: dict) -> dict:
    """Paired bootstrap of (cleaned - original) macro-F1 on all in-scope rows, stratified by intent."""
    true = next(iter(original.values()))["label"].to_numpy()[in_scope]

    def macro_f1(runs: dict):
        preds = [df["pred_label"].to_numpy()[in_scope] for df in runs.values()]

        def metric(rows: np.ndarray) -> float:  # mean over seeds, as in the main table
            return float(np.mean([f1_score(true[rows], p[rows], labels=label_names, average="macro", zero_division=0) for p in preds]))

        return metric

    return paired_bootstrap_diff(macro_f1(cleaned), macro_f1(original), strata=true,
                                 n_resamples=boot["resamples"], seed=boot["seed"], ci=boot["ci"])


def ranking(results: dict, cell: str) -> list[str]:
    """System names, best first, by mean macro-F1 in one cell."""
    return sorted(results, key=lambda name: -results[name]["cells"][cell]["macro_f1"]["mean"])


def s4_effect(llm_systems: list[dict], flagged_mixed: np.ndarray, eval_df: pd.DataFrame) -> list[dict]:
    """S4 on format-controlled validation, all rows versus flagged rows removed (cached replies only)."""
    out = []
    is_oos = eval_df["is_oos"].to_numpy(dtype=bool)
    for system in llm_systems:
        if 0 not in list_runs("val_fmt").get(system["tag"], []):
            continue
        df = load_runs(system["tag"], [0], "val_fmt", eval_df)[0]
        wrong = (df["pred_label"] != df["label"]).to_numpy()
        row = {"name": system["name"], "tag": system["tag"]}
        for label, keep in [("all", np.ones(len(df), dtype=bool)), ("flagged removed", ~flagged_mixed)]:
            row[label] = {"n_rows": int(keep.sum()), "error_rate": float(wrong[keep].mean()),
                          "in_scope_accuracy": float(1 - wrong[keep & ~is_oos].mean())}
        out.append(row)
    return out


def fewshot_examples_flagged(flagged: pd.DataFrame, label_names: list[str]) -> dict:
    """How many of few-shot S4's prompt examples are flagged train rows."""
    cfg = data.load_config("s4_llm_fewshot")
    examples = s4_llm.pick_examples(data.load_train(), label_names, cfg["examples_per_intent"], cfg["example_seed"])
    flagged_texts = {format_control(t) for t in flagged.loc[flagged["split"] == "train", "text"]}
    hits = [{"intent": name, "example": text} for name, texts in examples.items() for text in texts if text in flagged_texts]
    return {"n_examples": sum(len(t) for t in examples.values()), "n_flagged": len(hits), "flagged": hits}


def pct(summary: dict, key: str) -> str:
    return f"{100 * summary[key]['mean']:.2f} ± {100 * summary[key]['std']:.2f}"


def write_markdown(out: dict, meta: dict) -> str:
    split, results = out["split"], out["systems"]
    cell_names = list(out["rankings"])  # four cells, or two when no evaluated row is flagged (test)
    main = [{"system": name, "seeds": r["seeds"], **{c: pct(r["cells"][c], "macro_f1") for c in cell_names}} for name, r in results.items()]
    acc = [{"system": name, **{c: pct(r["cells"][c], "accuracy") for c in cell_names}} for name, r in results.items()]
    ranks = [{"cell": c, "ranking (best first, by mean macro-F1)": " > ".join(out["rankings"][c])} for c in cell_names]
    effect = [{"system": name, "cleaned - original macro-F1 (points)": f"{100 * r['training_effect']['diff']:+.2f}",
               "95% CI": f"[{100 * r['training_effect']['ci_low']:+.2f}, {100 * r['training_effect']['ci_high']:+.2f}]"}
              for name, r in results.items()]
    s4_rows = [{"system": s["name"], "rows (all / kept)": f"{s['all']['n_rows']} / {s['flagged removed']['n_rows']}",
                "error rate, all": f"{100 * s['all']['error_rate']:.2f}%", "error rate, flagged removed": f"{100 * s['flagged removed']['error_rate']:.2f}%",
                "in-scope accuracy, all": f"{100 * s['all']['in_scope_accuracy']:.2f}%",
                "in-scope accuracy, flagged removed": f"{100 * s['flagged removed']['in_scope_accuracy']:.2f}%"}
               for s in out["s4_evaluation_effect"]]
    few = out["fewshot_examples"]
    parts = [
        f"# E2 results on {split}: label noise",
        "Written by `scripts/evaluate_label_noise.py`. Do not edit by hand.",
        "## Provenance", metadata_markdown(meta),
        "## Setup",
        f"cleanlab flagged {out['n_flagged_train']} train rows and {out['n_flagged_eval']} of the {out['n_eval_in_scope']} "
        f"in-scope {split} rows (`results/e2_cleaner.md`). \"Cleaned\" means those rows are removed, never relabelled. "
        "Cleaned-label systems keep every other choice frozen. Flagged is not the same as wrong: the audit below measures how often a flag is a real error.",
        f"## In-scope macro-F1 ({split}, raw text), percent, mean ± std over seeds",
        md_table(main),
        "## In-scope accuracy, same cells",
        md_table(acc),
        "## Rankings",
        md_table(ranks),
        f"**Ranking stable in all {len(cell_names)} cells: {'yes' if out['ranking_stable'] else 'no'}.**",
        "## Training effect: cleaned minus original train labels, all rows",
        f"Paired bootstrap, stratified by intent, {out['bootstrap']['resamples']} resamples, {int(100 * out['bootstrap']['ci'])}% percentile interval.",
        md_table(effect),
    ]
    if out["n_flagged_eval"] == 0:
        parts += [f"No {split} row was checked for label errors (cleanlab checked train and validation only), so every row is "
                  "scored against its original label and the \"flagged removed\" cells are not reported. S4 has no training "
                  "data, so it has no cleaned version."]
    else:
        parts += [
            "## S4 evaluation effect (format-controlled validation, cached replies, no API calls)",
            "The error rate is on the mixed set (in-scope plus out-of-scope), the same quantity E6 uses as the matched risk. "
            "No flagged row is out of scope, so removing flagged rows only removes in-scope rows.",
            md_table(s4_rows) if s4_rows else "No S4 predictions found.",
        ]
    parts += [
        f"Few-shot S4 prompt examples that are flagged train rows: {few['n_flagged']} of {few['n_examples']}.",
        "## Audit (blind)",
        *audit_markdown(out["audit"], out["audit_by_annotator"], out["audit_agreement"]),
    ]
    return "\n\n".join(parts) + "\n"


def score_the_audit() -> tuple[dict | None, dict, dict | None]:
    """(all items together, each annotator on their own items, owner-vs-judge agreement).

    The agreement compares the owner's verdicts with the Claude judge's verdicts on
    the same items (results/e2_audit_claude.csv), which the judge made without
    seeing the owner's.
    """
    sheet_path = data.RESULTS_DIR / label_noise.AUDIT_FILE
    if not sheet_path.exists():
        return None, {}, None
    sheet = pd.read_csv(sheet_path, keep_default_na=False)
    key = pd.read_csv(data.RESULTS_DIR / label_noise.AUDIT_KEY_FILE, keep_default_na=False)
    judge_path = data.RESULTS_DIR / label_noise.AUDIT_JUDGE_FILE
    agreement = None
    if judge_path.exists() and "annotator" in sheet.columns:
        owner_part = sheet[sheet["annotator"] == label_noise.OWNER]
        agreement = label_noise.audit_agreement(owner_part, pd.read_csv(judge_path, keep_default_na=False), key)
    return label_noise.score_audit(sheet, key), label_noise.score_audit_by_annotator(sheet, key), agreement


def audit_markdown(audit: dict | None, by_annotator: dict, agreement: dict | None) -> list[str]:
    """The audit section of e2_<split>.md."""
    if audit is None:
        return ["No audit sheet found."]
    if not audit["complete"]:
        return [f"Not scored yet: {audit['n_done']} of {audit['n_items']} items have a verdict."]

    def sentence(a: dict) -> str:
        low, high = a["real_error_ci95"]
        return (f"{a['real_error']} of {a['n_items']} are real label errors "
                f"({100 * a['real_error_share']:.0f}%, 95% Wilson interval {100 * low:.0f} to {100 * high:.0f}%): "
                f"{a['picked_suggestion']} where the suggested label fits better, {a['neither']} where neither fits. "
                f"Ambiguous (both fit): {a['ambiguous']}. False flags (the given label fits): {a['false_flag']}.")

    parts = []
    if by_annotator:
        parts.append(
            "Who judged each item is in the `annotator` column of `results/e2_audit.csv` "
            f"(\"{label_noise.OWNER}\" or \"{label_noise.JUDGE}\"). The Claude judge saw only the sheet (query, two options in random order, "
            "three examples each), never the key or the owner's verdicts; its verdicts on every item are in "
            f"`results/{label_noise.AUDIT_JUDGE_FILE}`. README section 8.1 has the procedure.")
        rows = [{"annotator": name, "items": a["n_items"], "real errors": a["real_error"],
                 "suggestion fits": a["picked_suggestion"], "neither fits": a["neither"],
                 "ambiguous": a["ambiguous"], "false flags": a["false_flag"],
                 "real-error share (95% Wilson)": f"{100 * a['real_error_share']:.0f}% "
                 f"({100 * a['real_error_ci95'][0]:.0f} to {100 * a['real_error_ci95'][1]:.0f}%)"}
                for name, a in by_annotator.items()]
        parts.append(md_table(rows))
    parts.append(f"**All items together:** {sentence(audit)}")
    if agreement is not None:
        n = agreement["n_items"]
        parts.append(f"**Owner and Claude judge on the same {n} items:** same verdict on {agreement['same_verdict']} of {n}, "
                     f"same scored category on {agreement['same_category']} of {n}.")
        if agreement["disagreements"]:
            parts.append(md_table([{"item": d["item"], "owner": f"{d['first_verdict']} ({d['first_category']})",
                                    "Claude judge": f"{d['second_verdict']} ({d['second_category']})"}
                                   for d in agreement["disagreements"]]))
    return parts


def main(split: str) -> None:
    if split.startswith("test") and not (data.RESULTS_DIR / "preregistration.md").exists():
        sys.exit("Refusing to score a test split: results/preregistration.md does not exist yet (rigour rule 1).")
    cfg = data.load_config("e2_label_noise")
    meta = run_metadata({**data.load_config("data"), "e2_label_noise": cfg})
    label_names = label_names_in_id_order()
    flagged = label_noise.load_flagged()
    eval_df = data.load_eval_set(split, with_oos=True)
    in_scope = ~eval_df["is_oos"].to_numpy(dtype=bool)
    flagged_rows = label_noise.is_flagged(eval_df, flagged)
    keep = {"all": in_scope, "flagged removed": in_scope & ~flagged_rows}
    # Test was never checked for label errors, so "flagged removed" would equal "all": report the two "all" cells only.
    cells_used = CELLS if flagged_rows.any() else [cell for cell in CELLS if cell[1] == "all"]

    available = list_runs(split, include_cleaned=True)
    results = {}
    for system in cfg["systems"]:
        tag, clean = system["tag"], label_noise.cleaned_tag(system["tag"])
        if tag not in available or clean not in available:
            sys.exit(f"Missing predictions for {tag} or {clean} on {split}. Train them first.")
        if available[tag] != available[clean]:
            sys.exit(f"{tag} and {clean} have different seeds: {available[tag]} vs {available[clean]}")
        runs = {"original": load_runs(tag, available[tag], split, eval_df), "cleaned": load_runs(clean, available[clean], split, eval_df)}
        cells = {f"{t} train, {e}": cell_metrics(runs[t], keep[e], label_names) for t, e in cells_used}
        effect = training_effect(runs["original"], runs["cleaned"], in_scope, label_names, cfg["bootstrap"])
        results[system["name"]] = {"tag": tag, "cleaned_tag": clean, "seeds": len(available[tag]), "cells": cells, "training_effect": effect}
        print(f"{system['name']}: " + ", ".join(f"{c} {pct(v, 'macro_f1')}" for c, v in cells.items()))

    cell_names = [f"{t} train, {e}" for t, e in cells_used]
    rankings = {c: ranking(results, c) for c in cell_names}
    fmt_split = f"{split}_fmt"
    audit, audit_by_annotator, audit_agreement = score_the_audit()
    out = {
        "split": split,
        "n_eval_in_scope": int(in_scope.sum()), "n_flagged_eval": int(flagged_rows.sum()),
        "n_flagged_train": int((flagged["split"] == "train").sum()),
        "systems": results, "rankings": rankings,
        "ranking_stable": all(r == rankings[cell_names[0]] for r in rankings.values()),
        "bootstrap": cfg["bootstrap"],
        "s4_evaluation_effect": s4_effect(cfg["llm_systems"], flagged_rows, eval_df) if split == "val" else [],
        "s4_split": fmt_split,
        "fewshot_examples": fewshot_examples_flagged(flagged, label_names),
        "audit": audit,
        "audit_by_annotator": audit_by_annotator,
        "audit_agreement": audit_agreement,
    }
    with open(data.RESULTS_DIR / f"e2_{split}.json", "w") as f:
        json.dump({"metadata": meta, **out}, f, indent=2, default=str)
    (data.RESULTS_DIR / f"e2_{split}.md").write_text(write_markdown(out, meta))
    for c in cell_names:
        print(f"{c}: {' > '.join(rankings[c])}")
    print(f"Ranking stable: {out['ranking_stable']}. Wrote results/e2_{split}.json and results/e2_{split}.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    main(parser.parse_args().split)
