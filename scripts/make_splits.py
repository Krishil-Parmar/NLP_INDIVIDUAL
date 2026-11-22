"""Create the fixed data splits once (README section 6.2, rigour rule 6).

Run from the project root:
    uv run python scripts/make_splits.py          # refuses if splits already exist
    uv run python scripts/make_splits.py --force  # rebuild (only before any test run)

Steps, in the order of README section 6.2:
    1. load and pin       2. normalise text      3. drop empty rows and duplicates
    4. leakage check      5. validation split    6. token length check
    7. class balance      8. out-of-scope pools (put through steps 2 to 4 as well)

Writes:
    data/processed/banking77_{train,val,test}.csv, oos_{val,test}.csv
    data/processed/label_map.json, removed_rows.csv, token_lengths.json
    results/data_report.md
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from router import data
from router import preprocess as pp
from router.provenance import metadata_markdown, run_metadata
from router.tables import md_table

# Columns written to every split file, in this order.
COLUMNS = ["text", "label", "label_id", "orig_index", "source"]


def check(condition: bool, message: str) -> None:
    """Stop the script with a clear message if an invariant does not hold."""
    if not condition:
        sys.exit(f"CHECK FAILED: {message}")


def main(force: bool = False) -> None:
    cfg = data.load_config("data")
    seed: int = cfg["seed"]
    # Record git state before writing anything, so the report shows the code that ran.
    meta = run_metadata(cfg, seed=seed)

    split_paths = [data.PROCESSED_DIR / name for name in data.SPLIT_FILES.values()]
    if any(p.exists() for p in split_paths) and not force:
        sys.exit(
            "Splits already exist in data/processed/. They are created once and reused "
            "(rigour rule 6). Re-run with --force only if no test-set run has happened."
        )
    data.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    data.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: load and pin -------------------------------------------------
    print("Step 1: loading datasets at pinned revisions")
    train, test, label_names = data.load_banking77(cfg)
    expected = cfg["banking77"]["expected_sizes"]
    check(len(train) == expected["train"], f"Banking77 train has {len(train)} rows, expected {expected['train']}")
    check(len(test) == expected["test"], f"Banking77 test has {len(test)} rows, expected {expected['test']}")
    check(len(label_names) == 77, f"Banking77 has {len(label_names)} intents, expected 77")

    oos_raw = data.load_clinc_oos(cfg)
    for split, n_expected in cfg["clinc150"]["expected_oos_sizes"].items():
        n = len(oos_raw[split])
        check(n == n_expected, f"CLINC150 OOS {split} has {n} rows, expected {n_expected}")
    raw_sizes = {
        "banking77_train": len(train),
        "banking77_test": len(test),
        **{f"clinc_oos_{s}": len(df) for s, df in oos_raw.items()},
    }

    # ---- Step 2: normalise text -----------------------------------------------
    print("Step 2: normalising text")
    changed_by_normalising = {
        "banking77_train": int((pp.normalise_frame(train)["text"] != train["text"]).sum()),
        "banking77_test": int((pp.normalise_frame(test)["text"] != test["text"]).sum()),
    }
    train = pp.normalise_frame(train)
    test = pp.normalise_frame(test)
    oos_raw = {split: pp.normalise_frame(df) for split, df in oos_raw.items()}

    # ---- Step 3: empty rows and duplicates, within each split -----------------
    print("Step 3: dropping empty rows and duplicates")
    removed: list[pd.DataFrame] = []
    train, r = pp.drop_empty_and_duplicates(train)
    removed.append(r)
    test, r = pp.drop_empty_and_duplicates(test)
    removed.append(r)

    # ---- Step 4: leakage check (train vs test) --------------------------------
    print("Step 4: train/test leakage check")
    leaked_texts = pp.overlap_count(train, test)
    train, r = pp.remove_overlap(train, test, reason="leakage")  # removed from train, never test
    removed.append(r)

    # ---- Step 8: out-of-scope pools, through the same steps 2 to 4 ------------
    print("Step 8: building OOS pools")
    oos_val = pd.concat([oos_raw[s] for s in cfg["oos_pools"]["val"]])
    oos_test = pd.concat([oos_raw[s] for s in cfg["oos_pools"]["test"]])
    oos_val, r = pp.drop_empty_and_duplicates(oos_val)
    removed.append(r)
    oos_test, r = pp.drop_empty_and_duplicates(oos_test)
    removed.append(r)
    # An OOS validation query that also appears in the OOS test pool leaks: drop it from val.
    oos_val, r = pp.remove_overlap(oos_val, oos_test, reason="leakage")
    removed.append(r)
    # An "out-of-scope" text that also appears in Banking77 with an intent contradicts
    # itself. Banking77 is the primary benchmark, so the OOS copy is dropped.
    banking_all = pd.concat([train, test])
    oos_val, r = pp.remove_overlap(oos_val, banking_all, reason="in_scope_conflict")
    removed.append(r)
    oos_test, r = pp.remove_overlap(oos_test, banking_all, reason="in_scope_conflict")
    removed.append(r)

    # ---- Step 5: stratified validation split ----------------------------------
    print("Step 5: stratified validation split")
    train, val = pp.stratified_val_split(train, fraction=cfg["val_fraction"], seed=seed)

    removed_all = pd.concat(removed).reset_index(drop=True)

    # ---- Invariants ----------------------------------------------------------
    splits = {"train": train, "val": val, "test": test, "oos_val": oos_val, "oos_test": oos_test}
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            n = pp.overlap_count(splits[a], splits[b])
            check(n == 0, f"{n} texts appear in both {a} and {b}")
    for name in ["train", "val", "test"]:
        n_labels = splits[name]["label"].nunique()
        check(n_labels == 77, f"{name} has {n_labels} intents, expected 77")
    n_removed_train = int((removed_all["source"] == "banking77_train").sum())
    n_removed_test = int((removed_all["source"] == "banking77_test").sum())
    check(len(train) + len(val) + n_removed_train == expected["train"], "train rows do not add up")
    check(len(test) + n_removed_test == expected["test"], "test rows do not add up")

    # ---- Step 6: token length check (fitted on train only) --------------------
    print("Step 6: token length statistics")
    rng = np.random.default_rng(seed)
    train_texts = train["text"].tolist()
    # Random partner for each train query, to measure cross-encoder pair lengths for S3.
    partners = [train_texts[i] for i in rng.permutation(len(train_texts))]
    length_rows = []
    length_json = {}
    for tok_name in cfg["tokenisers"]:
        tok = AutoTokenizer.from_pretrained(tok_name)
        lengths = pp.token_lengths(train_texts, tok)
        stats = pp.length_stats(lengths)
        max_len = pp.choose_max_length(lengths, cfg["length_percentile"])
        pair_lengths = pp.token_lengths(train_texts, tok, pair_texts=partners)
        pair_max_len = pp.choose_max_length(pair_lengths, cfg["length_percentile"])
        trunc = {
            name: pp.truncation_share(pp.token_lengths(splits[name]["text"].tolist(), tok), max_len)
            for name in ["train", "val", "test"]
        }
        length_json[tok_name] = {
            "single": {**stats, "max_length": max_len},
            "pair": {**pp.length_stats(pair_lengths), "max_length": pair_max_len},
            "truncated_share": trunc,
        }
        length_rows.append(
            {
                "tokeniser": f"`{tok_name}`",
                "mean": f"{stats['mean']:.1f}",
                "p50": f"{stats['p50']:.0f}",
                "p95": f"{stats['p95']:.0f}",
                "p99": f"{stats['p99']:.0f}",
                "max": stats["max"],
                "max_length": max_len,
                "truncated train / val / test": " / ".join(f"{trunc[s]:.2%}" for s in ["train", "val", "test"]),
                "pair max_length (S3)": pair_max_len,
            }
        )

    # ---- Step 7: class balance -----------------------------------------------
    print("Step 7: class balance")
    balance = pd.DataFrame({name: pp.class_balance(splits[name]) for name in ["train", "val", "test"]})
    balance = balance.fillna(0).astype(int).sort_values("train")

    # ---- Save ----------------------------------------------------------------
    print("Saving splits")
    for name, df in splits.items():
        df[COLUMNS].to_csv(data.PROCESSED_DIR / data.SPLIT_FILES[name], index=False)
    data.save_label_map(label_names)
    removed_all[COLUMNS + ["reason"]].to_csv(data.PROCESSED_DIR / data.REMOVED_ROWS_FILE, index=False)
    with open(data.PROCESSED_DIR / "token_lengths.json", "w") as f:
        json.dump(length_json, f, indent=2)

    # ---- Report ----------------------------------------------------------------
    removed_counts = (
        removed_all.groupby(["source", "reason"]).size().unstack(fill_value=0)
        if len(removed_all)
        else pd.DataFrame()
    )
    removed_rows = []
    for source in raw_sizes:
        row = {"source": source, "raw rows": raw_sizes[source]}
        for reason in ["empty", "duplicate", "conflict", "leakage", "in_scope_conflict"]:
            has = source in removed_counts.index and reason in removed_counts.columns
            row[reason] = int(removed_counts.loc[source, reason]) if has else 0
        removed_rows.append(row)

    size_rows = [{"split": name, "rows": len(df)} for name, df in splits.items()]
    val_oos_share = len(oos_val) / (len(val) + len(oos_val))
    test_oos_share = len(oos_test) / (len(test) + len(oos_test))

    def balance_summary(col: str) -> str:
        counts = balance[col]
        return f"min {counts.min()}, median {counts.median():.0f}, max {counts.max()}"

    report = f"""# Data report

Written by `scripts/make_splits.py`. Do not edit by hand.

## Provenance

{metadata_markdown(meta)}

## Final split sizes

{md_table(size_rows)}

Validation is a stratified {cfg['val_fraction']:.0%} split of the cleaned Banking77 train set (seed {seed}).

Out-of-scope share when mixed (step 8): validation {val_oos_share:.1%} ({len(oos_val)} OOS of {len(val) + len(oos_val)}), test {test_oos_share:.1%} ({len(oos_test)} OOS of {len(test) + len(oos_test)}). Validation OOS comes from CLINC150 OOS {' + '.join(cfg['oos_pools']['val'])}, test OOS from CLINC150 OOS {' + '.join(cfg['oos_pools']['test'])}.

## Rows removed, by step

Changed by normalisation (step 2, not removed): train {changed_by_normalising['banking77_train']}, test {changed_by_normalising['banking77_test']}.

{md_table(removed_rows)}

Train/test leakage (step 4): {leaked_texts} distinct texts appeared in both train and test after normalisation. They were removed from train only. Every removed row is listed in `data/processed/removed_rows.csv` with its reason.

## Token lengths (step 6, measured on train)

`max_length` is the {cfg['length_percentile']}th percentile of train lengths rounded up to a multiple of 8. "Pair" is a query plus a random other train query, which is what the S3 cross-encoder sees. Full numbers are in `data/processed/token_lengths.json`.

{md_table(length_rows)}

## Class balance (step 7)

Queries per intent: train {balance_summary('train')}; val {balance_summary('val')}; test {balance_summary('test')}.

The intents are uneven in train, so macro averages (each intent counts equally) are the primary metric (README section 8.1).

<details><summary>Full table (77 intents, fewest train queries first)</summary>

{md_table([{"intent": i, **{c: int(balance.loc[i, c]) for c in balance.columns}} for i in balance.index])}

</details>
"""
    (data.RESULTS_DIR / "data_report.md").write_text(report)

    print("\nDone.")
    for row in size_rows:
        print(f"  {row['split']:>8}: {row['rows']}")
    print(f"  removed rows: {len(removed_all)} (see data/processed/removed_rows.csv)")
    print(f"  train/test leaked texts: {leaked_texts}")
    print("  report: results/data_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="overwrite existing splits")
    main(force=parser.parse_args().force)
