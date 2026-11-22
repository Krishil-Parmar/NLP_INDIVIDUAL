"""The Jeff set (README section 6.1): checks, the support/test split, and the label check.

The queries were written by a Claude agent from the intent definitions in
configs/jeff.yaml (data/jeff/jeff_written.csv). This module:
  1. check_written: checks the file against the config (counts, labels, duplicates).
  2. assign_splits: puts a random, seeded share of each intent into the support
     split (the examples a matcher may see) and the rest into test. Out-of-scope
     queries are test only. Chance decides, so no one picks easy examples for
     either side.
  3. label_agreement: compares the writer's labels with a second agent's blind
     labels. The writer's label stays the label; the agreement only measures how
     clear the labels are.
"""

import numpy as np
import pandas as pd

OOS_LABEL = "oos"
OOS_KINDS = {"unrelated": "oos_unrelated", "finance-adjacent": "oos_finance_adjacent"}  # notes value -> config key


def intent_names(cfg: dict) -> list[str]:
    """The in-scope intent names, in config order."""
    return [intent["name"] for intent in cfg["intents"]]


def check_written(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Problems with the written file (empty list: fine)."""
    problems = []
    names = intent_names(cfg)
    unknown = sorted(set(df["intent"]) - set(names) - {OOS_LABEL})
    if unknown:
        problems.append(f"unknown intents: {unknown}")
    counts = df["intent"].value_counts()
    for name in names:
        if counts.get(name, 0) != cfg["queries_per_intent"]:
            problems.append(f"{name}: {counts.get(name, 0)} queries, expected {cfg['queries_per_intent']}")
    oos_notes = df.loc[df["intent"] == OOS_LABEL, "notes"]
    for kind, key in OOS_KINDS.items():
        if int((oos_notes == kind).sum()) != cfg[key]:
            problems.append(f"oos {kind}: {int((oos_notes == kind).sum())}, expected {cfg[key]}")
    if len(oos_notes) != sum(cfg[key] for key in OOS_KINDS.values()):
        problems.append(f"oos rows: {len(oos_notes)}, some with notes other than {list(OOS_KINDS)}")
    if (df["text"].str.strip() == "").any():
        problems.append("empty text")
    normalised = df["text"].str.lower().str.split().str.join(" ")
    if normalised.duplicated().any():
        problems.append(f"duplicate texts: {df.loc[normalised.duplicated(), 'text'].tolist()}")
    return problems


def assign_splits(df: pd.DataFrame, support_per_intent: int, seed: int) -> pd.DataFrame:
    """Copy of df with a "split" column: support_per_intent random rows per intent go to
    "support"; the other in-scope rows and every out-of-scope row go to "test"."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["split"] = "test"
    for intent in sorted(set(out["intent"]) - {OOS_LABEL}):
        rows = out.index[out["intent"] == intent].to_numpy()
        out.loc[rng.choice(rows, size=support_per_intent, replace=False), "split"] = "support"
    return out


def label_agreement(queries: pd.DataFrame, check: pd.DataFrame) -> dict:
    """How often a blind checker's label matches the writer's, by query id.

    queries has columns id, intent (the writer's label); check has id, label.
    """
    merged = queries[["id", "text", "intent", "split"]].merge(check[["id", "label"]], on="id", how="inner")
    if len(merged) != len(queries):
        raise ValueError(f"the check covers {len(merged)} of {len(queries)} queries")
    same = merged["intent"] == merged["label"]
    per_label = merged.assign(same=same).groupby("intent")["same"].agg(["sum", "count"])
    return {
        "n_queries": len(merged),
        "agree": int(same.sum()),
        "per_label": {k: {"agree": int(r["sum"]), "n": int(r["count"])} for k, r in per_label.iterrows()},
        "disagreements": merged.loc[~same, ["id", "split", "text", "intent", "label"]]
                               .rename(columns={"intent": "writer_label", "label": "checker_label"})
                               .to_dict("records"),
    }
