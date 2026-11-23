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

from router import data

OOS_LABEL = "oos"
OOS_KINDS = {"unrelated": "oos_unrelated", "finance-adjacent": "oos_finance_adjacent"}  # notes value -> config key


def intent_names(cfg: dict) -> list[str]:
    """The in-scope intent names, in config order. A name's position is its label id."""
    return [intent["name"] for intent in cfg["intents"]]


def intent_descriptions(cfg: dict) -> dict[str, str]:
    """{intent name: its definition}, the text S4's Jeff prompt shows next to each name."""
    return {intent["name"]: f"{intent['description']} Not this: {intent['not_this']}" for intent in cfg["intents"]}


def oos_description(cfg: dict) -> str:
    """What counts as out of scope, in the words the writer and the checker were given."""
    return f"{cfg['oos']['unrelated']} Or {cfg['oos']['finance_adjacent'][0].lower()}{cfg['oos']['finance_adjacent'][1:]}"


def load_jeff(split: str) -> pd.DataFrame:
    """One split of the Jeff set ("support" or "test") with columns id, text, label, is_oos,
    the layout router.systems.base.save_predictions expects. Raw text."""
    cfg = data.load_config("jeff")
    df = pd.read_csv(data.PROJECT_ROOT / cfg["file"], keep_default_na=False)
    part = df[df["split"] == split]
    if part.empty:
        raise ValueError(f"no Jeff rows with split {split!r}")
    return pd.DataFrame({
        "id": part["id"].to_numpy(), "text": part["text"].to_numpy(), "label": part["intent"].to_numpy(),
        "is_oos": (part["intent"] == OOS_LABEL).to_numpy(),
    })


def jeff_support() -> tuple[pd.DataFrame, list[str]]:
    """(support split with a label_id column, intent names in id order): what S2 and S3 match against in E7."""
    names = intent_names(data.load_config("jeff"))
    support = load_jeff("support")
    support["label_id"] = support["label"].map({name: i for i, name in enumerate(names)})
    if support["label_id"].isna().any():
        raise ValueError("a support row has a label that is not one of the Jeff intents")
    support["label_id"] = support["label_id"].astype(int)
    return support, names


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


def routing_row(correct: np.ndarray, confidence: np.ndarray, is_oos: np.ndarray, r_star: float,
                frozen_threshold: float, fixed_risks: list[float]) -> dict[str, float]:
    """E7 routing metrics for one run (one system, one seed) on the Jeff test split.

    Two views of policy v1: the threshold recomputed on these queries at the
    matched risk r_star (the RQ6 number, as in RQ1), and the threshold frozen on
    Banking77 validation applied unchanged (what a deployed router would do).
    """
    from router.metrics import aurc, coverage_at_risk, risk_coverage_curve  # here to keep the module light

    at_matched = coverage_at_risk(correct, confidence, r_star)
    handled = confidence >= frozen_threshold
    row = {
        "coverage_at_matched_risk": at_matched["coverage"],
        "selective_risk_at_matched_risk": at_matched["risk"],
        "coverage_frozen_threshold": float(handled.mean()),
        "selective_risk_frozen_threshold": float((~correct[handled]).mean()) if handled.any() else 0.0,
        "oos_handled_frozen_threshold": float((handled & is_oos).sum()),
        "aurc": aurc(risk_coverage_curve(correct, confidence)),
    }
    for risk in fixed_risks:
        row[f"coverage_at_{risk:g}"] = coverage_at_risk(correct, confidence, risk)["coverage"]
    return row


def label_agreement(queries: pd.DataFrame, check: pd.DataFrame) -> dict:
    """How often a blind checker's label matches the writer's, by query id.

    queries has columns id, intent (the writer's label); check has id, label and
    optionally sure ("yes" or "no": whether the checker thought a careful reader
    could pick another label). Unsure items are listed even when the labels agree.
    """
    check = check if "sure" in check.columns else check.assign(sure="yes")
    merged = queries[["id", "text", "intent", "split"]].merge(check[["id", "label", "sure"]], on="id", how="inner")
    if len(merged) != len(queries):
        raise ValueError(f"the check covers {len(merged)} of {len(queries)} queries")
    same = merged["intent"] == merged["label"]
    per_label = merged.assign(same=same).groupby("intent")["same"].agg(["sum", "count"])
    columns = {"intent": "writer_label", "label": "checker_label"}
    return {
        "n_queries": len(merged),
        "agree": int(same.sum()),
        "per_label": {k: {"agree": int(r["sum"]), "n": int(r["count"])} for k, r in per_label.iterrows()},
        "disagreements": merged.loc[~same, ["id", "split", "text", "intent", "label"]].rename(columns=columns).to_dict("records"),
        "unsure": merged.loc[merged["sure"] == "no", ["id", "split", "text", "intent", "label"]].rename(columns=columns).to_dict("records"),
    }
