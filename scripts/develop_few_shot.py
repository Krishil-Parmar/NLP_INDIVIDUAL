"""E8 development: try every candidate fix for S2 on development data and choose one per tier.

Run from the project root:
    uv run python scripts/develop_few_shot.py            # full run
    uv run python scripts/develop_few_shot.py --smoke    # plumbing check: 1 seed, tiny fine-tune, nothing saved

Post-hoc experiment, not in the pre-registration. Design and selection rule: configs/e8_few_shot.yaml.

Data:
  - Jeff development data: the old Jeff test split (180 queries, format-controlled as in
    E7), now development data because its errors were read in the diagnosis. Support:
    the 140 Jeff support examples, plus the 14 intent definitions for "descriptions".
    Matched risk: few-shot S4's error on these queries (results/e7_jeff.json).
  - Banking77 validation check (report only): 10 random support examples per intent,
    five draws, at few-shot S4's validation error (results/policy_v1.json).

Checks: the S2 variant (frozen/examples/nearest) must reproduce the saved E7 numbers
exactly, or the script stops.

Writes results/e8_dev.json and results/e8_dev.md, and saves the fine-tuned Jeff
encoders to checkpoints/e8/ (one per support option and seed) so the final test uses
exactly these models.
"""

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from router import data, few_shot, jeff
from router.metrics import coverage_at_risk
from router.preprocess import format_control
from router.provenance import metadata_markdown, run_metadata
from router.seed import pick_device
from router.systems.s2_embedding import load_encoder
from router.tables import md_table

CHECKPOINTS = data.PROJECT_ROOT / "checkpoints" / "e8"
S2_TAG = "s2_all-MiniLM-L6-v2"


def jeff_dev() -> tuple[pd.DataFrame, list[str], dict[str, pd.DataFrame], np.ndarray, np.ndarray]:
    """(dev queries, intent names, {support option: support set}, true ids, is_oos) for the Jeff development data."""
    cfg = data.load_config("jeff")
    support, names = jeff.jeff_support()
    definitions = {intent["name"]: intent["description"] for intent in cfg["intents"]}  # the definition only, not "not this"
    supports = {"examples": support[["text", "label_id"]], "descriptions": few_shot.add_descriptions(support, definitions, names)}
    dev = jeff.load_jeff("test")
    dev = dev.assign(text=dev["text"].map(format_control))
    is_oos = dev["is_oos"].to_numpy(dtype=bool)
    true_ids = np.array([names.index(l) if not o else -1 for l, o in zip(dev["label"], is_oos)])
    return dev, names, supports, true_ids, is_oos


def banking77_val() -> tuple[list[str], pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    """(texts, train split, intent names, true ids, is_oos) for the Banking77 validation check."""
    label_map = data.load_label_map()
    names = sorted(label_map, key=label_map.get)
    eval_df = data.load_eval_set("val", with_oos=True)
    is_oos = eval_df["is_oos"].to_numpy(dtype=bool)
    true_ids = np.where(is_oos, -1, eval_df["label"].map(label_map).fillna(-1).to_numpy()).astype(int)
    return eval_df["text"].map(format_control).tolist(), data.load_train(), names, true_ids, is_oos


def run_all_scorings(encoder, support, texts, names, true_ids, is_oos, cfg, seed, r_star, encoder_kind, support_kind):
    """Run the three scoring rules with one encoder. Returns one row per variant."""
    rows = []
    for scoring in ["nearest", "margin", "head"]:
        variant = few_shot.Variant(encoder_kind, support_kind, scoring)
        pred, conf = few_shot.predict_variant(scoring, encoder, support, texts, len(names), cfg, seed)
        metrics = few_shot.run_metrics(pred, conf, true_ids, is_oos, names, r_star, cfg["fixed_risks"])
        correct = (pred == true_ids) & ~is_oos
        threshold = coverage_at_risk(correct, conf, r_star)["threshold"]  # the deployable threshold of this run
        rows.append({"variant": variant.name, "tier": variant.tier, "seed": seed, **metrics, "dev_threshold": threshold})
    return rows


def summarise(rows: list[dict]) -> pd.DataFrame:
    """Mean and standard deviation over seeds (or draws) of every metric, one row per variant."""
    df = pd.DataFrame(rows)
    metrics = [c for c in df.columns if c not in ("variant", "tier", "seed", "dev_threshold")]
    mean = df.groupby(["variant", "tier"], sort=False)[metrics].mean()
    std = df.groupby(["variant", "tier"], sort=False)[metrics].std(ddof=0).add_suffix("_std")
    return pd.concat([mean, std], axis=1).reset_index()


def check_reproduces_e7(rows: list[dict]) -> None:
    """The S2 variant must give exactly the saved E7 numbers, or the variant code differs from S2."""
    saved = json.loads((data.RESULTS_DIR / "e7_jeff.json").read_text())["systems"][S2_TAG]["per_seed"][0]
    ours = next(r for r in rows if r["variant"] == "frozen/examples/nearest")
    for key in ["in_scope_accuracy", "macro_f1", "oos_auroc", "aurc", "coverage_at_matched_risk"]:
        if abs(ours[key] - saved[key]) > 1e-9:
            sys.exit(f"frozen/examples/nearest gives {key}={ours[key]}, but E7 saved {saved[key]}: the variant code is not S2.")
    print("Check passed: frozen/examples/nearest reproduces the saved E7 numbers.")


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def table_rows(summary: pd.DataFrame, r_star: float, chosen: set[str] | None = None) -> list[dict]:
    out = []
    for _, r in summary.iterrows():
        out.append({
            "variant": r["variant"] + (" **(chosen)**" if chosen and r["variant"] in chosen else ""), "tier": r["tier"],
            "in-scope accuracy (%)": f"{pct(r['in_scope_accuracy'])} ± {pct(r['in_scope_accuracy_std'])}",
            "confidence AUROC (%)": pct(r["confidence_auroc"]), "OOS AUROC (%)": pct(r["oos_auroc"]),
            "AURC (%)": f"{pct(r['aurc'])} ± {pct(r['aurc_std'])}",
            f"coverage at {pct(r_star)}% risk (%)": pct(r["coverage_at_matched_risk"]),
            "coverage at 1% (%)": pct(r["coverage_at_0.01"]), "coverage at 5% (%)": pct(r["coverage_at_0.05"]),
        })
    return out


def report(result: dict) -> str:
    jeff_summary = pd.DataFrame(result["jeff"]["summary"])
    b77_summary = pd.DataFrame(result["banking77_check"]["summary"])
    chosen = {tier: name for tier, name in result["chosen"].items()}
    chosen_rows = [{"tier": t, "chosen variant": n, "dev AURC (%)": pct(jeff_summary.set_index("variant").loc[n, "aurc"])}
                   for t, n in chosen.items()]
    return "\n\n".join([
        "# E8 development: candidate fixes for S2 on a small new intent set",
        "Written by `scripts/develop_few_shot.py`. Do not edit by hand. **Post-hoc experiment, not pre-registered.** "
        "The pre-registered RQ6 verdict (not met) is unchanged.",
        "## Provenance", metadata_markdown(result["metadata"]),
        "## Setup",
        "Variants: encoder (frozen MiniLM, or fine-tuned on the support set SetFit-style) x support (the 140 examples, "
        "or those plus the 14 intent definitions) x scoring (nearest-example cosine as S2, top-two margin, or a "
        "logistic-regression head). Fine-tuned variants: five seeds, mean ± standard deviation. "
        f"Jeff development data: the old Jeff test split ({result['jeff']['n_queries']} queries, {result['jeff']['n_oos']} out of scope), "
        f"read during the diagnosis and therefore no longer used as a test. Matched risk: few-shot S4's error there, {pct(result['jeff']['r_star'])}%.",
        f"Check: frozen/examples/nearest (S2 itself) reproduces the saved E7 numbers exactly.",
        "## Jeff development data",
        md_table(table_rows(jeff_summary, result["jeff"]["r_star"], set(chosen.values()))),
        "## Chosen variants (rule fixed in `configs/e8_few_shot.yaml` before this run)",
        "Per tier, the lowest mean AURC; a simpler variant of the same tier within 0.5 AURC points wins instead.",
        md_table(chosen_rows),
        "## Banking77 validation check (report only, not used for selection)",
        f"10 random support examples per intent, five draws (fine-tuning seed = draw). Matched risk: few-shot S4's validation error, "
        f"{pct(result['banking77_check']['r_star'])}%. Descriptions do not exist for Banking77, so only the examples support is run.",
        md_table(table_rows(b77_summary, result["banking77_check"]["r_star"])),
    ]) + "\n"


def main(smoke: bool) -> None:
    cfg = data.load_config("e8_few_shot")
    if smoke:
        cfg = {**cfg, "seeds": [0], "fine_tune": {**cfg["fine_tune"], "pair_iterations": 1},
               "banking77_check": {**cfg["banking77_check"], "draws": [0]}}
    device = pick_device()
    frozen = load_encoder(cfg["encoder_model"], device)

    # Jeff development data
    dev, names, supports, true_ids, is_oos = jeff_dev()
    texts = dev["text"].tolist()
    r_star = json.loads((data.RESULTS_DIR / "e7_jeff.json").read_text())["matched_risk"]
    jeff_rows = []
    for support_kind in cfg["supports"]:
        support = supports[support_kind]
        jeff_rows += run_all_scorings(frozen, support, texts, names, true_ids, is_oos, cfg, 0, r_star, "frozen", support_kind)
        for seed in cfg["seeds"]:
            start = time.perf_counter()
            tuned = few_shot.fine_tune(cfg["encoder_model"], support["text"].tolist(), support["label_id"].to_numpy(), cfg, seed, device)
            print(f"Jeff {support_kind} seed {seed}: fine-tuned in {time.perf_counter() - start:.0f} s")
            if not smoke:
                tuned.save(str(CHECKPOINTS / "jeff" / support_kind / f"seed{seed}"))
            jeff_rows += run_all_scorings(tuned, support, texts, names, true_ids, is_oos, cfg, seed, r_star, "tuned", support_kind)
    check_reproduces_e7(jeff_rows)
    jeff_summary = summarise(jeff_rows)
    mean_aurc = {few_shot.Variant(*v.split("/")): a for v, a in zip(jeff_summary["variant"], jeff_summary["aurc"])}
    chosen = few_shot.select_per_tier(mean_aurc, cfg["selection"]["simpler_within"])

    # Banking77 validation check
    b77 = cfg["banking77_check"]
    b_texts, train, b_names, b_true, b_oos = banking77_val()
    b_r_star = json.loads((data.RESULTS_DIR / "policy_v1.json").read_text())["matched_risk"]
    b_rows = []
    for draw in b77["draws"]:
        support = few_shot.per_intent_sample(train, b77["per_intent"], draw)[["text", "label_id"]]
        b_rows += run_all_scorings(frozen, support, b_texts, b_names, b_true, b_oos, cfg, draw, b_r_star, "frozen", "examples")
        start = time.perf_counter()
        tuned = few_shot.fine_tune(cfg["encoder_model"], support["text"].tolist(), support["label_id"].to_numpy(), cfg, draw, device)
        print(f"Banking77 draw {draw}: fine-tuned in {time.perf_counter() - start:.0f} s")
        b_rows += run_all_scorings(tuned, support, b_texts, b_names, b_true, b_oos, cfg, draw, b_r_star, "tuned", "examples")

    result = {
        "metadata": run_metadata(cfg),
        "smoke": smoke,
        "jeff": {"n_queries": len(dev), "n_oos": int(is_oos.sum()), "r_star": r_star, "per_run": jeff_rows,
                 "summary": jeff_summary.to_dict("records")},
        "chosen": {tier: v.name for tier, v in chosen.items()},
        "banking77_check": {"r_star": b_r_star, "per_run": b_rows, "summary": summarise(b_rows).to_dict("records")},
    }
    if smoke:
        print("Smoke run finished; nothing written.")
        print(pd.DataFrame(jeff_rows)[["variant", "in_scope_accuracy", "aurc", "coverage_at_0.05"]].to_string())
        return
    (data.RESULTS_DIR / "e8_dev.json").write_text(json.dumps(result, indent=2, default=float))
    (data.RESULTS_DIR / "e8_dev.md").write_text(report(result))
    print("Chosen:", result["chosen"])
    print("Wrote results/e8_dev.json and results/e8_dev.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="1 seed, 1 pair iteration, 1 draw; writes nothing")
    main(parser.parse_args().smoke)
