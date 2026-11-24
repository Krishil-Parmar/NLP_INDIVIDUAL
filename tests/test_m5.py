"""Tests for the M5 code: exact split matching, the single-system bootstrap, the S4 prompt
with descriptions, the Jeff loaders, E7 routing metrics, and the test-split guards."""

import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from router import data, jeff
from router.metrics import bootstrap_ci
from router.systems import s4_llm
from router.systems.base import label_names_in_id_order

SCRIPTS = data.PROJECT_ROOT / "scripts"


def load_script(name: str):
    """Import a script from scripts/ as a module, without running it."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_list_runs_matches_the_split_exactly(tmp_path, monkeypatch):
    from router.systems import base

    for tag, files in {"s2_x": ["seed0_test.csv", "seed0_test_fmt.csv"], "s3_y": ["seed0_jeff_test_fmt.csv", "seed1_jeff_test_fmt.csv"]}.items():
        (tmp_path / tag).mkdir()
        for name in files:
            (tmp_path / tag / name).write_text("")
    monkeypatch.setattr(base, "PREDICTIONS_DIR", tmp_path)
    assert base.list_runs("test") == {"s2_x": [0]}  # "seed0_jeff_test_fmt.csv" is not a test file
    assert base.list_runs("test_fmt") == {"s2_x": [0]}
    assert base.list_runs("jeff_test_fmt") == {"s3_y": [0, 1]}


def test_bootstrap_ci_brackets_the_full_sample_value():
    rng = np.random.default_rng(0)
    values = rng.random(200)
    strata = np.repeat([0, 1], 100)
    result = bootstrap_ci(lambda rows: float(values[rows].mean()), strata, n_resamples=300, seed=1)
    assert result["value"] == pytest.approx(values.mean())
    assert result["ci_low"] < result["value"] < result["ci_high"]
    assert bootstrap_ci(lambda rows: float(values[rows].mean()), strata, n_resamples=300, seed=1) == result  # seeded


def test_prompt_lines_with_and_without_descriptions():
    template = "Intents:\n{intent_list}\nElse {oos_answer}."
    plain = s4_llm.build_system_prompt(template, ["a", "b"], "out_of_scope")
    assert plain == "Intents:\n- a\n- b\nElse out_of_scope."
    full = s4_llm.build_system_prompt("{intent_list} | {oos_description}", ["a"], "oos_x", {"a": ["x", "y"]},
                                      {"a": "Does A. Not this: B."}, "anything else")
    assert full == '- a: Does A. Not this: B. (examples: "x"; "y") | anything else'


def test_banking77_few_shot_prompt_still_matches_its_cached_replies():
    """Changing build_system_prompt must not change the Banking77 prompt, or the cache keys would miss."""
    cfg = data.load_config("s4_llm_fewshot")
    names = label_names_in_id_order()
    examples = s4_llm.pick_examples(data.load_split("train"), names, cfg["examples_per_intent"], cfg["example_seed"])
    template = (data.PROJECT_ROOT / cfg["prompt_file"]).read_text(encoding="utf-8")
    prompt = s4_llm.build_system_prompt(template, names, cfg["oos_answer"], examples)
    cache = s4_llm.ResponseCache(data.PROJECT_ROOT / cfg["cache_file"])
    first = next(iter(cache.records.values()))
    assert cache.get(s4_llm.cache_key(cfg["model_id"], prompt, first["text"])) is not None


def test_jeff_support_has_ids_for_the_14_intents():
    support, names = jeff.jeff_support()
    assert len(names) == 14 and len(support) == 140
    assert sorted(support["label_id"].unique()) == list(range(14))
    assert not support["is_oos"].any()
    assert set(support.columns) >= {"id", "text", "label", "is_oos", "label_id"}
    assert "Not this:" in jeff.intent_descriptions(data.load_config("jeff"))["safe_to_spend"]


def test_routing_row_on_toy_queries():
    correct = np.array([True, True, True, False, False])
    confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.95])
    is_oos = np.array([False, False, False, False, True])
    row = jeff.routing_row(correct, confidence, is_oos, r_star=0.25, frozen_threshold=0.75, fixed_risks=[0.01])
    # Highest coverage with risk <= 0.25: handle the top 4 by confidence (0.95 wrong, 0.9, 0.8, 0.7 right).
    assert row["coverage_at_matched_risk"] == pytest.approx(0.8)
    assert row["selective_risk_at_matched_risk"] == pytest.approx(0.25)
    # Frozen threshold 0.75 handles 0.95, 0.9, 0.8: one of three wrong, and it is the out-of-scope one.
    assert row["coverage_frozen_threshold"] == pytest.approx(0.6)
    assert row["selective_risk_frozen_threshold"] == pytest.approx(1 / 3)
    assert row["oos_handled_frozen_threshold"] == 1
    assert row["coverage_at_0.01"] == 0.0  # the most confident query is wrong, so no threshold reaches 1% risk


def test_verdict_rule():
    judge = load_script("report_verdicts").judge
    assert judge(0.65, 0.62, 0.68, 0.60) == "met"
    assert judge(0.61, 0.58, 0.64, 0.60).startswith("met (not clear-cut")
    assert judge(0.55, 0.50, 0.59, 0.60) == "not met"
    assert judge(0.15, 0.10, 0.25, 0.20, higher_is_better=False).startswith("met (not clear-cut")  # a drop: lower is better


@pytest.mark.parametrize("script, call", [
    ("predict", lambda m: m.main("test", "original", False)),
    ("predict_jeff", lambda m: m.main(None)),
    ("evaluate_jeff", lambda m: m.main()),
    ("run_s4", lambda m: m.main("test", None, "s4_jeff")),
    ("run_s4", lambda m: m.main("test_v2", None, "s4_jeff_fewshot")),  # E8: needs preregistration_e8.md
    ("evaluate_few_shot", lambda m: m.main()),
])
def test_every_new_test_path_refuses_without_preregistration(tmp_path, monkeypatch, script, call):
    module = load_script(script)
    monkeypatch.setattr(data, "RESULTS_DIR", tmp_path)  # an empty results folder: no preregistration.md
    with pytest.raises(SystemExit, match="preregistration"):
        call(module)


def test_retrieval_variant_on_test_needs_its_own_plan(tmp_path, monkeypatch):
    module = load_script("run_s4")
    monkeypatch.setattr(data, "RESULTS_DIR", tmp_path)
    (tmp_path / "preregistration.md").write_text("present")  # the main plan exists, the retrieval plan does not
    with pytest.raises(SystemExit, match="plan_llm_bar_test"):
        module.main("test", None, "s4_llm_retrieval")
    (tmp_path / "preregistration_e8.md").write_text("present")  # so the test_v2 call reaches the retrieval check
    with pytest.raises(SystemExit, match="val or test only"):
        module.main("test_v2", None, "s4_llm_retrieval")


def test_retrieval_message_lists_examples_then_query():
    from router.systems.s4_llm import retrieval_message
    message = retrieval_message("where is my card", [("where's my card", "card_arrival"), ("card not here", "card_delivery_estimate")])
    assert message.splitlines()[1] == '1. "where\'s my card" -> card_arrival'
    assert message.splitlines()[-1] == 'Customer message: "where is my card"'
