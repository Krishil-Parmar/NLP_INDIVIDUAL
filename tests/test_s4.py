"""Unit tests for S4's prompt, answer parsing, cost, cache and spend cap. No network calls."""

import numpy as np
import pytest

from router.data import OOS_LABEL
from router.systems import s4_llm

LABELS = ["card_arrival", "top_up_failed", "pin_blocked"]
OOS = "out_of_scope"
PRICES = {"input": 3.30, "output": 16.50, "cache_write": 4.125, "cache_read": 0.33}


def test_prompt_lists_every_intent_and_the_oos_answer():
    prompt = s4_llm.build_system_prompt("Pick one:\n{intent_list}\nElse {oos_answer}.", LABELS, OOS)
    for name in LABELS:
        assert f"- {name}" in prompt
    assert "Else out_of_scope." in prompt


def test_schema_allows_exactly_the_intents_plus_oos():
    enum = s4_llm.output_schema(LABELS, OOS)["properties"]["intent"]["enum"]
    assert enum == LABELS + [OOS]


@pytest.mark.parametrize("raw, expected", [
    ('{"intent": "card_arrival"}', "card_arrival"),
    ('{"intent": "out_of_scope"}', OOS_LABEL),
    ("top_up_failed", "top_up_failed"),            # plain-text fallback
    ("  pin_blocked\n", "pin_blocked"),
    ("Sure! card_arrival", s4_llm.INVALID),       # extra words are not accepted
    ('{"intent": "lost_card"}', s4_llm.INVALID),  # not in the list
    ("", s4_llm.INVALID),                         # e.g. a refusal with no text
])
def test_parse_answer(raw, expected):
    assert s4_llm.parse_answer(raw, LABELS, OOS) == expected


def test_call_cost_uses_per_million_prices():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    assert s4_llm.call_cost(usage, PRICES) == pytest.approx(3.30 + 16.50)
    small = {"input_tokens": 700, "output_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
    assert s4_llm.call_cost(small, PRICES) == pytest.approx((700 * 3.30 + 10 * 16.50) / 1e6)


def test_cache_key_changes_with_prompt_and_text_only():
    a = s4_llm.cache_key("model", "prompt", "my card")
    assert a == s4_llm.cache_key("model", "prompt", "my card")
    assert a != s4_llm.cache_key("model", "prompt v2", "my card")
    assert a != s4_llm.cache_key("model", "prompt", "my pin")


def test_spend_tracker_refuses_a_call_that_could_cross_the_cap():
    tracker = s4_llm.SpendTracker(cap_usd=1.0, spent_usd=0.80, first_call_estimate_usd=0.10)
    estimate = tracker.reserve()  # 0.80 + 0.15 <= 1.0
    tracker.settle(estimate, actual=0.10)
    assert tracker.spent == pytest.approx(0.90)
    with pytest.raises(s4_llm.SpendCapReached):
        tracker.reserve()  # 0.90 + 0.15 > 1.0


def _cache_with(tmp_path, prompt, replies):
    cache = s4_llm.ResponseCache(tmp_path / "cache.jsonl")
    for text, raw in replies.items():
        cache.add({"key": s4_llm.cache_key("m", prompt, text), "text": text, "raw": raw, "cost_usd": 0.01})
    return cache


def test_cache_round_trip_and_predictions(tmp_path):
    cfg = {"model_id": "m", "oos_answer": OOS}
    replies = {"where is my card": '{"intent": "card_arrival"}', "what is the weather": '{"intent": "out_of_scope"}'}
    _cache_with(tmp_path, "p", replies)
    reloaded = s4_llm.ResponseCache(tmp_path / "cache.jsonl")  # read back from disk
    assert reloaded.total_cost() == pytest.approx(0.02)
    preds = s4_llm.predictions_from_cache(list(replies), cfg, LABELS, "p", reloaded)
    assert preds.pred_label == ["card_arrival", OOS_LABEL]
    assert np.array_equal(preds.confidence, [1.0, 1.0])
    assert np.array_equal(preds.oos_score, [0.0, 1.0])


def test_predictions_fail_when_a_reply_is_missing(tmp_path):
    cache = _cache_with(tmp_path, "p", {"where is my card": '{"intent": "card_arrival"}'})
    with pytest.raises(KeyError):
        s4_llm.predictions_from_cache(["where is my card", "not cached"], {"model_id": "m", "oos_answer": OOS}, LABELS, "p", cache)


def test_few_shot_prompt_shows_each_intents_examples():
    examples = {"card_arrival": ["where is my card", "card not here"], "top_up_failed": ["top up failed"], "pin_blocked": ["pin locked"]}
    prompt = s4_llm.build_system_prompt("{intent_list}|{oos_answer}", LABELS, OOS, examples)
    assert '- card_arrival (examples: "where is my card"; "card not here")' in prompt


def test_pick_examples_is_seeded_and_format_controlled():
    import pandas as pd

    train = pd.DataFrame({"text": ["Where is my Card?", "Card not here.", "Still waiting!", "Top up failed.", "Why failed?", "PIN locked"],
                          "label": ["card_arrival"] * 3 + ["top_up_failed"] * 2 + ["pin_blocked"]})
    first = s4_llm.pick_examples(train, LABELS, per_intent=1, seed=0)
    assert first == s4_llm.pick_examples(train, LABELS, per_intent=1, seed=0)  # same seed, same picks
    assert all(text == text.lower() and "?" not in text and "." not in text for picks in first.values() for text in picks)
    assert first["pin_blocked"] == ["pin locked"]
