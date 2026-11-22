"""Unit tests for the Jeff set checks, split and label agreement, on toy data."""

import pandas as pd

from router import jeff

CFG = {
    "intents": [{"name": "safe_to_spend"}, {"name": "upcoming_bills"}],
    "queries_per_intent": 4, "support_per_intent": 2, "oos_unrelated": 1, "oos_finance_adjacent": 1, "split_seed": 0,
}


def toy_written() -> pd.DataFrame:
    rows = [(f"spend question {i}", "safe_to_spend", "") for i in range(4)]
    rows += [(f"bills question {i}", "upcoming_bills", "") for i in range(4)]
    rows += [("footy scores", "oos", "unrelated"), ("pay my rent", "oos", "finance-adjacent")]
    df = pd.DataFrame(rows, columns=["text", "intent", "notes"])
    df.insert(0, "id", range(1, len(df) + 1))
    return df


def test_a_correct_file_has_no_problems():
    assert jeff.check_written(toy_written(), CFG) == []


def test_check_written_catches_counts_labels_and_duplicates():
    df = toy_written()
    df.loc[0, "intent"] = "make_transfer"  # unknown, and safe_to_spend drops to 3
    df.loc[5, "text"] = "Bills  question 0"  # same as row 4 ("bills question 0") once case and spaces are ignored
    df.loc[9, "notes"] = "other"
    problems = jeff.check_written(df, CFG)
    joined = " ".join(problems)
    assert "make_transfer" in joined and "safe_to_spend: 3" in joined
    assert "duplicate" in joined and "finance-adjacent: 0" in joined


def test_assign_splits_counts_and_oos_is_test_only():
    out = jeff.assign_splits(toy_written(), support_per_intent=2, seed=0)
    support = out[out["split"] == "support"]
    assert support["intent"].value_counts().to_dict() == {"safe_to_spend": 2, "upcoming_bills": 2}
    assert (out.loc[out["intent"] == "oos", "split"] == "test").all()
    again = jeff.assign_splits(toy_written(), support_per_intent=2, seed=0)
    pd.testing.assert_frame_equal(out, again)  # same seed, same split


def test_label_agreement_counts_and_lists_disagreements():
    queries = jeff.assign_splits(toy_written(), support_per_intent=2, seed=0)
    check = queries[["id"]].assign(label=queries["intent"])
    check.loc[check["id"] == 1, "label"] = "upcoming_bills"
    agreement = jeff.label_agreement(queries, check)
    assert (agreement["n_queries"], agreement["agree"]) == (10, 9)
    assert agreement["per_label"]["safe_to_spend"] == {"agree": 3, "n": 4}
    assert [(d["id"], d["writer_label"], d["checker_label"]) for d in agreement["disagreements"]] == [
        (1, "safe_to_spend", "upcoming_bills")]
