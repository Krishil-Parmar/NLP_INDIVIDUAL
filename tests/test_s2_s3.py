"""Unit tests for the S2 and S3 logic that does not need a downloaded model."""

import numpy as np

from router.systems import s2_embedding, s3_pairwise
from router.systems.s2_embedding import SupportSet


def test_class_scores_take_the_closest_example_of_each_intent():
    # 1 query, 4 support examples: intents [0, 0, 1, 2]
    similarities = np.array([[0.2, 0.9, 0.5, -0.1]])
    scores = s2_embedding.class_scores(similarities, np.array([0, 0, 1, 2]), n_classes=4)
    assert np.allclose(scores[0, :3], [0.9, 0.5, -0.1])
    assert scores[0, 3] == -np.inf  # intent 3 has no examples, so it can never win


def test_candidate_examples_respects_intent_and_per_intent_limits():
    #               idx:   0    1    2    3    4    5
    labels = np.array([0, 0, 0, 1, 1, 2])
    sims = np.array([0.9, 0.8, 0.7, 0.85, 0.1, 0.95])
    # Most similar first: 5 (intent 2), 0 (intent 0), 3 (intent 1), 1 (0), 2 (0), 4 (1)
    chosen = s3_pairwise.candidate_examples(sims, labels, top_intents=2, per_intent=2)
    assert chosen == [5, 0, 1]  # intents 2 and 0 only; intent 2 has a single example
    assert s3_pairwise.candidate_examples(sims, labels, top_intents=2, per_intent=1, exclude=5) == [0, 3]


def _toy_support() -> SupportSet:
    # Three intents with two examples each, embeddings clustered by intent.
    embeddings = np.array([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0], [0.1, 0.99], [-1.0, 0.0], [-0.99, -0.1]])
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    return SupportSet(
        texts=["a1", "a2", "b1", "b2", "c1", "c2"],
        label_ids=np.array([0, 0, 1, 1, 2, 2]),
        embeddings=embeddings,
    )


def test_training_pairs_are_correctly_labelled_and_never_self_pairs():
    support = _toy_support()
    label_of = dict(zip(support.texts, support.label_ids))
    pairs = s3_pairwise.make_training_pairs(support, n_negatives=2, rng=np.random.default_rng(0))
    assert (pairs["text_a"] != pairs["text_b"]).all()
    for _, row in pairs.iterrows():
        same_intent = label_of[row["text_a"]] == label_of[row["text_b"]]
        assert same_intent == (row["label"] == 1.0)
    # Each of the 6 queries gets 1 positive and 2 negatives.
    assert len(pairs) == 6 * 3
    assert (pairs.groupby("text_a")["label"].sum() == 1.0).all()


def test_training_pairs_depend_on_the_seed_only():
    support = _toy_support()
    a = s3_pairwise.make_training_pairs(support, 2, np.random.default_rng(1))
    b = s3_pairwise.make_training_pairs(support, 2, np.random.default_rng(1))
    assert a.equals(b)
