import numpy as np
import pandas as pd

from scripts import evaluate_construct_weighted_candidates as construct


def candidate_frame():
    return pd.DataFrame(
        {
            "record_id": ["a", "a", "a", "b"],
            "core": [0.9, 0.5, 0.1, 0.8],
            "rank_virus_foreign_like": [0.2, 0.9, 0.5, 0.5],
        }
    )


def test_steep_weights_reduce_to_best_candidate_and_unweighted_sum():
    frame = candidate_frame()
    best = construct.construct_scores(frame, None, 0.0, 0.0, 1.0)
    total = construct.construct_scores(frame, None, 0.0, 1.0, 1.0)
    assert best["a"] == 0.9 and best["b"] == 0.8
    assert np.isclose(total["a"], 1.5)


def test_weights_decay_geometrically_by_rank():
    score = construct.construct_scores(candidate_frame(), None, 0.0, 0.5, 1.0)
    assert np.isclose(score["a"], 0.9 + 0.5 * 0.5 + 0.25 * 0.1)


def test_weirdo_term_is_additive_and_centred():
    frame = candidate_frame()
    base = construct.construct_scores(frame, None, 0.0, 0.0, 1.0)
    with_weirdo = construct.construct_scores(frame, "virus_foreign_like", 0.2, 0.0, 1.0)
    assert np.isclose(with_weirdo["a"], 0.9 + 0.2 * (0.2 - 0.5))
    assert np.isclose(with_weirdo["b"], base["b"])


def test_gamma_sharpens_the_score_dropoff():
    flat = construct.construct_scores(candidate_frame(), None, 0.0, 1.0, 1.0)
    sharp = construct.construct_scores(candidate_frame(), None, 0.0, 1.0, 6.0)
    assert sharp["a"] / sharp["b"] > flat["a"] / flat["b"]
