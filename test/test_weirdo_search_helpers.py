import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts import evaluate_lodo_weirdo_models as lodo
from scripts import evaluate_weirdo_term_null as null


def test_fast_auc_matches_sklearn_including_ties():
    rng = np.random.default_rng(0)
    outcome = rng.integers(0, 2, 200)
    score = np.round(rng.normal(size=200) + outcome * 0.3, 1)
    assert np.isclose(null.fast_auc(outcome, score), roc_auc_score(outcome, score))


def test_sample_weights_equalize_datasets_and_classes():
    frame = pd.DataFrame(
        {
            "dataset": ["a"] * 6 + ["b"] * 2,
            "outcome": [0, 0, 0, 0, 1, 1, 0, 1],
        }
    )
    weights = lodo.sample_weights(frame)
    for dataset in ("a", "b"):
        rows = frame["dataset"].eq(dataset)
        for label in (0, 1):
            assert np.isclose(weights[rows & frame["outcome"].eq(label)].sum(), 1.0)


def test_ecdf_is_label_free_rank_in_unit_interval():
    ranks = lodo.ecdf(np.array([3.0, 1.0, 2.0, 2.0]))
    assert ranks.max() == 1.0 and ranks.min() > 0
    assert ranks[0] > ranks[1]
