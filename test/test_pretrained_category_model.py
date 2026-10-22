"""Regression tests for the bundled swissprot-categories model.

The golden values pin the model's outputs for fixed peptides, so a change to the feature
definitions, the inference code or the weights that alters predictions fails here. If the model
is retrained on purpose, regenerate them and update the README/docs tables as well.
"""

import json

import numpy as np
import pytest

from weirdo import list_bundled_models, load_model, predict_categories
from weirdo.model_manager import DEFAULT_CATEGORY_MODEL

pytestmark = pytest.mark.skipif(
    DEFAULT_CATEGORY_MODEL not in [m["name"] for m in list_bundled_models()],
    reason="the bundled model is not installed",
)

CATEGORIES = [
    "archaea", "bacteria", "fungi", "human", "invertebrates",
    "mammals", "plants", "rodents", "vertebrates", "viruses",
]
GOLDEN = {
    "SIINFEKL": [0.07692, 0.38883, 0.14058, 0.04417, 0.15769, 0.02477, 0.10156, 0.04779, 0.02577, 0.05677],
    "NLVPMVATV": [0.02608, 0.46126, 0.10043, 0.09221, 0.06933, 0.04651, 0.11600, 0.10591, 0.04011, 0.02689],
    "GILGFVFTL": [0.04478, 0.43474, 0.11830, 0.08296, 0.07555, 0.04088, 0.11151, 0.09346, 0.03808, 0.03051],
    "MTMDKSELVQKAKLAE": [0.03301, 0.41992, 0.12890, 0.08748, 0.08999, 0.04375, 0.10772, 0.09874, 0.04152, 0.03227],
}


@pytest.fixture(scope="module")
def model():
    return load_model(DEFAULT_CATEGORY_MODEL)


def test_categories_and_shape(model):
    assert model.target_categories == CATEGORIES
    assert model.predict_proba(["SIINFEKL"]).shape == (1, 10)
    assert len(model.get_feature_names()) == 592


def test_golden_probabilities(model):
    peptides = list(GOLDEN)
    expected = np.array([GOLDEN[p] for p in peptides])
    np.testing.assert_allclose(model.predict_proba(peptides), expected, atol=2e-4)


def test_outputs_are_probabilities_with_a_real_spread(model):
    rng = np.random.RandomState(0)
    residues = list("ACDEFGHIKLMNPQRSTVWY")
    peptides = ["".join(rng.choice(residues, 9)) for _ in range(500)]
    probs = model.predict_proba(peptides)
    assert ((probs > 0) & (probs < 1)).all()
    # bacteria is the commonest label (about 40% of 8-mers); the old squared-loss head could not
    # leave a narrow band around 0.5-0.7 whatever the peptide
    assert 0.2 < probs[:, CATEGORIES.index("bacteria")].mean() < 0.6
    assert probs[:, CATEGORIES.index("bacteria")].std() > 0.02


def test_long_peptides_average_their_windows(model):
    peptide = "MTMDKSELVQKAKLAE"
    windows = [peptide[i:i + 8] for i in range(len(peptide) - 7)]
    expected = model.predict_proba(windows).mean(axis=0)
    np.testing.assert_allclose(model.predict_proba([peptide])[0], expected, atol=1e-6)


def test_pathogen_self_summary_matches_its_definition():
    frame = predict_categories(list(GOLDEN))
    pathogen = np.maximum(frame["bacteria"], frame["viruses"])
    mammalian = np.maximum.reduce([frame["human"], frame["rodents"], frame["mammals"]])
    np.testing.assert_allclose(frame["pathogen_max"], pathogen)
    np.testing.assert_allclose(frame["self_max"], mammalian)
    np.testing.assert_allclose(frame["foreignness"], pathogen / (pathogen + mammalian))


def test_summed_summary_is_available_through_combine():
    frame = predict_categories(list(GOLDEN), combine="sum")
    pathogen = frame["bacteria"] + frame["viruses"]
    mammalian = frame["human"] + frame["rodents"] + frame["mammals"]
    np.testing.assert_allclose(frame["pathogen_sum"], pathogen)
    np.testing.assert_allclose(frame["self_sum"], mammalian)
    np.testing.assert_allclose(frame["foreignness"], pathogen / (pathogen + mammalian))


def test_metadata_documents_the_training_and_evaluation():
    info = next(m for m in list_bundled_models() if m["name"] == DEFAULT_CATEGORY_MODEL)
    metadata = info["metadata"]
    assert metadata["categories"] == CATEGORIES
    assert "description" in metadata and "label_definition" in metadata
    test = metadata["evaluation"]["test"]
    assert test["n"] > 100_000
    assert abs(test["metrics"]["mean"]["auc"] - 0.6145497112412202) < 1e-9
    assert test["metrics"]["mean"]["ece"] < 0.02  # calibrated
    json.dumps(metadata)  # serializable
