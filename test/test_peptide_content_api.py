import numpy as np
import pytest

from weirdo import list_bundled_models, load_model, predict_peptide_content


CONTENT_CATEGORIES = [
    "viruses",
    "bacteria",
    "archaea",
    "fungi",
    "plants",
    "protists",
    "invertebrates",
    "fish",
    "non_mammalian_tetrapods",
    "mammals",
    "human",
    "mouse",
    "dog",
]


class FakeContentModel:
    target_categories = [
        "viruses",
        "bacteria",
        "archaea",
        "fungi",
        "protists",
        "mammals",
        "human",
    ]

    def predict_proba(self, peptides):
        return np.tile([0.8, 0.1, 0.1, 0.1, 0.1, 0.2, 0.9], (len(peptides), 1))


def test_predict_peptide_content_returns_named_summary_columns():
    result = predict_peptide_content("SIINFEKL", model=FakeContentModel())
    assert result.loc[0, "microbial_like"] == pytest.approx(0.8)
    assert result.loc[0, "self_like"] == pytest.approx(0.2)
    assert result.loc[0, "microbial_foreign_like"] == pytest.approx(0.64)


def test_predict_peptide_content_supports_species_augmented_self_bundle():
    result = predict_peptide_content(
        "SIINFEKL",
        model=FakeContentModel(),
        self_categories=("mammals", "human"),
    )
    assert result.loc[0, "self_like"] == pytest.approx(0.9)
    assert result.loc[0, "microbial_foreign_like"] == pytest.approx(0.08)


def test_exact_self_hit_overrides_learned_foreignness():
    result = predict_peptide_content(
        ["SIINFEKL", "GILGFVFTL"],
        model=FakeContentModel(),
        exact_self_hits=[True, False],
    )
    assert result.loc[0, "exact_self_hit"]
    assert result.loc[0, "self_like"] == 1.0
    assert result.loc[0, "microbial_foreign_like"] == 0.0


def test_exact_self_hit_length_must_match_peptides():
    with pytest.raises(ValueError, match="one value per peptide"):
        predict_peptide_content("SIINFEKL", model=FakeContentModel(), exact_self_hits=[])


@pytest.mark.parametrize("peptide", ["AAAAAAA", "AAAAAAAAAAAAAAAA"])
def test_predict_peptide_content_rejects_lengths_outside_training_range(peptide):
    with pytest.raises(ValueError, match="length 8--15"):
        predict_peptide_content(peptide, model=FakeContentModel())


def test_bundled_peptide_content_model_is_self_contained_and_usable():
    bundled = {entry["name"]: entry for entry in list_bundled_models()}
    assert "peptide-content-v1" in bundled
    metadata = bundled["peptide-content-v1"]["metadata"]
    assert metadata["experimental"] is True
    assert metadata["summary_scores"]["microbial_foreign_like"].startswith(
        "microbial_like"
    )

    model = load_model("peptide-content-v1")
    assert model.target_categories == CONTENT_CATEGORIES
    assert model.get_params()["whole_peptide"] is True

    result = predict_peptide_content(
        ["SIINFEKL", "GILGFVFTL", "ACDEFGHIKLMNPQR"],
        self_categories=("mammals", "human"),
    )
    assert list(result.columns) == [
        "peptide",
        *CONTENT_CATEGORIES,
        "microbial_like",
        "self_like",
        "microbial_foreign_like",
    ]
    assert result.shape == (3, 17)
    assert result.drop(columns="peptide").to_numpy().min() >= 0.0
    assert result.drop(columns="peptide").to_numpy().max() <= 1.0
