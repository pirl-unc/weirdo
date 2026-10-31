import numpy as np
import pandas as pd

from scripts import build_fresh_iedb_cohort as build
from scripts import evaluate_fresh_cohort as fresh


def test_outcome_parsing_treats_positive_variants_and_unknowns():
    assert build.positive("Positive-High") == 1
    assert build.positive("Negative") == 0
    assert build.positive("Equivocal") is None
    assert build.positive(float("nan")) is None


def test_peptide_and_allele_filters_match_the_frozen_protocol():
    assert build.CANONICAL.fullmatch("NLVPMVATV")
    assert not build.CANONICAL.fullmatch("NLVPMVAT*")
    assert not build.CANONICAL.fullmatch("NLVPMVA")
    assert build.EXACT_HLA_I.fullmatch("HLA-A*02:01")
    assert not build.EXACT_HLA_I.fullmatch("HLA-A2")


def test_eligible_studies_need_size_and_both_classes():
    frame = pd.DataFrame(
        {
            "pmid": ["big"] * 30 + ["tiny"] * 6 + ["onesided"] * 30,
            "outcome": [0, 1] * 15 + [0, 1] * 3 + [1] * 30,
        }
    )
    assert fresh.eligible(frame) == ["big"]


def test_within_study_gap_is_zero_for_identical_scores():
    frame = pd.DataFrame({"pmid": ["a"] * 8, "outcome": [0, 1] * 4})
    scores = pd.DataFrame({"x": np.arange(8.0), "y": np.arange(8.0)})
    assert fresh.within_study(frame, scores, ["a"], "x", "y") == 0.0
