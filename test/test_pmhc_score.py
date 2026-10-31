import numpy as np
import pytest

import weirdo
from weirdo import pmhc_score


def test_core_ranks_are_bounded_and_monotone_in_each_predictor():
    peptides = ["AAAAAAAAA"] * 3
    result = weirdo.pmhc_weirdo_score(
        peptides, [0.001, 0.05, 0.9], [60.0, 5.0, 0.1], weirdo_weight=0.0, face_weight=0.0
    )
    assert result["score"].between(0, 1).all()
    assert result["pmhc_core"].is_monotonic_increasing
    assert result["presentation_rank"].is_monotonic_increasing
    assert result["binding_rank"].is_monotonic_increasing
    assert result["weirdo_rank"].isna().all()
    assert (result["score"] == result["pmhc_core"]).all()


def test_weight_zero_needs_no_weirdo_model(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Weirdo model must not be called at weight 0")

    monkeypatch.setattr("weirdo.api.predict_peptide_content", fail)
    weirdo.pmhc_weirdo_score(["AAAAAAAAA"], [0.5], [1.0], weirdo_weight=0.0)  # face term only


def test_weirdo_term_is_additive_centred_and_bounded_by_weight():
    peptides = ["AAAAAAAAA"] * 2
    inputs = ([0.2, 0.2], [3.0, 3.0])
    low_high = [0.0, 1.0]
    result = weirdo.pmhc_weirdo_score(
        peptides, *inputs, weirdo_weight=0.2, weirdo_scores=low_high, face_weight=0.0
    )
    spread = result["score"].iloc[1] - result["score"].iloc[0]
    assert 0 < spread <= 0.2 + 1e-9
    reference = pmhc_score._rank([0.0, 1.0], "weirdo")
    assert np.isclose(spread, 0.2 * (reference[1] - reference[0]))


def test_default_weirdo_scores_come_from_the_shipped_ensemble():
    pytest.importorskip("torch")
    result = weirdo.pmhc_weirdo_score(
        ["NLVPMVATV", "AFVEEVME"], [0.3, 0.3], [1.0, 1.0]
    )
    assert result["weirdo_rank"].between(0, 1).all()
    assert result["score"].notna().all()


def test_inputs_are_validated():
    with pytest.raises(ValueError):
        weirdo.pmhc_weirdo_score(["AAAAAAAAA"], [0.1, 0.2], [1.0], weirdo_weight=0.0)
    with pytest.raises(ValueError):
        weirdo.pmhc_weirdo_score(["AAAAAAAAA"], [0.1], [1.0], weirdo_weight=2.0)
    with pytest.raises(ValueError):
        weirdo.pmhc_weirdo_score(["AAAAAAAAA"], [float("nan")], [1.0], weirdo_weight=0.0)


def test_face_surprise_uses_p3_to_penultimate_residues_only():
    assert pmhc_score.face_surprise("AAWWWWAA") == pmhc_score.face_surprise("CCWWWWAA")
    assert pmhc_score.face_surprise("AAWWWWAA") > pmhc_score.face_surprise("AALLLLAA")


def test_rare_face_residues_raise_the_score_and_weight_zero_disables_it():
    inputs = (["AAWWWWAA", "AALLLLAA"], [0.2, 0.2], [3.0, 3.0])
    on = weirdo.pmhc_weirdo_score(*inputs, weirdo_weight=0.0, face_weight=0.2)
    off = weirdo.pmhc_weirdo_score(*inputs, weirdo_weight=0.0, face_weight=0.0)
    assert on["score"].iloc[0] > on["score"].iloc[1]
    assert off["face_rank"].isna().all() and off["score"].nunique() == 1


def test_face_surprise_rejects_short_or_nonstandard_peptides():
    with pytest.raises(ValueError):
        pmhc_score.face_surprise("AAA")
    with pytest.raises(ValueError):
        pmhc_score.face_surprise("AAXXAAAA")
