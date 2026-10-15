"""Scientific guardrails for the exploratory viral assay benchmark."""

import numpy as np

from scripts.benchmark_viral_tcell import (
    human_tcr_kmer_surprisal,
    prepare_assays,
    vectorize,
    within_hla_auc,
)


def test_assay_selection_requires_exact_hla_and_preserves_exposure():
    base = {
        "host__name": "Homo sapiens (human)",
        "mhc_restriction__class": "I",
        "mhc_restriction__name": "HLA-A*02:01",
        "epitope__name": "YLQPRTFLL",
        "assay__qualitative_measurement": "Positive",
        "first_in_vivo_process__process_type": "Occurrence of infectious disease",
        "second_in_vivo_process__process_type": "Prophylactic vaccination",
    }
    unresolved = {**base, "mhc_restriction__name": "HLA class I"}
    class_ii = {**base, "mhc_restriction__class": "II"}
    result = prepare_assays([base, unresolved, class_ii], "35484232")
    assert len(result) == 1
    assert result[0]["exposure"] == "infected_then_vaccinated"


def test_within_hla_auc_does_not_pair_different_alleles():
    rows = [
        {"hla": "HLA-A*02:01", "outcome": "Positive", "foreignness": 0.9},
        {"hla": "HLA-A*02:01", "outcome": "Negative", "foreignness": 0.2},
        {"hla": "HLA-B*07:02", "outcome": "Positive", "foreignness": 0.1},
        {"hla": "HLA-B*07:02", "outcome": "Negative", "foreignness": 0.8},
    ]
    assert within_hla_auc(rows) == (0.5, 2)


def test_tcr_face_view_excludes_anchor_residues():
    same_face_different_anchors = ["AAVVVVVVA", "RKVVVVVVL"]
    assert np.array_equal(
        vectorize(same_face_different_anchors, "tcr_face")[0],
        vectorize(same_face_different_anchors, "tcr_face")[1],
    )
    assert not np.array_equal(
        vectorize(same_face_different_anchors, "whole")[0],
        vectorize(same_face_different_anchors, "whole")[1],
    )


def test_presentation_caliper_excludes_dissimilar_pairs():
    rows = [
        {
            "hla": "HLA-A*02:01", "outcome": "Positive", "score": 0.9,
            "mhcflurry_presentation_percentile": 0.05,
        },
        {
            "hla": "HLA-A*02:01", "outcome": "Negative", "score": 0.1,
            "mhcflurry_presentation_percentile": 0.08,
        },
        {
            "hla": "HLA-A*02:01", "outcome": "Negative", "score": 0.2,
            "mhcflurry_presentation_percentile": 1.0,
        },
    ]
    assert within_hla_auc(
        rows, "score", presentation_caliper=0.1) == (1.0, 1)


def test_human_kmer_surprisal_is_higher_for_unseen_tcr_face(tmp_path):
    swissprot = tmp_path / "swissprot.csv"
    swissprot.write_text(
        "identifier,sequence,category\n"
        "human-1,AAAAAAAAAAAA,human\n"
        "virus-1,CCCCCCCCCCCC,viruses\n"
    )
    values = human_tcr_kmer_surprisal(
        swissprot, ["AAAAAAAA", "AACCCCCA"], k_values=(3,))
    assert values["human_tcr_3mer_surprisal"][1] > (
        values["human_tcr_3mer_surprisal"][0])
