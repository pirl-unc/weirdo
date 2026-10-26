import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_improve_foreignness.py"
SPEC = importlib.util.spec_from_file_location("evaluate_improve_foreignness", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_patient_groups_keep_patients_independent():
    frame = pd.DataFrame(
        {
            "Patient": ["a", "a", "b", "c"],
            "Mut_peptide": ["AAAAAAAA", "CCCCCCCC", "CCCCCCCC", "DDDDDDDD"],
        }
    )
    groups = MODULE.patient_groups(frame)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]
    assert groups[3] not in groups[:3]


def test_purge_shared_test_peptides_removes_only_overlaps():
    peptides = np.asarray(["AAAAAAAA", "CCCCCCCC", "AAAAAAAA", "DDDDDDDD"])
    train = np.asarray([0, 1, 3])
    test = np.asarray([2])
    np.testing.assert_array_equal(
        MODULE.purge_shared_test_peptides(train, test, peptides), [1, 3]
    )


def test_content_scores_include_human_and_mammal_forms():
    categories = ("viruses", "bacteria", "mammals", "human")
    probabilities = np.asarray([[0.2, 0.3, 0.4, 0.1]])
    scores = MODULE._content_scores(probabilities, categories)
    np.testing.assert_allclose(scores["one_minus_mammals"], [0.6])
    np.testing.assert_allclose(scores["one_minus_human"], [0.9])
    np.testing.assert_allclose(scores["virus_vs_mammals"], [0.12])
    np.testing.assert_allclose(scores["virus_vs_human"], [0.18])
    np.testing.assert_allclose(scores["p_bacteria"], [0.3])
    np.testing.assert_allclose(scores["virus_bacteria_max"], [0.3])
    np.testing.assert_allclose(scores["bacteria_vs_mammals"], [0.18])
    np.testing.assert_allclose(scores["virus_bacteria_vs_human"], [0.27])


def test_content_scores_include_nested_viral_host_forms():
    categories = (
        "viruses",
        "bacteria",
        "mammals",
        "human",
        "mammal_host_viruses",
        "other_host_viruses",
    )
    probabilities = np.asarray([[0.8, 0.1, 0.25, 0.2, 0.7, 0.3]])
    scores = MODULE._content_scores(probabilities, categories)
    np.testing.assert_allclose(scores["p_mammal_host_viruses"], [0.7])
    np.testing.assert_allclose(scores["p_other_host_viruses"], [0.3])
    np.testing.assert_allclose(scores["host_virus_max"], [0.7])
    np.testing.assert_allclose(scores["mammal_host_virus_margin"], [0.4])
    np.testing.assert_allclose(scores["mammal_host_virus_specific"], [0.49])
    np.testing.assert_allclose(scores["mammal_host_virus_vs_mammals"], [0.525])
    np.testing.assert_allclose(scores["mammal_host_virus_vs_human"], [0.56])


def test_design_matrix_adds_tcr_chemistry_columns():
    frame = pd.DataFrame(
        {
            "RankEL_4.1": [0.1, 1.0],
            "RankBA_4.1": [0.2, 2.0],
            "PeptLen": [9, 10],
            "HydroCore": [0.1, 0.2],
            "PropAro": [0.0, 0.25],
            "HLA_allele": ["HLA-A01:01", "HLA-A02:01"],
            "cohort": ["a", "b"],
        }
    )
    base = MODULE.design_matrix(frame, include_tcr_chemistry=False)
    tcr = MODULE.design_matrix(frame, include_tcr_chemistry=True)
    assert tcr.shape == (2, base.shape[1] + 2)


def test_ranking_metrics_accept_negative_uncalibrated_scores():
    metrics = MODULE._ranking_metrics(
        np.asarray([0, 0, 1, 1]), np.asarray([-2.0, -1.0, 0.0, 1.0])
    )
    assert metrics == {"roc_auc": 1.0, "pr_auc": 1.0}


def test_published_prediction_join_is_idempotent(tmp_path):
    frame = pd.DataFrame(
        {
            "HLA_allele": ["HLA-A01:01"],
            "Norm_peptide": ["AAAAAAAAA"],
            "Mut_peptide": ["AAAAAAAAC"],
            "Expression": [1.23449],
            "response": [1],
            "published_iedb_calis": [-99.0],
        }
    )
    row = {
        "HLA_allele": "HLA-A01:01",
        "Norm_peptide": "AAAAAAAAA",
        "Mut_peptide": "AAAAAAAAC",
        "Expression": 1.234,
        "response": 1,
    }
    for index, (_, source, _) in enumerate(MODULE.PUBLISHED_COLUMN_SPECS):
        row[source] = float(index + 1)
    path = tmp_path / "published.tsv"
    pd.DataFrame([row]).to_csv(path, sep="\t", index=False)

    once = MODULE.attach_published_predictions(frame, path)
    twice = MODULE.attach_published_predictions(once, path)

    assert once.columns.tolist() == twice.columns.tolist()
    assert twice["published_iedb_calis"].iloc[0] == 1.0
    assert not any(column.endswith(("_x", "_y")) for column in twice.columns)
