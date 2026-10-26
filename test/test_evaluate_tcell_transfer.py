import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_tcell_transfer.py"
SPEC = importlib.util.spec_from_file_location("evaluate_tcell_transfer", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_continuous_mhc_features_have_stable_directions():
    frame = pd.DataFrame(
        {
            "mhcflurry_presentation_score": [0.1, 0.9],
            "mhcflurry_affinity": [1000.0, 10.0],
            "mhcflurry_processing_score": [0.2, 0.8],
            "tcr_face_hydrophobicity": [0.0, 1.0],
            "tcr_face_aromatic_fraction": [0.0, 0.5],
        }
    )
    values, names = module._continuous_mhc_features(frame, include_tcr_shape=True)
    assert names == [
        "presentation_logit",
        "negative_log10_affinity",
        "processing_logit",
        "tcr_face_hydrophobicity",
        "tcr_face_aromatic_fraction",
    ]
    assert values.shape == (2, len(names))
    assert np.all(values[1] > values[0])


def test_matched_pair_auc_is_chance_for_tied_scores():
    frame = pd.DataFrame(
        {
            "hla": ["A", "A", "B", "B"],
            "outcome": ["Positive", "Negative", "Positive", "Negative"],
            "mhcflurry_presentation_percentile": [1.0, 1.05, 2.0, 2.05],
        }
    )
    result = module._matched_pair_auc(frame, np.ones(len(frame)))
    assert result["n_pairs"] == 2
    assert result["auc"] == 0.5


def test_auc_direction_audit_reverses_affinity():
    frame = pd.DataFrame(
        {
            "outcome": ["Negative", "Positive"],
            "mhcflurry_affinity": [10.0, 1000.0],
        }
    )
    assert module._auc_if_defined(frame, "mhcflurry_affinity", reverse=True) == 0.0
    assert module._auc_if_defined(frame, "mhcflurry_affinity") == 1.0


def test_joint_components_connect_cross_table_seven_mer_neighbor():
    frame = pd.DataFrame(
        {
            "peptide": ["ABCDEFGHI", "XYZABCDEFG", "LMNOPQRST"],
            "peptide_cluster": ["d-1", "v-9", "other"],
        }
    )
    components = module._joint_peptide_components(frame)
    assert components.iloc[0] == components.iloc[1]
    assert components.iloc[0] != components.iloc[2]


def test_joint_components_connect_hamming_one_neighbor():
    frame = pd.DataFrame(
        {
            "peptide": ["ABCDEFGHI", "ABCDXFGHI"],
            "peptide_cluster": ["d-1", "v-9"],
        }
    )
    components = module._joint_peptide_components(frame)
    assert components.iloc[0] == components.iloc[1]
