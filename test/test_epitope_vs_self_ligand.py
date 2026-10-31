import numpy as np
import pandas as pd

from scripts import evaluate_epitope_vs_self_ligand as task


def frame(presentation, ba, hla="HLA-A*02:01", peptide="AAAAAAAAA"):
    return pd.DataFrame(
        {
            "peptide": peptide,
            "hla": hla,
            "mhcflurry_2_3_0_presentation_score": presentation,
            "netmhcpan_ba_rank": ba,
        }
    )


def test_matching_is_one_to_one_nearest_and_respects_hla_and_caliper():
    cases = frame([0.5, 0.5], [1.0, 1.0])
    controls = pd.concat(
        [frame([0.5, 0.01], [1.0, 50.0]), frame([0.5], [1.0], hla="HLA-B*07:02")], ignore_index=True
    )
    scale = np.array([1.0, 1.0])
    pairs = task.match(cases, controls, scale, caliper=0.5)
    assert len(pairs) == 1  # one close same-HLA control, used only once
    assert controls.loc[pairs["control"].iloc[0], "hla"] == "HLA-A*02:01"
    assert pairs["control"].iloc[0] == 0


def test_matching_requires_same_length():
    cases = frame([0.5], [1.0], peptide="AAAAAAAAA")
    controls = frame([0.5], [1.0], peptide="AAAAAAAAAA")
    assert task.match(cases, controls, np.array([1.0, 1.0])).empty
