import numpy as np
import pandas as pd

from scripts.aggregate_braun_candidate_pmcs import aggregate_constructs, candidate_evidence


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "record_id": ["a", "a", "a", "b"],
            "patient": ["p1", "p1", "p1", "p2"],
            "mutation_class": ["frameshift"] * 4,
            "outcome": [1, 1, 1, 0],
            "peptide": ["AAAAAAAA", "BBBBBBBB", "CCCCCCCC", "DDDDDDDD"],
            "hla": ["HLA-A*01:01"] * 4,
            "mhcflurry_2_3_0_presentation_score": [1.0, 0.81, 0.64, 0.25],
            "mhcflurry_2_3_0_presentation_percentile": [1.0, 2.0, 10.0, 1.0],
            "netmhcpan_ba_rank": [0.0, 1.0, 1.0, 10.0],
        }
    )


def test_candidate_evidence_requires_presentation_and_binding():
    values = candidate_evidence(_rows())
    assert np.allclose(values, [1.0, np.sqrt(0.81 * 0.99), np.sqrt(0.64 * 0.99), np.sqrt(0.25 * 0.9)])


def test_top_three_sum_adds_distinct_candidates_and_gate_eliminates_weak_windows():
    aggregates = aggregate_constructs(_rows())
    first = aggregates.loc[aggregates.record_id.eq("a")].iloc[0]
    assert first.n_distinct_candidate_pmhcs == 3
    assert np.isclose(first.top3_sum_candidate_evidence, 1.0 + np.sqrt(0.81 * 0.99) + np.sqrt(0.64 * 0.99))
    assert np.isclose(first.top3_sum_gated_candidate_evidence, 1.0 + np.sqrt(0.81 * 0.99))


def test_taxonomy_only_reweights_candidates_that_already_pass_the_same_gate():
    rows = _rows()
    rows["broad9__esm_160k__one_minus_mammals"] = [0.1, 0.9, 1.0, 1.0]
    aggregates = aggregate_constructs(rows)
    first = aggregates.loc[aggregates.record_id.eq("a")].iloc[0]
    assert np.isclose(
        first.top3_sum_gated_evidence_x_one_minus_mammals,
        0.1 + 0.9 * np.sqrt(0.81 * 0.99),
    )
