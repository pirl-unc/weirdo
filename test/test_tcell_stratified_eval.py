import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tcell_stratified_eval.py"
SPEC = importlib.util.spec_from_file_location("tcell_stratified_eval", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def _frame():
    return pd.DataFrame(
        {
            "peptide": ["AAACCCK", "AAACCDK", "GGGTTTK", "GGGTTTK"],
            "hla": ["HLA-A*01:01"] * 4,
            "cohort": ["study"] * 4,
            "exposure": ["infected"] * 4,
            "source_species": ["virus-a", "virus-a", "virus-b", "virus-b"],
            "outcome": ["Positive", "Negative", "Positive", "Negative"],
            "mhcflurry_presentation_score": [0.8, 0.8, 0.2, 0.2],
            "mhcflurry_processing_score": [0.8, 0.8, 0.2, 0.2],
            "n_positive_assays": [2, 0, 1, 0],
            "n_negative_assays": [0, 3, 0, 2],
            "n_assay_rows": [2, 3, 1, 2],
            "n_subjects_tested": [2, 3, 1, 2],
        }
    )


def test_prepare_table_preserves_counts_and_strata_fields():
    result, edges = module.prepare_table(_frame(), n_bins=2, face_k=4)
    assert edges["presentation_edges"][0] == -np.inf
    assert result["assay_trials"].tolist() == [2.0, 3.0, 1.0, 2.0]
    assert result["response_rate"].tolist() == [1.0, 0.0, 1.0, 0.0]
    assert set(result["stratum_key"].str.count("\\|").tolist()) == {6}
    assert result["holdout_group"].nunique() == 2


def test_bin_edges_can_be_frozen_on_development_only():
    development, edges = module.prepare_table(_frame(), n_bins=2)
    extended = pd.concat(
        (_frame(), _frame().assign(mhcflurry_presentation_score=100.0, mhcflurry_processing_score=100.0)),
        ignore_index=True,
    )
    reapplied, frozen = module.prepare_table(
        extended,
        presentation_edges=np.asarray(edges["presentation_edges"]),
        processing_edges=np.asarray(edges["processing_edges"]),
    )
    assert frozen == edges
    assert reapplied.loc[: len(development) - 1, "presentation_bin"].tolist() == development["presentation_bin"].tolist()


def test_missing_counts_are_explicit_binary_fallback():
    frame = _frame().drop(columns=["n_positive_assays", "n_negative_assays"])
    result, _ = module.prepare_table(frame)
    assert set(result["count_source"]) == {"binary_outcome_fallback"}
    assert result["n_positive_assays"].tolist() == [1.0, 0.0, 1.0, 0.0]


def test_matched_pairs_use_count_weights_within_strata():
    result, _ = module.prepare_table(_frame(), n_bins=2)
    pairs = module.matched_pairs(result)
    assert len(pairs) == 2
    assert sorted(pairs["pair_weight"].tolist()) == [2.0, 6.0]


def test_connected_holdout_keeps_face_component_together():
    frame = _frame()
    frame.loc[3, "source_species"] = "virus-c"
    result, _ = module.prepare_table(frame, n_bins=2, face_k=4)
    # The two identical faces are in one connected component despite their
    # different source family labels; the strict group therefore is shared.
    assert result.loc[2, "face_component"] == result.loc[3, "face_component"]
    assert result.loc[2, "holdout_group"] == result.loc[3, "holdout_group"]


def test_wt_mutant_pairs_require_explicit_wild_type_column():
    result, _ = module.prepare_table(_frame())
    pairs, info = module.wt_mutant_pairs(result)
    assert len(pairs) == 0
    assert info["available"] is False


def test_explicit_wt_mutant_table_is_normalized_without_outcomes(tmp_path):
    path = tmp_path / "pairs.csv"
    pd.DataFrame(
        {
            "wild_type_peptide": ["AAACCCK"],
            "mutant_peptide": ["AAACCDK"],
            "hla": ["HLA-A*01:01"],
        }
    ).to_csv(path, index=False)
    pairs = module.load_explicit_wt_mutant_pairs(path)
    assert pairs.loc[0, "pair_key"].startswith("HLA-A*01:01|AAACCCK|")
    assert pairs.loc[0, "wild_type_length"] == pairs.loc[0, "mutant_length"] == 7


def test_cached_iedb_wt_mutant_loader_filters_and_deduplicates(tmp_path):
    path = tmp_path / "iedb-1.json"
    path.write_text(
        '[{"assay_id": 7, "assay_antigen__name": "AAACCDKK", '
        '"related_object__name": "AAACCCKK", '
        '"related_object__epitope_relation": "in-frame neo-epitope"}, '
        '{"assay_id": 8, "assay_antigen__name": "AAAA", '
        '"related_object__name": "AAAT"}]'
    )
    pairs = module.load_iedb_wt_mutant_pairs(tmp_path, {7, 8})
    assert pairs["assay_id"].tolist() == [7]
    assert pairs.loc[0, "wild_type_peptide"] == "AAACCCKK"


def test_metrics_and_clustered_uncertainty_are_count_aware():
    result, _ = module.prepare_table(_frame(), n_bins=2)
    score = np.asarray([0.9, 0.1, 0.8, 0.2])
    metrics = module.within_stratum_metrics(result, score)
    assert metrics["within_stratum_micro_pr_auc"] is not None
    assert metrics["calibration"]["brier"] < 0.1
    uncertainty = module.clustered_uncertainty(result, score, draws=10)
    assert uncertainty["draws"] == 10
    assert "pr_auc_95ci" in uncertainty


def test_metrics_accept_nonconsecutive_frame_indices():
    result, _ = module.prepare_table(_frame(), n_bins=2)
    result.index = [10, 20, 30, 40]
    metrics = module.within_stratum_metrics(result, np.asarray([0.9, 0.1, 0.8, 0.2]))
    assert metrics["top_k_yield"]["0.1"]["n_strata"] >= 1


def test_lockbox_is_groupwise_and_reproducible():
    result, _ = module.prepare_table(_frame(), n_bins=2)
    first = module.reserve_lockbox(result, fraction=0.4, seed=7)
    second = module.reserve_lockbox(result, fraction=0.4, seed=7)
    assert first.equals(second)
    assert set(result.loc[first, "holdout_group"]).isdisjoint(set(result.loc[~first, "holdout_group"]))
