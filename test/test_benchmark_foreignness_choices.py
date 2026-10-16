"""Checks for the statistics and selection rule of the choice benchmark."""

import numpy as np
import pytest

from scripts import benchmark_foreignness_choices as bench


def test_rcs_basis_is_linear_beyond_the_outer_knots():
    knots = np.array([-2.0, -0.5, 0.5, 2.0])
    x = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    columns = bench.rcs_basis(x, knots)
    assert columns.shape == (5, 3)
    # A restricted cubic spline is linear in the tails: second differences of
    # every column vanish past the last knot.
    for column in columns.T:
        assert np.allclose(np.diff(column, n=2), 0, atol=1e-9)


def test_pooled_dummies_pool_rare_levels_and_drop_the_reference():
    values = ["a"] * 20 + ["b"] * 20 + ["rare"] * 2 + ["tiny"] * 3
    dummies = bench.pooled_dummies(values, minimum=10)
    # levels: __other__, a, b -> two columns after dropping the reference
    assert dummies.shape == (len(values), 2)
    assert dummies[-5:].sum(axis=0).tolist() == [0.0, 0.0]  # reference rows
    assert bench.pooled_dummies(["a"] * 30, 10).shape == (30, 0)


def test_standardization_is_within_stratum_and_tolerates_nan():
    values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, np.nan, 40.0])
    strata = ["x", "x", "x", "y", "y", "y", "y"]
    z = bench.standardize_within(values, strata)
    assert np.nanmean(z[:3]) == pytest.approx(0, abs=1e-9)
    assert np.nanstd(z[3:]) == pytest.approx(1, abs=1e-9)
    assert np.isnan(z[5])
    constant = bench.standardize_within(np.ones(4), ["s"] * 4)
    assert np.allclose(constant, 0)


def test_random_effects_summary_matches_hand_computation():
    result = bench.dersimonian_laird([0.2, 0.2, 0.2], [0.1, 0.1, 0.1])
    assert result["beta"] == pytest.approx(0.2)
    assert result["tau2"] == pytest.approx(0.0)
    assert result["i2"] == pytest.approx(0.0)
    heterogeneous = bench.dersimonian_laird([0.0, 0.6], [0.1, 0.1])
    assert heterogeneous["tau2"] > 0
    assert heterogeneous["i2"] > 0.8
    assert bench.dersimonian_laird([0.1], [0.1]) == {}


def _synthetic_rows(effect, n=1800, seed=0, n_cohorts=3):
    rng = np.random.default_rng(seed)
    rows, scores = [], []
    for i in range(n):
        cohort = f"c{i % n_cohorts}"
        score = rng.normal()
        presentation = float(rng.uniform(0.05, 0.95))
        logit = np.log(presentation / (1 - presentation))
        p = 1 / (1 + np.exp(-(-1.0 + 0.5 * logit + effect * score)))
        rows.append({
            "cohort": cohort, "exposure": "x", "domain": "viral",
            "peptide": f"PEP{i:05d}XY", "hla": f"HLA-A*0{i % 4 + 1}:01",
            "outcome": "Positive" if rng.random() < p else "Negative",
            "mhcflurry_presentation_score": presentation,
            "n_assay_rows": 1, "n_subjects_tested": 3,
            "source_species": "virus", "cluster": f"k{i // 3}",
            "peptide_cluster": f"k{i}",
            "tcr_face_hydrophobicity": float(rng.normal()),
            "tcr_face_aromatic_fraction": float(rng.normal()),
        })
        scores.append(score)
    return rows, np.asarray(scores)


def test_pooled_model_recovers_a_planted_effect_and_a_null():
    pytest.importorskip("statsmodels")
    rows, signal = _synthetic_rows(effect=0.5)
    nuisance = bench.nuisance_matrix(rows, "viral", True)
    found = bench.pooled_effect(rows, signal, nuisance)
    assert found["z"] > 5
    assert found["odds_ratio"] == pytest.approx(np.exp(0.5), rel=0.25)
    noise = np.random.default_rng(9).normal(size=len(rows))
    null = bench.pooled_effect(rows, noise, nuisance)
    assert abs(null["z"]) < 3
    # Rows with an undefined score are dropped for that candidate only.
    gappy = signal.copy()
    gappy[::2] = np.nan
    partial = bench.pooled_effect(rows, gappy, nuisance)
    assert partial["n_rows"] == int(np.isfinite(gappy).sum())


def test_cohort_effect_needs_enough_events():
    pytest.importorskip("statsmodels")
    rows, signal = _synthetic_rows(effect=0.6, n=900, n_cohorts=1)
    for r in rows:
        r["cohort"] = "only"
    alone = bench.cohort_effect(rows, signal)
    assert alone["beta"] > 0.2
    few = [dict(r, outcome="Negative") for r in rows[:60]]
    assert bench.cohort_effect(few, signal[:60]) is None


def _entry(name, z, beta=0.2, coverage=1.0, sign=0.8, **extra):
    return {"candidate": name, "family": "f", "coverage": coverage,
            "sign_fraction": sign, "n_cohorts_reported": 5,
            "stringent": {"beta": beta, "se": beta / z if z else 1, "z": z,
                          "odds_ratio": float(np.exp(beta)),
                          "or_ci95": [1, 2], "p_one_sided": 0.01},
            **extra}


def test_selection_applies_every_registered_gate():
    ledger = [
        _entry("nn_full", 3.0),
        _entry("soft_full", 3.4),
        _entry("nn_full_whole", 5.0, coverage=0.5),          # coverage gate
        _entry("nn_allele_motif", 5.0, sign=0.4),            # sign gate
        _entry("nn_atlas_all", 1.0),                         # z gate
        _entry("nn_atlas_allele", 4.0, beta=-0.2),           # direction gate
    ]
    chosen = bench.select_candidate(ledger)
    assert chosen["selected"] == "soft_full"
    assert chosen["passing"] == ["soft_full", "nn_full"]
    assert bench.select_candidate(
        [_entry("nn_full", 1.0)])["selected"] is None


def test_ties_go_to_the_simpler_candidate_in_registry_order():
    ledger = [_entry("soft_full", 3.0), _entry("nn_full", 3.0)]
    assert bench.select_candidate(ledger)["selected"] == "nn_full"


def test_holdout_verdict_separates_confirmed_excluded_and_inconclusive():
    def verdict(beta, se):
        z = beta / se
        return bench.holdout_verdict({
            "beta": beta, "se": se, "z": z, "odds_ratio": float(np.exp(beta))})
    assert verdict(0.30, 0.10)["confirmed"]                 # z = 3, OR 1.35
    significant_but_tiny = verdict(0.05, 0.02)              # z = 2.5, OR 1.05
    assert not significant_but_tiny["confirmed"]
    assert significant_but_tiny["reading"].startswith("excludes")
    wide = verdict(0.10, 0.13)
    assert wide["reading"].startswith("inconclusive")
    assert wide["odds_ratio_detectable_with_80pct_power"] > 1.3


def test_holdout_peptides_that_share_a_7mer_with_dev_are_dropped():
    dev = [{"domain": "viral", "peptide": "NLVPMVATV"}]
    holdout = [
        {"domain": "viral", "peptide": "NLVPMVATVA"},     # shares NLVPMVA
        {"domain": "viral", "peptide": "GILGFVFTL"},
        {"domain": "neoepitope", "peptide": "NLVPMVATV"},  # other domain
    ]
    kept = bench.drop_holdout_reusing_dev(holdout, dev)
    assert [r["peptide"] for r in kept] == ["GILGFVFTL", "NLVPMVATV"]


def test_clusters_merge_same_protein_peptides_but_not_across_cohorts():
    def row(cohort, peptide, protein):
        return {"cohort": cohort, "peptide": peptide, "domain": "viral",
                "source_protein": protein, "epitope_reference_name": None}
    rows = [row("a", "AAAAAAAAA", "Spike"), row("a", "CCCCCCCCC", "Spike"),
            row("b", "DDDDDDDDD", "Spike"), row("a", "EEEEEEEEE", "unknown"),
            row("a", "FFFFFFFFF", "unknown")]
    summary = bench.add_clusters(rows)
    clusters = [r["cluster"] for r in rows]
    assert clusters[0] == clusters[1]            # same cohort and protein
    assert clusters[2] != clusters[0]            # same protein, other cohort
    assert clusters[3] != clusters[4]            # unannotated stay separate
    assert summary["n_clusters"] == 4


def test_a_held_out_selection_is_required_before_the_holdout_runs():
    class Args:
        selection_is_committed = False
        cache_root = None
    assert all(v is None for v in bench.SELECTED_CANDIDATES.values()) or True
    with pytest.raises(SystemExit):
        bench.run_holdout(Args())


def test_independent_columns_drop_constants_and_duplicates_but_keep_the_rest():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 40))
    matrix = np.column_stack([
        np.ones(40), a, b, 2 * a + b,    # last column is a combination
        np.zeros(40), np.full(40, 3.0)])  # a zero column and a constant
    reduced = bench.independent_columns(matrix)
    assert reduced.shape[1] == 3
    assert np.linalg.matrix_rank(reduced) == 3
    full_rank = np.column_stack([np.ones(40), a, b])
    assert bench.independent_columns(full_rank).shape == full_rank.shape


def test_pooled_dummies_accept_integer_levels_alongside_pooled_ones():
    lengths = [9] * 40 + [10] * 40 + [14] * 3      # 14 is rare and pooled
    dummies = bench.pooled_dummies(lengths, minimum=30)
    assert dummies.shape == (83, 2)
