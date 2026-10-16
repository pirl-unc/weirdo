"""Scientific guardrails for the exploratory viral assay benchmark."""

import numpy as np

from weirdo.blosum import blosum62_dict

import pytest

from scripts.benchmark_viral_tcell import (
    HELD_OUT_VIRUS_FAMILIES,
    PRIMARY_VALIDATION_SCORE,
    _nearest_human_blosum,
    adjusted_enrichment_models,
    annotate_multiplicity,
    cached_permutation_test,
    cluster_bootstrap_auc,
    composition_baselines,
    deduplicate_assays,
    full_human_blosum_scores,
    human_proteome_stream,
    human_tcr_kmer_surprisal,
    minimum_detectable_auc,
    peptide_clusters,
    prepare_assays,
    primary_validation_outcome,
    stratified_permutation_test,
    tcr_face,
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
        "epitope__species": "Severe acute respiratory syndrome-related coronavirus",
        "first_in_vivo_process__process_type": "Occurrence of infectious disease",
        "second_in_vivo_process__process_type": "Prophylactic vaccination",
    }
    unresolved = {**base, "mhc_restriction__name": "HLA class I"}
    class_ii = {**base, "mhc_restriction__class": "II"}
    result, excluded = prepare_assays([base, unresolved, class_ii], "35484232")
    assert len(result) == 1
    assert result[0]["exposure"] == "infected_then_vaccinated"
    assert excluded["not_an_exact_hla_class_i_canonical_peptide"] == 2


def test_within_hla_auc_does_not_pair_different_alleles():
    rows = [
        {"hla": "HLA-A*02:01", "outcome": "Positive", "score": 0.9},
        {"hla": "HLA-A*02:01", "outcome": "Negative", "score": 0.2},
        {"hla": "HLA-B*07:02", "outcome": "Positive", "score": 0.1},
        {"hla": "HLA-B*07:02", "outcome": "Negative", "score": 0.8},
    ]
    assert within_hla_auc(rows, "score") == (0.5, 2)


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


def _iedb_row(**overrides):
    row = {
        "host__name": "Homo sapiens (human)",
        "mhc_restriction__class": "I",
        "mhc_restriction__name": "HLA-B*07:02",
        "epitope__name": "RPIDDRFGL",
        "assay__qualitative_measurement": "Positive-High",
        "epitope__species": "Yellow fever virus",
        "assay_id": 1,
        "first_in_vivo_process__process_type": "Administration in vivo",
        "second_in_vivo_process__process_type": None,
    }
    row.update(overrides)
    return row


def test_graded_positives_are_one_positive_class_with_provenance():
    rows = [
        _iedb_row(assay__qualitative_measurement=grade)
        for grade in (
            "Positive", "Positive-High", "Positive-Intermediate",
            "Positive-Low", "Negative", "Positive-Unknown", None,
        )
    ]
    result, excluded = prepare_assays(rows, "28386132")
    assert [r["outcome"] for r in result] == (
        ["Positive"] * 4 + ["Negative"])
    assert excluded["outcome_not_positive_or_negative"] == 2
    assert result[1]["qualitative_measurement"] == "Positive-High"
    # Live attenuated vaccine administration shares the vaccinated stratum.
    assert {r["exposure"] for r in result} == {"vaccinated"}


def test_one_shared_panel_is_a_single_cohort_not_two_replications():
    rows = (
        prepare_assays([_iedb_row()], "28386132")[0]
        + prepare_assays([_iedb_row(
            assay__qualitative_measurement="Positive",
            assay_id=2,
            epitope__source_organism="Yellow fever virus 17D",
            first_in_vivo_process__process_type="Prophylactic vaccination",
        )], "32983097")[0]
    )
    assert {r["cohort"] for r in rows} == {"yellow_fever_17d"}
    deduped, conflicts = deduplicate_assays(rows)
    assert not conflicts
    assert len(deduped) == 1
    assert deduped[0]["n_assay_rows"] == 2
    assert deduped[0]["contributing_pmids"] == "28386132|32983097"
    assert deduped[0]["role"] == "validation"
    # No annotation is resolved by fetch order: every value is kept.
    assert deduped[0]["assay_id"] == "1|2"
    assert deduped[0]["qualitative_measurement"] == "Positive|Positive-High"


def test_flavivirus_holdout_covers_the_validation_virus_family():
    flavivirus = HELD_OUT_VIRUS_FAMILIES["flavivirus"]
    assert flavivirus.search("POLG_YEFV1")
    assert flavivirus.search("POLG_DEN2H")
    assert flavivirus.search("POLG_ZIKV")
    # Dugbe virus is a nairovirus; the bare GBV substring must not catch it.
    assert not flavivirus.search("NCAP_DUGBV")
    assert not flavivirus.search("SPIKE_SARS2")


def _permutation_rows(n=12, positives=6, percentile=0.5, scores=None):
    return [
        {
            "peptide": f"PEPTIDE{i:03d}",
            "assay_id": i,
            "hla": "HLA-A*02:01",
            "outcome": "Positive" if i < positives else "Negative",
            PRIMARY_VALIDATION_SCORE: (
                1.0 - 0.05 * i if scores is None else scores[i]),
            "mhcflurry_presentation_percentile": (
                percentile if not isinstance(percentile, list)
                else percentile[i]),
        }
        for i in range(n)
    ]


def test_permutation_test_matches_the_reported_auc_and_detects_signal():
    separated = _permutation_rows()
    result = stratified_permutation_test(
        separated, n_permutations=500, seed=0)
    assert (result["auc"], result["n_pairs"]) == (1.0, 36)
    assert result["auc"] == within_hla_auc(
        separated, PRIMARY_VALIDATION_SCORE)[0]
    assert result["p_value"] < 0.01
    assert result["p_value_standard_error"] < 0.01
    assert result["resolved_at_alpha"]

    reversed_rows = [
        {**row, PRIMARY_VALIDATION_SCORE: -row[PRIMARY_VALIDATION_SCORE]}
        for row in separated
    ]
    reversed_result = stratified_permutation_test(
        reversed_rows, n_permutations=500, seed=0)
    assert reversed_result["auc"] == 0.0
    assert reversed_result["p_value"] > 0.9


def test_permutation_p_value_is_invariant_to_batching_and_row_order():
    rows = _permutation_rows(
        n=14, positives=5,
        percentile=[0.1 * i for i in range(14)],
        scores=[0.4, 0.9, 0.2, 0.7, 0.5, 0.3, 0.8, 0.1, 0.6, 0.45,
                0.95, 0.15, 0.55, 0.25])
    reference = stratified_permutation_test(
        rows, presentation_caliper=0.1, n_permutations=400, seed=3)
    # chunk only bounds memory, and upstream row order is not a scientific
    # input, so neither may move a pre-registered p-value.
    for chunk in (1, 7, 400, 1000):
        batched = stratified_permutation_test(
            rows, presentation_caliper=0.1, n_permutations=400, seed=3,
            chunk=chunk)
        assert batched == reference
    shuffled = stratified_permutation_test(
        list(reversed(rows)), presentation_caliper=0.1, n_permutations=400,
        seed=3)
    assert shuffled == reference


def test_permutations_without_an_eligible_pair_leave_the_denominator():
    # One positive sits beside one in-caliper negative; permutations that move
    # the positive onto the far row have no comparable pair at all.
    rows = _permutation_rows(
        n=3, positives=1, percentile=[0.0, 0.05, 9.0],
        scores=[0.9, 0.1, 0.2])
    result = stratified_permutation_test(
        rows, presentation_caliper=0.1, n_permutations=300, seed=0)
    assert (result["auc"], result["n_pairs"]) == (1.0, 1)
    assert result["n_undefined_permutations"] > 0
    assert result["n_permutations"] == 300 - result["n_undefined_permutations"]


def test_permutation_p_value_respects_the_presentation_caliper():
    rows = _permutation_rows(
        n=3, positives=1, percentile=[0.05, 0.08, 1.0],
        scores=[0.9, 0.1, 0.2])
    result = stratified_permutation_test(
        rows, presentation_caliper=0.1, n_permutations=200, seed=0)
    assert (result["auc"], result["n_pairs"]) == (1.0, 1)


def test_a_missing_validation_cohort_fails_the_conjunctive_rule():
    rows = [
        {**row, "cohort": "yellow_fever_17d", "exposure": "vaccinated",
         "source_species": "Yellow fever virus"}
        for row in _permutation_rows()
    ]
    try:
        primary_validation_outcome(
            rows, [], 200, 0, cached_permutation_test())
    except RuntimeError as error:
        assert "multivirus_concise_set" in str(error)
    else:
        raise AssertionError("a cohort with no assays must not be skipped")


def test_vectorized_nearest_human_distance_matches_the_definition():
    rng = np.random.default_rng(0)
    alphabet = "ACDEFGHIKLMNPQRSTVWY"

    def random_peptides(count, lengths):
        return [
            "".join(rng.choice(list(alphabet), size=int(length)))
            for length in rng.choice(lengths, size=count)
        ]

    reference = random_peptides(40, [9, 10, 11])
    queries = random_peptides(12, [9, 10, 11])
    expected = []
    for peptide in queries:
        face = tcr_face(peptide)
        best = max(
            sum(blosum62_dict[a][b] for a, b in zip(face, tcr_face(other)))
            for other in reference if len(other) == len(peptide)
        )
        expected.append(
            (sum(blosum62_dict[aa][aa] for aa in face) - best) / len(face))
    assert np.allclose(_nearest_human_blosum(reference, queries), expected)


def test_binding_only_readouts_are_excluded_from_response_outcomes():
    activation = _iedb_row(
        assay__method="biological activity",
        assay__qualitative_measurement="Negative",
        mhc_restriction__name="HLA-A*02:01",
        epitope__name="GLCPHCINV",
        epitope__species="Erythroparvovirus primate1",
        first_in_vivo_process__process_type="No immunization",
    )
    binding = {**activation, "assay__method": "multimer/tetramer",
               "assay__qualitative_measurement": "Positive"}
    result, excluded = prepare_assays([activation, binding], "38608022")
    assert excluded["readout_not_a_t_cell_response"] == 1
    assert len(result) == 1
    assert result[0]["outcome"] == "Negative"
    assert result[0]["assay_method"] == "biological activity"
    assert result[0]["exposure"] == "natural_exposure"


def test_benchmark_nearest_human_matches_the_shipped_similarity_metric():
    """Pin the pre-registered score to the package's own distance definition."""
    from weirdo.scorers.similarity import SimilarityScorer

    scorer = SimilarityScorer(matrix="blosum62")
    reference = ["AAKLVTGGCY", "MQWERTYIPA", "GGSSTTNNQQ"]
    queries = ["AAKLVSGGCY", "MQWEKTYIPA"]
    expected = [
        min(
            scorer._sequence_distance(tcr_face(query), tcr_face(candidate))
            for candidate in reference
        ) / len(tcr_face(query))
        for query in queries
    ]
    assert np.allclose(_nearest_human_blosum(reference, queries), expected)


def test_peptide_clusters_link_nested_and_overlapping_but_not_unrelated():
    clusters = peptide_clusters([
        "RPIDDRFGL",       # parent
        "RPIDDRFGLA",      # C-terminal extension
        "PIDDRFGL",        # nested substring
        "SPRERLVLTL",      # unrelated
    ])
    assert len({clusters["RPIDDRFGL"], clusters["RPIDDRFGLA"],
                clusters["PIDDRFGL"]}) == 1
    assert clusters["SPRERLVLTL"] != clusters["RPIDDRFGL"]
    # Reordering the input must not rename a cluster.
    assert peptide_clusters(["SPRERLVLTL", "PIDDRFGL", "RPIDDRFGLA",
                             "RPIDDRFGL"]) == clusters


def _clustered_rows(duplicates, n_clusters=12, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        outcome = "Positive" if c % 2 == 0 else "Negative"
        value = rng.normal(shift if outcome == "Positive" else 0.0)
        for d in range(duplicates):
            rows.append({
                "peptide": f"PEP{c:02d}X{d}", "peptide_cluster": f"c{c}",
                "assay_id": f"{c}-{d}", "hla": "HLA-A*02:01",
                "outcome": outcome,
                PRIMARY_VALIDATION_SCORE: value + rng.normal(0, 0.01),
                "mhcflurry_presentation_percentile": 0.5,
            })
    return rows


def test_cluster_bootstrap_widens_uncertainty_when_rows_are_not_independent():
    independent = cluster_bootstrap_auc(
        _clustered_rows(1, n_clusters=24), n_bootstrap=600, seed=1)
    # Same evidence repeated five times per cluster: the point estimate is
    # unchanged but the effective sample size is not.
    repeated = cluster_bootstrap_auc(
        _clustered_rows(5, n_clusters=24), n_bootstrap=600, seed=1)
    assert independent["n_clusters"] == repeated["n_clusters"] == 24
    assert repeated["standard_error"] > 0.5 * independent["standard_error"]
    assert repeated["minimum_detectable_auc"] > 0.5

    separated = cluster_bootstrap_auc(
        _clustered_rows(3, n_clusters=24, shift=5.0), n_bootstrap=600, seed=1)
    assert separated["auc"] > 0.95
    assert separated["ci_low"] > 0.5
    assert separated["p_value"] < 0.01
    # Bit-for-bit reproducible at a fixed seed.
    assert separated == cluster_bootstrap_auc(
        _clustered_rows(3, n_clusters=24, shift=5.0), n_bootstrap=600, seed=1)


def test_minimum_detectable_auc_grows_with_standard_error():
    assert minimum_detectable_auc(0.0) == 0.5
    assert minimum_detectable_auc(0.05) < minimum_detectable_auc(0.10)
    assert minimum_detectable_auc(0.05) == pytest.approx(
        0.5 + 2.4865 * 0.05, abs=1e-3)


def test_full_human_search_matches_brute_force_across_chunk_boundaries(tmp_path):
    swissprot = tmp_path / "swissprot.csv"
    proteins = ["MKTAYIAKQRQISFVKSHFSRQ", "GGSSTTNNQQAAKLVTGGCYNVA",
                "XXMKTAYXX"]  # ambiguous residues must never match
    swissprot.write_text(
        "identifier,sequence,category\n"
        + "".join(f"h{i},{s},human\n" for i, s in enumerate(proteins))
        + "v1,MQWERTYIPAAAAA,viruses\n")
    queries = ["MKTAYIAKQ", "AAKLVSGGCYN", "MQWERTYIP", "GGSSTTNN"]
    reference = {}
    for length in {len(tcr_face(q)) for q in queries}:
        reference[length] = [
            protein[i:i + length] for protein in proteins
            for i in range(len(protein) - length + 1)
            if set(protein[i:i + length]) <= set("ACDEFGHIKLMNPQRSTVWY")]
    stream = human_proteome_stream(swissprot)
    for chunk in (3, 7, 1000):
        scores = full_human_blosum_scores(
            stream, queries, beta=1.0, chunk=chunk, query_batch=2)
        for i, query in enumerate(queries):
            face = tcr_face(query)
            similarities = np.asarray([
                sum(blosum62_dict[a][b] for a, b in zip(face, window))
                for window in reference[len(face)]], dtype=float)
            self_similarity = sum(blosum62_dict[a][a] for a in face)
            assert scores["nearest_human_full_blosum62_tcr"][i] == (
                pytest.approx((self_similarity - similarities.max())
                              / len(face)))
            soft = np.log(np.exp(similarities).sum())
            assert scores["soft_nearest_human_full_blosum62_tcr"][i] == (
                pytest.approx((self_similarity - soft) / len(face)))
    # An exact human window has zero exact distance to self.
    exact = full_human_blosum_scores(stream, ["AMKTAYIAKQRQ"])
    assert exact["nearest_human_full_blosum62_tcr"][0] == pytest.approx(0.0)


def test_composition_baselines_need_no_self_reference():
    scores = composition_baselines(["AAFWYAAAL", "AAAAAAAAL"])
    assert scores["tcr_face_aromatic_fraction"][0] > (
        scores["tcr_face_aromatic_fraction"][1])
    assert scores["tcr_face_hydrophobicity"].shape == (2,)


def test_multiplicity_correction_scales_only_the_requested_scores():
    def row(score, p):
        return {"score": score, "permutation_tests": {
            "a": {"p_value": p}, "b": {"p_value": None}}}
    results = [row("kept", 0.01), row("kept", 0.2), row("ignored", 0.001)]
    family = annotate_multiplicity(results, ("kept",))
    assert family == 2
    assert results[0]["permutation_tests"]["a"]["p_bonferroni"] == (
        pytest.approx(0.02))
    assert results[1]["permutation_tests"]["a"]["p_bonferroni"] == (
        pytest.approx(0.4))
    assert "p_bonferroni" not in results[2]["permutation_tests"]["a"]


def test_adjusted_model_recovers_a_planted_effect_and_ignores_noise():
    pytest.importorskip("statsmodels")
    rng = np.random.default_rng(0)
    rows = []
    for i in range(240):
        signal = rng.normal()
        outcome = rng.random() < 1 / (1 + np.exp(-1.5 * signal))
        rows.append({
            "cohort": "c", "peptide": f"P{i:03d}",
            "peptide_cluster": f"k{i // 2}", "hla": f"A{i % 3}",
            "source_species": f"s{i % 4}",
            "outcome": "Positive" if outcome else "Negative",
            "mhcflurry_presentation_score": float(rng.uniform(0.05, 0.95)),
            "signal": float(signal), "noise": float(rng.normal()),
            "baseline": float(rng.normal()),
        })
    results = adjusted_enrichment_models(
        rows, ["signal", "noise"], ("baseline",), ("c",))
    by = {(r["variant"], r["score"]): r for r in results}
    assert by[("presentation_hla_species", "signal")]["fit_ok"]
    assert by[("presentation_hla_species", "signal")][
        "one_sided_p_positive"] < 0.001
    assert by[("presentation_hla_species", "signal")][
        "odds_ratio_per_sd"] > 2
    assert by[("plus_composition_baselines", "noise")][
        "one_sided_p_positive"] > 0.01
    assert ("presentation_hla_species", "baseline") not in by
    # Baselines are reported on their own, and scores carry a multiplicity
    # adjusted p scaled by the number of scores tried.
    assert by[("plus_composition_baselines", "__baseline__baseline")]["fit_ok"]
    signal = by[("presentation_hla_species", "signal")]
    assert signal["p_bonferroni_across_scores"] == pytest.approx(
        min(1.0, signal["one_sided_p_positive"] * 2))
