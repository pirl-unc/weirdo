"""Scientific guardrails for the exploratory viral assay benchmark."""

import numpy as np

from weirdo.blosum import blosum62_dict

from scripts.benchmark_viral_tcell import (
    HELD_OUT_VIRUS_FAMILIES,
    PRIMARY_VALIDATION_SCORE,
    _nearest_human_blosum,
    cached_permutation_test,
    deduplicate_assays,
    human_tcr_kmer_surprisal,
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
