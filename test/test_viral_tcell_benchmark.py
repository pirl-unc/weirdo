"""Scientific guardrails for the exploratory viral assay benchmark."""

import numpy as np

from weirdo.blosum import blosum62_dict

from scripts.benchmark_viral_tcell import (
    HELD_OUT_VIRUS_FAMILIES,
    PRIMARY_VALIDATION_SCORE,
    _nearest_human_blosum,
    deduplicate_assays,
    human_tcr_kmer_surprisal,
    prepare_assays,
    tcr_face,
    vectorize,
    within_hla_auc,
    within_hla_permutation_p_value,
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


def _iedb_row(**overrides):
    row = {
        "host__name": "Homo sapiens (human)",
        "mhc_restriction__class": "I",
        "mhc_restriction__name": "HLA-B*07:02",
        "epitope__name": "RPIDDRFGL",
        "assay__qualitative_measurement": "Positive-High",
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
    result = prepare_assays(rows, "28386132")
    assert [r["outcome"] for r in result] == (
        ["Positive"] * 4 + ["Negative"])
    assert result[1]["qualitative_measurement"] == "Positive-High"
    # Live attenuated vaccine administration shares the vaccinated stratum.
    assert {r["exposure"] for r in result} == {"vaccinated"}


def test_one_shared_panel_is_a_single_cohort_not_two_replications():
    rows = (
        prepare_assays([_iedb_row()], "28386132")
        + prepare_assays([_iedb_row(
            assay__qualitative_measurement="Positive",
            first_in_vivo_process__process_type="Prophylactic vaccination",
        )], "32983097")
    )
    assert {r["cohort"] for r in rows} == {"yellow_fever_17d"}
    deduped, conflicts = deduplicate_assays(rows)
    assert not conflicts
    assert len(deduped) == 1
    assert deduped[0]["n_assay_rows"] == 2
    assert deduped[0]["contributing_pmids"] == "28386132|32983097"
    assert deduped[0]["role"] == "validation"


def test_flavivirus_holdout_covers_the_validation_virus_family():
    flavivirus = HELD_OUT_VIRUS_FAMILIES["flavivirus"]
    assert flavivirus.search("POLG_YEFV1")
    assert flavivirus.search("POLG_DEN2H")
    assert flavivirus.search("POLG_ZIKV")
    # Dugbe virus is a nairovirus; the bare GBV substring must not catch it.
    assert not flavivirus.search("NCAP_DUGBV")
    assert not flavivirus.search("SPIKE_SARS2")


def test_permutation_p_value_matches_the_reported_auc_and_detects_signal():
    separated = [
        {
            "hla": "HLA-A*02:01",
            "outcome": "Positive" if i < 6 else "Negative",
            PRIMARY_VALIDATION_SCORE: 1.0 - 0.05 * i,
            "mhcflurry_presentation_percentile": 0.5,
        }
        for i in range(12)
    ]
    p_value, auc, n_pairs = within_hla_permutation_p_value(
        separated, n_permutations=500, seed=0)
    assert (auc, n_pairs) == (1.0, 36)
    assert auc == within_hla_auc(separated, PRIMARY_VALIDATION_SCORE)[0]
    assert p_value < 0.01

    reversed_rows = [
        {**row, PRIMARY_VALIDATION_SCORE: -row[PRIMARY_VALIDATION_SCORE]}
        for row in separated
    ]
    reversed_p, reversed_auc, _ = within_hla_permutation_p_value(
        reversed_rows, n_permutations=500, seed=0)
    assert reversed_auc == 0.0
    assert reversed_p > 0.9


def test_permutation_p_value_respects_the_presentation_caliper():
    rows = [
        {
            "hla": "HLA-A*02:01", "outcome": "Positive",
            PRIMARY_VALIDATION_SCORE: 0.9,
            "mhcflurry_presentation_percentile": 0.05,
        },
        {
            "hla": "HLA-A*02:01", "outcome": "Negative",
            PRIMARY_VALIDATION_SCORE: 0.1,
            "mhcflurry_presentation_percentile": 0.08,
        },
        {
            "hla": "HLA-A*02:01", "outcome": "Negative",
            PRIMARY_VALIDATION_SCORE: 2.0,
            "mhcflurry_presentation_percentile": 9.0,
        },
    ]
    _, auc, n_pairs = within_hla_permutation_p_value(
        rows, presentation_caliper=0.1, n_permutations=200, seed=0)
    assert (auc, n_pairs) == (1.0, 1)


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
        first_in_vivo_process__process_type="No immunization",
    )
    binding = {**activation, "assay__method": "multimer/tetramer",
               "assay__qualitative_measurement": "Positive"}
    result = prepare_assays([activation, binding], "38608022")
    assert len(result) == 1
    assert result[0]["outcome"] == "Negative"
    assert result[0]["assay_method"] == "biological activity"
    assert result[0]["exposure"] == "natural_exposure"
