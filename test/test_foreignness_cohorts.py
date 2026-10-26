"""Guardrails for cohort assignment and assay preparation."""

from scripts.foreignness_cohorts import (
    FORCED_DEV,
    FORCED_HOLDOUT,
    assign_role,
    cohort_specs,
    deduplicate,
    eligibility_audit,
    prepare_cohort,
)


def _row(**overrides):
    row = {
        "host__name": "Homo sapiens (human)",
        "mhc_restriction__class": "I",
        "mhc_restriction__name": "HLA-A*02:01",
        "epitope__name": "NLVPMVATV",
        "assay__qualitative_measurement": "Positive-High",
        "assay__method": "ELISPOT",
        "effector_cell__culture_condition": "Direct Ex Vivo",
        "epitope__species": "Human betaherpesvirus 5",
        "assay_id": 1,
        "reference__pmid": "1",
        "assay__number_of_subjects_tested": 3,
    }
    row.update(overrides)
    return row


def _spec(domain="viral", **extra):
    return {"cohort": "c", "domain": domain, "role": "dev", **extra}


def test_roles_are_deterministic_and_forced_cohorts_keep_their_side():
    assert all(assign_role(c) == "dev" for c in FORCED_DEV)
    assert all(assign_role(c) == "holdout" for c in FORCED_HOLDOUT)
    assert assign_role("12345678") == assign_role("12345678")
    held = sum(assign_role(str(10_000_000 + i)) == "holdout"
               for i in range(2000))
    assert 0.20 < held / 2000 < 0.30   # a quarter, by hash


def test_every_registered_cohort_has_exactly_one_domain_and_role():
    specs = cohort_specs()
    assert {s["domain"] for s in specs.values()} == {"viral", "neoepitope"}
    assert {s["role"] for s in specs.values()} == {
        "dev", "holdout", "confirm2"}
    assert specs["33038342"]["role"] == "holdout"
    assert specs["vaccinia_dryvax"]["role"] == "holdout"
    assert specs["38608022"]["assay_methods"] == ("biological activity",)


def test_preparation_logs_each_reason_a_row_is_excluded():
    rows = [
        _row(),
        _row(mhc_restriction__class="II"),
        _row(mhc_restriction__name="HLA class I"),
        _row(effector_cell__culture_condition="Engineered"),
        _row(assay__method="surface plasmon resonance (SPR)"),
        _row(assay__qualitative_measurement="Positive-Unknown"),
        _row(epitope__species=None, assay_antigen__species=None,
             epitope__source_organism=None, assay_antigen__source_organism=None),
    ]
    prepared, excluded = prepare_cohort(rows, _spec())
    assert len(prepared) == 1
    assert prepared[0]["outcome"] == "Positive"
    assert excluded["not_an_exact_hla_class_i_canonical_peptide"] == 2
    assert excluded["culture_not_direct_or_restimulated"] == 1
    assert excluded["binding_only_readout"] == 1
    assert excluded["outcome_not_positive_or_negative"] == 1
    assert excluded["missing_source_species"] == 1


def test_neoepitope_rows_need_no_source_species():
    row = _row(epitope__species=None, assay_antigen__species=None,
               epitope__source_organism=None,
               assay_antigen__source_organism=None)
    prepared, excluded = prepare_cohort([row], _spec("neoepitope"))
    assert prepared[0]["source_species"] == "tumour neoepitope"
    assert not excluded


def test_cohort_specific_readout_filter_is_applied():
    rows = [
        _row(assay__method="biological activity"),
        _row(assay__method="multimer/tetramer", assay_id=2),
    ]
    prepared, excluded = prepare_cohort(
        rows, _spec(assay_methods=("biological activity",)))
    assert len(prepared) == 1
    assert excluded["readout_not_a_t_cell_response"] == 1


def test_a_pair_is_positive_if_any_assay_is_positive():
    base = [
        _row(assay_id=1, assay__qualitative_measurement="Negative"),
        _row(assay_id=2, assay__qualitative_measurement="Positive-Low"),
        _row(assay_id=3, assay__qualitative_measurement="Negative",
             epitope__name="GILGFVFTL"),
    ]
    prepared, _ = prepare_cohort(base, _spec())
    deduped, mixed = deduplicate(prepared)
    by_peptide = {r["peptide"]: r for r in deduped}
    mixed_pair = by_peptide["NLVPMVATV"]
    assert mixed_pair["outcome"] == "Positive"
    assert (mixed_pair["n_positive_assays"], mixed_pair["n_negative_assays"]
            ) == (1, 1)
    assert mixed_pair["n_assay_rows"] == 2
    assert mixed_pair["assay_id"] == "1|2"
    assert by_peptide["GILGFVFTL"]["outcome"] == "Negative"
    assert mixed["c"] == 1   # counted, not dropped


def test_subjects_tested_keeps_the_largest_count_and_tolerates_missing():
    rows = [
        _row(assay_id=1, assay__number_of_subjects_tested=2),
        _row(assay_id=2, assay__number_of_subjects_tested=7),
        _row(assay_id=3, assay__number_of_subjects_tested=None,
             epitope__name="GILGFVFTL"),
    ]
    prepared, _ = prepare_cohort(rows, _spec())
    deduped, _ = deduplicate(prepared)
    counts = {r["peptide"]: r["n_subjects_tested"] for r in deduped}
    assert counts == {"NLVPMVATV": 7, "GILGFVFTL": None}


def test_eligibility_thresholds_separate_pooled_from_reported_cohorts():
    def rows(cohort, n_pos, n_neg):
        return [{"cohort": cohort, "domain": "viral", "role": "dev",
                 "outcome": "Positive" if i < n_pos else "Negative",
                 "peptide": f"P{i}", "hla": "HLA-A*02:01"}
                for i in range(n_pos + n_neg)]
    audit = {a["cohort"]: a for a in eligibility_audit(
        rows("big", 20, 30) + rows("small", 6, 6) + rows("onesided", 40, 2))}
    assert audit["big"]["included"] and audit["big"]["reported_alone"]
    assert audit["small"]["included"] and not audit["small"]["reported_alone"]
    assert not audit["onesided"]["included"]
    assert audit["onesided"]["reason"]


def test_the_second_confirmation_set_is_disjoint_from_everything_already_used():
    from scripts.foreignness_cohorts import (
        CONFIRM2_EXCLUDED_PMIDS,
        CONFIRM2_NEOEPITOPE_PMIDS,
        NEOEPITOPE_PMIDS,
        VIRAL_PMIDS,
    )
    confirm2 = set(CONFIRM2_NEOEPITOPE_PMIDS)
    assert len(confirm2) == len(CONFIRM2_NEOEPITOPE_PMIDS)       # no repeats
    assert not confirm2 & set(NEOEPITOPE_PMIDS) | confirm2 & set(VIRAL_PMIDS)
    assert not confirm2 & {"28386132", "32983097"}
    assert not confirm2 & set(CONFIRM2_EXCLUDED_PMIDS)
    specs = cohort_specs()
    assert {c for c, s in specs.items() if s["role"] == "confirm2"} == confirm2
    assert all(specs[c]["domain"] == "neoepitope" for c in confirm2)


def test_the_pooled_inclusion_threshold_can_be_loosened_without_touching_reporting():
    rows = [{"cohort": "c", "domain": "viral", "role": "dev",
             "outcome": "Positive" if i < 2 else "Negative",
             "peptide": f"P{i}", "hla": "HLA-A*02:01"} for i in range(12)]
    strict = eligibility_audit(rows)[0]
    loose = eligibility_audit(rows, minimum=1)[0]
    assert not strict["included"] and loose["included"]
    assert not loose["reported_alone"]           # still too small to stand alone
