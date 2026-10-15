"""Scientific guardrails for the exploratory viral assay benchmark."""

from scripts.benchmark_viral_tcell import prepare_assays, within_hla_auc


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
