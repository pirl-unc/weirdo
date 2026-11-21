import pandas as pd

from scripts.build_braun_mutation_aware_candidates import build_candidates, windows


def test_windows_have_expected_coordinates():
    assert windows("ABCDEFGHI", 8, 8) == [("ABCDEFGH", 0), ("BCDEFGHI", 1)]


def test_builder_keeps_only_reference_absence_as_an_explicit_auditable_flag():
    constructs = pd.DataFrame(
        {
            "record_id": ["row"], "patient": [1], "mutation": ["G|p.A1V"],
            "mutation_class": ["SNV"], "construct_id": ["x"],
            "vaccine_peptide": ["ABCDEFGH"], "outcome": [1],
            "peptide": ["ABCDEFGH"], "alleles": [("HLA-A*01:01",)],
        }
    )
    result = build_candidates(constructs, ["ZZABCDEFGHZZ"])
    assert len(result) == 1
    assert result.loc[0, "present_in_human_reference"]
    assert not result.loc[0, "reference_absent_candidate"]
