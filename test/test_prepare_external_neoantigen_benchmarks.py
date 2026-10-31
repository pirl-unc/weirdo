import pandas as pd

from scripts.prepare_external_neoantigen_benchmarks import (
    extract_braun,
    extract_braun_constructs,
    extract_tesla,
    parse_braun_alleles,
)


def test_parse_braun_alleles_accepts_compound_annotation():
    assert parse_braun_alleles("C0602, A6601") == ("HLA-A*66:01", "HLA-C*06:02")


def test_extract_tesla_removes_nonconflicting_duplicate_rows(tmp_path):
    path = tmp_path / "tesla.xlsx"
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(
            {
                "peptide": ["SYFPEITHI", "SYFPEITHI", "GLCTLVAML"],
                "target_value": [1, 1, 0],
                "allele": ["HLA-A02:01", "HLA-A02:01", "HLA-A02:01"],
            }
        ).to_excel(writer, sheet_name="TESLA_dataset", index=False)
    result = extract_tesla(path)
    assert len(result) == 2
    assert result["outcome"].tolist() == [1, 0]


def test_extract_braun_collapses_vaccine_construct_repeats(tmp_path):
    path = tmp_path / "braun.xlsx"
    base = {
        "Patient_ID": [101, 101, 102],
        "Gene_and_Protein_Change": ["GENE|p.A1fs", "GENE|p.A1fs", "GENE|p.R2W"],
        "Short_Epitope": ["SYFPEITHI", "SYFPEITHI", "GLCTLVAML"],
        "HLA_of_best_short_epitope": ["A0201", "A0201", "B0702"],
        "Variant_Type": ["DEL", "DEL", "SNV"],
        "Ttest_pvalue_InVitroStim": [0.01, 0.2, 0.01],
        "InVitro_PeptideStim_Replicate01": [8.0, 1.0, 2.0],
        "InVitro_NoStim_Replicate01": [1.0, 1.0, 1.0],
    }
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(base).to_excel(writer, sheet_name="In Vitro", index=False)
    result = extract_braun(path)
    assert len(result) == 2
    frameshift = result.loc[result["mutation_class"].eq("frameshift")].iloc[0]
    assert frameshift["outcome"] == 1
    assert frameshift["alleles"] == ("HLA-A*02:01",)


def test_extract_braun_constructs_retains_overlapping_vaccine_constructs(tmp_path):
    path = tmp_path / "braun.xlsx"
    frame = pd.DataFrame(
        {
            "Patient_ID": [101, 101],
            "Peptide_ID": ["one", "two"],
            "Vaccine_Peptide": ["AAASYFPEITHI", "BBBBSYFPEITHI"],
            "Gene_and_Protein_Change": ["GENE|p.A1fs", "GENE|p.A1fs"],
            "Short_Epitope": ["SYFPEITHI", "SYFPEITHI"],
            "HLA_of_best_short_epitope": ["A0201", "A0201"],
            "Variant_Type": ["DEL", "DEL"],
            "Ttest_pvalue_InVitroStim": [0.01, 0.2],
            "InVitro_PeptideStim_Replicate01": [8.0, 1.0],
            "InVitro_NoStim_Replicate01": [1.0, 1.0],
        }
    )
    with pd.ExcelWriter(path) as writer:
        frame.to_excel(writer, sheet_name="In Vitro", index=False)
    result = extract_braun_constructs(path)
    assert len(result) == 2
    assert result["peptide"].tolist() == ["SYFPEITHI", "SYFPEITHI"]
    assert result["vaccine_peptide"].tolist() == ["AAASYFPEITHI", "BBBBSYFPEITHI"]
