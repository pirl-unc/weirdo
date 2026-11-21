import numpy as np
import pandas as pd

from scripts import build_source_context_features as context
from scripts import fetch_uniprot_sequences as uniprot


def test_locate_returns_none_for_absent_and_hint_breaks_repeat_ties():
    sequence = "MKKLLPAAAAAAAAAAMKKLLPGGGG"
    assert context.locate(sequence, "WWWWWWWW", np.nan) is None
    assert context.locate(sequence, "MKKLLP", np.nan) == 0
    assert context.locate(sequence, "MKKLLP", 17.0) == 16  # second copy, 1-based hint 17


def test_build_maps_peptide_and_extracts_flanks_and_position():
    sequence = "ACDEFGHIKLMNPQRSTVWY" * 3
    pool = pd.DataFrame({"pmid": ["1"], "peptide": ["GHIKLMNPQ"], "hla": ["HLA-A*02:01"]})
    raw = pd.DataFrame(
        {
            "reference__pmid": ["1"], "assay_antigen__name": ["GHIKLMNPQ"], "epitope__name": ["GHIKLMNPQ"],
            "mhc_restriction__name": ["HLA-A*02:01"],
            "epitope__molecule_parent_iri": ["http://www.uniprot.org/uniprot/P12345"],
            "epitope__starting_position": [6],
        }
    )
    row = context.build(pool, raw, {"P12345": sequence}).iloc[0]
    assert row["ctx_mapped"] == 1 and row["accession"] == "P12345"
    assert row["ctx_protein_length"] == 60
    assert row["ctx_from_n"] == 5 and row["ctx_from_c"] == 60 - 14
    assert row["n_flank"] == sequence[:5] and len(row["c_flank"]) == 10


def test_unmapped_pairs_keep_nan_features():
    pool = pd.DataFrame({"pmid": ["1"], "peptide": ["AAAAAAAAA"], "hla": ["HLA-A*02:01"]})
    raw = pd.DataFrame(
        {
            "reference__pmid": ["1"], "assay_antigen__name": ["AAAAAAAAA"], "epitope__name": ["AAAAAAAAA"],
            "mhc_restriction__name": ["HLA-A*02:01"], "epitope__molecule_parent_iri": [None],
            "epitope__starting_position": [np.nan],
        }
    )
    row = context.build(pool, raw, {}).iloc[0]
    assert row["ctx_mapped"] == 0 and np.isnan(row["ctx_protein_length"])


def test_fasta_parsing_extracts_accessions_and_joins_wrapped_lines():
    text = ">sp|P03141|HBSAG_HBVA3 Large envelope protein\nMGGWSS\nKPRKGM\n>tr|A0A0|X desc\nAAAA\n"
    assert uniprot.parse_fasta(text) == [("P03141", "MGGWSSKPRKGM"), ("A0A0", "AAAA")]
