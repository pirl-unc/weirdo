import numpy as np

from scripts import build_hla_self_ligand_neighborhood as nbr


def test_nearest_ligand_statistics_exclude_identical_peptides():
    reference_peptides = np.array(["AAAAAAAAA", "AAKKKKKAA"])
    reference = nbr.encode(reference_peptides)
    query = nbr.encode(["AAAAAAAAA", "AAAAAAAAV"])
    stats = nbr.features_for(query, reference, np.array(["AAAAAAAAA", "AAAAAAAAV"]), reference_peptides)
    # identical peptide: flagged exact hit, nearest other ligand is the K-rich one
    assert stats[0, 4] == 1
    assert stats[0, 0] == 5  # face P3..P8 = AAAAAA vs KKKKKA -> 5 mismatches
    # one substitution at the last (non-face) position: face identical to the first ligand
    assert stats[1, 4] == 0
    assert stats[1, 0] == 0
    assert stats[1, 1] >= np.log1p(1)


def test_blosum_scores_identical_residues_highest():
    assert nbr.BLOSUM[nbr.INDEX["W"], nbr.INDEX["W"]] == 11
    assert nbr.BLOSUM[nbr.INDEX["A"], nbr.INDEX["A"]] == 4
    assert (nbr.BLOSUM == nbr.BLOSUM.T).all()
