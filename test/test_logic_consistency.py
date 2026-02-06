import importlib

from weirdo.chou_fasman import alpha_helix_score, beta_sheet_score, turn_score
from weirdo.residue_contact_energies import (
    helix_vs_coil,
    helix_vs_coil_dict,
    helix_vs_strand,
    helix_vs_strand_dict,
    strand_vs_coil,
    strand_vs_coil_dict,
)
from weirdo.scorers import list_scorers


def test_chou_fasman_scores_are_keyed_by_letter():
    assert alpha_helix_score["A"] == 142
    assert beta_sheet_score["V"] == 170
    assert turn_score["P"] == 152


def test_contact_energy_aliases_match_dict_exports():
    assert helix_vs_coil is helix_vs_coil_dict
    assert helix_vs_strand is helix_vs_strand_dict
    assert strand_vs_coil is strand_vs_coil_dict
    assert isinstance(helix_vs_coil["A"]["G"], float)


def test_pmbec_import_is_silent(capsys):
    import weirdo.pmbec as pmbec

    importlib.reload(pmbec)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_similarity_scorer_is_discoverable():
    scorers = list_scorers()
    assert "mlp" in scorers
    assert "similarity" in scorers
