"""Wave 13 coherence tests: pepdata deduplication, datacache CLI help, and --cache-root."""

import json
from pathlib import Path

import pytest
from datacache.cli_help import HelpArgumentParser

import pepdata.amino_acid as pepdata_amino_acid
import pepdata.amino_acid_alphabet as pepdata_alphabet
import pepdata.amino_acid_properties as pepdata_properties
import pepdata.blosum as pepdata_blosum
import pepdata.chou_fasman as pepdata_chou_fasman
import pepdata.common as pepdata_common
import pepdata.peptide_vectorizer as pepdata_vectorizer
import pepdata.pmbec as pepdata_pmbec
import pepdata.residue_contact_energies as pepdata_rce
import pepdata.static_data as pepdata_static_data

import weirdo
from weirdo import (
    amino_acid,
    amino_acid_alphabet,
    amino_acid_properties,
    blosum,
    chou_fasman,
    cli,
    common,
    peptide_vectorizer,
    pmbec,
    reduced_alphabet,
    residue_contact_energies,
    static_data,
)


def test_weirdo_matrices_directory_removed():
    package_dir = Path(weirdo.__file__).resolve().parent
    assert not (package_dir / "matrices").exists()
    assert static_data.MATRIX_DIR == pepdata_static_data.MATRIX_DIR
    assert Path(static_data.MATRIX_DIR).is_dir()


def test_weirdo_reexports_from_pepdata_and_blosum50_wt_fixed():
    assert amino_acid.AminoAcid is pepdata_amino_acid.AminoAcid
    assert amino_acid_alphabet.canonical_amino_acids is pepdata_alphabet.canonical_amino_acids
    assert amino_acid_alphabet.dict_to_amino_acid_matrix is pepdata_alphabet.dict_to_amino_acid_matrix
    assert amino_acid_properties.hydropathy is pepdata_properties.hydropathy
    assert amino_acid_properties.solvent_exposed_area is pepdata_properties.solvent_exposed_area
    assert blosum.blosum30_dict is pepdata_blosum.blosum30_dict
    assert blosum.blosum50_dict is pepdata_blosum.blosum50_dict
    assert blosum.blosum62_dict is pepdata_blosum.blosum62_dict
    assert blosum.blosum50_dict["W"]["T"] == -3
    assert blosum.blosum50_dict["T"]["W"] == -3
    assert chou_fasman.alpha_helix_score is pepdata_chou_fasman.alpha_helix_score
    assert chou_fasman.beta_sheet_score is pepdata_chou_fasman.beta_sheet_score
    assert chou_fasman.turn_score is pepdata_chou_fasman.turn_score
    assert common.transform_peptide is pepdata_common.transform_peptide
    assert common.transform_peptides is pepdata_common.transform_peptides
    assert peptide_vectorizer.PeptideVectorizer is pepdata_vectorizer.PeptideVectorizer
    assert weirdo.PeptideVectorizer is pepdata_vectorizer.PeptideVectorizer
    assert pmbec.pmbec_dict is pepdata_pmbec.pmbec_dict
    assert pmbec.pmbec_matrix is pepdata_pmbec.pmbec_matrix
    assert residue_contact_energies.strand_vs_coil_dict is pepdata_rce.strand_vs_coil_dict
    assert residue_contact_energies.strand_vs_coil is pepdata_rce.strand_vs_coil_dict
    assert residue_contact_energies.helix_vs_coil is pepdata_rce.helix_vs_coil_dict


def test_reduced_alphabet_order_and_representatives_preserved():
    expected_order = [
        "gbmr4",
        "sdm12",
        "hsdm17",
        "hp2",
        "murphy8",
        "murphy10",
        "murphy15",
        "alex6",
        "aromatic2",
        "hp_vs_aromatic",
    ]
    assert list(reduced_alphabet.alphabets.keys()) == expected_order
    # sorted(group)[0] representatives preserved
    assert reduced_alphabet.dict_from_list(["LVIM"]) == {
        "L": "I",
        "V": "I",
        "I": "I",
        "M": "I",
    }
    assert reduced_alphabet.alphabets["murphy10"]["L"] == "I"


def test_cli_uses_help_argument_parser_argcomplete_marker_and_version(capsys):
    cli_source = Path(cli.__file__).read_text(encoding="utf-8")
    assert cli_source.splitlines()[0] == "# PYTHON_ARGCOMPLETE_OK"

    parser = cli.create_parser()
    assert isinstance(parser, HelpArgumentParser)

    with pytest.raises(SystemExit) as help_exc:
        cli.run(["--help"])
    assert help_exc.value.code == 0
    help_out = capsys.readouterr().out
    assert "usage: weirdo" in help_out
    assert "--version" in help_out

    with pytest.raises(SystemExit) as ver_exc:
        cli.run(["--version"])
    assert ver_exc.value.code == 0
    ver_out = capsys.readouterr().out.strip()
    assert ver_out == f"weirdo {weirdo.__version__}"


def test_cli_data_commands_support_cache_root_and_data_dir_alias(tmp_path, monkeypatch, capsys):
    env_dir = tmp_path / "env_weirdo"
    override_dir = tmp_path / "explicit_cache"
    alias_dir = tmp_path / "alias_cache"
    monkeypatch.setenv("WEIRDO_DATA_DIR", str(env_dir))

    # data dirs --cache-root --json resolves override without creating directory
    assert cli.run(["data", "dirs", "--cache-root", str(override_dir), "--json"]) == 0
    dirs_payload = json.loads(capsys.readouterr().out)
    assert dirs_payload["data_dir"] == str(override_dir)
    assert dirs_payload["origin"] == "override"
    assert dirs_payload["exists"] is False
    assert not override_dir.exists()

    # data status / list / ls accept --cache-root and --data-dir
    assert cli.run(["data", "status", "--data-dir", str(alias_dir), "--json"]) == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["data_dir"] == str(alias_dir)
    assert not alias_dir.exists()

    # data verify --cache-root
    assert cli.run(["data", "verify", "--cache-root", str(override_dir), "--json"]) == 0
    verify_payload = json.loads(capsys.readouterr().out)
    assert verify_payload["data_dir"] == str(override_dir)
    assert verify_payload["ok"] is True

    # data prune --cache-root
    assert cli.run(["data", "prune", "--cache-root", str(override_dir), "--json"]) == 0
    prune_payload = json.loads(capsys.readouterr().out)
    assert prune_payload["data_dir"] == str(override_dir)
    assert prune_payload["removed_count"] == 0

    # data download --cache-root --offline
    assert cli.run(
        ["data", "download", "swissprot-sequences", "--cache-root", str(override_dir), "--offline"]
    ) == 1
    assert "Offline mode is enabled" in capsys.readouterr().out

    # data clear --cache-root -y
    assert cli.run(["data", "clear", "--cache-root", str(override_dir), "-y"]) == 0
    assert "Deleted 0 downloads" in capsys.readouterr().out

    # Subsequent call without --cache-root still resolves env_dir (singleton not poisoned)
    assert cli.run(["data", "dirs", "--json"]) == 0
    default_payload = json.loads(capsys.readouterr().out)
    assert default_payload["data_dir"] == str(env_dir)
