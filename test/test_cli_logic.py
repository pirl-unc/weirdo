import pytest

from weirdo.cli import create_parser, run
from weirdo.reduced_alphabet import alphabets


def test_score_command_requires_model_argument():
    parser = create_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["score", "MTMDKSEL"])


def test_translate_fasta_file(tmp_path, capsys):
    fasta = tmp_path / "input.fasta"
    fasta.write_text(">seq1\nACD\nEF\n>seq2\nWXYZ\n")

    code = run(["translate", "--input-fasta", str(fasta), "-a", "hp2"])
    assert code == 0

    out_lines = capsys.readouterr().out.strip().splitlines()
    alphabet = alphabets["hp2"]
    expected_seq1 = "".join(alphabet.get(aa, aa) for aa in "ACDEF")
    expected_seq2 = "".join(alphabet.get(aa, aa) for aa in "WXYZ")

    assert out_lines == [">seq1", expected_seq1, ">seq2", expected_seq2]


def test_translate_fasta_requires_header(tmp_path, capsys):
    fasta = tmp_path / "bad.fasta"
    fasta.write_text("ACDEFG\n")

    code = run(["translate", "--input-fasta", str(fasta), "-a", "hp2"])
    assert code == 1
    assert "must contain at least one header line" in capsys.readouterr().out


def test_data_clear_rejects_conflicting_flags(capsys):
    code = run(["data", "clear", "--downloads", "--all", "-y"])
    assert code == 1
    assert "Choose either --downloads or --all" in capsys.readouterr().out


def test_models_available_when_no_registry(capsys, monkeypatch):
    monkeypatch.setattr("weirdo.model_manager.list_pretrained_models", lambda: [])
    code = run(["models", "available"])
    assert code == 0
    assert "No built-in pretrained models are configured." in capsys.readouterr().out


def test_models_download_requires_name_without_url(capsys):
    code = run(["models", "download"])
    assert code == 1
    assert "Provide a pretrained model name" in capsys.readouterr().out


def test_models_download_url_requires_name(capsys):
    code = run(["models", "download", "--url", "https://example.com/model.tar.gz"])
    assert code == 1
    assert "provide a model name" in capsys.readouterr().out.lower()


def test_models_download_custom_url(monkeypatch, capsys):
    called = {}

    def fake_download_model_from_url(name, url, overwrite=False, expected_sha256=None):
        called["name"] = name
        called["url"] = url
        called["overwrite"] = overwrite
        called["expected_sha256"] = expected_sha256
        return f"/tmp/{name}"

    monkeypatch.setattr("weirdo.model_manager.download_model_from_url", fake_download_model_from_url)

    code = run(
        [
            "models",
            "download",
            "--url",
            "https://example.com/model.tar.gz",
            "--save-as",
            "my-model",
            "--sha256",
            "abc123",
            "--force",
        ]
    )
    assert code == 0
    assert called == {
        "name": "my-model",
        "url": "https://example.com/model.tar.gz",
        "overwrite": True,
        "expected_sha256": "abc123",
    }
    assert "Downloaded model 'my-model'" in capsys.readouterr().out
