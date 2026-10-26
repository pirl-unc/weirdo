import json

from weirdo.data_manager import DATASETS, DataManager


def _seed_download(dm: DataManager, name: str = "swissprot-8mers"):
    filepath = dm.downloads_dir / DATASETS[name]["filename"]
    filepath.write_text("dummy\n")
    dm._metadata.setdefault("downloads", {})[name] = {
        "path": str(filepath),
        "downloaded_at": "2026-01-01T00:00:00",
        "size_bytes": filepath.stat().st_size,
    }
    dm._save_metadata()
    return filepath


def test_clear_all_keeps_metadata_file_by_default(tmp_path):
    dm = DataManager(data_dir=tmp_path, verbose=False)
    filepath = _seed_download(dm)

    count = dm.clear_all()
    assert count == 1
    assert not filepath.exists()
    assert dm.metadata_file.exists()

    data = json.loads(dm.metadata_file.read_text())
    assert data["downloads"] == {}


def test_clear_all_with_metadata_reset_removes_metadata_file(tmp_path):
    dm = DataManager(data_dir=tmp_path, verbose=False)
    filepath = _seed_download(dm)

    count = dm.clear_all(include_metadata=True)
    assert count == 1
    assert not filepath.exists()
    assert not dm.metadata_file.exists()
    assert dm._metadata == {"downloads": {}}


# ---------------------------------------------------------------------------
# Building the data from UniProt (served from a local directory, see test_uniprot.py)
# ---------------------------------------------------------------------------

import csv  # noqa: E402

import pytest  # noqa: E402

from test_uniprot import make_release  # noqa: E402
from weirdo import data_manager  # noqa: E402
from weirdo.cli import run  # noqa: E402
from weirdo.uniprot import SWISSPROT_DIVISIONS  # noqa: E402
from weirdo.reference_proteomes import (  # noqa: E402
    DownloadedReferenceProteome,
    ReferenceProteome,
    get_reference_proteome,
    reference_proteome_filename,
)


@pytest.fixture
def mirror(tmp_path):
    return make_release(tmp_path / "mirror")


class TestDownloadFromUniProt:
    def test_datasets_are_built_from_uniprot_not_from_a_prebuilt_download(self):
        for info in DATASETS.values():
            assert "url" not in info
        assert DATASETS["swissprot-8mers"]["requires"] == "swissprot-sequences"

    def test_sequences_are_downloaded_and_recorded_with_their_release(self, tmp_path, mirror):
        dm = DataManager(data_dir=tmp_path / "data", verbose=False, uniprot_url=mirror)
        path = dm.download("swissprot-sequences")
        assert path == dm.downloads_dir / "swissprot-sequences.csv"
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert {row["category"] for row in rows} == set(SWISSPROT_DIVISIONS)
        assert [row["identifier"] for row in rows if row["category"] == "bacteria"] == ["RS19_ECOLI", "RL2_ECOLI"]
        record = dm._metadata["downloads"]["swissprot-sequences"]
        assert record["uniprot_release"] == "2099_01" and record["n_proteins"] == len(rows)
        assert record["source_url"] == mirror
        assert dm.is_downloaded("swissprot-sequences")
        assert (dm.uniprot_dir / "release-2099_01" / "uniprot_sprot_human.dat.gz").exists()

    def test_the_kmer_table_is_built_from_the_sequences(self, tmp_path, mirror):
        dm = DataManager(data_dir=tmp_path / "data", verbose=False, uniprot_url=mirror)
        path = dm.download("swissprot-8mers")  # downloads the sequences first
        assert dm.is_downloaded("swissprot-sequences") and dm.is_downloaded("swissprot-8mers")
        with open(path, newline="") as handle:
            rows = {row["seq"]: row for row in csv.DictReader(handle)}
        assert rows["MTMDKSEL"]["human"] == "True" and rows["MTMDKSEL"]["bacteria"] == "False"
        record = dm._metadata["downloads"]["swissprot-8mers"]
        assert record["built_from"] == "swissprot-sequences" and record["k"] == 8
        assert record["uniprot_release"] == "2099_01" and record["n_kmers"] == len(rows)

    def test_present_data_is_not_built_again_unless_forced(self, tmp_path, mirror, monkeypatch):
        dm = DataManager(data_dir=tmp_path / "data", verbose=False, uniprot_url=mirror)
        dm.download("swissprot-sequences")

        def fail(*args, **kwargs):
            raise AssertionError("must not download again")

        monkeypatch.setattr("weirdo.uniprot.fetch_swissprot_divisions", fail)
        assert dm.download("swissprot-sequences") == dm.downloads_dir / "swissprot-sequences.csv"
        monkeypatch.undo()
        assert dm.download("swissprot-sequences", force=True).exists()

    def test_get_data_path_downloads_when_asked_to(self, tmp_path, mirror):
        dm = DataManager(data_dir=tmp_path / "data", verbose=False, uniprot_url=mirror)
        with pytest.raises(FileNotFoundError, match="weirdo data download"):
            dm.get_data_path("swissprot-sequences")
        assert dm.get_data_path("swissprot-sequences", auto_download=True).exists()

    def test_an_unreachable_source_is_reported_as_a_failed_download(self, tmp_path):
        dm = DataManager(data_dir=tmp_path / "data", verbose=False, uniprot_url=(tmp_path / "none").as_uri())
        with pytest.raises(RuntimeError, match="Failed to download 'swissprot-sequences'"):
            dm.download("swissprot-sequences")
        assert not (dm.downloads_dir / "swissprot-sequences.csv").exists()
        assert not dm.is_downloaded("swissprot-sequences")

    def test_status_counts_the_raw_uniprot_files_and_clear_removes_them(self, tmp_path, mirror):
        dm = DataManager(data_dir=tmp_path / "data", verbose=False, uniprot_url=mirror)
        dm.download("swissprot-sequences")
        status = dm.status()
        assert status["raw_uniprot_bytes"] > 0
        assert status["total_size_bytes"] > status["downloads"]["swissprot-sequences"]["size_bytes"]
        assert dm.clear_all() == 1
        assert not dm.uniprot_dir.exists()
        assert not (dm.downloads_dir / "swissprot-sequences.csv").exists()

    def test_the_command_line_builds_the_data(self, tmp_path, mirror, monkeypatch, capsys):
        dm = DataManager(data_dir=tmp_path / "data", verbose=True, uniprot_url=mirror)
        monkeypatch.setattr(data_manager, "_default_manager", dm)
        assert run(["data", "download", "swissprot-sequences"]) == 0
        out = capsys.readouterr().out
        assert "UniProt release 2099_01" in out and dm.is_downloaded("swissprot-sequences")
        assert run(["data", "status"]) == 0

    def test_the_command_line_reports_failures_without_a_traceback(self, tmp_path, monkeypatch, capsys):
        dm = DataManager(data_dir=tmp_path / "data", verbose=False, uniprot_url=(tmp_path / "none").as_uri())
        monkeypatch.setattr(data_manager, "_default_manager", dm)
        assert run(["data", "download", "swissprot-sequences"]) == 1
        assert "Failed to download 'swissprot-sequences'" in capsys.readouterr().out
        assert run(["setup"]) == 1

    def test_the_command_line_only_offers_known_datasets(self):
        with pytest.raises(SystemExit):
            run(["data", "download", "everything"])


def _fake_host_reference(directory, host, force=False, show_progress=False):
    proteome = get_reference_proteome(host)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / reference_proteome_filename(proteome)
    path.write_bytes(b"fake compressed fasta")
    return DownloadedReferenceProteome(
        proteome=proteome,
        path=path,
        sha256="a" * 64,
        proteins=2,
        amino_acids=9,
        source_url="file:///fake.fasta.gz",
    )


class TestHostReferenceProteomes:
    def test_download_records_a_complete_host_reference(self, tmp_path, monkeypatch):
        dm = DataManager(data_dir=tmp_path / "data", verbose=False)
        monkeypatch.setattr(
            "weirdo.reference_proteomes.fetch_reference_proteome", _fake_host_reference
        )
        path = dm.download_host_reference("dog")
        assert path.exists()
        record = dm._metadata["downloads"]["reference-proteome:dog"]
        assert record["proteome_id"] == "UP000805418"
        assert record["n_proteins"] == 2 and record["amino_acids"] == 9
        assert dm.get_host_reference_path("dog") == path
        assert dm.status()["host_reference_proteomes"]["dog"]["metadata"] == record

    def test_cli_lists_and_downloads_host_references(self, tmp_path, monkeypatch, capsys):
        dm = DataManager(data_dir=tmp_path / "data", verbose=False)
        monkeypatch.setattr(data_manager, "_default_manager", dm)
        monkeypatch.setattr(
            "weirdo.reference_proteomes.fetch_reference_proteome", _fake_host_reference
        )
        assert run(["data", "reference-proteome", "--list"]) == 0
        assert "dog\tCanis lupus familiaris\ttaxon:9615\tUP000805418" in capsys.readouterr().out
        assert run(["data", "reference-proteome", "dog"]) == 0
        assert dm.get_host_reference_path("dog").exists()

    def test_cli_accepts_an_arbitrary_reference_proteome_accession(self, tmp_path, monkeypatch):
        dm = DataManager(data_dir=tmp_path / "data", verbose=False)
        cattle = ReferenceProteome("cattle", "Bos taurus", 9913, "UP000009136")
        monkeypatch.setattr(data_manager, "_default_manager", dm)
        monkeypatch.setattr(
            "weirdo.reference_proteomes.fetch_reference_proteome", _fake_host_reference
        )
        monkeypatch.setattr(
            "weirdo.reference_proteomes.reference_proteome_from_uniprot", lambda upid: cattle
        )
        assert run(["data", "reference-proteome", "--proteome", "UP000009136"]) == 0
        assert "reference-proteome:cattle" in dm._metadata["downloads"]

    def test_manifest_downloads_use_the_same_verified_reference_path(self, tmp_path, monkeypatch):
        dm = DataManager(data_dir=tmp_path / "data", verbose=False)
        monkeypatch.setattr(
            "weirdo.reference_proteomes.fetch_reference_proteome", _fake_host_reference
        )
        paths = dm.download_reference_proteome_manifest(groups=("cattle",))
        assert len(paths) == 1 and paths[0].exists()
        record = dm._metadata["downloads"]["reference-proteome-manifest:reference-proteomes-v1"]
        assert record["groups"] == ["cattle"]
        assert record["proteome_ids"] == ["UP000009136"]
        status = dm.status()["reference_proteome_manifests"]["reference-proteomes-v1"]
        assert status["selected_proteomes"] == 544
        assert status["downloaded_proteomes"] == 1

    def test_cli_downloads_a_manifest_group_and_supports_dry_run(self, tmp_path, monkeypatch, capsys):
        dm = DataManager(data_dir=tmp_path / "data", verbose=False)
        monkeypatch.setattr(data_manager, "_default_manager", dm)
        monkeypatch.setattr(
            "weirdo.reference_proteomes.fetch_reference_proteome", _fake_host_reference
        )
        assert run(["data", "reference-proteome", "--manifest", "--group", "cattle"]) == 0
        assert dm.get_host_reference_path("cattle").exists()
        assert run(["data", "reference-proteome", "--manifest", "--group", "viruses", "--dry-run"]) == 0
        assert "512 selected" in capsys.readouterr().out

    def test_clear_all_removes_host_references(self, tmp_path, monkeypatch):
        dm = DataManager(data_dir=tmp_path / "data", verbose=False)
        monkeypatch.setattr(
            "weirdo.reference_proteomes.fetch_reference_proteome", _fake_host_reference
        )
        dm.download_host_reference("cat")
        assert dm.clear_all() == 1
        assert not dm.reference_proteomes_dir.exists()
        assert "reference-proteome:cat" not in dm._metadata["downloads"]
