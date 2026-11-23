"""Tests for Weirdo cache resolution, read-only inspection, atomic metadata, and harmonized data CLI."""

import hashlib
import json
import os
from pathlib import Path

import datacache
import pytest

from weirdo import cli
from weirdo.data_manager import DataManager, get_data_manager, resolve_data_dir


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    from weirdo import data_manager

    for key in (
        "WEIRDO_DATA_DIR",
        "OPENVAX_DATA_CACHE",
        "DATACACHE_OFFLINE",
        "OPENVAX_OFFLINE",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(data_manager, "_default_manager", None)
    monkeypatch.setattr(data_manager, "_default_manager_env", None)


def test_lazy_env_resolution_and_no_eager_mkdir(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    monkeypatch.setenv("OPENVAX_DATA_CACHE", str(shared))

    loc = resolve_data_dir()
    assert loc.path == str(shared / "weirdo")
    assert loc.origin == "shared_env"
    assert loc.detail == "OPENVAX_DATA_CACHE"

    dm = DataManager(verbose=False)
    assert dm.data_dir == shared / "weirdo"
    assert not dm.data_dir.exists()
    assert dm.status()["total_size_bytes"] == 0
    assert not dm.data_dir.exists()

    # Runtime WEIRDO_DATA_DIR overrides OPENVAX_DATA_CACHE even after import
    custom = tmp_path / "custom_weirdo"
    monkeypatch.setenv("WEIRDO_DATA_DIR", str(custom))
    assert get_data_manager().data_dir == custom
    assert not custom.exists()


def test_atomic_metadata_save_leaves_no_tmp(tmp_path):
    dm = DataManager(data_dir=tmp_path / "store", verbose=False)
    dm._metadata["downloads"]["demo"] = {"size_bytes": 12}
    dm._save_metadata()
    assert dm.metadata_file.is_file()
    assert json.loads(dm.metadata_file.read_text())["downloads"]["demo"]["size_bytes"] == 12
    assert not list((tmp_path / "store").glob(".metadata-*.tmp"))


def test_cli_dirs_path_status_json_readonly(tmp_path, monkeypatch, capsys):
    shared = tmp_path / "shared"
    monkeypatch.setenv("OPENVAX_DATA_CACHE", str(shared))
    monkeypatch.setenv("OPENVAX_OFFLINE", "1")

    assert cli.run(["data", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(shared / "weirdo")
    assert not shared.exists()

    assert cli.run(["data", "dirs", "--json"]) == 0
    dirs_payload = json.loads(capsys.readouterr().out)
    assert dirs_payload["data_dir"] == str(shared / "weirdo")
    assert dirs_payload["origin"] == "shared_env"
    assert dirs_payload["exists"] is False
    assert dirs_payload["offline"] is True
    assert not shared.exists()

    assert cli.run(["data", "status", "--json"]) == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["data_dir"] == str(shared / "weirdo")
    assert not shared.exists()


def test_cli_verify_checks_sha256_and_corruption(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "weirdo"
    monkeypatch.setenv("WEIRDO_DATA_DIR", str(data_dir))
    dm = DataManager(data_dir=data_dir, verbose=False)
    dm._ensure_dirs()
    dm.reference_proteomes_dir.mkdir(parents=True, exist_ok=True)
    fasta = dm.reference_proteomes_dir / "dog.fasta.gz"
    payload = b">sp|P1|DOG\nMAAA\n"
    fasta.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    dm._metadata["downloads"]["reference-proteome:dog"] = {
        "path": str(fasta),
        "sha256": sha,
    }
    dm._save_metadata()

    assert cli.run(["data", "verify", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["checks"][0]["status"] == "verified"

    # Corrupt file -> exit 1
    fasta.write_bytes(b"corrupted")
    assert cli.run(["data", "verify", "--json"]) == 1
    bad = json.loads(capsys.readouterr().out)
    assert bad["ok"] is False
    assert bad["checks"][0]["status"] == "corrupt"


def test_cli_prune_dry_run_then_apply(tmp_path, monkeypatch, capsys):
    data_dir = tmp_path / "weirdo"
    monkeypatch.setenv("WEIRDO_DATA_DIR", str(data_dir))
    store = data_dir / "bundles" / "sample"

    src1 = tmp_path / "v1.bin"
    b1 = b"gen1" * 40
    src1.write_bytes(b1)
    datacache.install_bundle(
        store,
        {"a.bin": {"url": src1.as_uri(), "sha256": hashlib.sha256(b1).hexdigest(), "size": len(b1)}},
    )
    src2 = tmp_path / "v2.bin"
    b2 = b"gen2" * 50
    src2.write_bytes(b2)
    datacache.install_bundle(
        store,
        {"a.bin": {"url": src2.as_uri(), "sha256": hashlib.sha256(b2).hexdigest(), "size": len(b2)}},
        force=True,
    )

    assert cli.run(["data", "prune", "--json"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["dry_run"] is True
    assert dry["removed_count"] == 1

    assert cli.run(["data", "prune", "--yes", "--json"]) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["dry_run"] is False
    assert applied["removed_count"] == 1


def test_cli_download_offline_blocks_remote_and_restores_env(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WEIRDO_DATA_DIR", str(tmp_path / "weirdo"))
    rc = cli.run(["data", "download", "swissprot-sequences", "--offline"])
    assert rc == 1
    assert "Offline mode is enabled" in capsys.readouterr().out
    assert "DATACACHE_OFFLINE" not in os.environ
