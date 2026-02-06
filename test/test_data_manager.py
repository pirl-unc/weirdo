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
