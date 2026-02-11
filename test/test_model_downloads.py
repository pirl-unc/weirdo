import io
import json
import shutil
import tarfile

import pytest

from weirdo.model_manager import ModelManager


def _create_model_archive(tmp_path, archive_name="model.tar.gz", top_level="demo-model"):
    model_dir = tmp_path / top_level
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({"scorer_type": "MLPScorer", "params": {"k": 8}}))
    (model_dir / "model.pt").write_bytes(b"weights")
    (model_dir / "metadata.json").write_text(json.dumps({"n_train": 10}))

    archive_path = tmp_path / archive_name
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(model_dir, arcname=top_level)
    return archive_path


class TestModelDownloads:
    def test_download_from_url_extracts_model(self, tmp_path, monkeypatch):
        archive = _create_model_archive(tmp_path, top_level="demo-model")
        mm = ModelManager(model_dir=tmp_path / "models")

        def fake_urlretrieve(url, filename):
            shutil.copyfile(archive, filename)
            return filename, None

        monkeypatch.setattr("weirdo.model_manager.urlretrieve", fake_urlretrieve)

        path = mm.download_from_url(
            name="downloaded-model",
            url="https://example.com/model.tar.gz",
            overwrite=False,
        )

        assert path == mm.model_dir / "downloaded-model"
        assert (path / "config.json").exists()
        assert (path / "model.pt").exists()
        assert mm.get_model_info("downloaded-model") is not None

    def test_download_from_url_rejects_unsafe_archive_paths(self, tmp_path, monkeypatch):
        archive = tmp_path / "malicious.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            bad = tarfile.TarInfo("../evil.txt")
            payload = b"oops"
            bad.size = len(payload)
            tar.addfile(bad, io.BytesIO(payload))

        mm = ModelManager(model_dir=tmp_path / "models")

        def fake_urlretrieve(url, filename):
            shutil.copyfile(archive, filename)
            return filename, None

        monkeypatch.setattr("weirdo.model_manager.urlretrieve", fake_urlretrieve)

        with pytest.raises(RuntimeError, match="Unsafe archive path detected"):
            mm.download_from_url(
                name="bad-model",
                url="https://example.com/malicious.tar.gz",
            )

    def test_download_pretrained_uses_registry(self, tmp_path, monkeypatch):
        archive = _create_model_archive(tmp_path, top_level="registry-model")
        mm = ModelManager(model_dir=tmp_path / "models")

        def fake_urlretrieve(url, filename):
            shutil.copyfile(archive, filename)
            return filename, None

        monkeypatch.setattr("weirdo.model_manager.urlretrieve", fake_urlretrieve)
        monkeypatch.setattr(
            "weirdo.model_manager.PRETRAINED_MODELS",
            {
                "registry-model": {
                    "description": "test model",
                    "url": "https://example.com/registry-model.tar.gz",
                    "sha256": None,
                }
            },
        )

        path = mm.download_pretrained("registry-model")
        assert path == mm.model_dir / "registry-model"
        assert (path / "model.pt").exists()

    def test_download_pretrained_unknown_model(self, tmp_path, monkeypatch):
        mm = ModelManager(model_dir=tmp_path / "models")
        monkeypatch.setattr("weirdo.model_manager.PRETRAINED_MODELS", {})
        with pytest.raises(ValueError, match="Unknown pretrained model"):
            mm.download_pretrained("missing-model")
