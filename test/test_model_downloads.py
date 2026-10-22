import hashlib
import io
import json
import tarfile

import pytest

from weirdo.model_manager import ModelManager, _safe_extract_tar


@pytest.mark.parametrize("member_name,member_type", [
    ("../extract-sibling/escaped.txt", tarfile.REGTYPE),
    ("named-pipe", tarfile.FIFOTYPE),
])
def test_archive_preflight_rejects_prefix_siblings_and_special_files(
    tmp_path, monkeypatch, member_name, member_type,
):
    archive = tmp_path / "unsafe.tar"
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo(member_name)
        member.type = member_type
        tar.addfile(member)
    def forbidden(*args, **kwargs):
        pytest.fail("Unsafe member reached extractall")
    monkeypatch.setattr(tarfile.TarFile, "extractall", forbidden)
    with pytest.raises(RuntimeError, match="Unsafe archive path|unsupported special files"):
        _safe_extract_tar(archive, tmp_path / "extract")


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


# The downloads go through datacache, which handles file:// URLs like HTTP ones, so the tests
# use real archives on disk instead of replacing the download function.
class TestModelDownloads:
    def test_download_from_url_extracts_model(self, tmp_path):
        archive = _create_model_archive(tmp_path, top_level="demo-model")
        mm = ModelManager(model_dir=tmp_path / "models")

        path = mm.download_from_url(
            name="downloaded-model",
            url=archive.as_uri(),
            overwrite=False,
        )

        assert path == mm.model_dir / "downloaded-model"
        assert (path / "config.json").exists()
        assert (path / "model.pt").exists()
        assert mm.get_model_info("downloaded-model") is not None

    def test_download_from_url_checks_the_sha256(self, tmp_path):
        archive = _create_model_archive(tmp_path, top_level="demo-model")
        mm = ModelManager(model_dir=tmp_path / "models")
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()

        with pytest.raises(RuntimeError, match="failed verification"):
            mm.download_from_url("bad-checksum", archive.as_uri(), expected_sha256="0" * 64)
        assert not (mm.model_dir / "bad-checksum").exists()

        path = mm.download_from_url("good-checksum", archive.as_uri(), expected_sha256=digest.upper())
        assert (path / "model.pt").exists()

    def test_download_from_url_reports_unreachable_sources(self, tmp_path):
        mm = ModelManager(model_dir=tmp_path / "models")
        with pytest.raises(RuntimeError, match="Failed to download model archive"):
            mm.download_from_url("missing", (tmp_path / "no-such-archive.tar.gz").as_uri())
        assert not (mm.model_dir / "missing").exists()

    def test_download_from_url_refuses_to_overwrite_unless_asked(self, tmp_path):
        archive = _create_model_archive(tmp_path, top_level="demo-model")
        mm = ModelManager(model_dir=tmp_path / "models")
        mm.download_from_url("model", archive.as_uri())
        with pytest.raises(FileExistsError):
            mm.download_from_url("model", archive.as_uri())
        assert mm.download_from_url("model", archive.as_uri(), overwrite=True).exists()

    def test_download_from_url_rejects_unsafe_archive_paths(self, tmp_path):
        archive = tmp_path / "malicious.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            bad = tarfile.TarInfo("../evil.txt")
            payload = b"oops"
            bad.size = len(payload)
            tar.addfile(bad, io.BytesIO(payload))

        mm = ModelManager(model_dir=tmp_path / "models")

        with pytest.raises(RuntimeError, match="Unsafe archive path detected"):
            mm.download_from_url(
                name="bad-model",
                url=archive.as_uri(),
            )
        assert not (mm.model_dir / "bad-model").exists()

    def test_download_pretrained_uses_registry(self, tmp_path, monkeypatch):
        archive = _create_model_archive(tmp_path, top_level="registry-model")
        mm = ModelManager(model_dir=tmp_path / "models")

        monkeypatch.setattr(
            "weirdo.model_manager.PRETRAINED_MODELS",
            {
                "registry-model": {
                    "description": "test model",
                    "url": archive.as_uri(),
                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
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
