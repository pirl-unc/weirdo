"""Both public loading routes enforce the released artifact and inference contract."""

import hashlib
import json
import shutil
import socket

import numpy as np
import pytest

from weirdo import PretrainedModel, get_model_manifest, load_pretrained_model
from weirdo import released


LOADERS = [PretrainedModel, load_pretrained_model]


@pytest.fixture(params=LOADERS)
def loader(request):
    return request.param


@pytest.fixture(scope="module")
def model():
    return load_pretrained_model()


def test_load_and_score_offline_without_user_cache(loader, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network access is forbidden during inference")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(released.Path, "home", lambda: tmp_path / "missing-home")
    model = loader()
    manifest = get_model_manifest()
    frame = model.predict_categories([manifest["smoke"]["peptide"]])
    np.testing.assert_allclose(
        frame[manifest["categories"]].iloc[0], manifest["smoke"]["category_values"],
        atol=manifest["smoke"]["atol"], rtol=0,
    )
    np.testing.assert_allclose(model.score(frame.peptide), 1 - frame.human)
    assert frame.attrs["prediction_identity"] == model.prediction_identity
    assert not (tmp_path / "missing-home").exists()


def test_bounded_batches_duplicates_case_and_generator(model, monkeypatch):
    peptides = ["acdefghi", "MTMDKSELV", "ACDEFGHI", "MTMDKSELVQ", "ACDEFGHIKLM"]
    expected = model.score(peptides)
    calls = []
    original = model._scorer.predict_proba
    def counted(batch, **kwargs):
        calls.append(list(batch))
        return original(batch, **kwargs)
    monkeypatch.setattr(model._scorer, "predict_proba", counted)
    actual = model.score(iter(peptides), batch_size=2)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
    assert calls == [["ACDEFGHI", "MTMDKSELV"], ["MTMDKSELVQ", "ACDEFGHIKLM"]]
    assert actual[0] == actual[2]


@pytest.mark.parametrize("peptides", [[""], ["ACD"], ["A" * 12], ["ACDEFGHX"],
                                      ["ACDEFGHI "], ["ACDEFGH*"], [None], [123], None,
                                      ["ACDEFGHı"]])
@pytest.mark.parametrize("method", ["score", "predict_categories"])
def test_invalid_inputs_fail_before_inference(model, peptides, method, monkeypatch):
    monkeypatch.setattr(model._scorer, "predict_proba", lambda *a, **kw: pytest.fail("invalid input reached inference"))
    with pytest.raises(ValueError):
        getattr(model, method)(peptides)


@pytest.mark.parametrize("size", [0, -1, 1.5, True, "2"])
def test_invalid_batch_sizes(model, size):
    with pytest.raises(ValueError, match="batch_size"):
        model.score(["ACDEFGHI"], batch_size=size)


def test_empty_batch_and_single_sequence(model):
    empty = model.predict_categories([])
    assert list(empty.columns) == ["peptide"] + model.manifest["categories"]
    assert model.score([]).shape == (0,)
    assert model.score([]).dtype == np.float64
    assert len(model.score("acdefghi")) == 1


@pytest.mark.parametrize("kwargs", [{"model": "unknown"}, {"score_policy": "other"},
                                   {"host_taxon_id": 10090}, {"host_taxon_id": True}])
def test_unknown_model_policy_or_host_fails(loader, kwargs):
    with pytest.raises(ValueError):
        loader(**kwargs)


@pytest.mark.parametrize("filename", ["manifest.json", "model.pt", "metadata.json", "config.json"])
def test_corrupt_or_missing_artifacts_fail(loader, filename, tmp_path):
    directory = tmp_path / "model"
    shutil.copytree(released.MODEL_DIRECTORY, directory)
    target = directory / filename
    target.write_bytes(target.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        loader(model_dir=directory)
    target.unlink()
    with pytest.raises(FileNotFoundError):
        loader(model_dir=directory)


def test_self_consistent_replacement_is_not_the_named_release(loader, tmp_path):
    directory = tmp_path / "model"
    shutil.copytree(released.MODEL_DIRECTORY, directory)
    weights = directory / "model.pt"
    weights.write_bytes(b"replacement weights")
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["files"]["model.pt"] = hashlib.sha256(weights.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest checksum"):
        loader(model_dir=directory)


def test_manifest_copies_cannot_change_the_model_identity(model):
    original = model.prediction_identity
    modified = model.manifest
    modified["artifact_id"] = "different"
    modified["categories"].reverse()
    assert model.prediction_identity == original
    assert model.manifest == get_model_manifest()


def test_supported_lengths_average_category_windows(model):
    peptide = "ACDEFGHIKLM"
    windows = [peptide[i:i + 8] for i in range(4)]
    expected = model.predict_categories(windows)[model.manifest["categories"]].mean(axis=0)
    actual = model.predict_categories([peptide])[model.manifest["categories"]].iloc[0]
    np.testing.assert_allclose(actual, expected, atol=2e-6)
