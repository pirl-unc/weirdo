"""Tests for models that ship inside the package and the predict_categories API."""

import json

import numpy as np
import pandas as pd
import pytest

import weirdo
from weirdo import model_manager
from weirdo.api import predict_categories
from weirdo.cli import run
from weirdo.model_manager import ModelManager, _is_plain_name, list_bundled_models
from weirdo.scorers import MLPScorer
from weirdo.scorers.mlp import _NumpyMLP, _NumpyScaler

CATEGORIES = ['bacteria', 'human', 'viruses', 'rodents', 'mammals', 'plants']
PROBABILITIES = [0.9, 0.2, 0.3, 0.1, 0.4, 0.8]


def constant_scorer():
    """A trained scorer that ignores the peptide: P(category) is fixed."""
    scorer = MLPScorer(k=8)
    n_features = len(scorer.get_feature_names())
    probabilities = np.asarray(PROBABILITIES)
    logits = np.log(probabilities / (1.0 - probabilities))
    scorer._model = _NumpyMLP([np.zeros((n_features, len(CATEGORIES)))], [logits], 'relu')
    scorer._scaler = _NumpyScaler(np.zeros(n_features), np.ones(n_features))
    scorer._target_categories = list(CATEGORIES)
    scorer._is_trained = True
    scorer._is_fitted = True
    scorer._metadata = {'description': 'constant test model'}
    return scorer


@pytest.fixture
def bundled_dir(tmp_path, monkeypatch):
    """Point the package at a temporary 'pretrained' directory holding one model."""
    directory = tmp_path / 'pretrained'
    directory.mkdir()
    constant_scorer().save(directory / model_manager.DEFAULT_CATEGORY_MODEL)
    monkeypatch.setattr(model_manager, 'BUNDLED_MODEL_DIR', directory)
    return directory


@pytest.fixture
def empty_user_dir(tmp_path):
    return str(tmp_path / 'user-models')


class TestBundledLookup:
    def test_plain_names_only(self):
        assert _is_plain_name('swissprot-categories')
        for bad in ('', '.', '..', '../x', 'a/b', '/etc'):
            assert not _is_plain_name(bad)

    def test_bundled_models_are_listed_with_metadata(self, bundled_dir):
        models = list_bundled_models()
        assert [m['name'] for m in models] == [model_manager.DEFAULT_CATEGORY_MODEL]
        assert models[0]['scorer_type'] == 'MLPScorer'
        assert models[0]['metadata']['description'] == 'constant test model'
        assert [m['name'] for m in weirdo.list_bundled_models()] == [model_manager.DEFAULT_CATEGORY_MODEL]

    def test_no_bundled_directory_means_no_models(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manager, 'BUNDLED_MODEL_DIR', tmp_path / 'missing')
        assert list_bundled_models() == []

    def test_manager_loads_a_bundled_model_by_name(self, bundled_dir, empty_user_dir):
        manager = ModelManager(empty_user_dir)
        scorer = manager.load(model_manager.DEFAULT_CATEGORY_MODEL)
        assert scorer.target_categories == CATEGORIES
        assert manager.get_model_info(model_manager.DEFAULT_CATEGORY_MODEL).scorer_type == 'MLPScorer'
        assert manager.list_models() == []  # bundled models are not user models

    def test_user_models_take_precedence(self, bundled_dir, empty_user_dir):
        manager = ModelManager(empty_user_dir)
        other = constant_scorer()
        other._target_categories = list(reversed(CATEGORIES))
        manager.save(other, model_manager.DEFAULT_CATEGORY_MODEL)
        assert manager.load(model_manager.DEFAULT_CATEGORY_MODEL).target_categories == list(reversed(CATEGORIES))

    def test_missing_names_and_traversal_into_the_bundled_dir_are_not_found(self, bundled_dir, tmp_path):
        # The user directory is nested one level deeper than the bundled one, so a name like
        # '../pretrained/<model>' resolves to the bundled model only through the bundled
        # lookup, which must refuse names that are not plain.
        manager = ModelManager(tmp_path / 'deep' / 'user-models')
        traversal = '../pretrained/' + model_manager.DEFAULT_CATEGORY_MODEL
        assert (bundled_dir / traversal).exists()
        with pytest.raises(FileNotFoundError, match='Model not found'):
            manager.load('nope')
        with pytest.raises(FileNotFoundError, match='Model not found'):
            manager.load(traversal)
        assert manager.get_model_info(traversal) is None


class TestPredictCategories:
    def test_default_model_gives_probabilities_and_the_pathogen_self_summary(self, bundled_dir, empty_user_dir):
        frame = predict_categories(['SIINFEKL', 'NLVPMVATV'], model_dir=empty_user_dir)
        assert isinstance(frame, pd.DataFrame)
        assert list(frame.columns) == ['peptide'] + CATEGORIES + ['pathogen_max', 'self_max', 'foreignness']
        assert frame['peptide'].tolist() == ['SIINFEKL', 'NLVPMVATV']
        np.testing.assert_allclose(frame[CATEGORIES].iloc[0], PROBABILITIES, atol=1e-6)
        pathogen, mammalian = 0.9, 0.4  # largest of bacteria/viruses, largest of human/rodents/mammals
        assert frame['pathogen_max'].iloc[0] == pytest.approx(pathogen)
        assert frame['self_max'].iloc[0] == pytest.approx(mammalian)
        assert frame['foreignness'].iloc[0] == pytest.approx(pathogen / (pathogen + mammalian))

    def test_sum_is_available_through_combine(self, bundled_dir, empty_user_dir):
        frame = predict_categories(['SIINFEKL'], model_dir=empty_user_dir, combine='sum')
        assert list(frame.columns) == ['peptide'] + CATEGORIES + ['pathogen_sum', 'self_sum', 'foreignness']
        pathogen, mammalian = 0.9 + 0.3, 0.2 + 0.1 + 0.4
        assert frame['foreignness'].iloc[0] == pytest.approx(pathogen / (pathogen + mammalian))

    def test_single_string_and_options(self, bundled_dir, empty_user_dir):
        frame = predict_categories(
            'SIINFEKL', model_dir=empty_user_dir, pathogen_categories=['plants'],
            self_categories=['human'],
        )
        assert len(frame) == 1
        assert frame['pathogen_max'].iloc[0] == pytest.approx(0.8)
        assert frame['foreignness'].iloc[0] == pytest.approx(0.8 / 1.0)

    def test_an_explicit_scorer_is_used_as_is(self):
        frame = predict_categories(['SIINFEKL'], model=constant_scorer())
        assert frame['bacteria'].iloc[0] == pytest.approx(0.9, abs=1e-6)

    def test_untrained_and_single_output_models_are_rejected(self):
        with pytest.raises(RuntimeError, match='not trained'):
            predict_categories(['SIINFEKL'], model=MLPScorer(k=8))
        single = MLPScorer(k=8, hidden_layer_sizes=(4,))
        single.train(['ACDEFGHI', 'KLMNPQRS'] * 10, [0.0, 1.0] * 10, epochs=2, verbose=False)
        with pytest.raises(ValueError, match='target_categories'):
            predict_categories(['SIINFEKL'], model=single)


class TestScoreCommand:
    @staticmethod
    def _install(monkeypatch, tmp_path):
        monkeypatch.setattr(model_manager, 'DEFAULT_MODEL_DIR', tmp_path / 'user-models')
        monkeypatch.setattr(model_manager, '_manager_cache', {}, raising=False)

    def test_default_model_prints_a_table_and_the_definition(self, bundled_dir, tmp_path, monkeypatch, capsys):
        self._install(monkeypatch, tmp_path)
        assert run(['score', 'SIINFEKL']) == 0
        out = capsys.readouterr().out
        assert 'swissprot-categories' in out and 'pathogen_max' in out and 'foreignness' in out
        assert 'independent sigmoid probabilities' in out

    def test_csv_output_is_just_the_table(self, bundled_dir, tmp_path, monkeypatch, capsys):
        self._install(monkeypatch, tmp_path)
        assert run(['score', 'SIINFEKL', 'NLVPMVATV', '--csv']) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert lines[0].split(',') == ['peptide'] + CATEGORIES + ['pathogen_max', 'self_max', 'foreignness']
        assert len(lines) == 3

    def test_sum_combine_changes_the_columns(self, bundled_dir, tmp_path, monkeypatch, capsys):
        self._install(monkeypatch, tmp_path)
        assert run(['score', 'SIINFEKL', '--combine', 'sum', '--csv']) == 0
        header = capsys.readouterr().out.splitlines()[0].split(',')
        assert 'pathogen_sum' in header and 'pathogen_max' not in header

    def test_models_available_lists_bundled_models(self, bundled_dir, capsys):
        assert run(['models', 'available']) == 0
        out = capsys.readouterr().out
        assert 'bundled with WEIRDO' in out and model_manager.DEFAULT_CATEGORY_MODEL in out


def test_bundled_metadata_is_json_serializable(bundled_dir):
    json.dumps(list_bundled_models())


# Policy: weights that ship inside the package stay small enough for a pip install. Larger models
# are published as release assets and fetched with `weirdo models download` instead.
MAX_BUNDLED_MODEL_BYTES = 5 * 1024 * 1024


def test_bundled_models_stay_within_the_size_budget():
    from pathlib import Path

    models = list_bundled_models()
    assert models, "no bundled models found"
    for model in models:
        size = sum(f.stat().st_size for f in Path(model['path']).iterdir() if f.is_file())
        assert size <= MAX_BUNDLED_MODEL_BYTES, (
            f"bundled model {model['name']} is {size / 1e6:.1f} MB; models above "
            f"{MAX_BUNDLED_MODEL_BYTES / 2**20:.0f} MB must be a release asset, not shipped in the package"
        )
