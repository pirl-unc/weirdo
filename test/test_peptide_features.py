"""Tests for the vectorised feature extractor and the optional positional block."""

import json
import zipfile

import numpy as np
import pytest

from weirdo import list_bundled_models
from weirdo.model_manager import BUNDLED_MODEL_DIR, DEFAULT_CATEGORY_MODEL
from weirdo.scorers import MLPScorer
from weirdo.scorers.peptide_features import (
    NUM_AMINO_ACIDS,
    extract_features,
    extract_features_batch,
    feature_count,
    get_feature_names,
)

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
OTHER_CHARACTERS = list("XUBZJO*-.abc019 ")


def corpus(seed=0):
    """Peptides that exercise every branch: lengths, unknown characters, repeats, charge runs."""
    rng = np.random.RandomState(seed)

    def random_peptide(length, other_fraction=0.0):
        pool = AMINO_ACIDS + OTHER_CHARACTERS
        return "".join(
            rng.choice(pool) if rng.rand() < other_fraction else rng.choice(AMINO_ACIDS)
            for _ in range(length)
        )

    peptides = [random_peptide(8) for _ in range(150)]
    peptides += [random_peptide(8, 0.3) for _ in range(150)]
    peptides += [random_peptide(int(rng.randint(1, 31)), 0.1) for _ in range(150)]
    peptides += [random_peptide(int(rng.randint(100, 300))) for _ in range(5)]
    peptides += ["A" * n for n in range(1, 12)]
    peptides += ["KRKRKRKR", "DEDEDEDE", "KKKKKKKK", "RRRRRRRR", "KDKDKRER", "HHHHDDDD", "XXXXXXXX",
                 "????????", "aaaaaaaa", "C", "AC", "GGGGGGGG", "MTMDKSEL"]
    return peptides


class TestBatchMatchesTheDefinition:
    @pytest.mark.parametrize("use_dipeptides", [True, False])
    @pytest.mark.parametrize("positional", [False, True])
    def test_every_feature_agrees_with_the_per_peptide_functions(self, use_dipeptides, positional):
        peptides = corpus()
        expected = np.asarray(
            [extract_features(p, 8, use_dipeptides, positional) for p in peptides], dtype=np.float32
        )
        actual = extract_features_batch(peptides, 8, use_dipeptides, positional)
        assert actual.shape == expected.shape == (len(peptides), feature_count(use_dipeptides, positional, 8))
        assert actual.dtype == np.float32
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    def test_empty_inputs(self):
        assert extract_features_batch([]).shape == (0, 592)
        both = extract_features_batch(["", "MTMDKSEL", ""])
        np.testing.assert_allclose(both[0], extract_features(""), atol=1e-6)
        np.testing.assert_allclose(both[1], extract_features("MTMDKSEL"), atol=1e-6)
        np.testing.assert_array_equal(both[0], both[2])

    def test_rows_keep_their_order_whatever_the_chunking(self):
        peptides = corpus(seed=3)
        reference = extract_features_batch(peptides)
        np.testing.assert_array_equal(extract_features_batch(peptides, chunk_size=7), reference)
        np.testing.assert_array_equal(extract_features_batch(iter(peptides)), reference)
        with pytest.raises(ValueError, match="chunk_size"):
            extract_features_batch(peptides, chunk_size=0)

    def test_characters_outside_latin_1_are_unknown_residues(self):
        features = extract_features_batch(["ACDEFGHé", "ACDEFGH中"])
        np.testing.assert_allclose(features[0], extract_features("ACDEFGHé"), atol=1e-6)
        np.testing.assert_allclose(features[1], extract_features("ACDEFGH中"), atol=1e-6)

    def test_no_worker_processes_are_involved(self):
        # Importing and using the extractor must not need a __main__ guard or any pool: it is
        # plain array code in the calling process.
        import weirdo.scorers.mlp as mlp
        import weirdo.scorers.peptide_features as features
        for module in (mlp, features):
            assert not hasattr(module, "ProcessPoolExecutor")
            assert not hasattr(module, "mp")

    def test_it_is_much_faster_than_the_per_peptide_path(self):
        import time
        rng = np.random.RandomState(0)
        peptides = ["".join(rng.choice(AMINO_ACIDS, 8)) for _ in range(3000)]
        start = time.perf_counter()
        extract_features_batch(peptides)
        batch_seconds = time.perf_counter() - start
        start = time.perf_counter()
        for peptide in peptides[:300]:
            extract_features(peptide)
        per_peptide_seconds = (time.perf_counter() - start) * 10  # scaled to 3000 peptides
        assert batch_seconds * 5 < per_peptide_seconds


class TestFeatureNames:
    @pytest.mark.parametrize("use_dipeptides", [True, False])
    @pytest.mark.parametrize("positional", [False, True])
    @pytest.mark.parametrize("k", [8, 9])
    def test_one_name_per_column(self, use_dipeptides, positional, k):
        names = get_feature_names(use_dipeptides, positional, k)
        assert len(names) == len(set(names)) == feature_count(use_dipeptides, positional, k)

    def test_the_default_layout_is_the_one_the_bundled_model_was_trained_on(self):
        assert feature_count() == 592
        assert MLPScorer().get_feature_names() == get_feature_names()
        path = BUNDLED_MODEL_DIR / DEFAULT_CATEGORY_MODEL / "model.pt"
        if not path.exists():
            pytest.skip("the bundled model is not installed")
        with np.load(path, allow_pickle=False) as archive:
            stored = str(archive["feature_signature"])
        assert MLPScorer()._feature_signature() == stored

    def test_positional_names_are_appended_after_the_default_features(self):
        base = get_feature_names()
        names = get_feature_names(positional=True, k=8)
        assert names[:len(base)] == base
        assert names[len(base):][:2] == ["kmer_pos1_A", "kmer_pos1_C"]
        assert names[-1] == "kmer_pos8_X"
        assert len(names) == len(base) + 8 * NUM_AMINO_ACIDS


class TestPositionalFeatures:
    def test_an_8mer_is_one_hot_encoded_by_position(self):
        row = extract_features_batch(["ACDEFGHX"], 8, positional=True)[0][-8 * 21:].reshape(8, 21)
        assert (row.sum(axis=1) == 1).all()
        residues = "ACDEFGHIKLMNPQRSTVWYX"
        assert [residues[i] for i in row.argmax(axis=1)] == list("ACDEFGHX")

    def test_longer_peptides_average_their_windows_and_short_ones_are_padded(self):
        long = extract_features_batch(["ACDEFGHIK"], 8, positional=True)[0][-168:].reshape(8, 21)
        first = extract_features_batch(["ACDEFGHI"], 8, positional=True)[0][-168:].reshape(8, 21)
        second = extract_features_batch(["CDEFGHIK"], 8, positional=True)[0][-168:].reshape(8, 21)
        np.testing.assert_allclose(long, (first + second) / 2)
        short = extract_features_batch(["ACD"], 8, positional=True)[0][-168:].reshape(8, 21)
        assert (short[3:, 20] == 1).all()  # positions 4-8 are padding ("X")

    def test_scorer_option_changes_the_width_and_round_trips(self, tmp_path):
        scorer = MLPScorer(k=8, hidden_layer_sizes=(8,), positional=True, random_state=0)
        assert len(scorer.get_feature_names()) == 592 + 168
        rng = np.random.RandomState(0)
        peptides = ["".join(rng.choice(AMINO_ACIDS, 8)) for _ in range(120)]
        labels = np.array([[float(p[1] in "LI"), float(p[0] in "KR")] for p in peptides])
        scorer.train(peptides, labels, target_categories=["a", "b"], epochs=3, verbose=False)
        assert scorer._metadata["n_features"] == 760
        scorer.save(tmp_path / "positional")
        config = json.loads((tmp_path / "positional" / "config.json").read_text())
        assert config["params"]["positional"] is True
        loaded = MLPScorer.load(tmp_path / "positional")
        assert loaded.get_feature_names() == scorer.get_feature_names()
        np.testing.assert_allclose(loaded.predict_proba(peptides[:5]), scorer.predict_proba(peptides[:5]),
                                   atol=1e-6)

    def test_the_default_does_not_use_positional_features(self):
        assert MLPScorer()._params["positional"] is False
        assert len(MLPScorer().get_feature_names()) == 592


class TestLoadingModelsWithAnotherLayout:
    @staticmethod
    def _saved_squared_loss_model(directory, positional):
        scorer = MLPScorer(k=8, hidden_layer_sizes=(4,), positional=positional, loss="squared",
                           random_state=0)
        peptides = ["ACDEFGHI", "KLMNPQRS", "TVWYACDE", "GHIKLMNP"] * 8
        scorer.train(peptides, [0.1, 0.9, 0.2, 0.8] * 8, epochs=2, verbose=False)
        scorer.save(directory)
        return directory

    def test_a_model_with_another_feature_count_fails_at_load_with_a_clear_message(self, tmp_path):
        directory = self._saved_squared_loss_model(tmp_path / "model", positional=True)
        assert not zipfile.is_zipfile(directory / "model.pt")  # legacy pickle format
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text())
        config["params"]["positional"] = False  # like a config that predates the option
        config_path.write_text(json.dumps(config))
        with pytest.raises(ValueError, match=r"trained on 760 features.*builds 592"):
            MLPScorer.load(directory)

    def test_a_matching_pickled_model_still_loads(self, tmp_path):
        directory = self._saved_squared_loss_model(tmp_path / "model", positional=True)
        loaded = MLPScorer.load(directory)
        assert loaded.score(["ACDEFGHI"]).shape == (1,)

    def test_the_january_style_layout_is_reported_not_silently_accepted(self, tmp_path):
        # Reproduces the failure of a model saved by an unreleased development version with 684
        # features: it used to load and then fail inside scikit-learn when scoring.
        directory = self._saved_squared_loss_model(tmp_path / "model", positional=False)
        import pickle
        with open(directory / "model.pt", "rb") as handle:
            data = pickle.load(handle)
        data["scaler"].n_features_in_ = 684
        with open(directory / "model.pt", "wb") as handle:
            pickle.dump(data, handle)
        with pytest.raises(ValueError, match="684 features"):
            MLPScorer.load(directory)


class TestDeprecatedParallelOptions:
    def test_parallel_workers_are_ignored_with_a_warning(self):
        rows = [("ACDEFGHI", [1.0, 0.0]), ("KLMNPQRS", [0.0, 1.0])] * 10
        scorer = MLPScorer(k=8, hidden_layer_sizes=(4,), random_state=0)
        with pytest.warns(DeprecationWarning, match="parallel_workers"):
            scorer.train_streaming(
                lambda: iter(rows), target_categories=["a", "b"], epochs=1, batch_size=8,
                verbose=False, parallel_workers=4,
            )
        assert scorer.is_trained
        assert "parallel_workers" not in scorer._metadata

    def test_no_warning_when_the_options_are_not_used(self, recwarn):
        rows = [("ACDEFGHI", [1.0, 0.0]), ("KLMNPQRS", [0.0, 1.0])] * 10
        MLPScorer(k=8, hidden_layer_sizes=(4,), random_state=0).train_streaming(
            lambda: iter(rows), target_categories=["a", "b"], epochs=1, batch_size=8, verbose=False,
        )
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_the_bundled_model_is_listed():
    assert DEFAULT_CATEGORY_MODEL in [m["name"] for m in list_bundled_models()]
