"""Tests for the sigmoid / binary cross-entropy head of MLPScorer."""

import warnings
import zipfile

import numpy as np
import pytest

from sklearn.preprocessing import StandardScaler

from weirdo.scorers import MLPScorer
from weirdo.scorers.mlp import _NumpyMLP, _NumpyScaler

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


def make_two_class_data(n, seed=0):
    """K/R-rich peptides are class 0, D/E-rich peptides class 1 (multi-hot labels)."""
    rng = np.random.RandomState(seed)
    peptides, labels = [], []
    for _ in range(n):
        first = rng.rand() < 0.5
        pool = AMINO_ACIDS + (list("KR") if first else list("DE")) * 6
        peptides.append("".join(rng.choice(pool, 8)))
        labels.append([1.0, 0.0] if first else [0.0, 1.0])
    return peptides, np.array(labels, dtype=np.float32)


def constant_scorer(categories, probabilities):
    """A trained scorer whose output ignores the peptide: P(category) is fixed."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    scorer = MLPScorer(k=8)
    n_features = len(scorer.get_feature_names())
    logits = np.log(probabilities / (1.0 - probabilities))
    scorer._model = _NumpyMLP(
        [np.zeros((n_features, len(categories)))], [logits], activation='relu'
    )
    scorer._scaler = _NumpyScaler(np.zeros(n_features), np.ones(n_features))
    scorer._target_categories = list(categories)
    scorer._is_trained = True
    scorer._is_fitted = True
    return scorer


@pytest.fixture(scope='module')
def trained_scorer():
    peptides, labels = make_two_class_data(500, seed=1)
    scorer = MLPScorer(k=8, hidden_layer_sizes=(32, 16), random_state=0)
    scorer.train(
        peptides[:400], labels[:400],
        val_peptides=peptides[400:], val_labels=labels[400:],
        target_categories=['basic', 'acidic'], epochs=40, verbose=False,
    )
    return scorer, peptides[400:], labels[400:]


class TestObjective:
    def test_binary_multilabel_targets_use_cross_entropy_by_default(self, trained_scorer):
        scorer, _, _ = trained_scorer
        assert scorer._is_logistic
        assert scorer._metadata['loss'] == 'bce'

    def test_training_loss_is_averaged_over_labels(self):
        """The reported training loss is per label, comparable to the validation log loss."""
        rng = np.random.RandomState(13)
        peptides = ["".join(rng.choice(AMINO_ACIDS, 8)) for _ in range(300)]
        labels = (rng.rand(300, 6) < 0.5).astype(np.float32)  # unlearnable coin flips
        scorer = MLPScorer(k=8, hidden_layer_sizes=(8,), random_state=0)
        scorer.train(
            peptides[:240], labels[:240], val_peptides=peptides[240:], val_labels=labels[240:],
            target_categories=list("abcdef"), epochs=3, verbose=False,
        )
        first = scorer.training_history[0]
        # a coin-flip target costs about ln 2 = 0.69 per label (4.2 if summed over 6 labels)
        assert 0.5 < first["loss"] < 1.2
        assert 0.5 < first["val_loss"] < 1.2

    def test_scalar_scores_keep_the_squared_loss(self):
        peptides = ['MTMDKSEL', 'ACDEFGHI', 'KLMNPQRS', 'XXXXXXXX'] * 10
        scorer = MLPScorer(k=8, hidden_layer_sizes=(8,), random_state=0)
        scorer.train(peptides, [0.0, 0.0, 1.0, 1.0] * 10, epochs=3, verbose=False)
        assert not scorer._is_logistic
        assert scorer._metadata['loss'] == 'squared'

    def test_explicit_cross_entropy_rejects_continuous_labels(self):
        scorer = MLPScorer(k=8, hidden_layer_sizes=(8,), loss='bce')
        with pytest.raises(ValueError, match="binary"):
            scorer.train(['MTMDKSEL', 'ACDEFGHI'] * 5, [0.2, 0.7] * 5, epochs=2, verbose=False)

    def test_unknown_loss_is_rejected(self):
        with pytest.raises(ValueError, match="loss must be"):
            MLPScorer(loss='hinge')

    def test_probabilities_cover_the_unit_interval_and_separate_the_classes(self, trained_scorer):
        scorer, peptides, labels = trained_scorer
        probs = scorer.predict_proba(peptides)
        assert probs.shape == (len(peptides), 2)
        assert probs.min() >= 0.0 and probs.max() <= 1.0
        # A squared-loss model squeezed through a sigmoid cannot leave [0.5, 0.74].
        assert probs.min() < 0.3 and probs.max() > 0.8
        for column in range(2):
            positives = probs[labels[:, column] == 1, column].mean()
            negatives = probs[labels[:, column] == 0, column].mean()
            assert positives - negatives > 0.4

    def test_squared_loss_remains_available_and_warns_about_calibration(self):
        peptides, labels = make_two_class_data(200, seed=2)
        scorer = MLPScorer(k=8, hidden_layer_sizes=(16,), random_state=0, loss='squared')
        scorer.train(peptides, labels, target_categories=['basic', 'acidic'], epochs=10, verbose=False)
        assert not scorer._is_logistic
        with pytest.warns(UserWarning, match="not calibrated"):
            probs = scorer.predict_proba(peptides[:20])
        assert probs.min() >= 0.0 and probs.max() <= 1.0
        # sigmoid(0) = 0.5 and sigmoid(1) = 0.73: even a perfect fit of the 0/1 targets
        # could only produce those two values.
        assert abs(1 / (1 + np.exp(-1.0)) - 0.731) < 1e-3

    def test_best_epoch_weights_are_restored(self):
        peptides, labels = make_two_class_data(60, seed=3)
        noisy = np.random.RandomState(4).randint(0, 2, size=labels.shape).astype(np.float32)
        val_peptides, val_labels = make_two_class_data(300, seed=5)
        scorer = MLPScorer(k=8, hidden_layer_sizes=(64, 32), random_state=0, learning_rate_init=0.01)
        scorer.train(
            peptides, noisy, val_peptides=val_peptides, val_labels=val_labels,
            target_categories=['basic', 'acidic'], epochs=60, n_iter_no_change=3,
            verbose=False,
        )
        final_loss, _ = scorer._validation_metrics(val_peptides, val_labels, 64)
        assert final_loss == pytest.approx(scorer._metadata['best_val_loss'], abs=1e-4)
        assert scorer._metadata['best_epoch'] <= scorer._metadata['n_epochs']

    def test_cached_features_train_the_same_kind_of_model(self):
        peptides, labels = make_two_class_data(300, seed=6)
        scorer = MLPScorer(k=8, hidden_layer_sizes=(16,), random_state=0)
        scorer.train(
            peptides, labels, target_categories=['basic', 'acidic'], epochs=15,
            verbose=False, cache_features=True,
        )
        assert scorer._is_logistic
        assert scorer.predict_proba(peptides[:5]).shape == (5, 2)

    def test_in_place_scaling_matches_the_scikit_learn_scaler(self):
        peptides, _ = make_two_class_data(500, seed=11)
        scorer = MLPScorer(k=8)
        features = scorer._extract_matrix(peptides)
        expected_scaler = StandardScaler().fit(features)
        expected = expected_scaler.transform(features)
        scorer._fit_scaler_and_scale(features, chunk_rows=64)  # several chunks
        np.testing.assert_allclose(scorer._scaler.mean_, expected_scaler.mean_, rtol=1e-6)
        np.testing.assert_allclose(scorer._scaler.scale_, expected_scaler.scale_, rtol=1e-6)
        np.testing.assert_allclose(features, expected, atol=2e-4)

    def test_feature_matrix_handles_empty_and_ragged_chunks(self):
        scorer = MLPScorer(k=8)
        assert scorer._extract_matrix([]).shape == (0, 592)
        peptides, _ = make_two_class_data(10, seed=12)
        np.testing.assert_array_equal(
            scorer._extract_matrix(peptides, chunk_size=3),
            scorer._extract_matrix(peptides, chunk_size=100),
        )


class TestStreaming:
    def test_binary_rows_train_a_sigmoid_head(self):
        peptides, labels = make_two_class_data(120, seed=9)
        rows = list(zip(peptides, labels.tolist()))
        scorer = MLPScorer(k=8, hidden_layer_sizes=(16,), random_state=0)
        scorer.train_streaming(
            lambda: iter(rows), target_categories=['basic', 'acidic'],
            epochs=4, batch_size=32, verbose=False,
        )
        assert scorer._is_logistic and scorer._metadata['loss'] == 'bce'

    def test_soft_labels_are_rejected_by_the_sigmoid_head(self):
        rows = [('MTMDKSEL', [1.0, 0.0]), ('ACDEFGHI', [0.3, 0.7])] * 10
        scorer = MLPScorer(k=8, hidden_layer_sizes=(8,), random_state=0)
        with pytest.raises(ValueError, match="binary"):
            scorer.train_streaming(
                lambda: iter(rows), target_categories=['a', 'b'], epochs=1,
                batch_size=4, verbose=False,
            )

    def test_soft_labels_still_train_with_squared_loss(self):
        rows = [('MTMDKSEL', [1.0, 0.0]), ('ACDEFGHI', [0.3, 0.7])] * 10
        scorer = MLPScorer(k=8, hidden_layer_sizes=(8,), random_state=0, loss='squared')
        scorer.train_streaming(
            lambda: iter(rows), target_categories=['a', 'b'], epochs=1, batch_size=4, verbose=False,
        )
        assert not scorer._is_logistic


class TestPortableFormat:
    def test_round_trip_is_pickle_free_and_exact(self, trained_scorer, tmp_path):
        scorer, peptides, _ = trained_scorer
        scorer.save(tmp_path / 'model')
        archive_path = tmp_path / 'model' / 'model.pt'
        assert zipfile.is_zipfile(archive_path)
        with np.load(archive_path, allow_pickle=False) as archive:  # fails if it held objects
            assert 'coef_0' in archive.files

        loaded = MLPScorer.load(tmp_path / 'model')
        assert isinstance(loaded._model, _NumpyMLP)
        assert loaded.target_categories == ['basic', 'acidic']
        np.testing.assert_allclose(
            loaded.predict_proba(peptides), scorer.predict_proba(peptides), atol=1e-5
        )

    def test_changed_feature_definitions_are_detected(self, trained_scorer, tmp_path):
        scorer, _, _ = trained_scorer
        scorer.save(tmp_path / 'model')
        archive_path = tmp_path / 'model' / 'model.pt'
        with np.load(archive_path, allow_pickle=False) as archive:
            contents = {name: archive[name] for name in archive.files}
        contents['feature_signature'] = np.array('0000000000000000')
        with open(archive_path, 'wb') as handle:
            np.savez_compressed(handle, **contents)
        with pytest.raises(ValueError, match="feature"):
            MLPScorer.load(tmp_path / 'model')

    def test_legacy_pickled_models_still_load(self, tmp_path):
        peptides, labels = make_two_class_data(120, seed=10)
        scorer = MLPScorer(k=8, hidden_layer_sizes=(8,), random_state=0, loss='squared')
        scorer.train(peptides, labels, target_categories=['basic', 'acidic'], epochs=3, verbose=False)
        scorer.save(tmp_path / 'legacy')
        assert not zipfile.is_zipfile(tmp_path / 'legacy' / 'model.pt')
        loaded = MLPScorer.load(tmp_path / 'legacy')
        assert not loaded._is_logistic
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            np.testing.assert_allclose(
                loaded.predict_proba(peptides[:5]), scorer.predict_proba(peptides[:5]), atol=1e-5
            )

    def test_numpy_network_matches_scikit_learn(self, trained_scorer):
        scorer, peptides, _ = trained_scorer
        features = scorer._scaler.transform(scorer._extract_features(peptides)).astype(np.float32)
        portable = _NumpyMLP(scorer._model.coefs_, scorer._model.intercepts_, scorer._model.activation)
        np.testing.assert_allclose(
            portable.predict_proba(features), scorer._model.predict_proba(features), atol=1e-5
        )

    def test_unsupported_activation_is_rejected(self):
        with pytest.raises(ValueError, match="activation"):
            _NumpyMLP([np.zeros((3, 1))], [np.zeros(1)], activation='swish')


class TestForeignnessSummary:
    CATEGORIES = ['bacteria', 'human', 'viruses', 'rodents', 'mammals', 'plants']
    PROBABILITIES = [0.9, 0.2, 0.3, 0.1, 0.4, 0.8]

    def test_max_is_the_default(self):
        scorer = constant_scorer(self.CATEGORIES, self.PROBABILITIES)
        frame = scorer.predict_dataframe(['ACDEFGHI'])
        assert frame['pathogen_max'].iloc[0] == pytest.approx(0.9)   # max(bacteria, viruses)
        assert frame['self_max'].iloc[0] == pytest.approx(0.4)       # max(human, rodents, mammals)
        expected = 0.9 / (0.9 + 0.4)
        assert frame['foreignness'].iloc[0] == pytest.approx(expected)
        assert scorer.foreignness(['ACDEFGHI'])[0] == pytest.approx(expected)
        assert scorer.score(['ACDEFGHI'])[0] == pytest.approx(expected)
        assert 'pathogen_sum' not in frame.columns

    def test_sum_compares_expected_category_counts(self):
        scorer = constant_scorer(self.CATEGORIES, self.PROBABILITIES)
        frame = scorer.predict_dataframe(['ACDEFGHI'], combine='sum')
        pathogen = 0.9 + 0.3            # bacteria + viruses
        mammalian = 0.2 + 0.1 + 0.4     # human + rodents + mammals
        assert frame['pathogen_sum'].iloc[0] == pytest.approx(pathogen)
        assert frame['self_sum'].iloc[0] == pytest.approx(mammalian)
        expected = pathogen / (pathogen + mammalian)
        assert frame['foreignness'].iloc[0] == pytest.approx(expected)
        assert scorer.foreignness(['ACDEFGHI'], combine='sum')[0] == pytest.approx(expected)
        assert scorer.score(['ACDEFGHI'], combine='sum')[0] == pytest.approx(expected)

    def test_a_callable_can_combine_the_groups(self):
        scorer = constant_scorer(self.CATEGORIES, self.PROBABILITIES)
        mean = lambda values: values.mean(axis=1)  # noqa: E731
        frame = scorer.predict_dataframe(['ACDEFGHI'], combine=mean)
        pathogen, mammalian = (0.9 + 0.3) / 2, (0.2 + 0.1 + 0.4) / 3
        assert frame['pathogen_value'].iloc[0] == pytest.approx(pathogen)
        assert frame['self_value'].iloc[0] == pytest.approx(mammalian)
        assert frame['foreignness'].iloc[0] == pytest.approx(pathogen / (pathogen + mammalian))
        assert scorer.foreignness(['ACDEFGHI'], combine=mean)[0] == pytest.approx(
            pathogen / (pathogen + mammalian)
        )

    def test_custom_groups_skip_unknown_categories_and_count_duplicates_once(self):
        scorer = constant_scorer(self.CATEGORIES, self.PROBABILITIES)
        value = scorer.foreignness(
            ['ACDEFGHI'],
            pathogen_categories=['plants', 'plants', 'martians'],
            self_categories=['human'],
        )[0]
        assert value == pytest.approx(0.8 / (0.8 + 0.2))

    def test_both_groups_at_zero_is_neutral(self):
        scorer = constant_scorer(['bacteria', 'human'], [1e-12, 1e-12])
        scorer._model.intercepts_ = [np.full(2, -1e4, dtype=np.float32)]
        assert scorer.foreignness(['ACDEFGHI'])[0] == pytest.approx(0.5)

    def test_unknown_combine_is_rejected(self):
        scorer = constant_scorer(self.CATEGORIES, self.PROBABILITIES)
        with pytest.raises(ValueError, match="combine"):
            scorer.foreignness(['ACDEFGHI'], combine='median')

    def test_group_without_any_known_category_is_rejected(self):
        scorer = constant_scorer(['human', 'plants'], [0.5, 0.5])
        with pytest.raises(ValueError, match="No pathogen categories"):
            scorer.foreignness(['ACDEFGHI'])


class TestWindowAggregation:
    def test_windows_are_aggregated_per_peptide_without_mixing_peptides(self, monkeypatch):
        scorer = constant_scorer(['bacteria', 'human'], [0.5, 0.5])

        def by_first_residue(self, kmers):
            value = np.array([[AMINO_ACIDS.index(kmer[0]) / 20.0, 0.5] for kmer in kmers])
            return value

        monkeypatch.setattr(MLPScorer, '_predict_kmer_probabilities', by_first_residue)
        peptides = ['ACDEFGHIKL', 'KLMNPQRS', 'WYACDEFGHIKLM']  # 3, 1 and 6 windows
        expected = []
        for peptide in peptides:
            firsts = [AMINO_ACIDS.index(peptide[i]) / 20.0 for i in range(len(peptide) - 7)]
            expected.append((np.mean(firsts), np.max(firsts), np.min(firsts)))
        for column, aggregate in enumerate(('mean', 'max', 'min')):
            probs = scorer.predict_proba(peptides, aggregate=aggregate)
            np.testing.assert_allclose(probs[:, 0], [row[column] for row in expected])
            np.testing.assert_allclose(probs[:, 1], 0.5)

    def test_unknown_aggregate_is_rejected(self):
        scorer = constant_scorer(['bacteria', 'human'], [0.5, 0.5])
        with pytest.raises(ValueError, match="aggregate"):
            scorer.predict_proba(['ACDEFGHI'], aggregate='median')

    def test_empty_input_gives_an_empty_matrix(self):
        scorer = constant_scorer(['bacteria', 'human'], [0.5, 0.5])
        assert scorer.predict_proba([]).shape == (0, 2)

    def test_single_string_and_short_peptides_work(self):
        scorer = constant_scorer(['bacteria', 'human'], [0.25, 0.75])
        np.testing.assert_allclose(scorer.predict_proba('ACDEFGHI'), [[0.25, 0.75]], atol=1e-6)
        np.testing.assert_allclose(scorer.predict_proba(['ACD']), [[0.25, 0.75]], atol=1e-6)
