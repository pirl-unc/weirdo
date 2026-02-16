"""Tests for MLP-based scorer."""

import os
import tempfile
import warnings
import pytest
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from weirdo.scorers.mlp import (
    MLPScorer,
    extract_features,
    _compute_property_features,
    _compute_composition_features,
    _compute_dipeptide_features,
    _get_aa_properties,
    AMINO_ACIDS,
)


class TestFeatureExtraction:
    """Tests for feature extraction functions."""

    def test_compute_composition_features(self):
        """Test amino acid composition feature extraction."""
        features = _compute_composition_features('AAAA')
        assert len(features) == 20  # 20 amino acids
        assert features[0] == 1.0  # 100% A
        assert sum(features) == pytest.approx(1.0)

    def test_compute_dipeptide_features(self):
        """Test dipeptide composition feature extraction."""
        features = _compute_dipeptide_features('AAA')
        assert len(features) == 400  # 20 * 20
        # AA dipeptide should have frequency 1.0 (2 AAs out of 2 dipeptides)
        assert features[0] == 1.0  # AA is at index 0*20 + 0 = 0
        assert np.isfinite(features).all()

    def test_compute_property_features(self):
        """Test amino acid property feature extraction."""
        properties = _get_aa_properties()
        features = _compute_property_features('MTMDKSEL', properties)
        # 12 properties * 4 stats (mean, std, min, max) = 48 features
        assert len(features) == 12 * 4
        assert np.isfinite(features).all()

    def test_extract_features(self):
        """Test full feature extraction."""
        features = extract_features('MTMDKSEL', k=8, use_dipeptides=True)
        # 48 + 27 + 20 + 12 (seq stats) + 80 (reduced alph) + 5 (dipep stats) + 400 (dipeps) = 592
        expected_length = 48 + 27 + 20 + 12 + 80 + 5 + 400
        assert len(features) == expected_length
        assert np.isfinite(features).all()

    def test_extract_features_no_dipeptides(self):
        """Test feature extraction without dipeptides."""
        features = extract_features('MTMDKSEL', k=8, use_dipeptides=False)
        # 48 + 27 + 20 + 12 (seq stats) + 80 (reduced alph) = 187
        expected_length = 48 + 27 + 20 + 12 + 80
        assert len(features) == expected_length
        assert np.isfinite(features).all()

    def test_get_feature_names(self):
        """Test getting feature names from scorer."""
        scorer = MLPScorer(k=8, use_dipeptides=True)
        names = scorer.get_feature_names()

        # Should have names for all features
        expected_count = 48 + 27 + 20 + 12 + 80 + 5 + 400
        assert len(names) == expected_count

        # Check some specific feature names
        assert 'hydropathy_mean' in names
        assert 'helix_propensity_mean' in names
        assert 'frac_cysteine' in names
        assert 'arginine_ratio' in names
        assert 'aa_freq_A' in names
        assert 'seq_length' in names
        assert 'murphy8_freq_A' in names
        assert 'dipep_entropy' in names
        assert 'dipep_AA' in names
        assert all('kmer_pos' not in name for name in names)


class TestMLPScorer:
    """Tests for MLPScorer."""

    def test_create_scorer(self):
        """Test creating an MLP scorer."""
        scorer = MLPScorer(
            k=8,
            hidden_layer_sizes=(64, 32),
        )

        assert scorer.k == 8
        assert not scorer.is_trained

    def test_train_scorer(self):
        """Test training an MLP scorer."""
        # Create synthetic training data
        peptides = [
            'MTMDKSEL', 'ACDEFGHI', 'KLMNPQRS',
            'XXXXXXXX', 'YYYYYYYY', 'WWWWWWWW',
        ] * 10  # Need enough samples
        # Low scores for known, high for unknown
        labels = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0] * 10

        scorer = MLPScorer(
            k=8,
            hidden_layer_sizes=(32, 16),
            random_state=42,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            scorer.train(
                peptides=peptides,
                labels=labels,
                epochs=50,
                verbose=False,
            )

        assert scorer.is_trained
        assert len(scorer.training_history) > 0

    def test_train_uses_feature_batches(self, monkeypatch):
        """Training should process features in bounded-size batches."""
        peptides = ['MTMDKSEL', 'ACDEFGHI', 'KLMNPQRS', 'XXXXXXXX'] * 25
        labels = [0.0, 0.0, 1.0, 1.0] * 25

        calls = []
        original_extract = MLPScorer._extract_features

        def tracked_extract(self, batch_peptides):
            calls.append(len(batch_peptides))
            return original_extract(self, batch_peptides)

        monkeypatch.setattr(MLPScorer, '_extract_features', tracked_extract)

        scorer = MLPScorer(
            k=8,
            hidden_layer_sizes=(16,),
            batch_size=7,
            early_stopping=False,
            random_state=42,
        )
        scorer.train(peptides=peptides, labels=labels, epochs=5, verbose=False)

        assert calls, "Expected feature extraction calls during training."
        assert max(calls) <= 7

    def test_iter_feature_batches_casts_to_float32(self):
        """Batch iterator should cast feature and label arrays to float32."""
        scorer = MLPScorer(k=8, hidden_layer_sizes=(8,), random_state=1)
        peptides = ['MTMDKSEL', 'ACDEFGHI', 'KLMNPQRS', 'TVWYACDE', 'XXXXXXXX']
        labels = np.array([0, 1, 0, 1, 1], dtype=np.float64)

        batches = list(
            scorer._iter_feature_batches(
                peptides=peptides,
                labels=labels,
                batch_size=2,
                shuffle=False,
                rng=np.random.RandomState(0),
            )
        )

        assert batches
        assert all(X.dtype == np.float32 for X, _ in batches)
        assert all(y.dtype == np.float32 for _, y in batches)

    def test_train_streaming_multilabel(self):
        """Streaming training should work for multi-label data."""
        rows = (
            [('MTMDKSEL', [1.0, 0.0]), ('XXXXXXXX', [0.0, 1.0])]
            + [('ACDEFGHI', [1.0, 0.0]), ('WWWWWWWW', [0.0, 1.0])]
        ) * 20

        def row_iterator_factory():
            return iter(rows)

        scorer = MLPScorer(
            k=8,
            hidden_layer_sizes=(16,),
            random_state=42,
            early_stopping=False,
        )
        scorer.train_streaming(
            row_iterator_factory=row_iterator_factory,
            target_categories=['human', 'viruses'],
            epochs=8,
            batch_size=10,
            verbose=False,
        )

        assert scorer.is_trained
        assert scorer._metadata['n_train'] == len(rows)
        assert len(scorer.training_history) == 8

    def test_train_streaming_requires_target_categories_for_multilabel(self):
        """Multi-label row streams should require target_categories."""
        rows = [('MTMDKSEL', [1.0, 0.0]), ('XXXXXXXX', [0.0, 1.0])]

        def row_iterator_factory():
            return iter(rows)

        scorer = MLPScorer(k=8, hidden_layer_sizes=(8,), random_state=3)
        with pytest.raises(ValueError, match="requires target_categories"):
            scorer.train_streaming(
                row_iterator_factory=row_iterator_factory,
                epochs=2,
                verbose=False,
            )

    def test_train_streaming_progress_arguments_validation(self):
        """Streaming training should validate progress-related arguments."""
        rows = [('MTMDKSEL', 1.0), ('XXXXXXXX', 0.0)] * 4

        def row_iterator_factory():
            return iter(rows)

        scorer = MLPScorer(k=8, hidden_layer_sizes=(8,), random_state=5)
        with pytest.raises(ValueError, match="progress_every_samples must be positive"):
            scorer.train_streaming(
                row_iterator_factory=row_iterator_factory,
                epochs=1,
                batch_size=2,
                progress_every_samples=0,
                verbose=False,
            )
        with pytest.raises(ValueError, match="total_rows_hint must be positive"):
            scorer.train_streaming(
                row_iterator_factory=row_iterator_factory,
                epochs=1,
                batch_size=2,
                progress_every_samples=2,
                total_rows_hint=-1,
                verbose=False,
            )

    def test_train_streaming_emits_row_progress_logs(self, capsys):
        """Streaming training should print row-level progress during long passes."""
        rows = [('MTMDKSEL', 1.0), ('XXXXXXXX', 0.0)] * 12

        def row_iterator_factory():
            return iter(rows)

        scorer = MLPScorer(
            k=8,
            hidden_layer_sizes=(16,),
            random_state=42,
            early_stopping=False,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            scorer.train_streaming(
                row_iterator_factory=row_iterator_factory,
                epochs=1,
                batch_size=3,
                progress_every_samples=6,
                total_rows_hint=len(rows),
                verbose=True,
            )

        output = capsys.readouterr().out
        assert "[scaler]" in output
        assert "[epoch 1]" in output

    def test_score_peptides(self):
        """Test scoring peptides with trained model."""
        peptides = ['MTMDKSEL', 'ACDEFGHI'] * 20
        labels = [0.0, 1.0] * 20

        scorer = MLPScorer(k=8, hidden_layer_sizes=(32,), random_state=42)
        scorer.train(peptides=peptides, labels=labels, epochs=100, verbose=False)

        scores = scorer.score(['MTMDKSEL', 'ACDEFGHI'])

        assert len(scores) == 2
        assert all(np.isfinite(scores))

    def test_model_learns_difference(self):
        """Test that model learns to distinguish self from foreign peptides."""
        # Create peptides with distinct properties
        self_peptides = ['ACDEFGHI', 'KLMNPQRS', 'TVWYACDE'] * 30
        foreign_peptides = ['XXXXXXXX', 'WWWWWWWW', 'YYYYYYYY'] * 30

        peptides = self_peptides + foreign_peptides
        labels = [0.0] * len(self_peptides) + [1.0] * len(foreign_peptides)

        scorer = MLPScorer(
            k=8,
            hidden_layer_sizes=(64, 32),
            random_state=42,
            use_dipeptides=True,
        )
        scorer.train(peptides=peptides, labels=labels, epochs=200, verbose=False)

        # Test on training examples - model should separate them
        self_score = scorer.score(['ACDEFGHI'])[0]
        foreign_score = scorer.score(['XXXXXXXX'])[0]

        # Foreign peptides should have higher scores
        assert foreign_score > self_score

    def test_save_load_model(self):
        """Test saving and loading a model."""
        peptides = ['MTMDKSEL', 'ACDEFGHI'] * 10
        labels = [0.0, 1.0] * 10

        scorer = MLPScorer(k=8, hidden_layer_sizes=(32,), random_state=42)
        scorer.train(peptides=peptides, labels=labels, epochs=50, verbose=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_model')
            scorer.save(path)

            # Load and verify
            loaded = MLPScorer.load(path)
            assert loaded.is_trained
            assert loaded.k == 8

            # Check scores are similar
            original_scores = scorer.score(['MTMDKSEL'])
            loaded_scores = loaded.score(['MTMDKSEL'])
            np.testing.assert_allclose(original_scores, loaded_scores, rtol=1e-5)

    def test_different_hidden_sizes(self):
        """Test different hidden layer configurations."""
        peptides = ['MTMDKSEL'] * 20
        labels = [1.0] * 20

        for hidden in [(32,), (64, 32), (128, 64, 32)]:
            scorer = MLPScorer(k=8, hidden_layer_sizes=hidden, random_state=42)
            scorer.train(peptides=peptides, labels=labels, epochs=20, verbose=False)

            scores = scorer.score(['MTMDKSEL'])
            assert np.isfinite(scores[0])

    def test_different_activations(self):
        """Test different activation functions."""
        peptides = ['MTMDKSEL'] * 20
        labels = [1.0] * 20

        for activation in ['relu', 'tanh', 'logistic']:
            scorer = MLPScorer(
                k=8,
                hidden_layer_sizes=(32,),
                activation=activation,
                random_state=42
            )
            scorer.train(peptides=peptides, labels=labels, epochs=20, verbose=False)

            scores = scorer.score(['MTMDKSEL'])
            assert np.isfinite(scores[0])

    def test_train_with_external_validation_sets_metadata(self):
        """External validation data should produce validation-loss metadata."""
        peptides = ['MTMDKSEL', 'ACDEFGHI'] * 20
        labels = [0.0, 1.0] * 20
        val_peptides = ['MTMDKSEL', 'ACDEFGHI'] * 5
        val_labels = [0.0, 1.0] * 5

        scorer = MLPScorer(k=8, hidden_layer_sizes=(16,), random_state=7)
        scorer.train(
            peptides=peptides,
            labels=labels,
            val_peptides=val_peptides,
            val_labels=val_labels,
            epochs=30,
            verbose=False,
        )

        assert 'final_val_loss' in scorer._metadata
        assert 'best_val_loss' in scorer._metadata
        assert np.isfinite(scorer._metadata['final_val_loss'])
        assert scorer._metadata['final_val_loss'] >= 0.0

    def test_train_requires_both_validation_inputs(self):
        """Providing only one validation argument should fail fast."""
        peptides = ['MTMDKSEL', 'ACDEFGHI'] * 20
        labels = [0.0, 1.0] * 20
        scorer = MLPScorer(k=8, hidden_layer_sizes=(16,), random_state=11)

        with pytest.raises(ValueError, match="Provide both val_peptides and val_labels"):
            scorer.train(peptides=peptides, labels=labels, val_peptides=['MTMDKSEL'], epochs=10, verbose=False)

    def test_predict_dataframe_missing_category_group_raises(self):
        """predict_dataframe should reject invalid foreign/self category groups."""
        peptides = ['MTMDKSEL', 'ACDEFGHI'] * 20
        labels = [[1.0, 0.0], [0.0, 1.0]] * 20
        categories = ['human', 'viruses']
        scorer = MLPScorer(k=8, hidden_layer_sizes=(16,), random_state=17)
        scorer.train(
            peptides=peptides,
            labels=labels,
            target_categories=categories,
            epochs=20,
            verbose=False,
        )

        with pytest.raises(ValueError, match="No pathogen categories found"):
            scorer.predict_dataframe(['MTMDKSEL'], pathogen_categories=['bacteria'])


class TestModelManager:
    """Tests for model manager."""

    def test_list_empty(self):
        """Test listing models when none exist."""
        from weirdo.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mm = ModelManager(model_dir=tmpdir)
            models = mm.list_models()
            assert len(models) == 0

    def test_save_and_list(self):
        """Test saving and listing models."""
        from weirdo.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mm = ModelManager(model_dir=tmpdir)

            # Train and save a model
            peptides = ['MTMDKSEL'] * 20
            labels = [1.0] * 20

            scorer = MLPScorer(k=8, hidden_layer_sizes=(32,), random_state=42)
            scorer.train(peptides=peptides, labels=labels, epochs=20, verbose=False)

            mm.save(scorer, 'test-model')

            # List models
            models = mm.list_models()
            assert len(models) == 1
            assert models[0].name == 'test-model'
            assert models[0].scorer_type == 'MLPScorer'

    def test_load_model(self):
        """Test loading a model by name."""
        from weirdo.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mm = ModelManager(model_dir=tmpdir)

            peptides = ['MTMDKSEL'] * 20
            labels = [1.0] * 20

            scorer = MLPScorer(k=8, hidden_layer_sizes=(32,), random_state=42)
            scorer.train(peptides=peptides, labels=labels, epochs=20, verbose=False)
            mm.save(scorer, 'my-model')

            # Load by name
            loaded = mm.load('my-model')
            assert loaded.is_trained

    def test_delete_model(self):
        """Test deleting a model."""
        from weirdo.model_manager import ModelManager

        with tempfile.TemporaryDirectory() as tmpdir:
            mm = ModelManager(model_dir=tmpdir)

            peptides = ['MTMDKSEL'] * 20
            labels = [1.0] * 20

            scorer = MLPScorer(k=8, hidden_layer_sizes=(32,), random_state=42)
            scorer.train(peptides=peptides, labels=labels, epochs=20, verbose=False)
            mm.save(scorer, 'to-delete')

            assert len(mm.list_models()) == 1

            mm.delete('to-delete')

            assert len(mm.list_models()) == 0


class TestRegistration:
    """Test that MLP scorer is registered properly."""

    def test_scorer_registered(self):
        """Test MLP scorer is in registry."""
        from weirdo.scorers import list_scorers

        scorers = list_scorers()
        assert 'mlp' in scorers

    def test_get_scorer(self):
        """Test getting scorer class from registry."""
        from weirdo.scorers import get_scorer

        mlp_cls = get_scorer('mlp')
        assert mlp_cls is not None
