"""Tests for MLP-based scorer."""

import os
import tempfile
import pytest
import numpy as np

from weirdo.scorers.mlp import MLPScorer, _kmer_to_onehot, _peptide_to_features


class TestFeatureExtraction:
    """Tests for feature extraction functions."""

    def test_kmer_to_onehot(self):
        """Test one-hot encoding of k-mers."""
        onehot = _kmer_to_onehot('AA')
        assert len(onehot) == 2 * 21  # 2 positions * 21 amino acids
        assert onehot[0] == 1.0  # A at position 0
        assert onehot[21] == 1.0  # A at position 1
        assert sum(onehot) == 2.0  # Exactly 2 ones

    def test_peptide_to_features(self):
        """Test peptide feature extraction."""
        features = _peptide_to_features('MTMDKSEL', k=8)
        assert len(features) == 8 * 21  # k * num_amino_acids
        assert np.isfinite(features).all()


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
        labels = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0] * 10

        scorer = MLPScorer(
            k=8,
            hidden_layer_sizes=(32, 16),
            random_state=42,
        )

        scorer.train(
            peptides=peptides,
            labels=labels,
            epochs=50,
            verbose=False,
        )

        assert scorer.is_trained
        assert len(scorer.training_history) > 0

    def test_score_peptides(self):
        """Test scoring peptides with trained model."""
        peptides = ['MTMDKSEL', 'ACDEFGHI'] * 20
        labels = [0.0, 5.0] * 20

        scorer = MLPScorer(k=8, hidden_layer_sizes=(32,), random_state=42)
        scorer.train(peptides=peptides, labels=labels, epochs=100, verbose=False)

        scores = scorer.score(['MTMDKSEL', 'ACDEFGHI'])

        assert len(scores) == 2
        assert all(np.isfinite(scores))

    def test_save_load_model(self):
        """Test saving and loading a model."""
        peptides = ['MTMDKSEL', 'ACDEFGHI'] * 10
        labels = [0.0, 5.0] * 10

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
