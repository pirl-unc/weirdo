"""Tests for MLP-based scorer."""

import os
import tempfile
import pytest
import numpy as np

# Check if torch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Skip all tests if torch not available
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


class TestMLPScorer:
    """Tests for MLPScorer."""

    def test_create_scorer(self):
        """Test creating an MLP scorer."""
        from weirdo.scorers.mlp import MLPScorer

        scorer = MLPScorer(
            k=8,
            embedding_dim=32,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
        )

        assert scorer.k == 8
        assert not scorer.is_trained

    def test_train_scorer(self):
        """Test training an MLP scorer."""
        from weirdo.scorers.mlp import MLPScorer

        # Create synthetic training data
        peptides = [
            'MTMDKSEL', 'ACDEFGHI', 'KLMNPQRS',
            'XXXXXXXX', 'YYYYYYYY', 'WWWWWWWW',
        ]
        # Low scores for known, high for unknown
        labels = [0.0, 0.0, 0.0, 10.0, 10.0, 10.0]

        scorer = MLPScorer(
            k=8,
            embedding_dim=16,
            hidden_dim=32,
            num_layers=1,
        )

        scorer.train(
            peptides=peptides,
            labels=labels,
            epochs=10,
            verbose=False,
        )

        assert scorer.is_trained
        assert len(scorer.training_history) > 0

    def test_score_peptides(self):
        """Test scoring peptides with trained model."""
        from weirdo.scorers.mlp import MLPScorer

        peptides = ['MTMDKSEL', 'ACDEFGHI'] * 10
        labels = [0.0, 5.0] * 10

        scorer = MLPScorer(k=8, embedding_dim=16, hidden_dim=32, num_layers=1)
        scorer.train(peptides=peptides, labels=labels, epochs=20, verbose=False)

        scores = scorer.score(['MTMDKSEL', 'ACDEFGHI'])

        assert len(scores) == 2
        assert all(np.isfinite(scores))

    def test_save_load_model(self):
        """Test saving and loading a model."""
        from weirdo.scorers.mlp import MLPScorer

        peptides = ['MTMDKSEL', 'ACDEFGHI'] * 5
        labels = [0.0, 5.0] * 5

        scorer = MLPScorer(k=8, embedding_dim=16, hidden_dim=32, num_layers=1)
        scorer.train(peptides=peptides, labels=labels, epochs=5, verbose=False)

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

    def test_kmer_extraction(self):
        """Test k-mer extraction."""
        from weirdo.scorers.mlp import MLPScorer

        scorer = MLPScorer(k=8)
        kmers = scorer._extract_kmers('MTMDKSELVQKA')

        assert len(kmers) == 5  # 12 - 8 + 1
        assert kmers[0] == 'MTMDKSEL'

    def test_aggregate_methods(self):
        """Test different aggregation methods."""
        from weirdo.scorers.mlp import MLPScorer

        peptides = ['MTMDKSEL'] * 10
        labels = [1.0] * 10

        for agg in ['mean', 'max', 'min']:
            scorer = MLPScorer(k=8, hidden_dim=32, num_layers=1, aggregate=agg)
            scorer.train(peptides=peptides, labels=labels, epochs=5, verbose=False)

            scores = scorer.score(['MTMDKSELVQKA'])
            assert len(scores) == 1
            assert np.isfinite(scores[0])

    def test_validation_and_early_stopping(self):
        """Test training with validation data and early stopping."""
        from weirdo.scorers.mlp import MLPScorer

        peptides = ['MTMDKSEL', 'ACDEFGHI'] * 20
        labels = [0.0, 5.0] * 20

        val_peptides = ['KLMNPQRS', 'XXXXXXXX'] * 5
        val_labels = [2.0, 8.0] * 5

        scorer = MLPScorer(k=8, hidden_dim=32, num_layers=1)
        scorer.train(
            peptides=peptides,
            labels=labels,
            val_peptides=val_peptides,
            val_labels=val_labels,
            epochs=100,
            patience=5,
            verbose=False,
        )

        assert scorer.is_trained
        # Should have stopped early or completed
        assert len(scorer.training_history) <= 100

    def test_residual_architecture(self):
        """Test that deeper models work (residual connections enable this)."""
        from weirdo.scorers.mlp import MLPScorer

        peptides = ['MTMDKSEL'] * 20
        labels = [1.0] * 20

        # Deep model with 5 layers
        scorer = MLPScorer(
            k=8,
            embedding_dim=16,
            hidden_dim=64,
            num_layers=5,
        )
        scorer.train(peptides=peptides, labels=labels, epochs=10, verbose=False)

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
        from weirdo.scorers.mlp import MLPScorer

        with tempfile.TemporaryDirectory() as tmpdir:
            mm = ModelManager(model_dir=tmpdir)

            # Train and save a model
            peptides = ['MTMDKSEL'] * 5
            labels = [1.0] * 5

            scorer = MLPScorer(k=8, hidden_dim=32, num_layers=1)
            scorer.train(peptides=peptides, labels=labels, epochs=5, verbose=False)

            mm.save(scorer, 'test-model')

            # List models
            models = mm.list_models()
            assert len(models) == 1
            assert models[0].name == 'test-model'
            assert models[0].scorer_type == 'MLPScorer'

    def test_load_model(self):
        """Test loading a model by name."""
        from weirdo.model_manager import ModelManager
        from weirdo.scorers.mlp import MLPScorer

        with tempfile.TemporaryDirectory() as tmpdir:
            mm = ModelManager(model_dir=tmpdir)

            peptides = ['MTMDKSEL'] * 5
            labels = [1.0] * 5

            scorer = MLPScorer(k=8, hidden_dim=32, num_layers=1)
            scorer.train(peptides=peptides, labels=labels, epochs=5, verbose=False)
            mm.save(scorer, 'my-model')

            # Load by name
            loaded = mm.load('my-model')
            assert loaded.is_trained

    def test_delete_model(self):
        """Test deleting a model."""
        from weirdo.model_manager import ModelManager
        from weirdo.scorers.mlp import MLPScorer

        with tempfile.TemporaryDirectory() as tmpdir:
            mm = ModelManager(model_dir=tmpdir)

            peptides = ['MTMDKSEL'] * 5
            labels = [1.0] * 5

            scorer = MLPScorer(k=8, hidden_dim=32, num_layers=1)
            scorer.train(peptides=peptides, labels=labels, epochs=5, verbose=False)
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
