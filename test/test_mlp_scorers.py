"""Tests for MLP-based scorer."""

import os
import tempfile
import pytest
import numpy as np

from weirdo.scorers.mlp import (
    MLPScorer,
    extract_features,
    _compute_property_features,
    _compute_composition_features,
    _compute_dipeptide_features,
    _compute_kmer_onehot,
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

    def test_compute_kmer_onehot(self):
        """Test k-mer one-hot encoding."""
        features = _compute_kmer_onehot('AA', k=2)
        assert len(features) == 2 * 21  # k * 21 amino acids
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
        # 48 (properties) + 27 (structural) + 20 (composition) + 168 (8*21 kmer) + 400 (dipeptides) = 663
        expected_length = 48 + 27 + 20 + 8 * 21 + 400
        assert len(features) == expected_length
        assert np.isfinite(features).all()

    def test_extract_features_no_dipeptides(self):
        """Test feature extraction without dipeptides."""
        features = extract_features('MTMDKSEL', k=8, use_dipeptides=False)
        # 48 (properties) + 27 (structural) + 20 (composition) + 168 (8*21 kmer) = 263
        expected_length = 48 + 27 + 20 + 8 * 21
        assert len(features) == expected_length
        assert np.isfinite(features).all()

    def test_get_feature_names(self):
        """Test getting feature names from scorer."""
        scorer = MLPScorer(k=8, use_dipeptides=True)
        names = scorer.get_feature_names()

        # Should have names for all features
        expected_count = 48 + 27 + 20 + 8 * 21 + 400
        assert len(names) == expected_count

        # Check some specific feature names
        assert 'hydropathy_mean' in names
        assert 'helix_propensity_mean' in names
        assert 'frac_cysteine' in names
        assert 'arginine_ratio' in names
        assert 'aa_freq_A' in names
        assert 'kmer_pos0_A' in names
        assert 'dipep_AA' in names


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
