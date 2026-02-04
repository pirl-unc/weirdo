"""Tests for the scorer module."""

import warnings
import numpy as np
import pytest
from typing import Dict, Iterator, List, Optional, Tuple
from sklearn.exceptions import ConvergenceWarning

from weirdo.scorers import (
    BaseScorer,
    BatchScorer,
    BaseReference,
    StreamingReference,
    MLPScorer,
    ScorerConfig,
    registry,
    register_scorer,
    list_scorers,
    list_references,
    list_presets,
    get_preset,
)


# ====================
# Mock Reference for Testing
# ====================

class MockReference(BaseReference):
    """Simple mock reference for testing scorers."""

    def __init__(
        self,
        kmers: Optional[Dict[str, float]] = None,
        categories: Optional[List[str]] = None,
        k: int = 8
    ):
        super().__init__(categories=categories, k=k)
        self._kmers = kmers or {}

    def load(self) -> 'MockReference':
        self._is_loaded = True
        return self

    def contains(self, kmer: str) -> bool:
        return kmer in self._kmers

    def get_frequency(self, kmer: str, default: float = 0.0) -> float:
        return self._kmers.get(kmer, default)

    def get_categories(self) -> List[str]:
        return self._categories or ['mock']

    def iter_kmers(self) -> Iterator[str]:
        yield from self._kmers.keys()

    def __len__(self) -> int:
        return len(self._kmers)


# ====================
# Base Classes Tests
# ====================

class TestBaseScorer:
    """Tests for BaseScorer ABC."""

    def test_params_management(self):
        """Test get_params and set_params."""
        class SimpleScorer(BaseScorer):
            def __init__(self, k=8, alpha=0.5):
                super().__init__(k=k, alpha=alpha)

            def fit(self, reference):
                self._is_fitted = True
                return self

            def score(self, peptides):
                return np.zeros(len(self._ensure_list(peptides)))

        scorer = SimpleScorer(k=10, alpha=0.3)
        params = scorer.get_params()
        assert params['k'] == 10
        assert params['alpha'] == 0.3

        scorer.set_params(k=12)
        assert scorer.get_params()['k'] == 12
        assert not scorer.is_fitted  # Should invalidate fit

    def test_is_fitted_check(self):
        """Test that scoring without fitting raises error."""
        class SimpleScorer(BaseScorer):
            def fit(self, reference):
                self._is_fitted = True
                return self

            def score(self, peptides):
                self._check_is_fitted()
                return np.zeros(len(self._ensure_list(peptides)))

        scorer = SimpleScorer()
        with pytest.raises(RuntimeError, match="not fitted"):
            scorer.score(['MTMDKSEL'])

    def test_ensure_list(self):
        """Test single peptide to list conversion."""
        class SimpleScorer(BaseScorer):
            def fit(self, ref):
                self._is_fitted = True
                return self

            def score(self, peptides):
                peps = self._ensure_list(peptides)
                return np.zeros(len(peps))

        scorer = SimpleScorer()
        scorer.fit(None)

        # Single string
        scores = scorer.score('MTMDKSEL')
        assert len(scores) == 1

        # List of strings
        scores = scorer.score(['A', 'B', 'C'])
        assert len(scores) == 3


class TestBatchScorer:
    """Tests for BatchScorer."""

    def test_batch_size_param(self):
        """Test batch_size parameter."""
        class SimpleBatchScorer(BatchScorer):
            def fit(self, ref):
                self._is_fitted = True
                return self

            def score(self, peptides):
                return np.zeros(len(self._ensure_list(peptides)))

        scorer = SimpleBatchScorer(batch_size=100)
        assert scorer.batch_size == 100


# ====================
# Registry Tests
# ====================

class TestRegistry:
    """Tests for the scorer registry."""

    def test_builtin_scorers_registered(self):
        """Test that built-in scorers are registered."""
        scorers = list_scorers()
        assert 'mlp' in scorers

    def test_builtin_references_registered(self):
        """Test that built-in references are registered."""
        references = list_references()
        assert 'swissprot' in references

    def test_create_scorer(self):
        """Test creating scorer by name."""
        scorer = registry.create_scorer('mlp', k=8)
        assert isinstance(scorer, MLPScorer)
        assert scorer.k == 8

    def test_unknown_scorer_error(self):
        """Test error for unknown scorer."""
        with pytest.raises(KeyError, match="Unknown scorer"):
            registry.get_scorer('nonexistent')

    def test_custom_scorer_registration(self):
        """Test registering a custom scorer."""
        @register_scorer('test_custom', description='Test scorer')
        class TestCustomScorer(BaseScorer):
            def fit(self, ref):
                self._is_fitted = True
                return self

            def score(self, peptides):
                return np.zeros(len(self._ensure_list(peptides)))

        assert 'test_custom' in list_scorers()
        scorer = registry.create_scorer('test_custom')
        assert isinstance(scorer, TestCustomScorer)


# ====================
# Configuration Tests
# ====================

class TestScorerConfig:
    """Tests for ScorerConfig."""

    def test_presets_available(self):
        """Test that presets are available."""
        presets = list_presets()
        assert 'default' in presets
        assert 'fast' in presets

    def test_get_preset(self):
        """Test getting a preset config."""
        config = get_preset('default')
        assert config.scorer == 'mlp'
        assert config.reference == 'swissprot'
        assert config.k == 8

    def test_from_dict(self):
        """Test creating config from dict."""
        data = {
            'scorer': 'mlp',
            'reference': 'swissprot',
            'k': 10,
            'scorer_params': {'hidden_layer_sizes': (64, 32)},
        }
        config = ScorerConfig.from_dict(data)
        assert config.k == 10
        assert config.scorer_params['hidden_layer_sizes'] == (64, 32)

    def test_to_dict(self):
        """Test converting config to dict."""
        config = ScorerConfig(
            scorer='mlp',
            reference='swissprot',
            k=12
        )
        data = config.to_dict()
        assert data['k'] == 12
        assert data['scorer'] == 'mlp'

    def test_preset_returns_copy(self):
        """Test that get_preset returns a copy."""
        config1 = get_preset('default')
        config2 = get_preset('default')
        config1.k = 99
        assert config2.k == 8  # Should not be affected


# ====================
# Integration Tests
# ====================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self):
        """Test complete scoring workflow."""
        # Create simple training data
        peptides = ['MTMDKSEL', 'ACDEFGHI', 'XXXXXXXX', 'YYYYYYYY'] * 10
        labels = [0.0, 0.0, 1.0, 1.0] * 10

        scorer = MLPScorer(k=8, hidden_layer_sizes=(32,), random_state=1)
        scorer.train(peptides=peptides, labels=labels, epochs=50, verbose=False)

        scores = scorer.score(['MTMDKSEL', 'XXXXXXXX'])

        assert len(scores) == 2
        assert scores[0] < scores[1]  # Known vs unknown

    def test_batch_scoring(self):
        """Test batch scoring for large datasets."""
        peptides = ['MTMDKSEL', 'XXXXXXXX'] * 20
        labels = [0.0, 1.0] * 20
        scorer = MLPScorer(k=8, hidden_layer_sizes=(16,), batch_size=10, random_state=1)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            scorer.train(peptides=peptides, labels=labels, epochs=30, verbose=False)

        # Create many peptides
        peptides = ['AAAAAAAAAA' + str(i) for i in range(100)]
        scores = scorer.score_batch(peptides)

        assert len(scores) == 100
