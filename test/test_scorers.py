"""Tests for the scorer module."""

import numpy as np
import pytest
from typing import Dict, Iterator, List, Optional, Tuple

from weirdo.scorers import (
    BaseScorer,
    BatchScorer,
    BaseReference,
    StreamingReference,
    FrequencyScorer,
    SimilarityScorer,
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
        assert 'frequency' in scorers
        assert 'similarity' in scorers

    def test_builtin_references_registered(self):
        """Test that built-in references are registered."""
        references = list_references()
        assert 'swissprot' in references

    def test_create_scorer(self):
        """Test creating scorer by name."""
        scorer = registry.create_scorer('frequency', k=8)
        assert isinstance(scorer, FrequencyScorer)
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
        assert 'pathogen' in presets
        assert 'human' in presets
        assert 'similarity_blosum62' in presets

    def test_get_preset(self):
        """Test getting a preset config."""
        config = get_preset('default')
        assert config.scorer == 'frequency'
        assert config.reference == 'swissprot'
        assert config.k == 8

    def test_from_dict(self):
        """Test creating config from dict."""
        data = {
            'scorer': 'frequency',
            'reference': 'swissprot',
            'k': 10,
            'scorer_params': {'aggregate': 'max'},
        }
        config = ScorerConfig.from_dict(data)
        assert config.k == 10
        assert config.scorer_params['aggregate'] == 'max'

    def test_to_dict(self):
        """Test converting config to dict."""
        config = ScorerConfig(
            scorer='frequency',
            reference='swissprot',
            k=12
        )
        data = config.to_dict()
        assert data['k'] == 12
        assert data['scorer'] == 'frequency'

    def test_preset_returns_copy(self):
        """Test that get_preset returns a copy."""
        config1 = get_preset('default')
        config2 = get_preset('default')
        config1.k = 99
        assert config2.k == 8  # Should not be affected


# ====================
# FrequencyScorer Tests
# ====================

class TestFrequencyScorer:
    """Tests for FrequencyScorer."""

    def test_basic_scoring(self):
        """Test basic frequency scoring."""
        # Create mock reference with some k-mers
        ref = MockReference(kmers={
            'AAAAAAAA': 1.0,
            'BBBBBBBB': 0.5,
        }, k=8).load()

        scorer = FrequencyScorer(k=8, pseudocount=1e-10, aggregate='mean')
        scorer.fit(ref)

        # Score peptides
        scores = scorer.score(['AAAAAAAA', 'ZZZZZZZZ'])

        # Known k-mer should have lower score (less foreign)
        assert scores[0] < scores[1]

    def test_aggregation_methods(self):
        """Test different aggregation methods."""
        ref = MockReference(kmers={
            'AAAAAAAA': 1.0,
            'AAAAAAAB': 0.0,  # Not present
        }, k=8).load()

        # Test mean
        scorer_mean = FrequencyScorer(k=8, aggregate='mean').fit(ref)
        scores_mean = scorer_mean.score(['AAAAAAAAAB'])

        # Test max (should give higher score)
        scorer_max = FrequencyScorer(k=8, aggregate='max').fit(ref)
        scores_max = scorer_max.score(['AAAAAAAAAB'])

        # Test min (should give lower score)
        scorer_min = FrequencyScorer(k=8, aggregate='min').fit(ref)
        scores_min = scorer_min.score(['AAAAAAAAAB'])

        # Verify ordering
        assert scores_min[0] < scores_mean[0] < scores_max[0]

    def test_short_peptide(self):
        """Test handling of peptides shorter than k."""
        ref = MockReference(kmers={'AAAAAAAA': 1.0}, k=8).load()
        scorer = FrequencyScorer(k=8).fit(ref)

        scores = scorer.score(['SHORT'])
        assert scores[0] == float('inf')

    def test_get_kmer_scores(self):
        """Test getting individual k-mer scores."""
        ref = MockReference(kmers={
            'AAAAAAAA': 1.0,
            'AAAAAAAB': 0.0,
        }, k=8).load()

        scorer = FrequencyScorer(k=8).fit(ref)
        kmer_scores = scorer.get_kmer_scores('AAAAAAAAAB')

        assert len(kmer_scores) == 3  # 10-mer has 3 8-mers
        # K-mers from AAAAAAAAAB: AAAAAAAA (0-7), AAAAAAAA (1-8), AAAAAAAB (2-9)
        assert kmer_scores[0][0] == 'AAAAAAAA'
        assert kmer_scores[1][0] == 'AAAAAAAA'
        assert kmer_scores[2][0] == 'AAAAAAAB'

    def test_not_fitted_error(self):
        """Test error when scoring without fitting."""
        scorer = FrequencyScorer(k=8)
        with pytest.raises(RuntimeError, match="not fitted"):
            scorer.score(['MTMDKSEL'])

    def test_reference_not_loaded_error(self):
        """Test error when reference is not loaded."""
        ref = MockReference(kmers={'AAAAAAAA': 1.0}, k=8)  # Not loaded
        scorer = FrequencyScorer(k=8)
        with pytest.raises(RuntimeError, match="not loaded"):
            scorer.fit(ref)


# ====================
# SimilarityScorer Tests
# ====================

class TestSimilarityScorer:
    """Tests for SimilarityScorer."""

    def test_basic_scoring(self):
        """Test basic similarity scoring."""
        ref = MockReference(kmers={
            'AAAAAAAA': 1.0,
            'MMMMMMMM': 1.0,
        }, k=8).load()

        scorer = SimilarityScorer(k=8, matrix='blosum62')
        scorer.fit(ref)

        # Score peptides
        scores = scorer.score(['AAAAAAAA', 'ZZZZZZZZ'])

        # Identical k-mer should have lower score (less foreign)
        assert scores[0] < scores[1]

    def test_matrix_options(self):
        """Test different substitution matrices."""
        ref = MockReference(kmers={'AAAAAAAA': 1.0}, k=8).load()

        for matrix in ['blosum30', 'blosum50', 'blosum62', 'pmbec']:
            scorer = SimilarityScorer(k=8, matrix=matrix)
            scorer.fit(ref)
            scores = scorer.score(['AAAAAAAA'])
            assert np.isfinite(scores[0])

    def test_invalid_matrix(self):
        """Test error for invalid matrix name."""
        with pytest.raises(ValueError, match="Unknown matrix"):
            SimilarityScorer(k=8, matrix='invalid')

    def test_get_closest_reference(self):
        """Test finding closest reference k-mers."""
        ref = MockReference(kmers={
            'AAAAAAAA': 1.0,
            'AAAAMAAA': 1.0,
            'MMMMMMMM': 1.0,
        }, k=8).load()

        scorer = SimilarityScorer(k=8, matrix='blosum62')
        scorer.fit(ref)

        matches = scorer.get_closest_reference('AAAAAAAA', n=2)
        assert len(matches) == 2
        # Exact match should be first with distance 0
        assert matches[0][0] == 'AAAAAAAA'
        assert matches[0][1] == 0.0


# ====================
# Integration Tests
# ====================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self):
        """Test complete scoring workflow."""
        # Create reference
        ref = MockReference(kmers={
            'MTMDKSEL': 1.0,
            'ACDEFGHI': 0.8,
            'KLMNPQRS': 0.5,
        }, k=8).load()

        # Create and fit scorer
        scorer = FrequencyScorer(k=8, aggregate='mean')
        scorer.fit(ref)

        # Score peptides
        peptides = ['MTMDKSEL', 'XXXXXXXX']
        scores = scorer.score(peptides)

        # Verify results
        assert len(scores) == 2
        assert scores[0] < scores[1]  # Known vs unknown

    def test_fit_score_convenience(self):
        """Test fit_score convenience method."""
        ref = MockReference(kmers={'AAAAAAAA': 1.0}, k=8).load()
        scorer = FrequencyScorer(k=8)

        scores = scorer.fit_score(ref, ['AAAAAAAA', 'BBBBBBBB'])
        assert len(scores) == 2

    def test_batch_scoring(self):
        """Test batch scoring for large datasets."""
        ref = MockReference(kmers={'AAAAAAAA': 1.0}, k=8).load()
        scorer = FrequencyScorer(k=8, batch_size=10)
        scorer.fit(ref)

        # Create many peptides
        peptides = ['AAAAAAAAAA' + str(i) for i in range(100)]
        scores = scorer.score_batch(peptides)

        assert len(scores) == 100
