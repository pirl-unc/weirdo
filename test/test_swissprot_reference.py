"""Tests for SwissProtReference with real CSV data."""

import os
import warnings
import pytest
import numpy as np
from sklearn.exceptions import ConvergenceWarning

from weirdo.scorers import (
    SwissProtReference,
    MLPScorer,
)

# Path to test fixture
FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__),
    'fixtures',
    'swissprot-8mers-sample.csv'
)


class TestSwissProtReference:
    """Tests for SwissProtReference loading and querying."""

    def test_load_fixture(self):
        """Test loading the fixture CSV."""
        ref = SwissProtReference(data_path=FIXTURE_PATH)
        ref.load()

        assert ref.is_loaded
        assert len(ref) >= 500  # At least 500 k-mers

    def test_contains_known_kmer(self):
        """Test contains() for known k-mer."""
        ref = SwissProtReference(data_path=FIXTURE_PATH).load()

        # First k-mer in the fixture
        assert ref.contains('MTMDKSEL')
        assert 'MTMDKSEL' in ref  # Test __contains__

    def test_contains_unknown_kmer(self):
        """Test contains() for unknown k-mer."""
        ref = SwissProtReference(data_path=FIXTURE_PATH).load()

        assert not ref.contains('XXXXXXXX')
        assert 'XXXXXXXX' not in ref

    def test_get_frequency(self):
        """Test get_frequency() returns correct values."""
        ref = SwissProtReference(data_path=FIXTURE_PATH).load()

        # Known k-mer should return 1.0 (present)
        assert ref.get_frequency('MTMDKSEL') == 1.0

        # Unknown k-mer should return default
        assert ref.get_frequency('XXXXXXXX') == 0.0
        assert ref.get_frequency('XXXXXXXX', default=-1.0) == -1.0

    def test_get_categories(self):
        """Test get_categories() returns all categories."""
        ref = SwissProtReference(data_path=FIXTURE_PATH).load()
        categories = ref.get_categories()

        expected = [
            'archaea', 'bacteria', 'fungi', 'human', 'invertebrates',
            'mammals', 'plants', 'rodents', 'vertebrates', 'viruses'
        ]
        assert sorted(categories) == sorted(expected)

    def test_get_kmer_categories(self):
        """Test getting category breakdown for a k-mer."""
        ref = SwissProtReference(data_path=FIXTURE_PATH).load()
        cats = ref.get_kmer_categories('MTMDKSEL')

        # MTMDKSEL is in human, mammals, rodents (from CSV)
        assert cats['human'] is True
        assert cats['mammals'] is True
        assert cats['rodents'] is True
        assert cats['bacteria'] is False

    def test_iter_kmers(self):
        """Test iterating over all k-mers."""
        ref = SwissProtReference(data_path=FIXTURE_PATH).load()
        kmers = list(ref.iter_kmers())

        assert len(kmers) == len(ref)
        assert 'MTMDKSEL' in kmers

    def test_iter_kmers_with_counts(self):
        """Test iterating with category counts."""
        ref = SwissProtReference(data_path=FIXTURE_PATH).load()
        items = list(ref.iter_kmers_with_counts())

        assert len(items) == len(ref)
        # Check first item (MTMDKSEL has label_count=3)
        kmer, count = items[0]
        assert kmer == 'MTMDKSEL'
        assert count == 3

    def test_category_filter(self):
        """Test filtering by categories."""
        # Load without filter to get total count
        ref_all = SwissProtReference(data_path=FIXTURE_PATH).load()
        total_count = len(ref_all)

        # Load with human filter
        ref_human = SwissProtReference(
            data_path=FIXTURE_PATH,
            categories=['human']
        ).load()

        # Should only include k-mers present in human
        kmers_human = list(ref_human.iter_kmers())
        assert len(kmers_human) < total_count  # Fewer than total

        # Verify all returned k-mers are in human category
        ref_all = SwissProtReference(data_path=FIXTURE_PATH).load()
        for kmer in kmers_human:
            cats = ref_all.get_kmer_categories(kmer)
            assert cats['human'] is True

    def test_use_set_mode(self):
        """Test use_set mode (no category info)."""
        ref = SwissProtReference(
            data_path=FIXTURE_PATH,
            use_set=True
        ).load()

        assert ref.is_loaded
        assert ref.contains('MTMDKSEL')

        # Category info should not be available
        with pytest.raises(RuntimeError, match="use_set mode"):
            ref.get_kmer_categories('MTMDKSEL')

    def test_lazy_mode_contains(self):
        """Test lazy mode for contains()."""
        ref = SwissProtReference(
            data_path=FIXTURE_PATH,
            lazy=True
        ).load()

        assert ref.is_loaded
        assert ref.contains('MTMDKSEL')
        assert not ref.contains('XXXXXXXX')

    def test_lazy_mode_iter(self):
        """Test lazy mode for iteration."""
        ref = SwissProtReference(
            data_path=FIXTURE_PATH,
            lazy=True
        ).load()

        kmers = list(ref.iter_kmers())
        assert len(kmers) >= 500  # At least 500 k-mers

    def test_invalid_category_error(self):
        """Test error for invalid category."""
        with pytest.raises(ValueError, match="Invalid categories"):
            SwissProtReference(
                data_path=FIXTURE_PATH,
                categories=['invalid_category']
            )

    def test_file_not_found_error(self):
        """Test error when data file doesn't exist."""
        ref = SwissProtReference(data_path='/nonexistent/path.csv')
        with pytest.raises(FileNotFoundError):
            ref.load()


class TestSwissProtWithScorers:
    """Integration tests for SwissProtReference with scorers."""

    def test_mlp_scorer_with_swissprot(self):
        """Test MLPScorer with SwissProt data."""
        ref = SwissProtReference(data_path=FIXTURE_PATH).load()
        categories = ['human', 'viruses', 'bacteria']
        peptides, labels = ref.get_training_data(
            target_categories=categories,
            multi_label=True,
            max_samples=200,
            shuffle=True,
            seed=1,
        )

        scorer = MLPScorer(k=8, hidden_layer_sizes=(32,), random_state=1)
        scorer.train(peptides=peptides, labels=labels, target_categories=categories, epochs=50, verbose=False)

        scores = scorer.score(['MTMDKSEL', 'XXXXXXXX'])
        assert len(scores) == 2
        assert np.isfinite(scores[0])

    def test_mlp_scores_variable_length(self):
        """Test scoring peptides longer than k with aggregation."""
        ref = SwissProtReference(data_path=FIXTURE_PATH).load()
        categories = ['human', 'viruses', 'bacteria']
        peptides, labels = ref.get_training_data(
            target_categories=categories,
            multi_label=True,
            max_samples=200,
            shuffle=True,
            seed=2,
        )

        scorer = MLPScorer(k=8, hidden_layer_sizes=(32,), random_state=2)
        scorer.train(peptides=peptides, labels=labels, target_categories=categories, epochs=30, verbose=False)

        peptide = 'MTMDKSELTMDKSELVMDKSELVD'  # 24-mer
        score = scorer.score([peptide])[0]
        assert np.isfinite(score)

    def test_predict_dataframe_columns(self):
        """Test predict_dataframe output columns."""
        ref = SwissProtReference(data_path=FIXTURE_PATH).load()
        categories = ['human', 'viruses', 'bacteria']
        peptides, labels = ref.get_training_data(
            target_categories=categories,
            multi_label=True,
            max_samples=100,
            shuffle=True,
            seed=3,
        )

        scorer = MLPScorer(k=8, hidden_layer_sizes=(16,), random_state=3)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            scorer.train(peptides=peptides, labels=labels, target_categories=categories, epochs=20, verbose=False)

        df = scorer.predict_dataframe(['MTMDKSEL'])
        assert 'foreignness' in df.columns
        for cat in categories:
            assert cat in df.columns


class TestSwissProtPerformance:
    """Performance-related tests."""

    def test_memory_efficient_modes(self):
        """Compare memory usage of different modes."""
        # Standard mode
        ref_standard = SwissProtReference(data_path=FIXTURE_PATH).load()

        # Set mode (less memory)
        ref_set = SwissProtReference(
            data_path=FIXTURE_PATH,
            use_set=True
        ).load()

        # Both should work for basic operations
        assert ref_standard.contains('MTMDKSEL')
        assert ref_set.contains('MTMDKSEL')

    def test_streaming_iteration(self):
        """Test that streaming doesn't load all into memory."""
        ref = SwissProtReference(
            data_path=FIXTURE_PATH,
            lazy=True
        ).load()

        # Should be able to iterate without loading all data
        count = 0
        for kmer in ref.iter_kmers():
            count += 1
            if count >= 10:
                break

        assert count == 10
