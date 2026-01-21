"""Frequency-based foreignness scorer.

Scores peptides based on k-mer frequency in reference dataset.
"""

from typing import Any, List, Literal, Optional, Sequence, Union

import numpy as np

from .base import BatchScorer
from .reference import BaseReference
from .registry import register_scorer


AggregateMethod = Literal['mean', 'max', 'min', 'sum']


@register_scorer('frequency', description='Frequency-based foreignness scoring')
class FrequencyScorer(BatchScorer):
    """Frequency-based foreignness scorer.

    Scores peptides by computing -log10(frequency + pseudocount) for
    each k-mer, then aggregating across the peptide.

    Higher scores indicate more "foreign" peptides (rarer k-mers).

    Parameters
    ----------
    k : int, default=8
        K-mer size for decomposing peptides.
    pseudocount : float, default=1e-10
        Small value added to frequencies to avoid log(0).
        Smaller values give higher scores for unseen k-mers.
    aggregate : str, default='mean'
        How to combine k-mer scores: 'mean', 'max', 'min', 'sum'.
    category : str, optional
        If set, only consider k-mers from this category when scoring.

    Example
    -------
    >>> ref = SwissProtReference(categories=['human']).load()
    >>> scorer = FrequencyScorer(k=8, aggregate='mean')
    >>> scorer.fit(ref)
    >>> scores = scorer.score(['MTMDKSEL', 'XXXXXXXX'])
    >>> # XXXXXXXX will have higher score (more foreign)
    """

    def __init__(
        self,
        k: int = 8,
        pseudocount: float = 1e-10,
        aggregate: AggregateMethod = 'mean',
        category: Optional[str] = None,
        batch_size: int = 10000,
        **kwargs
    ):
        super().__init__(batch_size=batch_size, **kwargs)
        self._params.update({
            'k': k,
            'pseudocount': pseudocount,
            'aggregate': aggregate,
            'category': category,
        })

    @property
    def k(self) -> int:
        """Get k-mer size."""
        return self._params['k']

    @property
    def pseudocount(self) -> float:
        """Get pseudocount value."""
        return self._params['pseudocount']

    @property
    def aggregate(self) -> AggregateMethod:
        """Get aggregation method."""
        return self._params['aggregate']

    @property
    def category(self) -> Optional[str]:
        """Get category filter."""
        return self._params.get('category')

    def fit(self, reference: BaseReference) -> 'FrequencyScorer':
        """Fit the scorer to a reference dataset.

        Parameters
        ----------
        reference : BaseReference
            Reference dataset providing k-mer frequencies.

        Returns
        -------
        self : FrequencyScorer
            Returns self for method chaining.
        """
        if not reference.is_loaded:
            raise RuntimeError(
                "Reference is not loaded. Call reference.load() first."
            )
        self._reference = reference
        self._is_fitted = True
        return self

    def score(self, peptides: Union[str, Sequence[str]]) -> np.ndarray:
        """Score peptide(s) for foreignness.

        Parameters
        ----------
        peptides : str or sequence of str
            Single peptide or list of peptides to score.

        Returns
        -------
        scores : np.ndarray
            Array of foreignness scores. Higher = more foreign.
        """
        self._check_is_fitted()
        peptides = self._ensure_list(peptides)

        scores = np.array([self._score_peptide(p) for p in peptides])
        return scores

    def _score_peptide(self, peptide: str) -> float:
        """Score a single peptide.

        Parameters
        ----------
        peptide : str
            Peptide sequence.

        Returns
        -------
        score : float
            Foreignness score.
        """
        k = self.k
        if len(peptide) < k:
            # Peptide too short for k-mers
            return float('inf')

        # Extract k-mers
        kmers = [peptide[i:i+k] for i in range(len(peptide) - k + 1)]

        # Score each k-mer
        kmer_scores = []
        for kmer in kmers:
            freq = self._reference.get_frequency(kmer, default=0.0)
            # -log10(freq + pseudocount)
            # Higher score = lower frequency = more foreign
            score = -np.log10(freq + self.pseudocount)
            kmer_scores.append(score)

        # Aggregate scores
        kmer_scores = np.array(kmer_scores)
        return self._aggregate_scores(kmer_scores)

    def _aggregate_scores(self, scores: np.ndarray) -> float:
        """Aggregate k-mer scores into a single score.

        Parameters
        ----------
        scores : np.ndarray
            Array of k-mer scores.

        Returns
        -------
        score : float
            Aggregated score.
        """
        if len(scores) == 0:
            return float('inf')

        agg = self.aggregate
        if agg == 'mean':
            return float(np.mean(scores))
        elif agg == 'max':
            return float(np.max(scores))
        elif agg == 'min':
            return float(np.min(scores))
        elif agg == 'sum':
            return float(np.sum(scores))
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

    def _score_batch_impl(self, batch: List[str]) -> np.ndarray:
        """Score a batch of peptides.

        Default implementation - could be optimized for specific
        reference implementations that support batch lookups.
        """
        return np.array([self._score_peptide(p) for p in batch])

    def get_kmer_scores(self, peptide: str) -> List[tuple]:
        """Get individual k-mer scores for a peptide.

        Useful for debugging and understanding which k-mers
        contribute most to the foreignness score.

        Parameters
        ----------
        peptide : str
            Peptide sequence.

        Returns
        -------
        kmer_scores : list of (str, float)
            List of (k-mer, score) tuples.
        """
        self._check_is_fitted()
        k = self.k

        if len(peptide) < k:
            return []

        results = []
        for i in range(len(peptide) - k + 1):
            kmer = peptide[i:i+k]
            freq = self._reference.get_frequency(kmer, default=0.0)
            score = -np.log10(freq + self.pseudocount)
            results.append((kmer, score))

        return results
