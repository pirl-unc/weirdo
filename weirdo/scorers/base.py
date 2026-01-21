"""Base classes for foreignness scorers.

Provides abstract base classes defining the scorer interface,
following sklearn-style fit/score patterns.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np


class BaseScorer(ABC):
    """Abstract base class for foreignness scorers.

    Scorers follow a fit/score pattern similar to sklearn:
    1. Initialize with configuration parameters
    2. Call fit() with a reference dataset
    3. Call score() on new peptides

    Example
    -------
    >>> scorer = MyScorer(k=8, aggregate='mean')
    >>> scorer.fit(reference)
    >>> scores = scorer.score(['MTMDKSEL', 'ACDEFGHI'])
    """

    def __init__(self, **params):
        """Initialize scorer with parameters.

        Parameters
        ----------
        **params : dict
            Scorer-specific configuration parameters.
        """
        self._params = params
        self._is_fitted = False
        self._reference = None

    @abstractmethod
    def fit(self, reference: Any) -> 'BaseScorer':
        """Fit the scorer to a reference dataset.

        Parameters
        ----------
        reference : BaseReference
            Reference dataset providing k-mer frequencies or other data.

        Returns
        -------
        self : BaseScorer
            Returns self for method chaining.
        """
        pass

    @abstractmethod
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
            Shape: (n_peptides,)
        """
        pass

    def fit_score(self, reference: Any, peptides: Union[str, Sequence[str]]) -> np.ndarray:
        """Fit to reference and score peptides in one call.

        Parameters
        ----------
        reference : BaseReference
            Reference dataset to fit.
        peptides : str or sequence of str
            Peptides to score.

        Returns
        -------
        scores : np.ndarray
            Foreignness scores.
        """
        self.fit(reference)
        return self.score(peptides)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get scorer parameters.

        Parameters
        ----------
        deep : bool, default=True
            If True, return parameters of nested objects.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return self._params.copy()

    def set_params(self, **params) -> 'BaseScorer':
        """Set scorer parameters.

        Parameters
        ----------
        **params : dict
            Scorer parameters to update.

        Returns
        -------
        self : BaseScorer
            Returns self for method chaining.
        """
        self._params.update(params)
        self._is_fitted = False  # Invalidate fit when params change
        return self

    @property
    def is_fitted(self) -> bool:
        """Check if scorer has been fitted."""
        return self._is_fitted

    def _check_is_fitted(self) -> None:
        """Raise error if scorer is not fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. "
                "Call fit() before score()."
            )

    def _ensure_list(self, peptides: Union[str, Sequence[str]]) -> List[str]:
        """Convert single peptide to list if needed."""
        if isinstance(peptides, str):
            return [peptides]
        return list(peptides)


class BatchScorer(BaseScorer):
    """Base class for scorers that support efficient batch operations.

    Extends BaseScorer with score_batch() for vectorized scoring
    of large peptide sets.
    """

    def __init__(self, batch_size: int = 10000, **params):
        """Initialize batch scorer.

        Parameters
        ----------
        batch_size : int, default=10000
            Number of peptides to process per batch.
        **params : dict
            Additional scorer parameters.
        """
        super().__init__(**params)
        self._params['batch_size'] = batch_size

    @property
    def batch_size(self) -> int:
        """Get batch size for vectorized operations."""
        return self._params.get('batch_size', 10000)

    def score_batch(
        self,
        peptides: Sequence[str],
        show_progress: bool = False
    ) -> np.ndarray:
        """Score peptides in batches for memory efficiency.

        Parameters
        ----------
        peptides : sequence of str
            Peptides to score.
        show_progress : bool, default=False
            If True, show progress bar (requires tqdm).

        Returns
        -------
        scores : np.ndarray
            Foreignness scores.
        """
        self._check_is_fitted()
        peptides = self._ensure_list(peptides)
        n_peptides = len(peptides)
        scores = np.zeros(n_peptides)

        # Create batch iterator
        batches = range(0, n_peptides, self.batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                batches = tqdm(batches, desc="Scoring", unit="batch")
            except ImportError:
                pass

        for i in batches:
            batch = peptides[i:i + self.batch_size]
            scores[i:i + len(batch)] = self._score_batch_impl(batch)

        return scores

    def _score_batch_impl(self, batch: List[str]) -> np.ndarray:
        """Implementation of batch scoring.

        Override this for efficient vectorized scoring.
        Default implementation calls score() on each peptide.

        Parameters
        ----------
        batch : list of str
            Batch of peptides to score.

        Returns
        -------
        scores : np.ndarray
            Scores for the batch.
        """
        return self.score(batch)
