"""Base classes for reference datasets.

Provides abstract base classes for loading and querying
reference k-mer data (e.g., from SwissProt).
"""

from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union


class BaseReference(ABC):
    """Abstract base class for reference datasets.

    A reference dataset provides k-mer frequency information
    used by scorers to compute foreignness scores.

    Example
    -------
    >>> ref = MyReference(categories=['human'])
    >>> ref.load()
    >>> ref.contains('MTMDKSEL')
    True
    >>> ref.get_frequency('MTMDKSEL')
    0.00123
    """

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        k: int = 8,
        **kwargs
    ):
        """Initialize reference.

        Parameters
        ----------
        categories : list of str, optional
            Filter to specific organism categories.
            If None, use all available categories.
        k : int, default=8
            K-mer size expected in this reference.
        **kwargs : dict
            Additional implementation-specific parameters.
        """
        self._categories = categories
        self._k = k
        self._is_loaded = False
        self._kwargs = kwargs

    @abstractmethod
    def load(self) -> 'BaseReference':
        """Load reference data into memory.

        Returns
        -------
        self : BaseReference
            Returns self for method chaining.
        """
        pass

    @abstractmethod
    def contains(self, kmer: str) -> bool:
        """Check if k-mer exists in reference.

        Parameters
        ----------
        kmer : str
            K-mer sequence to look up.

        Returns
        -------
        exists : bool
            True if k-mer is in reference.
        """
        pass

    @abstractmethod
    def get_frequency(self, kmer: str, default: float = 0.0) -> float:
        """Get frequency of k-mer in reference.

        Parameters
        ----------
        kmer : str
            K-mer sequence to look up.
        default : float, default=0.0
            Value to return if k-mer not found.

        Returns
        -------
        frequency : float
            Frequency of k-mer (0.0 to 1.0) or default if not found.
        """
        pass

    @abstractmethod
    def get_categories(self) -> List[str]:
        """Get list of available organism categories.

        Returns
        -------
        categories : list of str
            Available category names (e.g., ['human', 'bacteria']).
        """
        pass

    @abstractmethod
    def iter_kmers(self) -> Iterator[str]:
        """Iterate over all k-mers in reference.

        Yields
        ------
        kmer : str
            Each k-mer sequence in the reference.
        """
        pass

    @property
    def k(self) -> int:
        """Get k-mer size."""
        return self._k

    @property
    def categories(self) -> Optional[List[str]]:
        """Get filtered categories, or None for all."""
        return self._categories

    @property
    def is_loaded(self) -> bool:
        """Check if reference data is loaded."""
        return self._is_loaded

    def _check_is_loaded(self) -> None:
        """Raise error if reference is not loaded."""
        if not self._is_loaded:
            raise RuntimeError(
                f"{self.__class__.__name__} is not loaded. "
                "Call load() first."
            )

    def __len__(self) -> int:
        """Return number of k-mers in reference."""
        raise NotImplementedError("Subclass must implement __len__")

    def __contains__(self, kmer: str) -> bool:
        """Support 'in' operator."""
        return self.contains(kmer)


class StreamingReference(BaseReference):
    """Base class for references that support streaming operations.

    Extends BaseReference with methods for memory-efficient
    iteration over large datasets.
    """

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        k: int = 8,
        lazy: bool = False,
        use_set: bool = False,
        **kwargs
    ):
        """Initialize streaming reference.

        Parameters
        ----------
        categories : list of str, optional
            Filter to specific organism categories.
        k : int, default=8
            K-mer size.
        lazy : bool, default=False
            If True, don't load data into memory; stream from disk.
        use_set : bool, default=False
            If True, only track k-mer presence (not frequencies).
            Reduces memory by ~50% but loses frequency information.
        **kwargs : dict
            Additional parameters.
        """
        super().__init__(categories=categories, k=k, **kwargs)
        self._lazy = lazy
        self._use_set = use_set

    @property
    def lazy(self) -> bool:
        """Check if using lazy (streaming) mode."""
        return self._lazy

    @property
    def use_set(self) -> bool:
        """Check if only tracking presence (no frequencies)."""
        return self._use_set

    @abstractmethod
    def iter_kmers_with_counts(self) -> Iterator[Tuple[str, int]]:
        """Iterate over k-mers with their counts.

        Yields
        ------
        kmer : str
            K-mer sequence.
        count : int
            Number of times k-mer appears in reference.
        """
        pass

    @abstractmethod
    def iter_kmers_with_categories(
        self
    ) -> Iterator[Tuple[str, Dict[str, bool]]]:
        """Iterate over k-mers with category presence.

        Yields
        ------
        kmer : str
            K-mer sequence.
        categories : dict
            Mapping of category name to presence (True/False).
        """
        pass

    def sample_kmers(
        self,
        n: int,
        seed: Optional[int] = None
    ) -> List[str]:
        """Sample random k-mers from reference.

        Parameters
        ----------
        n : int
            Number of k-mers to sample.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        kmers : list of str
            Sampled k-mer sequences.
        """
        import random
        if seed is not None:
            random.seed(seed)

        # Default implementation: collect and sample
        # Subclasses can override for memory-efficient sampling
        all_kmers = list(self.iter_kmers())
        return random.sample(all_kmers, min(n, len(all_kmers)))

    def get_kmers_for_category(self, category: str) -> Iterator[str]:
        """Get k-mers present in a specific category.

        Parameters
        ----------
        category : str
            Category name to filter by.

        Yields
        ------
        kmer : str
            K-mers present in the specified category.
        """
        for kmer, cats in self.iter_kmers_with_categories():
            if cats.get(category, False):
                yield kmer
