"""SwissProt reference dataset implementation.

Loads pre-computed k-mer data from SwissProt protein database.
"""

import csv
import os
from typing import Dict, Iterator, List, Optional, Set, Tuple

from .reference import StreamingReference
from .registry import register_reference


def _get_default_data_path(auto_download: bool = False) -> str:
    """Get default data path, using data manager if available.

    First checks for local repo data, then falls back to managed data.
    """
    # Check local repo path first (for development)
    local_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'data',
        'swissprot-8mers.csv'
    )
    if os.path.exists(local_path):
        return local_path

    # Use data manager for installed package
    try:
        from ..data_manager import get_data_manager
        dm = get_data_manager(auto_download=auto_download, verbose=auto_download)
        return str(dm.get_data_path('swissprot-8mers', auto_download=auto_download))
    except (ImportError, FileNotFoundError):
        # Fall back to local path (will fail with helpful error if not found)
        return local_path


# Legacy constant for backwards compatibility
DEFAULT_DATA_PATH = _get_default_data_path()

# All available categories in the CSV
ALL_CATEGORIES = [
    'archaea', 'bacteria', 'fungi', 'human', 'invertebrates',
    'mammals', 'plants', 'rodents', 'vertebrates', 'viruses'
]


@register_reference('swissprot', description='SwissProt k-mer reference data')
class SwissProtReference(StreamingReference):
    """Reference dataset from SwissProt protein database.

    Loads pre-computed k-mer presence from a CSV file containing
    organism category membership for each k-mer.

    Parameters
    ----------
    categories : list of str, optional
        Filter to specific organism categories.
        Available: archaea, bacteria, fungi, human, invertebrates,
        mammals, plants, rodents, vertebrates, viruses.
        If None, consider k-mer present if in ANY category.
    k : int, default=8
        K-mer size (must match data file).
    data_path : str, optional
        Path to k-mer CSV file. Defaults to bundled data.
    lazy : bool, default=False
        If True, don't load data into memory; stream from disk.
    use_set : bool, default=False
        If True, only track k-mer presence (not category counts).
        Reduces memory but loses category breakdown.
    auto_download : bool, default=False
        If True, automatically download reference data if not present.

    Example
    -------
    >>> ref = SwissProtReference(categories=['human', 'mammals'])
    >>> ref.load()
    >>> ref.contains('MTMDKSEL')
    True
    >>> ref.get_frequency('MTMDKSEL')
    1.0  # Present in filtered categories
    """

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        k: int = 8,
        data_path: Optional[str] = None,
        lazy: bool = False,
        use_set: bool = False,
        auto_download: bool = False,
        **kwargs
    ):
        super().__init__(categories=categories, k=k, lazy=lazy, use_set=use_set, **kwargs)
        self._auto_download = auto_download
        if data_path:
            self._data_path = data_path
        else:
            self._data_path = _get_default_data_path(auto_download=auto_download)
        self._kmers: Dict[str, Dict[str, bool]] = {}  # kmer -> {category: bool}
        self._kmer_set: Set[str] = set()  # For use_set mode
        self._total_kmers = 0

        # Validate categories
        if categories is not None:
            invalid = set(categories) - set(ALL_CATEGORIES)
            if invalid:
                raise ValueError(
                    f"Invalid categories: {invalid}. "
                    f"Available: {ALL_CATEGORIES}"
                )

    @property
    def data_path(self) -> str:
        """Get path to data file."""
        return self._data_path

    def load(self) -> 'SwissProtReference':
        """Load k-mer data from CSV file.

        Returns
        -------
        self : SwissProtReference
            Returns self for method chaining.
        """
        if self._lazy:
            # In lazy mode, just verify file exists
            if not os.path.exists(self._data_path):
                raise FileNotFoundError(f"Data file not found: {self._data_path}")
            self._is_loaded = True
            return self

        self._kmers.clear()
        self._kmer_set.clear()
        self._total_kmers = 0

        with open(self._data_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kmer = row['seq']

                # Check if k-mer is present in any of the filtered categories
                if self._categories is not None:
                    present = any(
                        row.get(cat, 'False') == 'True'
                        for cat in self._categories
                    )
                    if not present:
                        continue
                else:
                    # No filter - include if present in any category
                    present = any(
                        row.get(cat, 'False') == 'True'
                        for cat in ALL_CATEGORIES
                    )
                    if not present:
                        continue

                if self._use_set:
                    self._kmer_set.add(kmer)
                else:
                    self._kmers[kmer] = {
                        cat: row.get(cat, 'False') == 'True'
                        for cat in ALL_CATEGORIES
                    }
                self._total_kmers += 1

        self._is_loaded = True
        return self

    def contains(self, kmer: str) -> bool:
        """Check if k-mer exists in reference.

        Parameters
        ----------
        kmer : str
            K-mer sequence to look up.

        Returns
        -------
        exists : bool
            True if k-mer is in reference (after category filtering).
        """
        if self._lazy:
            # Stream through file to check
            return self._lazy_contains(kmer)

        if self._use_set:
            return kmer in self._kmer_set
        return kmer in self._kmers

    def _lazy_contains(self, kmer: str) -> bool:
        """Check containment by streaming file (lazy mode)."""
        with open(self._data_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['seq'] == kmer:
                    if self._categories is not None:
                        return any(
                            row.get(cat, 'False') == 'True'
                            for cat in self._categories
                        )
                    return True
        return False

    def get_frequency(self, kmer: str, default: float = 0.0) -> float:
        """Get frequency of k-mer in reference.

        For SwissProt data, returns 1.0 if present, 0.0 if not.
        True frequency computation requires count data.

        Parameters
        ----------
        kmer : str
            K-mer sequence to look up.
        default : float, default=0.0
            Value to return if k-mer not found.

        Returns
        -------
        frequency : float
            1.0 if present, default if not found.
        """
        if self.contains(kmer):
            return 1.0
        return default

    def get_categories(self) -> List[str]:
        """Get list of available organism categories.

        Returns
        -------
        categories : list of str
            Available category names.
        """
        if self._categories is not None:
            return list(self._categories)
        return ALL_CATEGORIES.copy()

    def get_kmer_categories(self, kmer: str) -> Dict[str, bool]:
        """Get category presence for a specific k-mer.

        Parameters
        ----------
        kmer : str
            K-mer sequence to look up.

        Returns
        -------
        categories : dict
            Mapping of category name to presence (True/False).
            Empty dict if k-mer not found.
        """
        self._check_is_loaded()

        if self._use_set:
            raise RuntimeError(
                "Category information not available in use_set mode. "
                "Set use_set=False to access category data."
            )

        if self._lazy:
            return self._lazy_get_categories(kmer)

        return self._kmers.get(kmer, {})

    def _lazy_get_categories(self, kmer: str) -> Dict[str, bool]:
        """Get categories by streaming file (lazy mode)."""
        with open(self._data_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['seq'] == kmer:
                    return {
                        cat: row.get(cat, 'False') == 'True'
                        for cat in ALL_CATEGORIES
                    }
        return {}

    def iter_kmers(self) -> Iterator[str]:
        """Iterate over all k-mers in reference.

        Yields
        ------
        kmer : str
            Each k-mer sequence in the reference.
        """
        self._check_is_loaded()

        if self._lazy:
            yield from self._lazy_iter_kmers()
            return

        if self._use_set:
            yield from self._kmer_set
        else:
            yield from self._kmers.keys()

    def _lazy_iter_kmers(self) -> Iterator[str]:
        """Iterate k-mers by streaming file (lazy mode)."""
        with open(self._data_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kmer = row['seq']
                if self._categories is not None:
                    present = any(
                        row.get(cat, 'False') == 'True'
                        for cat in self._categories
                    )
                    if present:
                        yield kmer
                else:
                    yield kmer

    def iter_kmers_with_counts(self) -> Iterator[Tuple[str, int]]:
        """Iterate over k-mers with their category counts.

        Yields
        ------
        kmer : str
            K-mer sequence.
        count : int
            Number of categories the k-mer appears in.
        """
        self._check_is_loaded()

        if self._lazy:
            yield from self._lazy_iter_with_counts()
            return

        if self._use_set:
            for kmer in self._kmer_set:
                yield kmer, 1
        else:
            for kmer, cats in self._kmers.items():
                count = sum(1 for v in cats.values() if v)
                yield kmer, count

    def _lazy_iter_with_counts(self) -> Iterator[Tuple[str, int]]:
        """Iterate with counts by streaming file (lazy mode)."""
        with open(self._data_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kmer = row['seq']
                try:
                    count = int(row.get('label_count', 0))
                except (ValueError, TypeError):
                    count = 0

                if self._categories is not None:
                    present = any(
                        row.get(cat, 'False') == 'True'
                        for cat in self._categories
                    )
                    if present:
                        yield kmer, count
                else:
                    yield kmer, count

    def iter_kmers_with_categories(self) -> Iterator[Tuple[str, Dict[str, bool]]]:
        """Iterate over k-mers with category presence.

        Yields
        ------
        kmer : str
            K-mer sequence.
        categories : dict
            Mapping of category name to presence (True/False).
        """
        self._check_is_loaded()

        if self._use_set:
            raise RuntimeError(
                "Category information not available in use_set mode."
            )

        if self._lazy:
            yield from self._lazy_iter_with_categories()
            return

        for kmer, cats in self._kmers.items():
            yield kmer, cats.copy()

    def _lazy_iter_with_categories(self) -> Iterator[Tuple[str, Dict[str, bool]]]:
        """Iterate with categories by streaming file (lazy mode)."""
        with open(self._data_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                kmer = row['seq']
                cats = {
                    cat: row.get(cat, 'False') == 'True'
                    for cat in ALL_CATEGORIES
                }

                if self._categories is not None:
                    present = any(cats.get(cat, False) for cat in self._categories)
                    if present:
                        yield kmer, cats
                else:
                    yield kmer, cats

    def __len__(self) -> int:
        """Return number of k-mers in reference."""
        self._check_is_loaded()

        if self._lazy:
            # Count by streaming
            return sum(1 for _ in self.iter_kmers())

        if self._use_set:
            return len(self._kmer_set)
        return len(self._kmers)

    def get_training_data(
        self,
        target_categories: Optional[List[str]] = None,
        max_samples: Optional[int] = None,
        multi_label: bool = False,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[List[str], 'np.ndarray']:
        """Generate training data for MLP scorer.

        Parameters
        ----------
        target_categories : list of str, optional
            Categories to use as targets. Default: ['human', 'viruses', 'bacteria'].
        max_samples : int, optional
            Maximum number of samples to return (for memory efficiency).
            If 0 or negative, use all available samples.
        multi_label : bool, default=False
            If True, return multi-label array (one column per category).
            If False, return single foreignness score (1 if not in first target category).
        shuffle : bool, default=False
            If True and max_samples is set, randomly sample k-mers using
            reservoir sampling instead of taking the first N.
        seed : int, optional
            Random seed for reproducible sampling.

        Returns
        -------
        peptides : list of str
            K-mer sequences.
        labels : np.ndarray
            If multi_label=True: shape (n_samples, n_categories) with binary labels.
            If multi_label=False: shape (n_samples,) with foreignness scores.

        Example
        -------
        >>> ref = SwissProtReference().load()
        >>> peptides, labels = ref.get_training_data(
        ...     target_categories=['human', 'viruses', 'bacteria'],
        ...     multi_label=True,
        ...     max_samples=100000
        ... )
        >>> print(labels.shape)  # (100000, 3)
        """
        import numpy as np

        if target_categories is None:
            target_categories = ['human', 'viruses', 'bacteria']

        if max_samples is not None and max_samples <= 0:
            max_samples = None

        # Validate categories
        for cat in target_categories:
            if cat not in ALL_CATEGORIES:
                raise ValueError(
                    f"Unknown category: {cat}. "
                    f"Available: {ALL_CATEGORIES}"
                )

        self._check_is_loaded()

        if self._use_set:
            raise RuntimeError(
                "Training data requires category info. Set use_set=False."
            )

        if shuffle or seed is not None:
            import random
            if seed is not None:
                random.seed(seed)

        peptides: List[str] = []
        labels_list: List[List[float]] = []

        def make_row(cats: Dict[str, bool]) -> List[float]:
            if multi_label:
                return [1.0 if cats.get(cat, False) else 0.0 for cat in target_categories]
            in_self = cats.get(target_categories[0], False)
            return [0.0 if in_self else 1.0]

        for idx, (kmer, cats) in enumerate(self.iter_kmers_with_categories()):
            if max_samples and shuffle:
                if len(peptides) < max_samples:
                    peptides.append(kmer)
                    labels_list.append(make_row(cats))
                else:
                    j = random.randint(0, idx)
                    if j < max_samples:
                        peptides[j] = kmer
                        labels_list[j] = make_row(cats)
            else:
                peptides.append(kmer)
                labels_list.append(make_row(cats))
                if max_samples and len(peptides) >= max_samples:
                    break

        labels = np.array(labels_list, dtype=np.float32)
        if not multi_label:
            labels = labels.ravel()

        return peptides, labels
