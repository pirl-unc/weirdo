"""Similarity-based foreignness scorer.

Scores peptides based on minimum distance to reference k-mers
using substitution matrices (BLOSUM, PMBEC).
"""

from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from .base import BatchScorer
from .reference import BaseReference
from .registry import register_scorer


DistanceMetric = Literal['min_distance', 'mean_distance', 'max_similarity']


def _get_matrix(name: str) -> Tuple[np.ndarray, Dict[str, int]]:
    """Load a substitution matrix by name.

    Parameters
    ----------
    name : str
        Matrix name: 'blosum30', 'blosum50', 'blosum62', or 'pmbec'.

    Returns
    -------
    matrix : np.ndarray
        The substitution matrix.
    aa_indices : dict
        Mapping from amino acid letter to matrix index.
    """
    name = name.lower()

    if name == 'blosum30':
        from ..blosum import blosum30_matrix
        matrix = blosum30_matrix
    elif name == 'blosum50':
        from ..blosum import blosum50_matrix
        matrix = blosum50_matrix
    elif name == 'blosum62':
        from ..blosum import blosum62_matrix
        matrix = blosum62_matrix
    elif name == 'pmbec':
        from ..pmbec import pmbec_matrix
        matrix = pmbec_matrix
    else:
        raise ValueError(
            f"Unknown matrix '{name}'. Available: blosum30, blosum50, blosum62, pmbec"
        )

    # Build amino acid index mapping (canonical 20 amino acids)
    from ..amino_acid_alphabet import canonical_amino_acid_letters
    aa_indices = {aa: i for i, aa in enumerate(canonical_amino_acid_letters)}

    return matrix, aa_indices


@register_scorer('similarity', description='Similarity-based scoring using substitution matrices')
class SimilarityScorer(BatchScorer):
    """Similarity-based foreignness scorer.

    Scores peptides by computing the minimum distance (or related metric)
    to reference k-mers using substitution matrices like BLOSUM or PMBEC.

    Higher scores indicate more "foreign" peptides (more distant from reference).

    Parameters
    ----------
    k : int, default=8
        K-mer size for decomposing peptides.
    matrix : str, default='blosum62'
        Substitution matrix to use: 'blosum30', 'blosum50', 'blosum62', 'pmbec'.
    distance_metric : str, default='min_distance'
        How to compute distance:
        - 'min_distance': minimum distance to any reference k-mer (default)
        - 'mean_distance': mean distance to sampled reference k-mers
        - 'max_similarity': negative of maximum similarity
    max_candidates : int, default=1000
        Maximum reference k-mers to compare against per query k-mer.
        Used for efficiency when reference is large.
    aggregate : str, default='mean'
        How to aggregate k-mer distances: 'mean', 'max', 'min', 'sum'.

    Example
    -------
    >>> ref = SwissProtReference(categories=['human']).load()
    >>> scorer = SimilarityScorer(matrix='blosum62')
    >>> scorer.fit(ref)
    >>> scores = scorer.score(['MTMDKSEL', 'XXXXXXXX'])
    """

    def __init__(
        self,
        k: int = 8,
        matrix: str = 'blosum62',
        distance_metric: DistanceMetric = 'min_distance',
        max_candidates: int = 1000,
        aggregate: str = 'mean',
        batch_size: int = 1000,
        **kwargs
    ):
        super().__init__(batch_size=batch_size, **kwargs)
        self._params.update({
            'k': k,
            'matrix': matrix,
            'distance_metric': distance_metric,
            'max_candidates': max_candidates,
            'aggregate': aggregate,
        })

        # Load matrix
        self._matrix, self._aa_indices = _get_matrix(matrix)
        self._ref_kmers: Optional[List[str]] = None

    @property
    def k(self) -> int:
        """Get k-mer size."""
        return self._params['k']

    @property
    def matrix_name(self) -> str:
        """Get substitution matrix name."""
        return self._params['matrix']

    @property
    def distance_metric(self) -> DistanceMetric:
        """Get distance metric."""
        return self._params['distance_metric']

    @property
    def max_candidates(self) -> int:
        """Get maximum candidates per k-mer."""
        return self._params['max_candidates']

    @property
    def aggregate(self) -> str:
        """Get aggregation method."""
        return self._params['aggregate']

    def fit(self, reference: BaseReference) -> 'SimilarityScorer':
        """Fit the scorer to a reference dataset.

        Parameters
        ----------
        reference : BaseReference
            Reference dataset providing k-mers to compare against.

        Returns
        -------
        self : SimilarityScorer
            Returns self for method chaining.
        """
        if not reference.is_loaded:
            raise RuntimeError(
                "Reference is not loaded. Call reference.load() first."
            )
        self._reference = reference

        # Cache reference k-mers for efficient lookup
        # For large references, we sample to max_candidates
        all_kmers = list(reference.iter_kmers())

        if len(all_kmers) <= self.max_candidates:
            self._ref_kmers = all_kmers
        else:
            # Sample reference k-mers
            import random
            self._ref_kmers = random.sample(all_kmers, self.max_candidates)

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
            Array of foreignness scores. Higher = more foreign/distant.
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
            Foreignness score (distance-based).
        """
        k = self.k
        if len(peptide) < k:
            return float('inf')

        # Extract k-mers
        kmers = [peptide[i:i+k] for i in range(len(peptide) - k + 1)]

        # Score each k-mer
        kmer_distances = []
        for kmer in kmers:
            dist = self._kmer_distance(kmer)
            kmer_distances.append(dist)

        # Aggregate distances
        kmer_distances = np.array(kmer_distances)
        return self._aggregate_scores(kmer_distances)

    def _kmer_distance(self, kmer: str) -> float:
        """Compute distance of a k-mer to reference.

        Parameters
        ----------
        kmer : str
            K-mer to score.

        Returns
        -------
        distance : float
            Distance to reference (metric depends on distance_metric setting).
        """
        if not self._ref_kmers:
            return float('inf')

        metric = self.distance_metric

        if metric == 'min_distance':
            min_dist = float('inf')
            for ref_kmer in self._ref_kmers:
                dist = self._sequence_distance(kmer, ref_kmer)
                if dist < min_dist:
                    min_dist = dist
                    if dist == 0:
                        break  # Can't get better than 0
            return min_dist

        elif metric == 'mean_distance':
            distances = [
                self._sequence_distance(kmer, ref_kmer)
                for ref_kmer in self._ref_kmers
            ]
            return np.mean(distances)

        elif metric == 'max_similarity':
            max_sim = float('-inf')
            for ref_kmer in self._ref_kmers:
                sim = self._sequence_similarity(kmer, ref_kmer)
                if sim > max_sim:
                    max_sim = sim
            # Return negative similarity (so higher = more foreign)
            return -max_sim

        else:
            raise ValueError(f"Unknown distance metric: {metric}")

    def _sequence_distance(self, seq1: str, seq2: str) -> float:
        """Compute distance between two sequences.

        Distance is computed as sum of (max_score - pairwise_score)
        for each position.

        Parameters
        ----------
        seq1, seq2 : str
            Sequences to compare (same length).

        Returns
        -------
        distance : float
            Total distance between sequences.
        """
        if len(seq1) != len(seq2):
            return float('inf')

        total_dist = 0.0
        for a, b in zip(seq1, seq2):
            idx_a = self._aa_indices.get(a)
            idx_b = self._aa_indices.get(b)

            if idx_a is None or idx_b is None:
                # Unknown amino acid - maximum penalty
                total_dist += 10.0  # Arbitrary high penalty
                continue

            # Get similarity score
            score = self._matrix[idx_a, idx_b]

            # Get max possible score (diagonal)
            max_score = self._matrix[idx_a, idx_a]

            # Distance is gap from max score
            total_dist += max_score - score

        return total_dist

    def _sequence_similarity(self, seq1: str, seq2: str) -> float:
        """Compute similarity between two sequences.

        Parameters
        ----------
        seq1, seq2 : str
            Sequences to compare (same length).

        Returns
        -------
        similarity : float
            Total similarity score.
        """
        if len(seq1) != len(seq2):
            return float('-inf')

        total_sim = 0.0
        for a, b in zip(seq1, seq2):
            idx_a = self._aa_indices.get(a)
            idx_b = self._aa_indices.get(b)

            if idx_a is None or idx_b is None:
                total_sim -= 10.0  # Penalty for unknown
                continue

            total_sim += self._matrix[idx_a, idx_b]

        return total_sim

    def _aggregate_scores(self, scores: np.ndarray) -> float:
        """Aggregate k-mer scores into a single score."""
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

    def get_closest_reference(self, kmer: str, n: int = 5) -> List[Tuple[str, float]]:
        """Find closest reference k-mers to a query k-mer.

        Useful for understanding why a k-mer has a particular score.

        Parameters
        ----------
        kmer : str
            K-mer to find matches for.
        n : int, default=5
            Number of closest matches to return.

        Returns
        -------
        matches : list of (str, float)
            List of (reference_kmer, distance) tuples, sorted by distance.
        """
        self._check_is_fitted()

        if not self._ref_kmers:
            return []

        distances = []
        for ref_kmer in self._ref_kmers:
            dist = self._sequence_distance(kmer, ref_kmer)
            distances.append((ref_kmer, dist))

        # Sort by distance and return top n
        distances.sort(key=lambda x: x[1])
        return distances[:n]
