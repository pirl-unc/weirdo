#!/usr/bin/env python
"""
Train and test foreignness scoring models.

This script demonstrates the complete workflow:
1. Load SwissProt reference data
2. Train FrequencyScorer and SimilarityScorer
3. Test on sample peptides
4. Compare results across models

Usage:
    python examples/train_and_test_models.py
"""

import time
import numpy as np
from typing import List, Tuple

# Import weirdo components
from weirdo.scorers import (
    SwissProtReference,
    FrequencyScorer,
    SimilarityScorer,
    ScorerConfig,
    list_presets,
)


def load_reference(categories: List[str] = None, use_set: bool = False) -> SwissProtReference:
    """Load SwissProt reference data.

    Parameters
    ----------
    categories : list of str, optional
        Filter to specific categories (e.g., ['human']).
        If None, loads all categories (slow, ~7.5GB).
    use_set : bool, default=False
        If True, only track presence (faster, less memory).

    Returns
    -------
    ref : SwissProtReference
        Loaded reference dataset.
    """
    print(f"Loading SwissProt reference...")
    print(f"  Categories: {categories or 'ALL'}")
    print(f"  Use set mode: {use_set}")

    start = time.time()
    ref = SwissProtReference(
        categories=categories,
        use_set=use_set
    ).load()
    elapsed = time.time() - start

    print(f"  Loaded {len(ref):,} k-mers in {elapsed:.1f}s")
    return ref


def train_frequency_scorer(ref: SwissProtReference, **params) -> FrequencyScorer:
    """Train a FrequencyScorer on reference data.

    Parameters
    ----------
    ref : SwissProtReference
        Reference dataset.
    **params : dict
        Scorer parameters (k, pseudocount, aggregate).

    Returns
    -------
    scorer : FrequencyScorer
        Trained scorer.
    """
    params.setdefault('k', 8)
    params.setdefault('pseudocount', 1e-10)
    params.setdefault('aggregate', 'mean')

    print(f"\nTraining FrequencyScorer...")
    print(f"  k={params['k']}, aggregate={params['aggregate']}")

    start = time.time()
    scorer = FrequencyScorer(**params).fit(ref)
    elapsed = time.time() - start

    print(f"  Trained in {elapsed:.3f}s")
    return scorer


def train_similarity_scorer(ref: SwissProtReference, **params) -> SimilarityScorer:
    """Train a SimilarityScorer on reference data.

    Parameters
    ----------
    ref : SwissProtReference
        Reference dataset.
    **params : dict
        Scorer parameters (k, matrix, max_candidates).

    Returns
    -------
    scorer : SimilarityScorer
        Trained scorer.
    """
    params.setdefault('k', 8)
    params.setdefault('matrix', 'blosum62')
    params.setdefault('max_candidates', 1000)

    print(f"\nTraining SimilarityScorer...")
    print(f"  k={params['k']}, matrix={params['matrix']}, max_candidates={params['max_candidates']}")

    start = time.time()
    scorer = SimilarityScorer(**params).fit(ref)
    elapsed = time.time() - start

    print(f"  Trained in {elapsed:.3f}s")
    return scorer


def test_scorers(
    scorers: List[Tuple[str, object]],
    peptides: List[str],
    labels: List[str] = None
) -> None:
    """Test scorers on sample peptides and compare results.

    Parameters
    ----------
    scorers : list of (name, scorer) tuples
        Trained scorers to test.
    peptides : list of str
        Peptide sequences to score.
    labels : list of str, optional
        Labels for each peptide.
    """
    labels = labels or peptides

    print("\n" + "="*70)
    print("SCORING RESULTS")
    print("="*70)

    # Header
    header = f"{'Peptide':<30}"
    for name, _ in scorers:
        header += f" {name:>15}"
    print(header)
    print("-"*70)

    # Score each peptide
    for peptide, label in zip(peptides, labels):
        row = f"{label:<30}"
        for name, scorer in scorers:
            start = time.time()
            score = scorer.score([peptide])[0]
            elapsed = time.time() - start
            row += f" {score:>15.3f}"
        print(row)

    print("-"*70)


def analyze_kmer_breakdown(scorer: FrequencyScorer, peptide: str) -> None:
    """Analyze individual k-mer contributions to score.

    Parameters
    ----------
    scorer : FrequencyScorer
        Trained frequency scorer.
    peptide : str
        Peptide to analyze.
    """
    print(f"\nK-mer breakdown for: {peptide}")
    print("-"*50)

    kmer_scores = scorer.get_kmer_scores(peptide)
    for kmer, score in kmer_scores:
        print(f"  {kmer}: {score:.3f}")

    total = scorer.score([peptide])[0]
    print(f"\n  Aggregated ({scorer.aggregate}): {total:.3f}")


def find_closest_matches(scorer: SimilarityScorer, kmer: str, n: int = 5) -> None:
    """Find closest reference k-mers.

    Parameters
    ----------
    scorer : SimilarityScorer
        Trained similarity scorer.
    kmer : str
        K-mer to find matches for.
    n : int, default=5
        Number of matches to show.
    """
    print(f"\nClosest reference k-mers to: {kmer}")
    print("-"*50)

    matches = scorer.get_closest_reference(kmer, n=n)
    for ref_kmer, distance in matches:
        print(f"  {ref_kmer}: distance={distance:.3f}")


def main():
    """Main training and testing workflow."""
    print("="*70)
    print("WEIRDO Foreignness Scorer - Training and Testing")
    print("="*70)

    # Sample peptides for testing
    test_peptides = [
        # Real human protein fragments (should score low = less foreign)
        'MTMDKSELVQKAKLAE',  # Human serine/threonine-protein kinase
        'MRLTVLCALLAALLLA',  # Human signal peptide
        'MVLSPADKTNVKAAWG',  # Human hemoglobin alpha

        # Random/synthetic sequences (should score high = more foreign)
        'XXXXXXXXYYYYYYYY',  # All unknown
        'AAAAAAAAAAAAAAAA',  # Polyalanine
        'WWWWWWWWWWWWWWWW',  # Polytryptophan (rare)

        # Mixed sequences
        'MTMDKSELXXXXXXXX',  # Half known, half unknown
        'ACDEFGHIKLMNPQRS',  # Alphabetical (synthetic)
    ]

    test_labels = [
        'Human kinase',
        'Human signal peptide',
        'Human hemoglobin',
        'All X (unknown)',
        'Polyalanine',
        'Polytryptophan',
        'Half known/unknown',
        'Alphabetical',
    ]

    # -------------------------------------------------------------------------
    # OPTION 1: Load subset (fast, for testing)
    # -------------------------------------------------------------------------
    print("\n" + "-"*70)
    print("Loading human-only reference (faster for demo)")
    print("-"*70)

    ref_human = load_reference(categories=['human'], use_set=False)

    # Train scorers
    freq_scorer = train_frequency_scorer(ref_human, aggregate='mean')
    sim_scorer = train_similarity_scorer(ref_human, matrix='blosum62', max_candidates=500)

    # Test scorers
    test_scorers(
        scorers=[
            ('Frequency', freq_scorer),
            ('Similarity', sim_scorer),
        ],
        peptides=test_peptides,
        labels=test_labels,
    )

    # -------------------------------------------------------------------------
    # Detailed analysis
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("DETAILED ANALYSIS")
    print("="*70)

    # Analyze k-mer breakdown for a human peptide vs random
    analyze_kmer_breakdown(freq_scorer, 'MTMDKSELVQKAKLAE')
    analyze_kmer_breakdown(freq_scorer, 'XXXXXXXXYYYYYYYY')

    # Find closest matches for similarity scorer
    find_closest_matches(sim_scorer, 'MTMDKSEL')
    find_closest_matches(sim_scorer, 'XXXXXXXX')

    # -------------------------------------------------------------------------
    # Compare aggregation methods
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("COMPARING AGGREGATION METHODS")
    print("="*70)

    freq_mean = train_frequency_scorer(ref_human, aggregate='mean')
    freq_max = train_frequency_scorer(ref_human, aggregate='max')
    freq_min = train_frequency_scorer(ref_human, aggregate='min')

    test_scorers(
        scorers=[
            ('Freq-Mean', freq_mean),
            ('Freq-Max', freq_max),
            ('Freq-Min', freq_min),
        ],
        peptides=test_peptides[:4],
        labels=test_labels[:4],
    )

    # -------------------------------------------------------------------------
    # Compare substitution matrices
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("COMPARING SUBSTITUTION MATRICES")
    print("="*70)

    sim_blosum62 = train_similarity_scorer(ref_human, matrix='blosum62', max_candidates=500)
    sim_blosum50 = train_similarity_scorer(ref_human, matrix='blosum50', max_candidates=500)
    sim_pmbec = train_similarity_scorer(ref_human, matrix='pmbec', max_candidates=500)

    test_scorers(
        scorers=[
            ('BLOSUM62', sim_blosum62),
            ('BLOSUM50', sim_blosum50),
            ('PMBEC', sim_pmbec),
        ],
        peptides=test_peptides[:4],
        labels=test_labels[:4],
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Key observations:
1. Known human peptides score LOWER (less foreign)
2. Random/synthetic sequences score HIGHER (more foreign)
3. FrequencyScorer is fast and effective for presence-based scoring
4. SimilarityScorer considers amino acid substitutability
5. Different aggregation methods emphasize different aspects:
   - mean: overall foreignness
   - max: worst (most foreign) k-mer
   - min: best (least foreign) k-mer
""")

    print("\nAvailable presets:", list_presets())
    print("\nDone!")


if __name__ == '__main__':
    main()
