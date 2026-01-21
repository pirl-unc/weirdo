"""High-level convenience API for foreignness scoring.

Provides simple functions for common use cases without
needing to understand the full scorer architecture.

Example
-------
>>> from weirdo import score_peptide
>>> score = score_peptide('MTMDKSEL')  # Uses default preset

>>> from weirdo import score_peptides
>>> scores = score_peptides(['MTMDKSEL', 'ACDEFGHI'])
"""

from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from .scorers import ScorerConfig, BaseScorer


# Cache for scorer instances by preset
_scorer_cache: Dict[str, BaseScorer] = {}


def create_scorer(
    preset: str = 'default',
    cache: bool = True,
    auto_download: bool = False,
    **overrides
) -> BaseScorer:
    """Create a scorer from a preset configuration.

    Parameters
    ----------
    preset : str, default='default'
        Preset name (e.g., 'default', 'pathogen', 'human', 'similarity_blosum62').
    cache : bool, default=True
        If True, cache the scorer instance for reuse.
        Set to False if you need multiple independent instances.
    auto_download : bool, default=False
        If True, automatically download reference data if not present.
    **overrides : dict
        Override specific config parameters (e.g., k=10, aggregate='max').

    Returns
    -------
    scorer : BaseScorer
        Configured and fitted scorer ready for scoring.

    Example
    -------
    >>> scorer = create_scorer('human')
    >>> scores = scorer.score(['MTMDKSEL'])

    >>> scorer = create_scorer('default', k=10, aggregate='max')

    >>> # Auto-download data on first use
    >>> scorer = create_scorer('default', auto_download=True)
    """
    # Build cache key from preset and overrides
    cache_key = f"{preset}:{sorted(overrides.items())}:auto={auto_download}"

    if cache and cache_key in _scorer_cache:
        return _scorer_cache[cache_key]

    # Get preset config
    config = ScorerConfig.from_preset(preset)

    # Apply overrides
    if overrides:
        # Check which params go to scorer vs reference
        scorer_params = {'k', 'aggregate', 'pseudocount', 'matrix', 'distance_metric', 'max_candidates'}
        reference_params = {'categories', 'lazy', 'use_set', 'data_path'}

        for key, value in overrides.items():
            if key == 'k':
                config.k = value
            elif key == 'scorer':
                config.scorer = value
            elif key == 'reference':
                config.reference = value
            elif key in scorer_params:
                config.scorer_params[key] = value
            elif key in reference_params:
                config.reference_params[key] = value
            else:
                # Assume it's a scorer param
                config.scorer_params[key] = value

    # Add auto_download to reference params
    if auto_download:
        config.reference_params['auto_download'] = True

    # Build scorer
    scorer = config.build()

    if cache:
        _scorer_cache[cache_key] = scorer

    return scorer


def score_peptide(
    peptide: str,
    preset: str = 'default',
    auto_download: bool = False,
    **kwargs
) -> float:
    """Score a single peptide for foreignness.

    Parameters
    ----------
    peptide : str
        Peptide sequence to score.
    preset : str, default='default'
        Scoring preset to use.
    auto_download : bool, default=False
        If True, automatically download reference data if not present.
    **kwargs : dict
        Additional arguments passed to create_scorer().

    Returns
    -------
    score : float
        Foreignness score. Higher = more foreign.

    Example
    -------
    >>> score = score_peptide('MTMDKSEL')
    >>> print(f"Foreignness: {score:.2f}")

    >>> # Auto-download data on first use
    >>> score = score_peptide('MTMDKSEL', auto_download=True)
    """
    scorer = create_scorer(preset, auto_download=auto_download, **kwargs)
    scores = scorer.score([peptide])
    return float(scores[0])


def score_peptides(
    peptides: Sequence[str],
    preset: str = 'default',
    auto_download: bool = False,
    **kwargs
) -> np.ndarray:
    """Score multiple peptides for foreignness.

    Parameters
    ----------
    peptides : sequence of str
        Peptide sequences to score.
    preset : str, default='default'
        Scoring preset to use.
    auto_download : bool, default=False
        If True, automatically download reference data if not present.
    **kwargs : dict
        Additional arguments passed to create_scorer().

    Returns
    -------
    scores : np.ndarray
        Array of foreignness scores. Higher = more foreign.

    Example
    -------
    >>> scores = score_peptides(['MTMDKSEL', 'ACDEFGHI', 'XXXXXXXX'])
    >>> for pep, score in zip(peptides, scores):
    ...     print(f"{pep}: {score:.2f}")

    >>> # Auto-download data on first use
    >>> scores = score_peptides(['MTMDKSEL'], auto_download=True)
    """
    scorer = create_scorer(preset, auto_download=auto_download, **kwargs)
    return scorer.score(peptides)


def clear_cache() -> None:
    """Clear the scorer cache.

    Use this to free memory or reset state.
    """
    _scorer_cache.clear()


def get_available_presets() -> List[str]:
    """Get list of available preset names.

    Returns
    -------
    presets : list of str
        Available preset names.
    """
    from .scorers import list_presets
    return list_presets()


def get_preset_info(preset: str) -> Dict[str, Any]:
    """Get information about a preset configuration.

    Parameters
    ----------
    preset : str
        Preset name.

    Returns
    -------
    info : dict
        Preset configuration details.
    """
    config = ScorerConfig.from_preset(preset)
    return config.to_dict()
