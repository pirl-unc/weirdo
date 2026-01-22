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


# =============================================================================
# Model Management Functions
# =============================================================================

def list_models(model_dir: Optional[str] = None) -> List[Any]:
    """List all available trained models.

    Parameters
    ----------
    model_dir : str, optional
        Custom model directory. Defaults to ~/.weirdo/models.

    Returns
    -------
    models : list of ModelInfo
        Information about each saved model.

    Example
    -------
    >>> models = list_models()
    >>> for m in models:
    ...     print(f"{m.name}: {m.scorer_type}")
    """
    from .model_manager import list_models as _list_models
    return _list_models(model_dir)


def load_model(name: str, model_dir: Optional[str] = None) -> BaseScorer:
    """Load a trained model by name.

    Parameters
    ----------
    name : str
        Model name.
    model_dir : str, optional
        Custom model directory.

    Returns
    -------
    scorer : TrainableScorer
        Loaded model ready for scoring.

    Example
    -------
    >>> model = load_model('my-mlp')
    >>> scores = model.score(['MTMDKSEL'])
    """
    from .model_manager import load_model as _load_model
    return _load_model(name, model_dir)


def save_model(
    scorer: BaseScorer,
    name: str,
    model_dir: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Save a trained model.

    Parameters
    ----------
    scorer : TrainableScorer
        Trained model to save.
    name : str
        Name for the saved model.
    model_dir : str, optional
        Custom model directory.
    overwrite : bool, default=False
        Overwrite existing model.

    Returns
    -------
    path : str
        Path where model was saved.

    Example
    -------
    >>> scorer = MLPScorer()
    >>> scorer.train(peptides, labels)
    >>> save_model(scorer, 'my-mlp')
    """
    from .model_manager import save_model as _save_model
    return str(_save_model(scorer, name, model_dir, overwrite))


def get_available_scorers() -> List[str]:
    """Get list of available scorer types.

    Returns both lookup-based and ML-based scorers.

    Returns
    -------
    scorers : list of str
        Available scorer names.

    Example
    -------
    >>> print(get_available_scorers())
    ['frequency', 'similarity', 'mlp', 'modern-mlp']
    """
    from .scorers import list_scorers
    return list_scorers()
