"""High-level convenience API for foreignness scoring.

Provides simple functions for common use cases without
needing to understand the full scorer architecture.

Example
-------
>>> from weirdo import score_peptide, load_model
>>> scorer = load_model('my-mlp')
>>> score = score_peptide('MTMDKSEL', model=scorer)

>>> from weirdo import score_peptides
>>> scores = score_peptides(['MTMDKSEL', 'ACDEFGHI'], model=scorer)
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from .scorers import ScorerConfig, BaseScorer, TrainableScorer
from .peptide_content import MICROBIAL_CATEGORIES, summarize_peptide_content


# Cache for scorer instances by preset
_scorer_cache: Dict[str, BaseScorer] = {}


def create_scorer(
    preset: str = 'default',
    cache: bool = True,
    auto_download: bool = False,
    train_data: Optional[Sequence[str]] = None,
    train_labels: Optional[Any] = None,
    target_categories: Optional[List[str]] = None,
    **overrides
) -> BaseScorer:
    """Create a scorer from a preset configuration.

    Parameters
    ----------
    preset : str, default='default'
        Preset name (e.g., 'default', 'fast').
    cache : bool, default=True
        If True, cache the scorer instance for reuse.
        Set to False if you need multiple independent instances.
    auto_download : bool, default=False
        If True, automatically download reference data if not present.
    train_data : sequence of str, optional
        Training peptides for trainable scorers.
    train_labels : array-like, optional
        Training labels for trainable scorers.
    target_categories : list of str, optional
        Category names for multi-label training.
    **overrides : dict
        Override specific config parameters (e.g., k=10, hidden_layer_sizes=(128, 64)).

    Returns
    -------
    scorer : BaseScorer
        Configured scorer. Trainable scorers are returned untrained unless
        train_data and train_labels are provided.

    Example
    -------
    >>> scorer = create_scorer('default', use_dipeptides=False)
    >>> scorer.train(peptides, labels, target_categories=['human', 'viruses'])

    >>> # Auto-download data on first use
    >>> scorer = create_scorer('default', auto_download=True)
    """
    # Build cache key from preset and overrides
    cache_key = f"{preset}:{sorted(overrides.items())}:auto={auto_download}"

    if cache and cache_key in _scorer_cache and train_data is None and train_labels is None:
        return _scorer_cache[cache_key]

    # Get preset config
    config = ScorerConfig.from_preset(preset)

    # Apply overrides
    if overrides:
        # Check which params go to scorer vs reference
        scorer_params = {
            'hidden_layer_sizes',
            'activation',
            'alpha',
            'learning_rate_init',
            'max_iter',
            'early_stopping',
            'use_dipeptides',
            'batch_size',
            'random_state',
        }
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

    # Build scorer (trainable scorers are returned untrained unless training data provided)
    scorer = config.build(
        train_data=list(train_data) if train_data is not None else None,
        train_labels=train_labels,
        target_categories=target_categories,
    )

    if cache and train_data is None and train_labels is None:
        _scorer_cache[cache_key] = scorer

    return scorer


def score_peptide(
    peptide: str,
    model: Optional[Union[str, BaseScorer]] = None,
    model_dir: Optional[str] = None,
    preset: Optional[str] = None,
    aggregate: str = 'mean',
    **kwargs
) -> float:
    """Score a single peptide.

    Parameters
    ----------
    peptide : str
        Peptide sequence to score.
    model : str or BaseScorer, optional
        Model name (from ModelManager) or an instantiated scorer.
    model_dir : str, optional
        Custom model directory when loading by name.
    preset : str, optional
        Scoring preset used to construct a scorer.
        For trainable presets, provide training data via kwargs
        or pass a trained model.
    aggregate : str, default='mean'
        How to aggregate k-mer probabilities for long peptides.
    **kwargs : dict
        Additional arguments passed to create_scorer().

    Returns
    -------
    score : float
        Foreignness score. Higher = more foreign.

    Example
    -------
    >>> scorer = load_model('my-mlp')
    >>> score = score_peptide('MTMDKSEL', model=scorer)
    """
    if model is None:
        if preset is None:
            raise ValueError("Provide a trained model or a preset (with training data if trainable).")
        scorer = create_scorer(preset, **kwargs)
    elif isinstance(model, str):
        scorer = load_model(model, model_dir)
    else:
        scorer = model

    if isinstance(scorer, TrainableScorer) and not scorer.is_trained:
        raise RuntimeError(
            "Scorer is not trained. Train it (e.g. via create_scorer(train_data=..., "
            "train_labels=...)) or load a trained model before scoring."
        )

    try:
        scores = scorer.score([peptide], aggregate=aggregate)
    except TypeError:
        scores = scorer.score([peptide])
    return float(scores[0])


def score_peptides(
    peptides: Sequence[str],
    model: Optional[Union[str, BaseScorer]] = None,
    model_dir: Optional[str] = None,
    preset: Optional[str] = None,
    aggregate: str = 'mean',
    **kwargs
) -> np.ndarray:
    """Score multiple peptides.

    Parameters
    ----------
    peptides : sequence of str
        Peptide sequences to score.
    model : str or BaseScorer, optional
        Model name (from ModelManager) or an instantiated scorer.
    model_dir : str, optional
        Custom model directory when loading by name.
    preset : str, optional
        Scoring preset used to construct a scorer.
        For trainable presets, provide training data via kwargs
        or pass a trained model.
    aggregate : str, default='mean'
        How to aggregate k-mer probabilities for long peptides.
    **kwargs : dict
        Additional arguments passed to create_scorer().

    Returns
    -------
    scores : np.ndarray
        Array of foreignness scores. Higher = more foreign.

    Example
    -------
    >>> scorer = load_model('my-mlp')
    >>> scores = score_peptides(['MTMDKSEL'], model=scorer)
    """
    if model is None:
        if preset is None:
            raise ValueError("Provide a trained model or a preset (with training data if trainable).")
        scorer = create_scorer(preset, **kwargs)
    elif isinstance(model, str):
        scorer = load_model(model, model_dir)
    else:
        scorer = model

    if isinstance(scorer, TrainableScorer) and not scorer.is_trained:
        raise RuntimeError(
            "Scorer is not trained. Train it (e.g. via create_scorer(train_data=..., "
            "train_labels=...)) or load a trained model before scoring."
        )

    try:
        return scorer.score(peptides, aggregate=aggregate)
    except TypeError:
        return scorer.score(peptides)


def predict_categories(
    peptides: Union[str, Sequence[str]],
    model: Optional[Union[str, BaseScorer]] = None,
    model_dir: Optional[str] = None,
    pathogen_categories: Optional[List[str]] = None,
    self_categories: Optional[List[str]] = None,
    aggregate: str = 'mean',
    combine: Union[str, Callable[[Any], Any]] = 'max',
):
    """Probability that each peptide came from each species category.

    Runs a multi-label predictor with one independent sigmoid output per
    category, by default the ``'swissprot-categories'`` model that ships with
    WEIRDO (archaea, bacteria, fungi, human, invertebrates, mammals, plants,
    rodents, vertebrates, viruses). Column ``c`` estimates the probability that
    the peptide occurs in SwissProt proteins of category ``c``. Columns are not
    mutually exclusive and need not sum to one.

    The result also contains a probability for the pathogen group and one for the
    self group and their ratio, ``foreignness = pathogen / (pathogen + self)``; by
    default each group value is the largest probability in the group (``combine``
    chooses ``'sum'`` or your own function); see
    :meth:`weirdo.scorers.MLPScorer.foreignness`.

    Parameters
    ----------
    peptides : str or sequence of str
        Peptide sequence(s), 8 residues or longer.
    model : str or MLPScorer, optional
        Model name (bundled or from the model directory) or a trained scorer.
        Defaults to the bundled ``'swissprot-categories'`` model.
    model_dir : str, optional
        Custom model directory when loading by name.
    pathogen_categories : list of str, optional
        Defaults to ``['bacteria', 'viruses']``.
    self_categories : list of str, optional
        Defaults to ``['human', 'rodents', 'mammals']``.
    aggregate : {'mean', 'max', 'min'}, default='mean'
        How to combine the 8-mer windows of longer peptides.
    combine : {'max', 'sum'} or callable, default='max'
        How to combine probabilities within each group: the largest, their sum (the
        expected number of categories that contain the peptide), or a function that
        maps an (n_peptides, n_categories_in_group) array to one value per peptide.

    Returns
    -------
    pandas.DataFrame
        One row per peptide: ``peptide``, one probability column per category,
        ``pathogen_<combine>``, ``self_<combine>`` and ``foreignness``.

    Example
    -------
    >>> from weirdo import predict_categories
    >>> predict_categories(['SIINFEKL', 'NLVPMVATV']).round(2)  # doctest: +SKIP
    """
    from .model_manager import DEFAULT_CATEGORY_MODEL

    if model is None:
        model = DEFAULT_CATEGORY_MODEL
    scorer = load_model(model, model_dir) if isinstance(model, str) else model

    if isinstance(scorer, TrainableScorer) and not scorer.is_trained:
        raise RuntimeError(
            "Scorer is not trained. Train it or load a trained model before predicting."
        )
    if not getattr(scorer, 'target_categories', None):
        raise ValueError(
            "predict_categories needs a model trained with target_categories "
            "(one output per category)."
        )

    if isinstance(peptides, str):
        peptides = [peptides]
    return scorer.predict_dataframe(
        list(peptides),
        pathogen_categories=pathogen_categories,
        self_categories=self_categories,
        aggregate=aggregate,
        combine=combine,
    )


def predict_peptide_content(
    peptides: Union[str, Sequence[str]],
    model: Optional[Union[str, BaseScorer]] = None,
    model_dir: Optional[str] = None,
    self_categories: Sequence[str] = ("mammals",),
    microbial_categories: Sequence[str] = MICROBIAL_CATEGORIES,
    exact_self_hits: Optional[Sequence[bool]] = None,
):
    """Score complete 8--15mers with independent taxonomy outputs.

    The returned ``microbial_foreign_like`` value is
    ``max(microbial categories) * (1 - max(self categories))``.  It is a
    bounded ranking score, not a posterior probability of foreignness.

    ``exact_self_hits`` may be supplied by a caller that has searched a
    pinned host reference proteome. Those exact matches are marked explicitly
    and override the learned summary (``self_like=1`` and
    ``microbial_foreign_like=0``); this API does not silently choose a host
    proteome or infer exact matches from model probabilities.
    """
    import pandas as pd

    from .model_manager import DEFAULT_PEPTIDE_CONTENT_MODEL

    if isinstance(peptides, str):
        peptides = [peptides]
    peptides = list(peptides)
    invalid = [peptide for peptide in peptides if not 8 <= len(peptide) <= 15]
    if invalid:
        raise ValueError(
            "predict_peptide_content supports complete peptides of length 8--15; "
            f"received lengths {sorted({len(peptide) for peptide in invalid})}"
        )

    if model is None:
        model = DEFAULT_PEPTIDE_CONTENT_MODEL
    scorer = load_model(model, model_dir) if isinstance(model, str) else model
    if isinstance(scorer, TrainableScorer) and not scorer.is_trained:
        raise RuntimeError(
            "Scorer is not trained. Train it or load a trained model before predicting."
        )
    categories = getattr(scorer, 'target_categories', None)
    if not categories:
        raise ValueError(
            "predict_peptide_content needs a model trained with target_categories."
        )

    probabilities = np.asarray(scorer.predict_proba(peptides), dtype=float)
    summary = summarize_peptide_content(
        probabilities,
        categories,
        self_categories=self_categories,
        microbial_categories=microbial_categories,
    )
    frame = pd.DataFrame(probabilities, columns=list(categories))
    frame.insert(0, "peptide", peptides)
    frame["microbial_like"] = summary["microbial_like"]
    frame["self_like"] = summary["self_like"]
    frame["microbial_foreign_like"] = summary["microbial_foreign_like"]
    if exact_self_hits is not None:
        exact = np.asarray(list(exact_self_hits), dtype=bool)
        if len(exact) != len(peptides):
            raise ValueError("exact_self_hits must have one value per peptide")
        frame["exact_self_hit"] = exact
        frame.loc[exact, "self_like"] = 1.0
        frame.loc[exact, "microbial_foreign_like"] = 0.0
    return frame


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
    scorer: TrainableScorer,
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


def list_bundled_models() -> List[Dict[str, Any]]:
    """List the models that ship inside the package (name, path, metadata)."""
    from .model_manager import list_bundled_models as _list_bundled_models
    return _list_bundled_models()


def list_pretrained_models(model_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """List configured downloadable pretrained model descriptors."""
    from .model_manager import list_pretrained_models as _list_pretrained_models
    return _list_pretrained_models(model_dir)


def download_pretrained_model(
    name: str,
    model_dir: Optional[str] = None,
    overwrite: bool = False,
) -> str:
    """Download and install a pretrained model from the registry."""
    from .model_manager import download_pretrained_model as _download_pretrained_model
    return str(_download_pretrained_model(name, model_dir, overwrite))


def download_model_from_url(
    name: str,
    url: str,
    model_dir: Optional[str] = None,
    overwrite: bool = False,
    expected_sha256: Optional[str] = None,
) -> str:
    """Download and install a model archive from a direct URL."""
    from .model_manager import download_model_from_url as _download_model_from_url
    return str(_download_model_from_url(name, url, model_dir, overwrite, expected_sha256))


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
    ['mlp', 'similarity']
    """
    from .scorers import list_scorers
    return list_scorers()
