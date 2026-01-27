"""Configuration system for scorers.

Provides dataclass-based configuration with preset support.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union


@dataclass
class ScorerConfig:
    """Configuration for creating a scorer instance.

    Supports loading from dict, YAML, or JSON, and provides
    preset configurations for common use cases.

    Attributes
    ----------
    scorer : str
        Name of the scorer to use (e.g., 'mlp').
    reference : str
        Name of the reference to use (e.g., 'swissprot').
    k : int
        K-mer size.
    scorer_params : dict
        Additional parameters for the scorer.
    reference_params : dict
        Additional parameters for the reference.
    training_params : dict
        Optional training parameters for trainable scorers.

    Example
    -------
    >>> config = ScorerConfig.from_preset('default')
    >>> scorer = config.build()
    >>> scorer.train(peptides, labels, target_categories=['human', 'viruses'])
    >>> scores = scorer.score(['MTMDKSEL'])
    """

    scorer: str = 'mlp'
    reference: str = 'swissprot'
    k: int = 8
    scorer_params: Dict[str, Any] = field(default_factory=dict)
    reference_params: Dict[str, Any] = field(default_factory=dict)
    training_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary.

        Returns
        -------
        config_dict : dict
            Configuration as a dictionary.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScorerConfig':
        """Create config from dictionary.

        Parameters
        ----------
        data : dict
            Configuration dictionary.

        Returns
        -------
        config : ScorerConfig
            Configuration instance.
        """
        return cls(
            scorer=data.get('scorer', 'mlp'),
            reference=data.get('reference', 'swissprot'),
            k=data.get('k', 8),
            scorer_params=data.get('scorer_params', {}),
            reference_params=data.get('reference_params', {}),
            training_params=data.get('training_params', {}),
        )

    @classmethod
    def from_yaml(cls, path: str) -> 'ScorerConfig':
        """Load config from YAML file.

        Parameters
        ----------
        path : str
            Path to YAML configuration file.

        Returns
        -------
        config : ScorerConfig
            Configuration instance.
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required for YAML config files")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> 'ScorerConfig':
        """Load config from JSON file.

        Parameters
        ----------
        path : str
            Path to JSON configuration file.

        Returns
        -------
        config : ScorerConfig
            Configuration instance.
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_json(self, path: str, indent: int = 2) -> None:
        """Save config to JSON file.

        Parameters
        ----------
        path : str
            Path to save JSON configuration.
        indent : int, default=2
            JSON indentation level.
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=indent)

    @classmethod
    def from_preset(cls, name: str) -> 'ScorerConfig':
        """Create config from a named preset.

        Parameters
        ----------
        name : str
            Preset name (e.g., 'default', 'fast').

        Returns
        -------
        config : ScorerConfig
            Configuration instance.

        Raises
        ------
        KeyError
            If preset name is not found.
        """
        return get_preset(name)

    def build(
        self,
        auto_load: bool = True,
        train_data: Optional[List[str]] = None,
        train_labels: Optional[Any] = None,
        target_categories: Optional[List[str]] = None,
        **train_overrides: Any
    ):
        """Build scorer from this configuration.

        Parameters
        ----------
        auto_load : bool, default=True
            If True, automatically load the reference.
        train_data : list of str, optional
            Training peptides for trainable scorers.
        train_labels : array-like, optional
            Training labels for trainable scorers.
        target_categories : list of str, optional
            Category names for multi-label training.
        **train_overrides : dict
            Overrides for training parameters.

        Returns
        -------
        scorer : BaseScorer
            Configured scorer. Trainable scorers are returned untrained
            unless train_data and train_labels are provided.
        """
        from .registry import create_scorer, create_reference
        from .trainable import TrainableScorer

        # Create and fit scorer
        scorer_params = {'k': self.k, **self.scorer_params}
        scorer = create_scorer(self.scorer, **scorer_params)
        if isinstance(scorer, TrainableScorer):
            if train_data is not None and train_labels is not None:
                train_kwargs = {**self.training_params, **train_overrides}
                scorer.train(
                    peptides=train_data,
                    labels=train_labels,
                    target_categories=target_categories,
                    **train_kwargs,
                )
            return scorer
        # Non-trainable scorers require a reference
        ref_params = {'k': self.k, **self.reference_params}
        reference = create_reference(self.reference, **ref_params)
        if auto_load:
            reference.load()

        scorer.fit(reference)
        return scorer


# Preset configurations
PRESETS: Dict[str, ScorerConfig] = {
    'default': ScorerConfig(
        scorer='mlp',
        reference='swissprot',
        k=8,
        scorer_params={
            'hidden_layer_sizes': (256, 128, 64),
            'activation': 'relu',
            'alpha': 0.0001,
            'early_stopping': True,
            'use_dipeptides': True,
        },
        reference_params={
            'categories': None,  # All categories available in SwissProt
        },
        training_params={
            'epochs': 200,
            'learning_rate': 0.001,
            'verbose': True,
        },
    ),
    'fast': ScorerConfig(
        scorer='mlp',
        reference='swissprot',
        k=8,
        scorer_params={
            'hidden_layer_sizes': (128, 64),
            'activation': 'relu',
            'alpha': 0.0001,
            'early_stopping': True,
            'use_dipeptides': False,
        },
        reference_params={
            'categories': None,
        },
        training_params={
            'epochs': 50,
            'learning_rate': 0.001,
            'verbose': True,
        },
    ),
}


def get_preset(name: str) -> ScorerConfig:
    """Get a preset configuration by name.

    Parameters
    ----------
    name : str
        Preset name.

    Returns
    -------
    config : ScorerConfig
        Configuration instance (a copy).

    Raises
    ------
    KeyError
        If preset name is not found.
    """
    if name not in PRESETS:
        available = list_presets()
        raise KeyError(
            f"Unknown preset '{name}'. Available: {available}"
        )
    # Return a copy to prevent modification of preset
    preset = PRESETS[name]
    return ScorerConfig(
        scorer=preset.scorer,
        reference=preset.reference,
        k=preset.k,
        scorer_params=preset.scorer_params.copy(),
        reference_params=preset.reference_params.copy(),
        training_params=preset.training_params.copy(),
    )


def list_presets() -> List[str]:
    """List available preset names.

    Returns
    -------
    names : list of str
        Available preset names.
    """
    return sorted(PRESETS.keys())
