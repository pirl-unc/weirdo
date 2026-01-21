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
        Name of the scorer to use (e.g., 'frequency', 'similarity').
    reference : str
        Name of the reference to use (e.g., 'swissprot').
    k : int
        K-mer size.
    scorer_params : dict
        Additional parameters for the scorer.
    reference_params : dict
        Additional parameters for the reference.

    Example
    -------
    >>> config = ScorerConfig.from_preset('default')
    >>> scorer = config.build()
    >>> scores = scorer.score(['MTMDKSEL'])
    """

    scorer: str = 'frequency'
    reference: str = 'swissprot'
    k: int = 8
    scorer_params: Dict[str, Any] = field(default_factory=dict)
    reference_params: Dict[str, Any] = field(default_factory=dict)

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
            scorer=data.get('scorer', 'frequency'),
            reference=data.get('reference', 'swissprot'),
            k=data.get('k', 8),
            scorer_params=data.get('scorer_params', {}),
            reference_params=data.get('reference_params', {}),
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
            Preset name (e.g., 'default', 'pathogen', 'similarity_blosum62').

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

    def build(self, auto_load: bool = True):
        """Build scorer from this configuration.

        Parameters
        ----------
        auto_load : bool, default=True
            If True, automatically load the reference.

        Returns
        -------
        scorer : BaseScorer
            Configured and fitted scorer.
        """
        from .registry import create_scorer, create_reference

        # Create reference
        ref_params = {'k': self.k, **self.reference_params}
        reference = create_reference(self.reference, **ref_params)
        if auto_load:
            reference.load()

        # Create and fit scorer
        scorer_params = {'k': self.k, **self.scorer_params}
        scorer = create_scorer(self.scorer, **scorer_params)
        scorer.fit(reference)

        return scorer


# Preset configurations
PRESETS: Dict[str, ScorerConfig] = {
    'default': ScorerConfig(
        scorer='frequency',
        reference='swissprot',
        k=8,
        scorer_params={
            'aggregate': 'mean',
            'pseudocount': 1e-10,
        },
        reference_params={
            'categories': None,  # All categories
        },
    ),
    'pathogen': ScorerConfig(
        scorer='frequency',
        reference='swissprot',
        k=8,
        scorer_params={
            'aggregate': 'mean',
            'pseudocount': 1e-10,
        },
        reference_params={
            'categories': ['bacteria', 'viruses'],
        },
    ),
    'human': ScorerConfig(
        scorer='frequency',
        reference='swissprot',
        k=8,
        scorer_params={
            'aggregate': 'mean',
            'pseudocount': 1e-10,
        },
        reference_params={
            'categories': ['human'],
        },
    ),
    'similarity_blosum62': ScorerConfig(
        scorer='similarity',
        reference='swissprot',
        k=8,
        scorer_params={
            'matrix': 'blosum62',
            'distance_metric': 'min_distance',
            'max_candidates': 1000,
        },
        reference_params={
            'categories': None,
        },
    ),
    'similarity_pmbec': ScorerConfig(
        scorer='similarity',
        reference='swissprot',
        k=8,
        scorer_params={
            'matrix': 'pmbec',
            'distance_metric': 'min_distance',
            'max_candidates': 1000,
        },
        reference_params={
            'categories': None,
        },
    ),
    'fast': ScorerConfig(
        scorer='frequency',
        reference='swissprot',
        k=8,
        scorer_params={
            'aggregate': 'mean',
            'pseudocount': 1e-10,
        },
        reference_params={
            'categories': ['human'],
            'use_set': True,  # No frequencies, faster lookup
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
    )


def list_presets() -> List[str]:
    """List available preset names.

    Returns
    -------
    names : list of str
        Available preset names.
    """
    return sorted(PRESETS.keys())
