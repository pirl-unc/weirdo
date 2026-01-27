"""Extensible foreignness scoring system.

This module provides a plugin-style architecture for scoring peptides
based on how "foreign" they are relative to a reference dataset.

Quick Start
-----------
>>> from weirdo.scorers import MLPScorer
>>> scorer = MLPScorer(k=8, hidden_layer_sizes=(128, 64))
>>> scorer.train(peptides, labels, target_categories=['human', 'viruses'])
>>> scores = scorer.score(['MTMDKSEL', 'ACDEFGHI'])

Using Presets
-------------
>>> from weirdo.scorers import ScorerConfig
>>> config = ScorerConfig.from_preset('default')
>>> scorer = config.build()
>>> scorer.train(peptides, labels, target_categories=['human', 'viruses'])
>>> scores = scorer.score(['MTMDKSEL'])

Adding Custom Scorers
---------------------
>>> from weirdo.scorers import register_scorer, BaseScorer
>>>
>>> @register_scorer('my_scorer', description='My custom scorer')
... class MyScorer(BaseScorer):
...     def fit(self, reference): ...
...     def score(self, peptides): ...
"""

# Base classes
from .base import BaseScorer, BatchScorer

# Reference classes
from .reference import BaseReference, StreamingReference

# Registry
from .registry import (
    ScorerRegistry,
    registry,
    register_scorer,
    register_reference,
    get_scorer,
    get_reference,
    create_scorer,
    create_reference,
    list_scorers,
    list_references,
)

# Configuration
from .config import (
    ScorerConfig,
    PRESETS,
    get_preset,
    list_presets,
)

# Trainable base class
from .trainable import TrainableScorer

# Concrete implementations (import to trigger registration)
from .swissprot import SwissProtReference

# ML-based scorer
from .mlp import MLPScorer

__all__ = [
    # Base classes
    'BaseScorer',
    'BatchScorer',
    'BaseReference',
    'StreamingReference',
    'TrainableScorer',
    # Registry
    'ScorerRegistry',
    'registry',
    'register_scorer',
    'register_reference',
    'get_scorer',
    'get_reference',
    'create_scorer',
    'create_reference',
    'list_scorers',
    'list_references',
    # Configuration
    'ScorerConfig',
    'PRESETS',
    'get_preset',
    'list_presets',
    # Implementations
    'SwissProtReference',
    # ML scorer
    'MLPScorer',
]
