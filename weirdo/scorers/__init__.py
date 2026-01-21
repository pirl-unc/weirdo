"""Extensible foreignness scoring system.

This module provides a plugin-style architecture for scoring peptides
based on how "foreign" they are relative to a reference dataset.

Quick Start
-----------
>>> from weirdo.scorers import FrequencyScorer, SwissProtReference
>>> ref = SwissProtReference(categories=['human']).load()
>>> scorer = FrequencyScorer(k=8).fit(ref)
>>> scores = scorer.score(['MTMDKSEL', 'ACDEFGHI'])

Using Presets
-------------
>>> from weirdo.scorers import ScorerConfig
>>> config = ScorerConfig.from_preset('default')
>>> scorer = config.build()
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

# Concrete implementations (import to trigger registration)
from .swissprot import SwissProtReference
from .frequency import FrequencyScorer
from .similarity import SimilarityScorer

__all__ = [
    # Base classes
    'BaseScorer',
    'BatchScorer',
    'BaseReference',
    'StreamingReference',
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
    'FrequencyScorer',
    'SimilarityScorer',
]
