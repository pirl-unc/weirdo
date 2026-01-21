"""Registry for scorer and reference implementations.

Provides a plugin-style registration system for scorers and references.
"""

from typing import Any, Callable, Dict, List, Optional, Type

from .base import BaseScorer
from .reference import BaseReference


class ScorerRegistry:
    """Registry for scorer and reference implementations.

    Provides decorator-based registration and factory methods
    for creating scorer instances.

    Example
    -------
    >>> from weirdo.scorers import registry, BaseScorer
    >>>
    >>> @registry.register_scorer('my_scorer', description='My custom scorer')
    ... class MyScorer(BaseScorer):
    ...     def fit(self, reference): ...
    ...     def score(self, peptides): ...
    >>>
    >>> scorer = registry.create_scorer('my_scorer', k=8)
    """

    def __init__(self):
        self._scorers: Dict[str, Dict[str, Any]] = {}
        self._references: Dict[str, Dict[str, Any]] = {}

    def register_scorer(
        self,
        name: str,
        description: str = '',
        aliases: Optional[List[str]] = None
    ) -> Callable[[Type[BaseScorer]], Type[BaseScorer]]:
        """Decorator to register a scorer class.

        Parameters
        ----------
        name : str
            Unique identifier for the scorer.
        description : str, optional
            Human-readable description.
        aliases : list of str, optional
            Alternative names for the scorer.

        Returns
        -------
        decorator : callable
            Decorator function that registers the class.

        Example
        -------
        >>> @registry.register_scorer('frequency', description='Frequency-based scoring')
        ... class FrequencyScorer(BaseScorer):
        ...     pass
        """
        aliases = aliases or []

        def decorator(cls: Type[BaseScorer]) -> Type[BaseScorer]:
            entry = {
                'class': cls,
                'name': name,
                'description': description,
                'aliases': aliases,
            }
            self._scorers[name] = entry
            for alias in aliases:
                self._scorers[alias] = entry
            return cls

        return decorator

    def register_reference(
        self,
        name: str,
        description: str = '',
        aliases: Optional[List[str]] = None
    ) -> Callable[[Type[BaseReference]], Type[BaseReference]]:
        """Decorator to register a reference class.

        Parameters
        ----------
        name : str
            Unique identifier for the reference.
        description : str, optional
            Human-readable description.
        aliases : list of str, optional
            Alternative names for the reference.

        Returns
        -------
        decorator : callable
            Decorator function that registers the class.
        """
        aliases = aliases or []

        def decorator(cls: Type[BaseReference]) -> Type[BaseReference]:
            entry = {
                'class': cls,
                'name': name,
                'description': description,
                'aliases': aliases,
            }
            self._references[name] = entry
            for alias in aliases:
                self._references[alias] = entry
            return cls

        return decorator

    def get_scorer(self, name: str) -> Type[BaseScorer]:
        """Get scorer class by name.

        Parameters
        ----------
        name : str
            Scorer name or alias.

        Returns
        -------
        scorer_class : type
            The scorer class.

        Raises
        ------
        KeyError
            If scorer name is not registered.
        """
        if name not in self._scorers:
            available = self.list_scorers()
            raise KeyError(
                f"Unknown scorer '{name}'. Available: {available}"
            )
        return self._scorers[name]['class']

    def get_reference(self, name: str) -> Type[BaseReference]:
        """Get reference class by name.

        Parameters
        ----------
        name : str
            Reference name or alias.

        Returns
        -------
        reference_class : type
            The reference class.

        Raises
        ------
        KeyError
            If reference name is not registered.
        """
        if name not in self._references:
            available = self.list_references()
            raise KeyError(
                f"Unknown reference '{name}'. Available: {available}"
            )
        return self._references[name]['class']

    def create_scorer(self, name: str, **params) -> BaseScorer:
        """Create scorer instance by name.

        Parameters
        ----------
        name : str
            Scorer name or alias.
        **params : dict
            Parameters to pass to scorer constructor.

        Returns
        -------
        scorer : BaseScorer
            Instantiated scorer.
        """
        cls = self.get_scorer(name)
        return cls(**params)

    def create_reference(self, name: str, **params) -> BaseReference:
        """Create reference instance by name.

        Parameters
        ----------
        name : str
            Reference name or alias.
        **params : dict
            Parameters to pass to reference constructor.

        Returns
        -------
        reference : BaseReference
            Instantiated reference.
        """
        cls = self.get_reference(name)
        return cls(**params)

    def list_scorers(self) -> List[str]:
        """List registered scorer names (excluding aliases).

        Returns
        -------
        names : list of str
            Registered scorer names.
        """
        seen = set()
        names = []
        for name, entry in self._scorers.items():
            canonical = entry['name']
            if canonical not in seen:
                seen.add(canonical)
                names.append(canonical)
        return sorted(names)

    def list_references(self) -> List[str]:
        """List registered reference names (excluding aliases).

        Returns
        -------
        names : list of str
            Registered reference names.
        """
        seen = set()
        names = []
        for name, entry in self._references.items():
            canonical = entry['name']
            if canonical not in seen:
                seen.add(canonical)
                names.append(canonical)
        return sorted(names)

    def get_scorer_info(self, name: str) -> Dict[str, Any]:
        """Get metadata about a scorer.

        Parameters
        ----------
        name : str
            Scorer name or alias.

        Returns
        -------
        info : dict
            Scorer metadata (name, description, aliases, class).
        """
        if name not in self._scorers:
            raise KeyError(f"Unknown scorer '{name}'")
        return self._scorers[name].copy()

    def get_reference_info(self, name: str) -> Dict[str, Any]:
        """Get metadata about a reference.

        Parameters
        ----------
        name : str
            Reference name or alias.

        Returns
        -------
        info : dict
            Reference metadata (name, description, aliases, class).
        """
        if name not in self._references:
            raise KeyError(f"Unknown reference '{name}'")
        return self._references[name].copy()


# Global registry instance
registry = ScorerRegistry()

# Convenience functions that delegate to global registry
register_scorer = registry.register_scorer
register_reference = registry.register_reference
get_scorer = registry.get_scorer
get_reference = registry.get_reference
create_scorer = registry.create_scorer
create_reference = registry.create_reference
list_scorers = registry.list_scorers
list_references = registry.list_references
