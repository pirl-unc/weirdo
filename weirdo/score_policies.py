"""Versioned, inspectable definitions for category-derived scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np


MICROBIAL_CONTENT_CATEGORIES = (
    "viruses",
    "bacteria",
    "archaea",
    "fungi",
    "protists",
)
DEFAULT_SCORE_POLICY = "human-self-complement-v1"
DEFAULT_PEPTIDE_CONTENT_SCORE_POLICY = "mammal-microbial-content-v1"


@dataclass(frozen=True)
class ScorePolicy:
    """Complete versioned definition of a scalar category-derived score.

    A policy describes interpretation and mechanics; it does not make a score
    an immune-response probability. Category outputs remain independent and a
    peptide may be high in both the source and self groups.
    """

    policy_id: str
    model_id: str
    host_taxon_id: int
    host_name: str
    source_categories: tuple[str, ...]
    self_categories: tuple[str, ...]
    category_combiner: str
    formula: str
    output_name: str
    exact_self_override: bool
    overlap_handling: str
    direction: str
    interpretation: str
    content_validation: str
    immune_response_validation: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_categories", tuple(self.source_categories))
        object.__setattr__(self, "self_categories", tuple(self.self_categories))
        if self.schema_version != 1:
            raise ValueError("Unsupported score-policy schema version")
        if not self.policy_id or not self.model_id or not self.output_name:
            raise ValueError("Policy, model, and output names cannot be empty")
        if (
            isinstance(self.host_taxon_id, bool)
            or not isinstance(self.host_taxon_id, int)
            or self.host_taxon_id < 1
        ):
            raise ValueError("host_taxon_id must be a positive integer")
        if self.category_combiner != "max":
            raise ValueError("Version 1 score policies support only the max combiner")
        if self.formula not in {"one_minus_self", "source_times_one_minus_self"}:
            raise ValueError(f"Unsupported score formula: {self.formula!r}")
        if not self.self_categories or len(set(self.self_categories)) != len(self.self_categories):
            raise ValueError("self_categories must be non-empty and unique")
        if len(set(self.source_categories)) != len(self.source_categories):
            raise ValueError("source_categories must be unique")
        if self.formula == "source_times_one_minus_self" and not self.source_categories:
            raise ValueError("source_times_one_minus_self requires source categories")
        if set(self.source_categories) & set(self.self_categories):
            raise ValueError("source and self categories must not overlap by name")

    def to_dict(self) -> dict:
        """Return a JSON-compatible definition for provenance and caching."""
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "model_id": self.model_id,
            "host_taxon_id": self.host_taxon_id,
            "host_name": self.host_name,
            "source_categories": list(self.source_categories),
            "self_categories": list(self.self_categories),
            "category_combiner": self.category_combiner,
            "formula": self.formula,
            "output_name": self.output_name,
            "exact_self_override": self.exact_self_override,
            "overlap_handling": self.overlap_handling,
            "direction": self.direction,
            "interpretation": self.interpretation,
            "content_validation": self.content_validation,
            "immune_response_validation": self.immune_response_validation,
        }


_INDEPENDENT_OVERLAP = (
    "Independent multi-label category outputs may overlap. Max selects the "
    "strongest source and self heads; self does not erase a source label."
)
_CONTENT_VALIDATION = (
    "Held-out protein-family taxonomy/content evaluation is recorded with the model."
)
_NO_IMMUNE_VALIDATION = (
    "No independently confirmed incremental T-cell-response benefit beyond MHC baselines."
)


def _content_policy(
    policy_id: str,
    host_taxon_id: int,
    host_name: str,
    self_categories: tuple[str, ...],
) -> ScorePolicy:
    return ScorePolicy(
        policy_id=policy_id,
        model_id="peptide-content-v1",
        host_taxon_id=host_taxon_id,
        host_name=host_name,
        source_categories=MICROBIAL_CONTENT_CATEGORIES,
        self_categories=self_categories,
        category_combiner="max",
        formula="source_times_one_minus_self",
        output_name="microbial_foreign_like",
        exact_self_override=True,
        overlap_handling=(
            f"{_INDEPENDENT_OVERLAP} An exact selected-host proteome hit "
            "overrides the learned score to zero when supplied."
        ),
        direction=(
            "Higher means more microbial-like and less self-like under the model; "
            "the bounded 0--1 value is a ranking score, not a biological probability."
        ),
        interpretation=(
            "Descriptive sequence-content contrast under the balanced training distribution."
        ),
        content_validation=_CONTENT_VALIDATION,
        immune_response_validation=_NO_IMMUNE_VALIDATION,
    )


_POLICIES = {
    DEFAULT_SCORE_POLICY: ScorePolicy(
        policy_id=DEFAULT_SCORE_POLICY,
        model_id="swissprot-categories-v1",
        host_taxon_id=9606,
        host_name="Homo sapiens",
        source_categories=(),
        self_categories=("human",),
        category_combiner="max",
        formula="one_minus_self",
        output_name="foreignness",
        exact_self_override=False,
        overlap_handling=(
            "The score uses only the independent human head; other category "
            "memberships are returned separately and do not change it."
        ),
        direction=(
            "Higher means less human-like under the model; the bounded 0--1 "
            "value is not a biological or immune-response probability."
        ),
        interpretation="Complement of the SwissProt human-category output.",
        content_validation=(
            "Human-category discrimination and calibration were measured on held-out "
            "protein-mnemonic families."
        ),
        immune_response_validation=_NO_IMMUNE_VALIDATION,
    ),
    DEFAULT_PEPTIDE_CONTENT_SCORE_POLICY: _content_policy(
        DEFAULT_PEPTIDE_CONTENT_SCORE_POLICY,
        40674,
        "Mammalia (generic clade; not species-specific)",
        ("mammals",),
    ),
    "human-microbial-content-v1": _content_policy(
        "human-microbial-content-v1", 9606, "Homo sapiens", ("mammals", "human")
    ),
    "mouse-microbial-content-v1": _content_policy(
        "mouse-microbial-content-v1", 10090, "Mus musculus", ("mammals", "mouse")
    ),
    "dog-microbial-content-v1": _content_policy(
        "dog-microbial-content-v1", 9615, "Canis lupus familiaris", ("mammals", "dog")
    ),
}


def get_score_policy(policy: Union[str, ScorePolicy]) -> ScorePolicy:
    """Resolve a built-in policy name or validate a supplied policy object."""
    if isinstance(policy, ScorePolicy):
        return policy
    if not isinstance(policy, str) or policy not in _POLICIES:
        available = ", ".join(sorted(_POLICIES))
        raise ValueError(f"Unknown score policy {policy!r}; available: {available}")
    return _POLICIES[policy]


def list_score_policies(model_id: Optional[str] = None) -> list[dict]:
    """Return JSON-compatible definitions, optionally filtered by model ID."""
    policies = sorted(_POLICIES.values(), key=lambda item: item.policy_id)
    if model_id is not None:
        policies = [policy for policy in policies if policy.model_id == model_id]
    return [policy.to_dict() for policy in policies]


def apply_score_policy(
    probabilities: np.ndarray,
    categories: Sequence[str],
    policy: Union[str, ScorePolicy],
    *,
    exact_self_hits: Optional[Sequence[bool]] = None,
) -> dict[str, np.ndarray]:
    """Apply a policy to independent category probabilities."""
    resolved = get_score_policy(policy)
    values = np.asarray(probabilities, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    category_names = tuple(categories)
    if values.ndim != 2 or values.shape[1] != len(category_names):
        raise ValueError("probability columns must match the one-dimensional categories")
    if len(set(category_names)) != len(category_names):
        raise ValueError("categories must be unique")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")

    index = {name: position for position, name in enumerate(category_names)}

    def maximum(names: Sequence[str], label: str) -> np.ndarray:
        missing = [name for name in names if name not in index]
        if missing:
            raise ValueError(f"Policy {resolved.policy_id!r} needs missing {label}: {missing}")
        return values[:, [index[name] for name in names]].max(axis=1)

    self_like = maximum(resolved.self_categories, "self categories")
    result = {"self_like": self_like}
    if resolved.formula == "one_minus_self":
        score = 1.0 - self_like
    else:
        source_like = maximum(resolved.source_categories, "source categories")
        result["source_like"] = source_like
        score = source_like * (1.0 - self_like)

    if exact_self_hits is not None:
        if not resolved.exact_self_override:
            raise ValueError(
                f"Policy {resolved.policy_id!r} does not define an exact-self override"
            )
        exact = np.asarray(list(exact_self_hits), dtype=bool)
        if exact.ndim != 1 or len(exact) != len(values):
            raise ValueError("exact_self_hits must have one value per peptide")
        self_like = self_like.copy()
        score = score.copy()
        self_like[exact] = 1.0
        score[exact] = 0.0
        result["self_like"] = self_like
        result["exact_self_hit"] = exact

    result["score"] = score
    return result
