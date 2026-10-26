"""Derived scores for multi-label peptide-content predictions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


MICROBIAL_CATEGORIES = (
    "viruses",
    "bacteria",
    "archaea",
    "fungi",
    "protists",
)


def summarize_peptide_content(
    probabilities: np.ndarray,
    categories: Sequence[str],
    *,
    self_categories: Sequence[str] = ("mammals",),
    microbial_categories: Sequence[str] = MICROBIAL_CATEGORIES,
) -> dict[str, np.ndarray]:
    """Combine independent category outputs without rewarding overlap.

    ``microbial_like`` and ``self_like`` are maxima rather than sums because a
    peptide can occur in several taxonomic categories at once.  The bounded
    ``microbial_foreign_like`` ranking score is then::

        microbial_like * (1 - self_like)

    The result is not a posterior probability of biological foreignness.  The
    caller controls what counts as self; for example, mammalian hosts can use
    ``("mammals",)`` while a human-specific view can use
    ``("mammals", "human")``.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("probabilities must be a one- or two-dimensional array")
    category_names = tuple(categories)
    if values.shape[1] != len(category_names):
        raise ValueError(
            "probability columns must match categories: "
            f"{values.shape[1]} != {len(category_names)}"
        )
    if len(set(category_names)) != len(category_names):
        raise ValueError("categories must be unique")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")

    index = {name: position for position, name in enumerate(category_names)}

    def selected(names: Sequence[str], label: str) -> np.ndarray:
        names = tuple(names)
        if not names:
            raise ValueError(f"{label} cannot be empty")
        missing = [name for name in names if name not in index]
        if missing:
            raise ValueError(f"Unknown {label}: {missing}")
        return values[:, [index[name] for name in names]].max(axis=1)

    microbial_like = selected(microbial_categories, "microbial_categories")
    self_like = selected(self_categories, "self_categories")
    return {
        "microbial_like": microbial_like,
        "self_like": self_like,
        "microbial_foreign_like": microbial_like * (1.0 - self_like),
    }
