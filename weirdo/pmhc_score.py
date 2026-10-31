"""Rank-normalized pMHC score with an optional Weirdo foreignness term.

MHCflurry presentation and NetMHCpan BA rank live on very different scales, so a
plain geometric mean is effectively presentation alone. Here each is converted to
a rank against a frozen, label-free reference (``pmhc-rank-reference-v1.json``)
and the two ranks are averaged into a core score in (0, 1]. A small additive term
from the shipped ``peptide-content-v1-ensemble`` ``virus_foreign_like`` head
breaks ties toward more foreign-looking peptides:

    score = core + weirdo_weight * (rank(virus_foreign_like) - 0.5)

This is a fixed combination with no fitted response coefficients. On the pooled
IMPROVE, viral, TESLA and Braun benchmark the two terms together improved macro
AUC over the core by about 0.0065, positive in every family; that gain is small
and the weights were fixed after exploratory screens on the same data (see
``reports/auto-weirdo-v1/analysis.md``). It is not an immunogenicity predictor.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

REFERENCE_PATH = Path(__file__).resolve().parent / "pretrained" / "pmhc-rank-reference-v1.json"
WEIRDO_MODEL = "peptide-content-v1-ensemble"
WEIRDO_HEAD = "virus_foreign_like"
DEFAULT_WEIRDO_WEIGHT = 0.1
DEFAULT_FACE_WEIGHT = 0.1


@lru_cache(maxsize=1)
def _reference() -> dict:
    with open(REFERENCE_PATH, "r") as handle:
        return json.load(handle)


def _rank(values: Sequence[float], key: str) -> np.ndarray:
    """Rank in [0, 1] of ``values`` against the frozen reference table ``key``."""
    table = _reference()[key]
    array = np.asarray(list(values), dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{key} values must be finite")
    return np.interp(array, table["values"], table["probs"])


@lru_cache(maxsize=1)
def _ensemble():
    from .peptide_content_models import load_peptide_content_model

    return load_peptide_content_model(Path(__file__).resolve().parent / "pretrained" / WEIRDO_MODEL)


def _weirdo_head_scores(peptides: Sequence[str]) -> np.ndarray:
    """``virus_foreign_like`` of the packaged ensemble (8-15mers; needs torch)."""
    from .peptide_content_models import standard_peptide_content_scores

    invalid = sorted({len(p) for p in peptides if not 8 <= len(p) <= 15})
    if invalid:
        raise ValueError(f"Weirdo scores need peptides of length 8-15; received lengths {invalid}")
    model = _ensemble()
    probabilities = np.asarray(model.predict_proba(list(peptides)), dtype=float)
    scores = standard_peptide_content_scores(
        probabilities, model.target_categories, required_scores=[WEIRDO_HEAD]
    )
    return np.asarray(scores[WEIRDO_HEAD], dtype=float)


def face_surprise(peptide: str) -> float:
    """Mean -log human residue frequency over the TCR-facing positions P3..P(L-1)."""
    face = peptide[2:-1]
    if len(peptide) < 4 or not face:
        raise ValueError(f"Peptide too short for a TCR-facing window: {peptide!r}")
    table = _reference()["residue_surprise"]
    try:
        return float(np.mean([table[residue] for residue in face]))
    except KeyError as error:
        raise ValueError(f"Non-standard residue {error} in {peptide!r}") from error


def pmhc_weirdo_score(
    peptides: Sequence[str],
    mhcflurry_presentation_scores: Sequence[float],
    netmhcpan_ba_ranks: Sequence[float],
    weirdo_weight: float = DEFAULT_WEIRDO_WEIGHT,
    weirdo_scores: Optional[Sequence[float]] = None,
    face_weight: float = DEFAULT_FACE_WEIGHT,
) -> pd.DataFrame:
    """Combine MHCflurry presentation, NetMHCpan BA rank and Weirdo foreignness.

    Parameters
    ----------
    peptides
        Peptide sequences (8-15 residues when Weirdo is used).
    mhcflurry_presentation_scores
        MHCflurry 2.3 presentation probabilities, one per peptide.
    netmhcpan_ba_ranks
        NetMHCpan BA percentile ranks (lower is better), one per peptide.
    weirdo_weight
        Weight on the centred Weirdo rank. ``0`` gives the pMHC core only and
        needs no torch dependency.
    weirdo_scores
        Optional precomputed ``virus_foreign_like`` values; if omitted and
        ``weirdo_weight > 0`` they are computed with the bundled ensemble
        (requires ``weirdo[peptide-content]``).

    face_weight
        Weight on the centred self-relative TCR-face surprise rank (no model or
        extra dependency). ``0`` disables it.

    Returns
    -------
    pandas.DataFrame
        ``peptide``, ``presentation_rank``, ``binding_rank``, ``pmhc_core``,
        ``weirdo_rank`` and ``face_rank`` (NaN when unused) and the final ``score``.
    """
    peptides = list(peptides)
    presentation = np.asarray(list(mhcflurry_presentation_scores), dtype=float)
    ba_rank = np.asarray(list(netmhcpan_ba_ranks), dtype=float)
    if presentation.shape != (len(peptides),) or ba_rank.shape != (len(peptides),):
        raise ValueError("Presentation scores and BA ranks must have one value per peptide")
    if not (0.0 <= weirdo_weight <= 1.0 and 0.0 <= face_weight <= 1.0):
        raise ValueError("weirdo_weight and face_weight must be between 0 and 1")
    presentation_rank = _rank(presentation, "presentation")
    binding_rank = _rank(-ba_rank, "negative_ba_rank")
    core = (presentation_rank + binding_rank) / 2.0
    weirdo_rank = np.full(len(peptides), np.nan)
    score = core.copy()
    if weirdo_weight > 0:
        if weirdo_scores is None:
            weirdo_scores = _weirdo_head_scores(peptides)
        weirdo_rank = _rank(weirdo_scores, "weirdo")
        score = core + weirdo_weight * (weirdo_rank - 0.5)
    face_rank = np.full(len(peptides), np.nan)
    if face_weight > 0:
        face_rank = _rank([face_surprise(peptide) for peptide in peptides], "face_surprise")
        score = score + face_weight * (face_rank - 0.5)
    return pd.DataFrame(
        {
            "peptide": peptides,
            "presentation_rank": presentation_rank,
            "binding_rank": binding_rank,
            "pmhc_core": core,
            "weirdo_rank": weirdo_rank,
            "face_rank": face_rank,
            "score": score,
        }
    )
