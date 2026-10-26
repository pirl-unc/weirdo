#!/usr/bin/env python3
"""Build matched pMHC response strata and evaluate a count-aware baseline.

This module is intentionally independent of the foreignness model.  It makes
the unit of analysis explicit before any neural model is fit:

* strata are ``(HLA, length, presentation bin, processing bin, cohort,
  exposure, source family)``;
* positive and negative assay counts are retained as binomial observations;
* source-family and TCR-face components are available as separate strict
  holdout axes (with a connected-component audit group); and
* a spline logistic (GAM-like) baseline is fit on aggregate assay counts.

The command-line entry point also reserves a deterministic, label-blind
lockbox from development data.  It is not used to choose features.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer


DEFAULT_COUNT_COLUMNS = (
    "n_positive_assays",
    "n_negative_assays",
    "n_assay_rows",
    "n_subjects_tested",
)
DEFAULT_PRESENTATION = "mhcflurry_presentation_score"
DEFAULT_PROCESSING = "mhcflurry_processing_score"


def _text(series: pd.Series, fallback: str = "unknown") -> pd.Series:
    """Normalize identifiers without collapsing missing values into NaN."""
    result = series.astype("string").fillna("").str.replace(r"\s+", " ", regex=True).str.strip()
    return result.mask(result.eq(""), fallback)


def _first_column(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _identifier(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    column = _first_column(frame, names)
    if column is None:
        return pd.Series("unknown", index=frame.index, dtype="string")
    return _text(frame[column])


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").fillna(default).clip(lower=0.0)


def _stable_id(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _peptide_face(peptide: str) -> str:
    # For canonical class-I ligands, P3..PΩ-1 is a useful first TCR-facing
    # approximation.  Short/ambiguous peptides are kept intact rather than
    # being silently discarded.
    return peptide[2:-1] if len(peptide) > 3 else peptide


class _UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}
        self.rank = {value: 0 for value in self.parent}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _face_components(
    peptides: pd.Series,
    faces: pd.Series,
    k: int = 4,
    min_shared_kmers: int = 2,
) -> pd.Series:
    """Assign deterministic connected components to TCR-facing sequences.

    Two faces are connected when they share at least ``min_shared_kmers``
    k-mers (or the entire face when it is shorter than ``k``).  Requiring two
    k-mers avoids the pathological giant component caused by common single
    amino-acid motifs.  Components are calculated over every table passed to
    the function, so a validation face cannot accidentally be treated as novel
    after a split.
    """
    if k < 1 or min_shared_kmers < 1:
        raise ValueError("k and min_shared_kmers must be positive")
    peptide_values = peptides.astype(str).tolist()
    face_values = faces.astype(str).tolist()
    uf = _UnionFind(f"p:{peptide}" for peptide in sorted(set(peptide_values)))
    index: dict[str, set[str]] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    for peptide, face in zip(peptide_values, face_values):
        tokens = {face} if len(face) < k else {face[i : i + k] for i in range(len(face) - k + 1)}
        for token in tokens:
            previous = index.setdefault(token, set())
            # Common short motifs can have many members.  Counting pairs is
            # still bounded by the observed posting list and avoids a dense
            # all-to-all graph from one shared motif.
            for other in previous:
                pair = tuple(sorted((f"p:{peptide}", other)))
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            previous.add(f"p:{peptide}")
    for (left, right), shared in pair_counts.items():
        if shared >= min_shared_kmers:
            uf.union(left, right)
    members: dict[str, list[str]] = {}
    for peptide in sorted(set(peptide_values)):
        members.setdefault(uf.find(f"p:{peptide}"), []).append(peptide)
    names = {}
    for root, values in members.items():
        names[root] = f"face-{_stable_id(min(values))}"
    return pd.Series(
        [names[uf.find(f"p:{peptide}")] for peptide in peptide_values],
        index=peptides.index,
        dtype="string",
    )


def _connected_holdout_groups(source_family: pd.Series, face_component: pd.Series) -> pd.Series:
    """Union source families and face components so neither can cross folds."""
    labels = [
        *(f"family:{value}" for value in sorted(set(source_family.astype(str)))),
        *(f"face:{value}" for value in sorted(set(face_component.astype(str)))),
    ]
    uf = _UnionFind(labels)
    for family, face in zip(source_family.astype(str), face_component.astype(str)):
        uf.union(f"family:{family}", f"face:{face}")
    roots = {label: uf.find(label) for label in labels}
    return pd.Series(
        [f"holdout-{_stable_id(roots[f'family:{family}'])}" for family in source_family.astype(str)],
        index=source_family.index,
        dtype="string",
    )


def _fit_edges(values: pd.Series, n_bins: int) -> np.ndarray:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if len(finite) == 0:
        return np.asarray([-np.inf, np.inf])
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    inner = np.unique(np.quantile(finite, quantiles[1:-1]))
    return np.concatenate(([-np.inf], inner, [np.inf]))


def _apply_edges(values: pd.Series, edges: np.ndarray) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
    result = np.full(len(values), -1, dtype=int)
    finite = np.isfinite(numeric)
    result[finite] = np.searchsorted(edges[1:-1], numeric[finite], side="right")
    return pd.Series(result, index=values.index, dtype=int)


def prepare_table(
    frame: pd.DataFrame,
    *,
    presentation_edges: np.ndarray | None = None,
    processing_edges: np.ndarray | None = None,
    face_k: int = 4,
    min_shared_face_kmers: int = 2,
    n_bins: int = 5,
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """Normalize one or more assay tables and derive leakage-safe features."""
    result = frame.copy().reset_index(drop=True)
    if "peptide" not in result or "hla" not in result:
        raise ValueError("input requires peptide and hla columns")
    result["peptide"] = _text(result["peptide"], fallback="")
    result["hla"] = _text(result["hla"])
    result["cohort"] = _identifier(result, ("cohort", "study", "pmid"))
    result["assay_method"] = _identifier(result, ("assay_method", "assay", "assay_type"))
    result["assay_context"] = result[["assay_method", "cohort"]].astype(str).agg("::".join, axis=1)
    result["exposure"] = _identifier(result, ("exposure", "culture_condition"))
    species = _identifier(result, ("source_species", "source_organism"))
    protein = _identifier(result, ("source_protein", "epitope_reference_name"))
    # source_family is deliberately the organism/species family.  The protein
    # family is retained separately for neoepitope stratification and audits.
    result["source_family"] = species
    result["source_protein_family"] = protein
    result["peptide_length"] = result["peptide"].str.len().astype(int)
    result["tcr_face"] = result["peptide"].map(_peptide_face)

    result["presentation_score"] = pd.to_numeric(
        result.get(DEFAULT_PRESENTATION, pd.Series(np.nan, index=result.index)), errors="coerce"
    )
    result["processing_score"] = pd.to_numeric(
        result.get(DEFAULT_PROCESSING, pd.Series(np.nan, index=result.index)), errors="coerce"
    )
    if presentation_edges is None:
        presentation_edges = _fit_edges(result["presentation_score"], n_bins)
    if processing_edges is None:
        processing_edges = _fit_edges(result["processing_score"], n_bins)
    result["presentation_bin"] = _apply_edges(result["presentation_score"], presentation_edges)
    result["processing_bin"] = _apply_edges(result["processing_score"], processing_edges)

    positive = _numeric(result, "n_positive_assays")
    negative = _numeric(result, "n_negative_assays")
    assay_rows = _numeric(result, "n_assay_rows")
    missing_counts = (positive + negative) <= 0
    # Validation tables often intentionally contain only a binary outcome.  A
    # single tested assay is an explicit fallback, never an invented donor
    # count; provenance is retained in count_source.
    binary_positive = _text(result.get("outcome", pd.Series("", index=result.index))).str.casefold().eq("positive")
    positive = positive.where(~missing_counts, binary_positive.astype(float))
    negative = negative.where(~missing_counts, (~binary_positive).astype(float))
    trials = positive + negative
    trials = trials.where(trials > 0, assay_rows.where(assay_rows > 0, 1.0))
    result["n_positive_assays"] = positive.astype(float)
    result["n_negative_assays"] = negative.astype(float)
    result["n_assay_rows"] = assay_rows.where(assay_rows > 0, trials).astype(float)
    result["n_subjects_tested"] = _numeric(result, "n_subjects_tested")
    result["assay_trials"] = trials.astype(float)
    result["response_rate"] = (positive / trials).fillna(0.0)
    result["count_source"] = np.where(missing_counts, "binary_outcome_fallback", "reported_assays")

    result["face_component"] = _face_components(
        result["peptide"], result["tcr_face"], k=face_k, min_shared_kmers=min_shared_face_kmers
    )
    result["holdout_group"] = _connected_holdout_groups(result["source_family"], result["face_component"])
    fields = [
        "hla",
        "peptide_length",
        "presentation_bin",
        "processing_bin",
        "assay_context",
        "exposure",
        "source_family",
    ]
    result["stratum_key"] = result[fields].astype(str).agg("|".join, axis=1)
    result["stratum_size"] = result.groupby("stratum_key")["stratum_key"].transform("size").astype(int)
    result["row_id"] = np.arange(len(result), dtype=int)
    return result, {
        "presentation_edges": [float(value) for value in presentation_edges],
        "processing_edges": [float(value) for value in processing_edges],
    }


def matched_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Return all within-stratum positive/negative row pairs with count weights."""
    rows = []
    for stratum, group in frame.groupby("stratum_key", sort=True):
        positive = group.loc[group["n_positive_assays"] > 0]
        negative = group.loc[group["n_negative_assays"] > 0]
        for _, left in positive.iterrows():
            for _, right in negative.iterrows():
                if left["row_id"] == right["row_id"]:
                    continue
                rows.append(
                    {
                        "stratum_key": stratum,
                        "positive_row_id": int(left["row_id"]),
                        "negative_row_id": int(right["row_id"]),
                        "positive_peptide": str(left["peptide"]),
                        "negative_peptide": str(right["peptide"]),
                        "hla": str(left["hla"]),
                        "pair_weight": float(left["n_positive_assays"] * right["n_negative_assays"]),
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "stratum_key",
            "positive_row_id",
            "negative_row_id",
            "positive_peptide",
            "negative_peptide",
            "hla",
            "pair_weight",
        ],
    )


def wt_mutant_pairs(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Annotate explicit WT/mutant pairs; never infer WT from nearest self."""
    column = _first_column(frame, ("wild_type_peptide", "wt_peptide", "parent_peptide", "wild_type"))
    if column is None:
        return pd.DataFrame(columns=["pair_key", "wt_row_id", "mutant_row_id"]), {
            "available": False,
            "reason": "no explicit wild-type peptide column",
            "n_pairs": 0,
        }
    rows = frame.copy()
    rows["wild_type_peptide"] = _text(rows[column], fallback="")
    rows["wt_mutant_status"] = np.where(
        rows["peptide"].astype(str).eq(rows["wild_type_peptide"].astype(str)), "wild_type", "mutant"
    )
    rows["wt_mutant_pair_key"] = rows[["hla", "wild_type_peptide", "cohort", "exposure"]].astype(str).agg("|".join, axis=1)
    pairs = []
    for key, group in rows.groupby("wt_mutant_pair_key"):
        wt = group.loc[group["wt_mutant_status"].eq("wild_type"), "row_id"]
        mutant = group.loc[group["wt_mutant_status"].eq("mutant"), "row_id"]
        for wt_id in wt:
            for mutant_id in mutant:
                pairs.append({"pair_key": key, "wt_row_id": int(wt_id), "mutant_row_id": int(mutant_id)})
    return pd.DataFrame(pairs, columns=["pair_key", "wt_row_id", "mutant_row_id"]), {
        "available": True,
        "column": column,
        "n_pairs": len(pairs),
    }


def load_explicit_wt_mutant_pairs(path: str | Path) -> pd.DataFrame:
    """Load a separate mutation table without fabricating assay outcomes."""
    pairs = pd.read_csv(path)
    required = {"wild_type_peptide", "mutant_peptide", "hla"}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"WT/mutant table is missing columns: {sorted(missing)}")
    for column in ("wild_type_peptide", "mutant_peptide", "hla"):
        pairs[column] = _text(pairs[column], fallback="")
    for column in ("cohort", "exposure"):
        if column not in pairs:
            pairs[column] = "unknown"
        pairs[column] = _text(pairs[column])
    pairs["pair_key"] = pairs[["hla", "wild_type_peptide", "cohort", "exposure"]].astype(str).agg("|".join, axis=1)
    pairs["wild_type_length"] = pairs["wild_type_peptide"].str.len().astype(int)
    pairs["mutant_length"] = pairs["mutant_peptide"].str.len().astype(int)
    return pairs


def load_iedb_wt_mutant_pairs(raw_dir: str | Path, assay_ids: set[int] | None = None) -> pd.DataFrame:
    """Extract explicit WT/mutant sequences from cached IEDB JSON exports."""
    records = []
    sequence = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")
    for path in sorted(Path(raw_dir).glob("iedb-*.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        rows = payload if isinstance(payload, list) else payload.get("data", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            try:
                assay_id = int(row.get("assay_id"))
            except (TypeError, ValueError):
                continue
            if assay_ids is not None and assay_id not in assay_ids:
                continue
            mutant = str(row.get("assay_antigen__name") or row.get("epitope__name") or "").upper().strip()
            wild_type = str(row.get("related_object__name") or "").upper().strip()
            if not (
                8 <= len(mutant) <= 15
                and len(mutant) == len(wild_type)
                and sequence.fullmatch(mutant)
                and sequence.fullmatch(wild_type)
                and mutant != wild_type
            ):
                continue
            records.append(
                {
                    "assay_id": assay_id,
                    "mutant_peptide": mutant,
                    "wild_type_peptide": wild_type,
                    "pair_relation": row.get("related_object__epitope_relation") or "unknown",
                    "reference_pmid": row.get("reference__pmid"),
                    "source_file": str(path),
                }
            )
    columns = [
        "assay_id",
        "mutant_peptide",
        "wild_type_peptide",
        "pair_relation",
        "reference_pmid",
        "source_file",
    ]
    return pd.DataFrame(records, columns=columns).drop_duplicates(
        subset=["assay_id", "mutant_peptide", "wild_type_peptide"]
    )


def reserve_lockbox(
    frame: pd.DataFrame,
    fraction: float = 0.2,
    seed: int = 42,
    group_column: str = "holdout_group",
) -> pd.Series:
    """Reserve whole holdout groups by a reproducible label-blind hash."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between zero and one")
    if group_column not in frame:
        raise ValueError(f"missing lockbox group column: {group_column}")
    group_values = frame[group_column].astype(str)
    groups = group_values.drop_duplicates().tolist()
    ranks = sorted(
        groups,
        key=lambda group: hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest(),
    )
    target = max(1, math.ceil(len(frame) * fraction))
    selected: set[str] = set()
    count = 0
    for group in ranks:
        selected.add(group)
        count += int(group_values.eq(group).sum())
        if count >= target:
            break
    return group_values.isin(selected)


def _expanded_counts(frame: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    scores = np.asarray(score, dtype=float)
    if len(scores) != len(frame):
        raise ValueError("score length does not match frame")
    positive = pd.DataFrame(
        {
            "stratum_key": frame["stratum_key"].to_numpy(),
            "y": 1,
            "weight": frame["n_positive_assays"].to_numpy(float),
            "score": scores,
        }
    )
    negative = pd.DataFrame(
        {
            "stratum_key": frame["stratum_key"].to_numpy(),
            "y": 0,
            "weight": frame["n_negative_assays"].to_numpy(float),
            "score": scores,
        }
    )
    return pd.concat((positive, negative), ignore_index=True).loc[lambda value: value["weight"] > 0]


def within_stratum_metrics(frame: pd.DataFrame, score: np.ndarray, top_fractions: tuple[float, ...] = (0.1, 0.2)) -> dict:
    """Compute count-weighted PR-AUC, top-k yield, and calibration."""
    expanded = _expanded_counts(frame, score)
    valid_groups = []
    macro_pr = []
    macro_weights = []
    for stratum, group in expanded.groupby("stratum_key"):
        if group["y"].nunique() < 2:
            continue
        value = average_precision_score(group["y"], group["score"], sample_weight=group["weight"])
        macro_pr.append(value)
        macro_weights.append(float(group.loc[group["y"].eq(1), "weight"].sum()))
        valid_groups.append(stratum)
    micro = (
        float(average_precision_score(expanded["y"], expanded["score"], sample_weight=expanded["weight"]))
        if expanded["y"].nunique() == 2
        else None
    )
    result = {
        "n_rows": int(len(frame)),
        "n_strata": int(frame["stratum_key"].nunique()),
        "n_comparable_strata": int(len(valid_groups)),
        "within_stratum_macro_pr_auc": float(np.mean(macro_pr)) if macro_pr else None,
        "within_stratum_positive_weighted_pr_auc": float(np.average(macro_pr, weights=macro_weights)) if macro_pr else None,
        "within_stratum_micro_pr_auc": micro,
        "calibration": {},
        "top_k_yield": {},
    }
    trials = frame["assay_trials"].to_numpy(float)
    rates = frame["response_rate"].to_numpy(float)
    probabilities = np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
    # ``response_rate`` is a binomial proportion, not a hard class label;
    # sklearn's brier_score_loss intentionally rejects fractional targets.
    result["calibration"]["brier"] = float(np.average((rates - probabilities) ** 2, weights=trials))
    bins = np.linspace(0.0, 1.0, 11)
    calibration_bins = []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= low) & ((probabilities < high) if high < 1 else (probabilities <= high))
        if not np.any(mask):
            continue
        weight = trials[mask]
        calibration_bins.append(
            {
                "lower": float(low),
                "upper": float(high),
                "n": int(mask.sum()),
                "predicted": float(np.average(probabilities[mask], weights=weight)),
                "observed": float(np.average(rates[mask], weights=weight)),
            }
        )
    result["calibration"]["bins"] = calibration_bins
    for fraction in top_fractions:
        yields = []
        lifts = []
        for _, group in frame.groupby("stratum_key"):
            if len(group) < 2:
                continue
            n_top = max(1, int(math.ceil(len(group) * fraction)))
            ranked_indices = pd.Series(probabilities, index=frame.index).loc[group.index].sort_values(ascending=False).index[:n_top]
            top = group.loc[ranked_indices]
            top_trials = float(top["assay_trials"].sum())
            all_trials = float(group["assay_trials"].sum())
            if top_trials <= 0 or all_trials <= 0:
                continue
            top_rate = float(top["n_positive_assays"].sum() / top_trials)
            all_rate = float(group["n_positive_assays"].sum() / all_trials)
            yields.append(top_rate)
            lifts.append(top_rate / all_rate if all_rate > 0 else np.nan)
        result["top_k_yield"][str(fraction)] = {
            "fraction": float(fraction),
            "macro_positive_rate": float(np.nanmean(yields)) if yields else None,
            "macro_lift": float(np.nanmean([value for value in lifts if np.isfinite(value)]))
            if any(np.isfinite(value) for value in lifts)
            else None,
            "n_strata": len(yields),
        }
    return result


def clustered_uncertainty(frame: pd.DataFrame, score: np.ndarray, draws: int = 200, seed: int = 42) -> dict:
    """Cluster bootstrap over leakage groups, retaining all count weights."""
    if draws <= 0:
        return {"draws": 0}
    groups = frame["holdout_group"].astype(str).to_numpy()
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values = {"pr_auc": [], "brier": [], "top_10_yield": []}
    for _ in range(draws):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        selected = np.concatenate([members[group] for group in sampled])
        sub = frame.iloc[selected].reset_index(drop=True)
        metric = within_stratum_metrics(sub, np.asarray(score)[selected])
        if metric["within_stratum_micro_pr_auc"] is not None:
            values["pr_auc"].append(metric["within_stratum_micro_pr_auc"])
        values["brier"].append(metric["calibration"]["brier"])
        top = metric["top_k_yield"].get("0.1", {}).get("macro_positive_rate")
        if top is not None:
            values["top_10_yield"].append(top)
    result = {"draws": draws, "successful_draws": len(values["brier"])}
    for name, observations in values.items():
        if observations:
            result[name + "_95ci"] = [float(value) for value in np.quantile(observations, [0.025, 0.975])]
    return result


class GAMBaseline:
    """Count-aware spline logistic baseline with categorical group effects."""

    numeric = ["presentation_score", "processing_score", "mhcflurry_affinity", "peptide_length"]
    categorical = ["hla", "cohort", "assay_context", "exposure", "source_family"]

    def __init__(self):
        self.preprocessor: ColumnTransformer | None = None
        self.model: object | None = None
        self.fallback = False

    def _features(self, frame: pd.DataFrame, fit: bool) -> np.ndarray:
        work = frame.copy()
        if "mhcflurry_affinity" not in work:
            work["mhcflurry_affinity"] = np.nan
        work["mhcflurry_affinity"] = pd.to_numeric(work["mhcflurry_affinity"], errors="coerce").replace(0, np.nan)
        work["mhcflurry_affinity"] = -np.log10(work["mhcflurry_affinity"].clip(lower=1e-3))
        for column in self.numeric:
            if column not in work:
                work[column] = np.nan
        for column in self.categorical:
            if column not in work:
                work[column] = "unknown"
            work[column] = _text(work[column])
        if fit or self.preprocessor is None:
            self.preprocessor = ColumnTransformer(
                [
                    (
                        "spline",
                        make_pipeline(SimpleImputer(strategy="median"), make_spline()),
                        ["presentation_score", "processing_score", "mhcflurry_affinity"],
                    ),
                    (
                        "linear",
                        SimpleImputer(strategy="median"),
                        ["peptide_length"],
                    ),
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False),
                        self.categorical,
                    ),
                ],
                remainder="drop",
            )
            transformed = self.preprocessor.fit_transform(work)
        else:
            transformed = self.preprocessor.transform(work)
        return np.asarray(transformed, dtype=float)

    def fit(self, frame: pd.DataFrame) -> "GAMBaseline":
        x = self._features(frame, fit=True)
        y = frame["response_rate"].to_numpy(float)
        weights = frame["assay_trials"].to_numpy(float)
        try:
            import statsmodels.api as sm

            self.model = sm.GLM(
                y,
                sm.add_constant(x, has_constant="add"),
                family=sm.families.Binomial(),
                freq_weights=weights,
            ).fit_regularized(alpha=0.01, L1_wt=0.0, maxiter=1000)
            self.fallback = False
        except (ImportError, ValueError, np.linalg.LinAlgError):
            self.model = LogisticRegression(C=0.5, max_iter=2000, random_state=42)
            self.model.fit(x, (y >= 0.5).astype(int), sample_weight=weights)
            self.fallback = True
        return self

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("GAMBaseline must be fit before prediction")
        x = self._features(frame, fit=False)
        if self.fallback:
            return self.model.predict_proba(x)[:, 1]
        import statsmodels.api as sm

        return np.clip(
            np.asarray(self.model.predict(sm.add_constant(x, has_constant="add")), dtype=float),
            1e-6,
            1.0 - 1e-6,
        )


def make_spline() -> SplineTransformer:
    return SplineTransformer(n_knots=4, degree=2, include_bias=False)


def _filter_rows(frame: pd.DataFrame) -> pd.DataFrame:
    outcome = _text(frame.get("outcome", pd.Series("", index=frame.index))).str.casefold()
    required = frame.get(DEFAULT_PRESENTATION, pd.Series(np.nan, index=frame.index)).notna()
    required &= frame.get(DEFAULT_PROCESSING, pd.Series(np.nan, index=frame.index)).notna()
    return frame.loc[required & outcome.isin(("positive", "negative"))].reset_index(drop=True)


def run(args: argparse.Namespace) -> dict:
    development_raw = _filter_rows(pd.read_csv(args.development))
    validation_raw = _filter_rows(pd.read_csv(args.validation))
    if "role" in validation_raw:
        validation_raw = validation_raw.loc[validation_raw["role"].astype(str).eq("validation")].reset_index(drop=True)
    # Fit numeric bin boundaries on development only.  Face components are
    # intentionally computed on the combined table below so the similarity
    # holdout cannot be contaminated by a validation face appearing novel.
    _, development_edges = prepare_table(
        development_raw,
        n_bins=args.presentation_bins,
        face_k=args.face_k,
        min_shared_face_kmers=args.min_shared_face_kmers,
    )
    combined_raw = pd.concat((development_raw.assign(split="development"), validation_raw.assign(split="validation")), ignore_index=True)
    combined, edges = prepare_table(
        combined_raw,
        presentation_edges=np.asarray(development_edges["presentation_edges"], dtype=float),
        processing_edges=np.asarray(development_edges["processing_edges"], dtype=float),
        face_k=args.face_k,
        min_shared_face_kmers=args.min_shared_face_kmers,
    )
    development = combined.loc[combined["split"].eq("development")].reset_index(drop=True)
    validation = combined.loc[combined["split"].eq("validation")].reset_index(drop=True)
    validation_families = set(validation["source_family"].astype(str))
    validation_faces = set(validation["face_component"].astype(str))
    family_overlap_mask = development["source_family"].astype(str).isin(validation_families)
    face_overlap_mask = development["face_component"].astype(str).isin(validation_faces)
    # The registered validation set remains untouched.  We report source-family
    # and face-component overlap rather than deleting most of development when
    # a common face motif links otherwise unrelated families.  New lockboxes
    # below are strict and are created only within development.
    family_lockbox_mask = reserve_lockbox(
        development, args.lockbox_fraction, args.seed, "source_family"
    )
    face_lockbox_mask = reserve_lockbox(
        development, args.lockbox_fraction, args.seed + 1, "face_component"
    )
    family_train = development.loc[~family_lockbox_mask].reset_index(drop=True)
    face_train = development.loc[~face_lockbox_mask].reset_index(drop=True)
    family_lockbox = development.loc[family_lockbox_mask].reset_index(drop=True)
    face_lockbox = development.loc[face_lockbox_mask].reset_index(drop=True)
    family_leakage = set(family_train["source_family"].astype(str)) & set(
        family_lockbox["source_family"].astype(str)
    )
    face_leakage = set(face_train["face_component"].astype(str)) & set(
        face_lockbox["face_component"].astype(str)
    )

    all_rows = pd.concat((development, validation), ignore_index=True)
    pairs = matched_pairs(all_rows)
    explicit_pairs = None
    pair_source = None
    if args.neoepitope_pairs:
        explicit_pairs = load_explicit_wt_mutant_pairs(args.neoepitope_pairs)
        pair_source = str(args.neoepitope_pairs)
        pair_info = {
            "available": True,
            "source": pair_source,
            "n_pairs": int(len(explicit_pairs)),
        }
    elif args.iedb_raw_dir:
        assay_ids = set(pd.to_numeric(all_rows.get("assay_id", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))
        explicit_pairs = load_iedb_wt_mutant_pairs(args.iedb_raw_dir, assay_ids)
        pair_source = str(args.iedb_raw_dir)
        if len(explicit_pairs):
            context = all_rows[["assay_id", "hla", "cohort", "exposure", "outcome"]].copy()
            context["assay_id"] = pd.to_numeric(context["assay_id"], errors="coerce")
            explicit_pairs = explicit_pairs.merge(context.drop_duplicates("assay_id"), on="assay_id", how="left")
            wt_context = all_rows[
                [
                    "peptide",
                    "hla",
                    "cohort",
                    "exposure",
                    "outcome",
                    "n_positive_assays",
                    "n_negative_assays",
                ]
            ].rename(columns={"peptide": "wild_type_peptide", "outcome": "wild_type_outcome"})
            wt_context = wt_context.drop_duplicates(
                ["wild_type_peptide", "hla", "cohort", "exposure"]
            )
            explicit_pairs = explicit_pairs.merge(
                wt_context,
                on=["wild_type_peptide", "hla", "cohort", "exposure"],
                how="left",
                suffixes=("", "_wt"),
            )
            explicit_pairs["wild_type_tested"] = explicit_pairs["wild_type_outcome"].notna()
        pair_info = {
            "available": bool(len(explicit_pairs)),
            "source": pair_source,
            "n_pairs": int(len(explicit_pairs)),
            "n_pairs_with_wild_type_response": int(explicit_pairs["wild_type_tested"].sum())
            if len(explicit_pairs)
            else 0,
            "wild_type_response_counts_available": bool(
                len(explicit_pairs) and explicit_pairs["wild_type_tested"].any()
            ),
        }
    else:
        pair_info = wt_mutant_pairs(all_rows)[1]
    baselines = {
        "family_lockbox": (GAMBaseline().fit(family_train), family_lockbox, "source_family"),
        "face_lockbox": (GAMBaseline().fit(face_train), face_lockbox, "face_component"),
    }
    reports = {}
    for name, (baseline, frame, cluster_column) in baselines.items():
        if len(frame):
            scores = baseline.predict_proba(frame)
            metric = within_stratum_metrics(frame, scores)
            metric["clustered_uncertainty"] = clustered_uncertainty(
                frame.assign(holdout_group=frame[cluster_column]),
                scores,
                args.bootstrap_draws,
                args.seed,
            )
        else:
            metric = {"n_rows": 0}
        reports[name] = metric
    validation_baseline = GAMBaseline().fit(development)
    validation_scores = validation_baseline.predict_proba(validation)
    reports["validation_external"] = within_stratum_metrics(validation, validation_scores)
    reports["validation_external"]["clustered_uncertainty"] = clustered_uncertainty(
        validation.assign(holdout_group=validation["face_component"]),
        validation_scores,
        args.bootstrap_draws,
        args.seed,
    )
    lockbox_membership = np.select(
        [family_lockbox_mask & face_lockbox_mask, family_lockbox_mask, face_lockbox_mask],
        ["family_and_face", "family", "face"],
        default="none",
    )
    lockbox = development.loc[family_lockbox_mask | face_lockbox_mask].copy()
    lockbox["lockbox_kind"] = lockbox_membership[family_lockbox_mask.to_numpy() | face_lockbox_mask.to_numpy()]
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    development.to_csv(output / "stratified-development.csv", index=False)
    validation.to_csv(output / "stratified-validation.csv", index=False)
    lockbox.to_csv(output / "lockbox.csv", index=False)
    pairs.to_csv(output / "matched-pairs.csv", index=False)
    if explicit_pairs is not None:
        explicit_pairs.to_csv(output / "neoepitope-wt-mutant-pairs.csv", index=False)
    report = {
        "protocol": "count-aware matched strata; independent source-family and TCR-face lockboxes; label-blind development lockbox",
        "development_input": str(args.development),
        "validation_input": str(args.validation),
        "presentation_processing_bins": edges,
        "bin_edges_fit_on": "development_only",
        "face_k": args.face_k,
        "min_shared_face_kmers": args.min_shared_face_kmers,
        "n_input_development": int(len(development_raw)),
        "n_development": int(len(development)),
        "n_development_rows_sharing_registered_validation_source_family": int(family_overlap_mask.sum()),
        "n_development_rows_sharing_registered_validation_face_component": int(face_overlap_mask.sum()),
        "n_family_train": int(len(family_train)),
        "n_face_train": int(len(face_train)),
        "n_family_lockbox": int(len(family_lockbox)),
        "n_face_lockbox": int(len(face_lockbox)),
        "n_lockbox_union": int(len(lockbox)),
        "family_lockbox_group_overlap": len(family_leakage),
        "face_lockbox_component_overlap": len(face_leakage),
        "n_validation": int(len(validation)),
        "n_matched_pairs": int(len(pairs)),
        "lockbox_fraction_target": args.lockbox_fraction,
        "count_sources": development["count_source"].value_counts().to_dict(),
        "validation_count_sources": validation["count_source"].value_counts().to_dict(),
        "wt_mutant": pair_info,
        "baseline": "statsmodels binomial GLM with spline terms and HLA/cohort/exposure/source-family effects" if not baseline.fallback else "fallback binary logistic spline model",
        "metrics": reports,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--development", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--neoepitope-pairs",
        help="Optional CSV with explicit wild_type_peptide, mutant_peptide, and hla columns",
    )
    parser.add_argument(
        "--iedb-raw-dir",
        help="Optional directory of cached iedb-*.json files containing related WT peptides",
    )
    parser.add_argument("--presentation-bins", type=int, default=5)
    parser.add_argument("--face-k", type=int, default=4)
    parser.add_argument("--min-shared-face-kmers", type=int, default=2)
    parser.add_argument("--lockbox-fraction", type=float, default=0.2)
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
