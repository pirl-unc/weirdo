#!/usr/bin/env python3
"""Evaluate TCR-visible non-self representations over MHC predictors.

Every response adapter is trained on other cohorts from the same domain after
exact held-out peptides are purged.  The frozen baseline is MHCflurry 2.3
presentation plus NetMHCpan BA.  This report compares residue-position/HLA
features, PRIME, human-proteome self neighborhoods, a restricting-HLA thymic
proxy, an observed-allele pseudo-haplotype, and response-positive similarity
kernels under the same transfer protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/weirdo-tcr-visible-nonself")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_tcell_predictor_comparison import cluster_metrics  # noqa: E402


COHORTS = {
    "bladder": ("IMPROVE bladder", "tumor"),
    "melanoma": ("IMPROVE melanoma", "tumor"),
    "Basket": ("IMPROVE basket", "tumor"),
    "yellow_fever_17d": ("Yellow fever 17D", "viral"),
    "multivirus_concise_set": ("Multi-virus panel", "viral"),
}
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
BLOSUM_SOURCE = substitution_matrices.load("BLOSUM62")
BLOSUM62 = np.asarray(
    [[BLOSUM_SOURCE[a, b] for b in AMINO_ACIDS] for a in AMINO_ACIDS],
    dtype=float,
)

UNGATED_SELF = (
    "human_self_face_mismatch",
    "human_self_full_mismatch",
    "human_self_weighted_mismatch",
    "human_self_anchor_matched_face_mismatch",
    "human_self_weighted_blosum",
    "human_self_anchor_matched_blosum",
    "human_self_face_exact_log_density",
    "human_self_face_oneoff_log_density",
    "human_self_anchor_face_exact_log_density",
    "human_self_anchor_face_oneoff_log_density",
)
SAME_HLA_SELF = (
    "human_self_hla_joint_distance",
    "human_self_hla_crossreactivity_max",
    "human_self_hla_crossreactivity_log_density",
    "human_self_hla_anchor_crossreactivity_max",
)
HAPLOTYPE_SELF = tuple(
    name.replace("human_self_hla_", "human_self_haplotype_")
    for name in SAME_HLA_SELF
)

MODEL_SPECS = (
    ("mhc_baseline", "MHC baseline", False, None, None, 1.0),
    ("taxonomy", "+ host-virus taxonomy", False, "taxonomy", None, 1.0),
    ("prime", "+ PRIME 2.1", True, None, None, 1.0),
    ("hla_length", "+ length + HLA family", False, None, "calibration", 0.003),
    ("hla_anchors", "+ HLA family + anchors", False, None, "anchors", 0.003),
    (
        "tcr_face_composition",
        "+ TCR-face composition + HLA family",
        False,
        None,
        "face_composition",
        0.003,
    ),
    (
        "tcr_face_exact_allele",
        "+ TCR-face composition + exact HLA",
        False,
        None,
        "allele_face_composition",
        0.003,
    ),
    (
        "tcr_face_exact_allele_interactions",
        "+ TCR-face × exact-HLA interactions",
        False,
        None,
        "allele_face_interactions",
        0.003,
    ),
    (
        "tcr_face_positions",
        "+ TCR-face positions",
        False,
        None,
        "face_positions",
        0.003,
    ),
    (
        "tcr_face_hla_interactions",
        "+ face × HLA interactions",
        False,
        None,
        "face_hla_interactions",
        0.003,
    ),
    ("ungated_self", "+ nearest human (ungated)", False, "ungated", None, 0.003),
    ("same_hla_self", "+ same-HLA human self", False, "same_hla", None, 0.003),
    (
        "pseudo_haplotype_self",
        "+ observed-HLA pseudo-haplotype",
        False,
        "haplotype",
        None,
        0.003,
    ),
    (
        "tcr_face_same_hla",
        "+ face/HLA + same-HLA self",
        False,
        "same_hla",
        "allele_face_composition",
        0.003,
    ),
    (
        "prime_tcr_face",
        "+ PRIME + face/HLA",
        True,
        None,
        "allele_face_composition",
        0.003,
    ),
    (
        "prime_tcr_face_same_hla",
        "+ PRIME + face/HLA + same-HLA self",
        True,
        "same_hla",
        "allele_face_composition",
        0.003,
    ),
    (
        "response_kernel",
        "+ positive-response similarity kernel",
        False,
        "response_kernel",
        None,
        0.003,
    ),
    (
        "response_kernel_strict",
        "+ response kernel (1-mutation purge)",
        False,
        "response_kernel_strict",
        None,
        0.003,
    ),
)

BLUE = "#2D6FBB"
ORANGE = "#D3871C"
GREEN = "#2E8B62"
RED = "#C44E52"
PURPLE = "#7A58B5"
INK = "#17263D"
MUTED = "#63758E"
GRID = "#D7E1EA"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logit(values: pd.Series | np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1 - values))


def hla_family(hla: str) -> str:
    match = re.match(r"HLA-([ABC])\*(\d{2})", str(hla))
    return "".join(match.groups()) if match else str(hla)


def hla_locus(hla: str) -> str:
    return str(hla)[4] if str(hla).startswith("HLA-") else str(hla)[:1]


def sequence_features(peptide: str, hla: str, representation: str) -> dict:
    """Create sparse residue/HLA features with explicit anchor separation."""
    family = hla_family(hla)
    length = len(peptide)
    exact_allele = representation.startswith("allele_")
    result: dict[str, float] = {f"length={length}": 1.0}
    if exact_allele:
        result[f"hla_allele={hla}"] = 1.0
    else:
        result[f"hla_family={family}"] = 1.0
    if representation == "calibration":
        return result
    result[f"P2={peptide[1]}"] = 1.0
    result[f"Pomega={peptide[-1]}"] = 1.0
    if representation == "anchors":
        return result
    face = peptide[2:-1]
    if representation in {
        "face_composition",
        "face_hla_interactions",
        "allele_face_composition",
        "allele_face_interactions",
    }:
        for amino_acid in AMINO_ACIDS:
            frequency = face.count(amino_acid) / len(face)
            result[f"face_composition={amino_acid}"] = frequency
            if representation == "face_hla_interactions":
                result[f"{family}|face={amino_acid}"] = frequency
            elif representation == "allele_face_interactions":
                result[f"{hla}|face={amino_acid}"] = frequency
        return result
    if representation == "face_positions":
        positions = (
            ("N3", 2),
            ("N4", 3),
            ("N5", 4),
            ("C5", length - 5),
            ("C4", length - 4),
            ("C3", length - 3),
            ("C2", length - 2),
        )
        for label, index in positions:
            if 0 <= index < length:
                result[f"face_position:{label}={peptide[index]}"] = 1.0
        return result
    raise ValueError(f"Unknown sequence representation: {representation}")


def base_numeric(frame: pd.DataFrame, include_prime: bool) -> np.ndarray:
    columns = [
        logit(frame["mhcflurry_2_3_0_presentation_score"]),
        -np.log10(np.clip(frame["netmhcpan_ba_rank"].to_numpy(float), 1e-5, 100)),
    ]
    if include_prime:
        columns.append(
            -np.log10(np.clip(frame["prime_2_1_rank"].to_numpy(float), 1e-5, 100))
        )
    return np.column_stack(columns)


def numeric_features(
    frame: pd.DataFrame,
    *,
    include_prime: bool,
    extra: str | None,
    kernel: np.ndarray | None,
) -> np.ndarray:
    result = base_numeric(frame, include_prime)
    if extra == "taxonomy":
        result = np.column_stack(
            (
                result,
                logit(
                    frame[
                        "ablation__mammal_host_virus_mammal_softmax_120k__mammal_host_viruses"
                    ]
                ),
            )
        )
    elif extra == "ungated":
        result = np.column_stack((result, frame[list(UNGATED_SELF)].to_numpy(float)))
    elif extra == "same_hla":
        result = np.column_stack((result, frame[list(SAME_HLA_SELF)].to_numpy(float)))
    elif extra == "haplotype":
        result = np.column_stack((result, frame[list(HAPLOTYPE_SELF)].to_numpy(float)))
    elif extra in {"response_kernel", "response_kernel_strict"}:
        if kernel is None:
            raise ValueError(f"Missing dynamic features for {extra}")
        result = np.column_stack((result, kernel))
    elif extra is not None:
        raise ValueError(f"Unknown numeric feature set: {extra}")
    return result


def encode_peptides(peptides: pd.Series) -> np.ndarray:
    return np.asarray([[AA_INDEX[aa] for aa in peptide] for peptide in peptides], dtype=np.int8)


def response_similarity_features(
    train: pd.DataFrame,
    query: pd.DataFrame,
    *,
    purge_distance: int,
) -> np.ndarray:
    """Compare queries with response-positive motifs from the training cohorts."""
    output = np.full((len(query), 6), np.nan)
    train = train.copy()
    query = query.copy()
    train["_family"] = train["hla"].map(hla_family)
    query["_family"] = query["hla"].map(hla_family)
    train["_locus"] = train["hla"].map(hla_locus)
    query["_locus"] = query["hla"].map(hla_locus)
    for length in sorted(query["peptide"].str.len().unique()):
        length_train = train.loc[train["peptide"].str.len().eq(length)]
        length_query = query.loc[query["peptide"].str.len().eq(length)]
        for family, indices in length_query.groupby("_family").groups.items():
            part = query.loc[indices]
            candidates = length_train.loc[
                length_train["_family"].eq(family) & length_train["outcome"].eq(1)
            ]
            if len(candidates) < 3:
                candidates = length_train.loc[
                    length_train["_locus"].eq(part["_locus"].iloc[0])
                    & length_train["outcome"].eq(1)
                ]
            if len(candidates) < 3:
                candidates = length_train.loc[length_train["outcome"].eq(1)]
            if candidates.empty:
                continue
            candidate_values = encode_peptides(candidates["peptide"])
            query_values = encode_peptides(part["peptide"])
            weights = np.ones(length)
            weights[[0, 1, -1]] = (0.50, 0.25, 0.25)
            denominator = weights.sum()
            for start in range(0, len(query_values), 256):
                values = query_values[start : start + 256]
                similarity = np.zeros((len(values), len(candidate_values)))
                mismatch = np.zeros_like(similarity)
                for position, weight in enumerate(weights):
                    similarity += weight * BLOSUM62[
                        values[:, None, position], candidate_values[None, :, position]
                    ]
                    mismatch += (
                        values[:, None, position] != candidate_values[None, :, position]
                    )
                similarity /= denominator
                similarity[mismatch <= purge_distance] = -np.inf
                finite = np.isfinite(similarity)
                maximum = np.max(similarity, axis=1, where=finite, initial=-20)
                k = min(5, len(candidate_values))
                top = np.partition(similarity, -k, axis=1)[:, -k:]
                top[~np.isfinite(top)] = np.nan
                top_mean = np.nanmean(top, axis=1)
                shifted = np.zeros_like(similarity)
                np.exp(similarity - maximum[:, None], where=finite, out=shifted)
                soft = maximum + np.log(
                    np.maximum(shifted.sum(axis=1), 1) / len(candidate_values)
                )
                face_distance = (
                    values[:, None, 2:-1] != candidate_values[None, :, 2:-1]
                ).sum(axis=2)
                face_distance[~finite] = 99
                anchor_match = (
                    (values[:, None, 1] == candidate_values[None, :, 1])
                    & (values[:, None, -1] == candidate_values[None, :, -1])
                    & finite
                )
                anchor_similarity = np.max(
                    similarity, axis=1, where=anchor_match, initial=-20
                )
                anchor_similarity[~anchor_match.any(axis=1)] = np.nan
                source_indices = np.asarray(list(indices))[start : start + len(values)]
                positions = query.index.get_indexer(source_indices)
                output[positions] = np.column_stack(
                    (
                        maximum,
                        top_mean,
                        soft,
                        face_distance.min(axis=1),
                        anchor_similarity,
                        np.full(len(values), np.log1p(len(candidate_values))),
                    )
                )
    return output


def fit_adapter(
    train: pd.DataFrame,
    test: pd.DataFrame,
    spec: tuple,
    *,
    train_kernel: np.ndarray | None = None,
    test_kernel: np.ndarray | None = None,
) -> np.ndarray:
    model_id, _, include_prime, extra, representation, regularization = spec
    train_numeric = numeric_features(
        train, include_prime=include_prime, extra=extra, kernel=train_kernel
    )
    test_numeric = numeric_features(
        test, include_prime=include_prime, extra=extra, kernel=test_kernel
    )
    imputer = SimpleImputer(strategy="median", add_indicator=True).fit(train_numeric)
    train_numeric = imputer.transform(train_numeric)
    test_numeric = imputer.transform(test_numeric)
    scaler = StandardScaler().fit(train_numeric)
    train_matrix = csr_matrix(scaler.transform(train_numeric))
    test_matrix = csr_matrix(scaler.transform(test_numeric))
    if representation is not None:
        vectorizer = DictVectorizer()
        train_sparse = vectorizer.fit_transform(
            [
                sequence_features(peptide, hla, representation)
                for peptide, hla in zip(train["peptide"], train["hla"])
            ]
        )
        test_sparse = vectorizer.transform(
            [
                sequence_features(peptide, hla, representation)
                for peptide, hla in zip(test["peptide"], test["hla"])
            ]
        )
        train_matrix = hstack((train_matrix, train_sparse))
        test_matrix = hstack((test_matrix, test_sparse))
    model = LogisticRegression(
        C=regularization,
        class_weight="balanced",
        max_iter=4_000,
        random_state=84_342,
        solver="liblinear" if representation is not None or regularization < 1 else "lbfgs",
    )
    model.fit(train_matrix, train["outcome"].to_numpy(int))
    prediction = model.predict_proba(test_matrix)[:, 1]
    if len(prediction) != len(test) or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid predictions from {model_id}")
    return prediction


def metric_row(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y, prediction)),
        "average_precision": float(average_precision_score(y, prediction)),
        "log_loss": float(log_loss(y, prediction, labels=[0, 1])),
        "brier": float(brier_score_loss(y, prediction)),
    }


def transfer_predictions(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_rows = []
    prediction_rows = []
    for cohort, (cohort_label, domain) in COHORTS.items():
        test = frame.loc[frame["cohort"].eq(cohort)].copy().reset_index(drop=True)
        train_cohorts = [
            name
            for name, (_, candidate_domain) in COHORTS.items()
            if name != cohort and candidate_domain == domain
        ]
        train = frame.loc[frame["cohort"].isin(train_cohorts)].copy()
        before = len(train)
        train = train.loc[~train["peptide"].isin(set(test["peptide"]))].reset_index(
            drop=True
        )
        purged = before - len(train)
        kernels = {
            "response_kernel": (
                response_similarity_features(train, train, purge_distance=0),
                response_similarity_features(train, test, purge_distance=0),
            ),
            "response_kernel_strict": (
                response_similarity_features(train, train, purge_distance=1),
                response_similarity_features(train, test, purge_distance=1),
            ),
        }
        y = test["outcome"].to_numpy(int)
        cohort_predictions = {}
        for spec in MODEL_SPECS:
            model_id, model_label, _, extra, _, regularization = spec
            train_kernel, test_kernel = kernels.get(extra, (None, None))
            prediction = fit_adapter(
                train,
                test,
                spec,
                train_kernel=train_kernel,
                test_kernel=test_kernel,
            )
            cohort_predictions[model_id] = prediction
            result_rows.append(
                {
                    "heldout_cohort": cohort,
                    "heldout_label": cohort_label,
                    "domain": domain,
                    "model_id": model_id,
                    "model_label": model_label,
                    "regularization_C": regularization,
                    "n_test": len(test),
                    "n_positive": int(y.sum()),
                    "n_train": len(train),
                    "n_exact_peptide_train_rows_purged": purged,
                    **metric_row(y, prediction),
                }
            )
            for row_index, score in enumerate(prediction):
                prediction_rows.append(
                    {
                        "heldout_cohort": cohort,
                        "row_in_cohort": row_index,
                        "group": str(test.loc[row_index, "group"]),
                        "peptide": test.loc[row_index, "peptide"],
                        "outcome": int(y[row_index]),
                        "model_id": model_id,
                        "score": float(score),
                    }
                )
        if set(cohort_predictions) != {spec[0] for spec in MODEL_SPECS}:
            raise AssertionError("A model failed to produce predictions")
    results = pd.DataFrame(result_rows)
    baseline = results.loc[results["model_id"].eq("mhc_baseline")].set_index(
        "heldout_cohort"
    )
    for metric in ("roc_auc", "average_precision", "log_loss", "brier"):
        results[f"baseline_{metric}"] = results["heldout_cohort"].map(baseline[metric])
        results[f"delta_{metric}"] = results[metric] - results[f"baseline_{metric}"]
    return results, pd.DataFrame(prediction_rows)


def bootstrap_deltas(
    results: pd.DataFrame, predictions: pd.DataFrame, draws: int
) -> pd.DataFrame:
    rows = []
    for cohort_index, cohort in enumerate(COHORTS):
        cohort_predictions = predictions.loc[
            predictions["heldout_cohort"].eq(cohort)
        ]
        baseline = cohort_predictions.loc[
            cohort_predictions["model_id"].eq("mhc_baseline")
        ].sort_values("row_in_cohort")
        scores = {"mhc_baseline": baseline["score"].to_numpy(float)}
        for model_id in results.loc[
            results["heldout_cohort"].eq(cohort), "model_id"
        ]:
            if model_id == "mhc_baseline":
                continue
            part = cohort_predictions.loc[
                cohort_predictions["model_id"].eq(model_id)
            ].sort_values("row_in_cohort")
            scores[model_id] = part["score"].to_numpy(float)
        _, samples = cluster_metrics(
            baseline["outcome"].to_numpy(int),
            scores,
            baseline["group"].to_numpy(str),
            draws,
            seed=20261200 + cohort_index,
        )
        cohort_results = results.loc[
            results["heldout_cohort"].eq(cohort)
            & ~results["model_id"].eq("mhc_baseline")
        ]
        for result in cohort_results.itertuples(index=False):
            distribution = samples[result.model_id] - samples["mhc_baseline"]
            bounds = np.nanquantile(distribution, [0.025, 0.975], axis=0)
            rows.append(
                {
                    "heldout_cohort": cohort,
                    "model_id": result.model_id,
                    "delta_roc_auc": result.delta_roc_auc,
                    "delta_roc_auc_low": bounds[0, 0],
                    "delta_roc_auc_high": bounds[1, 0],
                    "delta_average_precision": result.delta_average_precision,
                    "delta_average_precision_low": bounds[0, 1],
                    "delta_average_precision_high": bounds[1, 1],
                    "bootstrap_valid": int(np.isfinite(distribution[:, 0]).sum()),
                }
            )
    return pd.DataFrame(rows)


def macro_table(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby(["model_id", "model_label"], as_index=False)
        .agg(
            mean_roc_auc=("roc_auc", "mean"),
            mean_delta_roc_auc=("delta_roc_auc", "mean"),
            mean_average_precision=("average_precision", "mean"),
            mean_delta_average_precision=("delta_average_precision", "mean"),
            mean_delta_log_loss=("delta_log_loss", "mean"),
            mean_delta_brier=("delta_brier", "mean"),
            positive_auc_cohorts=("delta_roc_auc", lambda values: int((values > 0).sum())),
        )
        .sort_values("mean_delta_roc_auc", ascending=False)
    )


def style_axis(axis) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=INK)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def plot_heatmap(results: pd.DataFrame, output: Path) -> None:
    selected = (
        "taxonomy",
        "prime",
        "tcr_face_composition",
        "tcr_face_exact_allele",
        "tcr_face_positions",
        "same_hla_self",
        "pseudo_haplotype_self",
        "tcr_face_same_hla",
        "prime_tcr_face",
        "prime_tcr_face_same_hla",
        "response_kernel_strict",
    )
    labels = {spec[0]: spec[1].removeprefix("+ ") for spec in MODEL_SPECS}
    matrix = (
        results.loc[results["model_id"].isin(selected)]
        .pivot(index="model_id", columns="heldout_cohort", values="delta_roc_auc")
        .reindex(index=selected, columns=list(COHORTS))
    )
    figure, axis = plt.subplots(figsize=(10.5, 6.2))
    limit = max(0.11, float(np.nanmax(np.abs(matrix.to_numpy()))))
    image = axis.imshow(matrix, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(COHORTS)), [COHORTS[c][0] for c in COHORTS], rotation=25, ha="right")
    axis.set_yticks(range(len(selected)), [labels[m] for m in selected])
    for row in range(len(selected)):
        for column in range(len(COHORTS)):
            value = matrix.iloc[row, column]
            axis.text(column, row, f"{value:+.3f}", ha="center", va="center", fontsize=8)
    axis.set_title("Held-out gain over MHCflurry 2.3 + NetMHCpan BA", color=INK, weight="bold")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Δ ROC AUC")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_macro(macro: pd.DataFrame, output: Path) -> None:
    selected = (
        "taxonomy",
        "prime",
        "hla_length",
        "hla_anchors",
        "tcr_face_composition",
        "tcr_face_exact_allele",
        "tcr_face_positions",
        "tcr_face_hla_interactions",
        "tcr_face_exact_allele_interactions",
        "same_hla_self",
        "tcr_face_same_hla",
        "prime_tcr_face",
        "prime_tcr_face_same_hla",
    )
    labels = {spec[0]: spec[1].removeprefix("+ ") for spec in MODEL_SPECS}
    part = macro.set_index("model_id").loc[list(selected)]
    order = part["mean_delta_roc_auc"].sort_values().index
    y = np.arange(len(order))
    colors = [ORANGE if "prime" in model else BLUE for model in order]
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 6.5), sharey=True)
    metrics = (
        ("mean_delta_roc_auc", "Mean held-out Δ ROC AUC"),
        ("mean_delta_average_precision", "Mean held-out Δ average precision"),
    )
    for axis, (metric, xlabel) in zip(axes, metrics):
        axis.barh(y, part.loc[order, metric], color=colors, height=0.7)
        axis.axvline(0, color=INK, linewidth=0.8)
        axis.set_xlabel(xlabel)
        axis.set_yticks(y, [labels[model] for model in order])
        axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].tick_params(axis="y", labelsize=9)
    figure.suptitle("Exploratory feature-family trade-offs", color=INK, weight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_self_gating(results: pd.DataFrame, output: Path) -> None:
    selected = ("ungated_self", "same_hla_self", "pseudo_haplotype_self")
    labels = {
        "ungated_self": "Nearest human\n(no HLA gate)",
        "same_hla_self": "Restricting-HLA\nself neighborhood",
        "pseudo_haplotype_self": "Observed-HLA\npseudo-haplotype",
    }
    colors = (MUTED, GREEN, PURPLE)
    x = np.arange(len(COHORTS))
    width = 0.24
    figure, axis = plt.subplots(figsize=(10, 5.2))
    for offset, (model, color) in enumerate(zip(selected, colors)):
        values = (
            results.loc[results["model_id"].eq(model)]
            .set_index("heldout_cohort")
            .loc[list(COHORTS), "delta_roc_auc"]
        )
        axis.bar(x + (offset - 1) * width, values, width, label=labels[model], color=color)
    axis.axhline(0, color=INK, linewidth=0.8)
    axis.set_xticks(x, [COHORTS[c][0] for c in COHORTS], rotation=20, ha="right")
    axis.set_ylabel("Δ ROC AUC over MHC baseline")
    axis.set_title("Self similarity only helps after restricting-HLA gating", color=INK, weight="bold")
    axis.legend(frameon=False, ncols=3, loc="upper left")
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_roc(predictions: pd.DataFrame, output: Path) -> None:
    models = (
        ("mhc_baseline", "MHC baseline", MUTED),
        ("tcr_face_exact_allele", "+ face/exact HLA", BLUE),
        ("prime_tcr_face_same_hla", "+ PRIME + face/HLA + self", ORANGE),
    )
    figure, axes = plt.subplots(1, len(COHORTS), figsize=(16, 3.4), sharex=True, sharey=True)
    for axis, (cohort, (label, _)) in zip(axes, COHORTS.items()):
        part = predictions.loc[predictions["heldout_cohort"].eq(cohort)]
        for model, model_label, color in models:
            scores = part.loc[part["model_id"].eq(model)].sort_values("row_in_cohort")
            false_positive, true_positive, _ = roc_curve(scores["outcome"], scores["score"])
            auc = roc_auc_score(scores["outcome"], scores["score"])
            axis.plot(false_positive, true_positive, color=color, label=f"{model_label} ({auc:.3f})")
        axis.plot([0, 1], [0, 1], linestyle="--", color=GRID)
        axis.set_title(label, fontsize=10, color=INK)
        axis.set_xlabel("False-positive rate")
        axis.grid(color=GRID, linewidth=0.6)
    axes[0].set_ylabel("True-positive rate")
    axes[-1].legend(frameon=False, fontsize=7, loc="lower right")
    figure.suptitle("Held-out cohort ROC curves", color=INK, weight="bold")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_analysis(output: Path, results: pd.DataFrame, macro: pd.DataFrame) -> None:
    lookup = macro.set_index("model_id")
    selected = results.loc[results["model_id"].eq("tcr_face_exact_allele")]
    combined = results.loc[results["model_id"].eq("prime_tcr_face_same_hla")]
    same = lookup.loc["same_hla_self"]
    ungated = lookup.loc["ungated_self"]
    haplotype = lookup.loc["pseudo_haplotype_self"]
    kernel = lookup.loc["response_kernel_strict"]
    text = f"""# TCR-visible non-self: held-out cohort diagnosis

## Outcome

The first representation that improves all five held-out cohorts is a strongly
regularized adapter containing peptide length, the exact HLA allele, P2/PΩ anchors, and
amino-acid composition across P3…PΩ−1. It adds **{lookup.loc['tcr_face_exact_allele', 'mean_delta_roc_auc']:+.4f} mean ROC AUC** and
**{lookup.loc['tcr_face_exact_allele', 'mean_delta_average_precision']:+.4f} mean AP** over MHCflurry 2.3 presentation plus
NetMHCpan BA. Per-cohort Δ AUC ranges from {selected['delta_roc_auc'].min():+.4f} to
{selected['delta_roc_auc'].max():+.4f}; all five point estimates are positive.

The exploratory highest macro endpoint is PRIME 2.1 plus that face/HLA adapter
and the continuous same-HLA self-neighborhood proxy: **{combined['delta_roc_auc'].mean():+.4f} mean Δ AUC**,
**{combined['delta_average_precision'].mean():+.4f} mean Δ AP**, and
**{combined['delta_log_loss'].mean():+.4f} mean Δ log loss**. It is not uniformly robust: bladder remains
negative ({combined.loc[combined['heldout_cohort'].eq('bladder'), 'delta_roc_auc'].iloc[0]:+.4f} Δ AUC), so it should not replace the face/HLA-only model without new validation.

## What the ablations say

- Generic taxonomy is effectively null here ({lookup.loc['taxonomy', 'mean_delta_roc_auc']:+.4f} mean Δ AUC).
- Ungated nearest-human similarity is harmful in ranking ({ungated['mean_delta_roc_auc']:+.4f}), despite some AP/calibration gains.
- Conditioning the same candidate self peptides on the restricting HLA rescues the direction ({same['mean_delta_roc_auc']:+.4f} mean Δ AUC; {int(same['positive_auc_cohorts'])}/5 positive cohorts).
- Pooling all observed group alleles as a pseudo-haplotype is worse ({haplotype['mean_delta_roc_auc']:+.4f}); restricting-HLA selection is the better proxy. Viral group allele sets are incomplete, so this is an ablation rather than a definitive haplotype test.
- Exact-allele calibration beats two-digit HLA-family calibration; unseen held-out alleles are handled as unknown sparse categories. Position-specific face one-hot features are less stable than face composition. The benchmark supports residue weighting, but not a high-dimensional positional code at its present sample size.
- A response-positive BLOSUM cross-reactivity kernel fails even after removing exact and one-mutation neighbors ({kernel['mean_delta_roc_auc']:+.4f}). Known-responder similarity is cohort-biased here.

## Interpretation

“Foreign source” is not the quantity selected TCR repertoires see. A useful
feature must separate MHC-facing anchors from the exposed peptide face and ask
whether similar *presented* self ligands existed during selection. The data
support the first half strongly and the same-HLA self term weakly. They do not
support generic virus probability, raw distance to the human proteome, or
pooling every observed allele.

This is compatible with direct experimental and modeling work: PRIME combines
HLA presentation with residues at positions less important for HLA binding;
large TCR ligand screens find conserved central, up-facing motifs with more
tolerance at MHC-contact residues; and thymic-selection models predict fewer
available clones near displayed self. See the [PRIME2.0 study](https://pmc.ncbi.nlm.nih.gov/articles/PMC9811684/),
[direct TCR cross-reactivity mapping](https://pmc.ncbi.nlm.nih.gov/articles/PMC4071348/), and
[thymic selection model](https://pmc.ncbi.nlm.nih.gov/articles/PMC1857316/).

## Experimental adapters (not a recommended default)

The exact-HLA/TCR-face specification is fitted on all maintained data as
separate tumor and viral response adapters in `models/`. They require only
`peptide`, `hla`, `mhcflurry_2_3_0_presentation_score`, and
`netmhcpan_ba_rank`; `scripts/train_tcr_visible_response_adapter.py` also
provides the load-and-predict function. Training metrics in the model manifest
are descriptive—the held-out table in this report is the performance evidence.

## Validation boundary

All feature families and regularization choices in this report are exploratory.
PRIME has known training overlap with part of this benchmark. The self index is
reviewed human Swiss-Prot, candidate retrieval is approximate, and the thymic
proxy uses predicted binding under the restricting HLA rather than thymic
expression, peptide processing, or an observed TCR repertoire.

## External confirmation: failed

The frozen tumor adapter was scored once, unchanged, on TESLA and Braun, which
never entered fitting or selection (`scripts/evaluate_tcr_visible_external.py`,
[`tables/external_confirmation.csv`](tables/external_confirmation.csv)). Paired
group-bootstrap intervals are for AUC(adapter) − AUC(baseline).

| Endpoint | n (pos) | Adapter AUC | MHCflurry presentation AUC | Δ [95% interval] | NetMHCpan BA AUC | Δ [95% interval] |
|---|---:|---:|---:|---:|---:|---:|
| TESLA | 711 (33) | 0.673 | 0.841 | −0.168 [−0.264, −0.066] | 0.817 | −0.144 [−0.247, −0.055] |
| Braun, all | 112 (53) | 0.625 | 0.635 | −0.011 [−0.112, +0.079] | 0.644 | −0.020 [−0.088, +0.050] |
| Braun, frameshift | 19 (5) | 0.757 | 0.714 | +0.043 [−0.188, +0.214] | 0.729 | +0.029 [−0.067, +0.133] |

The within-IMPROVE gain does not transfer: on TESLA the adapter is significantly
worse than either raw MHC predictor, and on Braun it is indistinguishable from
them. The earlier “+0.030 mean AUC” is therefore a cross-cohort result within one
benchmark family, selected from many exploratory specifications, and must not be
read as evidence of a general improvement. Use MHC presentation/binding scores
directly for external neoepitope ranking; these adapters stay experimental.

## Figures

![Held-out delta heatmap](figures/01_cohort_delta_auc_heatmap.png)

![Macro trade-offs](figures/02_macro_tradeoffs.png)

![Self gating](figures/03_self_gating.png)

![Selected ROC curves](figures/04_selected_roc_curves.png)
"""
    (output / "analysis.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--human-self", type=Path, required=True)
    parser.add_argument("--same-hla-self", type=Path, required=True)
    parser.add_argument("--pseudo-haplotype-self", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=1_000)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores, low_memory=False)
    scores["group_key"] = scores["cohort"].astype(str) + "|" + scores[
        "group"
    ].astype(str)
    human = pd.read_csv(args.human_self)
    same_hla = pd.read_csv(args.same_hla_self)
    haplotype = pd.read_csv(args.pseudo_haplotype_self)
    frame = (
        scores.merge(human, on="peptide", how="left", validate="many_to_one")
        .merge(same_hla, on=["peptide", "hla"], how="left", validate="many_to_one")
        .merge(
            haplotype,
            on=["group_key", "peptide"],
            how="left",
            validate="many_to_one",
        )
    )
    required = {
        "peptide",
        "hla",
        "cohort",
        "outcome",
        "group",
        "prime_2_1_rank",
        "mhcflurry_2_3_0_presentation_score",
        "netmhcpan_ba_rank",
        *UNGATED_SELF,
        *SAME_HLA_SELF,
        *HAPLOTYPE_SELF,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    unexpected = sorted(set(frame["cohort"]) - set(COHORTS))
    if unexpected:
        raise ValueError(f"Unexpected cohorts: {unexpected}")

    args.output.mkdir(parents=True, exist_ok=True)
    tables = args.output / "tables"
    figures = args.output / "figures"
    raw = args.output / "raw"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    raw.mkdir(exist_ok=True)

    results, predictions = transfer_predictions(frame)
    bootstrap = bootstrap_deltas(results, predictions, args.bootstrap_draws)
    macro = macro_table(results)
    results.to_csv(tables / "heldout_results.csv", index=False)
    bootstrap.to_csv(tables / "paired_group_bootstrap.csv", index=False)
    macro.to_csv(tables / "macro_results.csv", index=False)
    predictions.to_csv(raw / "heldout_predictions.csv.gz", index=False)
    plot_heatmap(results, figures / "01_cohort_delta_auc_heatmap.png")
    plot_macro(macro, figures / "02_macro_tradeoffs.png")
    plot_self_gating(results, figures / "03_self_gating.png")
    plot_roc(predictions, figures / "04_selected_roc_curves.png")
    write_analysis(args.output, results, macro)
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "inputs": {
                    str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
                    for path in (
                        args.scores,
                        args.human_self,
                        args.same_hla_self,
                        args.pseudo_haplotype_self,
                    )
                },
                "cohorts": COHORTS,
                "models": [
                    {
                        "model_id": spec[0],
                        "label": spec[1],
                        "include_prime": spec[2],
                        "numeric_extra": spec[3],
                        "sequence_representation": spec[4],
                        "regularization_C": spec[5],
                    }
                    for spec in MODEL_SPECS
                ],
                "bootstrap_draws": args.bootstrap_draws,
                "split_rule": (
                    "held out cohort; train on other same-domain cohorts; purge all "
                    "training rows with an exact held-out peptide"
                ),
                "status": "exploratory feature-family sweep",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(macro.to_string(index=False, float_format=lambda value: f"{value:+.5f}"))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()
