#!/usr/bin/env python3
"""Evaluate content scores after broad-development MHC calibration.

The registered viral benchmark is deliberately held out.  Rows from the broad
development table whose joint exact/7-mer/Hamming peptide component appears in
the registered validation table are removed before fitting the response model.
This prevents separately generated cluster labels from allowing near-duplicate
peptides across the transfer boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "evaluate_peptide_content_immunogenicity.py"


def _joint_peptide_components(frame: pd.DataFrame) -> pd.Series:
    """Build overlap components jointly across development and validation.

    Per-file ``peptide_cluster`` labels are not safe for transfer filtering:
    the same sequence or a one-residue/7-mer neighbor can receive unrelated
    labels in separately generated tables.  We connect existing labels,
    exact sequences, shared 7-mers, and same-length Hamming-one sequences, then
    return deterministic component IDs.
    """
    peptides = frame["peptide"].fillna("").astype(str).str.upper().tolist()
    unique = sorted(set(peptides))
    parent = {peptide: peptide for peptide in unique}
    rank = {peptide: 0 for peptide in unique}

    def find(value: str) -> str:
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left == right:
            return
        if rank[left] < rank[right]:
            left, right = right, left
        parent[right] = left
        if rank[left] == rank[right]:
            rank[left] += 1

    if "peptide_cluster" in frame:
        for _, group in frame.assign(_peptide=peptides).groupby("peptide_cluster", dropna=True):
            members = sorted(set(group["_peptide"]))
            for left, right in zip(members, members[1:]):
                union(left, right)

    token_index: dict[str, list[str]] = {}
    for peptide in unique:
        for token in {peptide[i : i + 7] for i in range(max(0, len(peptide) - 6))}:
            token_index.setdefault(token, []).append(peptide)
    for members in token_index.values():
        for left, right in zip(members, members[1:]):
            union(left, right)

    wildcard_index: dict[tuple[int, int, str], list[str]] = {}
    for peptide in unique:
        for position in range(len(peptide)):
            key = (len(peptide), position, peptide[:position] + "_" + peptide[position + 1 :])
            wildcard_index.setdefault(key, []).append(peptide)
    for members in wildcard_index.values():
        for left, right in itertools.combinations(sorted(set(members)), 2):
            union(left, right)

    roots = {peptide: find(peptide) for peptide in unique}
    component_names: dict[str, str] = {}
    for root in set(roots.values()):
        members = sorted(peptide for peptide in unique if roots[peptide] == root)
        component_names[root] = "joint-" + hashlib.sha256(members[0].encode()).hexdigest()[:16]
    return pd.Series(
        [component_names[roots[peptide]] for peptide in peptides],
        index=frame.index,
        dtype="string",
    )


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("weirdo_tcell_evaluator", EVALUATOR)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _logit(values: pd.Series | np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), 1e-5, 1.0 - 1e-5)
    return np.log(values / (1.0 - values))


def _continuous_mhc_features(
    frame: pd.DataFrame, *, include_tcr_shape: bool = False
) -> tuple[np.ndarray, list[str]]:
    """Use score directions that are stable across HLA/cohort domains."""
    columns = [
        _logit(frame["mhcflurry_presentation_score"]),
        -np.log10(np.clip(frame["mhcflurry_affinity"].to_numpy(float), 1e-3, None)),
        _logit(frame["mhcflurry_processing_score"]),
    ]
    names = ["presentation_logit", "negative_log10_affinity", "processing_logit"]
    if include_tcr_shape:
        columns.extend(
            [
                frame["tcr_face_hydrophobicity"].fillna(0).to_numpy(float),
                frame["tcr_face_aromatic_fraction"].fillna(0).to_numpy(float),
            ]
        )
        names.extend(("tcr_face_hydrophobicity", "tcr_face_aromatic_fraction"))
    return np.column_stack(columns), names


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray):
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.25, max_iter=1000, random_state=42)
    )
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1], model[-1].coef_[0].tolist()


def _auc_if_defined(frame: pd.DataFrame, column: str, *, reverse: bool = False) -> float | None:
    """Return a raw score AUC, or ``None`` when a split has one class."""
    if column not in frame:
        return None
    usable = frame.loc[
        frame["outcome"].isin(("Positive", "Negative")) & frame[column].notna()
    ]
    if usable["outcome"].nunique() < 2:
        return None
    values = usable[column].to_numpy(dtype=float)
    if reverse:
        values = -values
    return float(roc_auc_score(usable["outcome"] == "Positive", values))


def _cluster_bootstrap(
    outcome: np.ndarray,
    baseline: np.ndarray,
    augmented: np.ndarray,
    groups: np.ndarray,
    draws: int,
    seed: int = 42,
) -> dict:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        selected = np.concatenate([members[group] for group in sampled])
        y = outcome[selected]
        if len(np.unique(y)) < 2:
            continue
        deltas.append(
            roc_auc_score(y, augmented[selected])
            - roc_auc_score(y, baseline[selected])
        )
    values = np.asarray(deltas, dtype=float)
    return {
        "draws": int(len(values)),
        "delta_auc_95ci": [float(x) for x in np.quantile(values, [0.025, 0.975])],
        "probability_delta_auc_positive": float(np.mean(values > 0)),
    }


def _matched_pair_auc(
    frame: pd.DataFrame, score: np.ndarray, caliper: float = 0.1
) -> dict:
    """Compare positive/negative pairs with the same HLA and matched presentation."""
    rows = frame.reset_index(drop=True)
    pairs = []
    for _, group in rows.groupby("hla"):
        positives = np.flatnonzero(group["outcome"].to_numpy() == "Positive")
        negatives = np.flatnonzero(group["outcome"].to_numpy() == "Negative")
        if not len(positives) or not len(negatives):
            continue
        positive_presentation = group.iloc[positives][
            "mhcflurry_presentation_percentile"
        ].to_numpy(float)
        negative_presentation = group.iloc[negatives][
            "mhcflurry_presentation_percentile"
        ].to_numpy(float)
        ii, jj = np.where(
            np.abs(positive_presentation[:, None] - negative_presentation[None, :])
            <= caliper
        )
        pairs.extend((group.index[positives[i]], group.index[negatives[j]]) for i, j in zip(ii, jj))
    if not pairs:
        return {"n_pairs": 0, "auc": None}
    comparisons = np.asarray(
        [
            1.0 if score[positive] > score[negative]
            else 0.5 if score[positive] == score[negative]
            else 0.0
            for positive, negative in pairs
        ]
    )
    return {"n_pairs": int(len(comparisons)), "auc": float(comparisons.mean())}


def evaluate(args: argparse.Namespace) -> dict:
    evaluator = _load_evaluator()
    evaluator.set_data_script(args.data_script)
    development = pd.read_csv(args.development)
    registered = pd.read_csv(args.validation)
    validation = registered.copy()
    if "role" in validation and "validation" in set(validation["role"]):
        validation = validation.loc[validation["role"] == "validation"].copy()
    validation = validation.loc[
        validation["outcome"].isin(("Positive", "Negative"))
        & validation["mhcflurry_presentation_score"].notna()
        & validation["mhcflurry_affinity"].notna()
        & validation["mhcflurry_processing_score"].notna()
    ].reset_index(drop=True)
    development = development.loc[
        development["outcome"].isin(("Positive", "Negative"))
        & development["mhcflurry_presentation_score"].notna()
        & development["mhcflurry_affinity"].notna()
        & development["mhcflurry_processing_score"].notna()
    ].reset_index(drop=True)

    all_rows = pd.concat((development, validation), ignore_index=True)
    joint_components = _joint_peptide_components(all_rows)
    development_components = joint_components.iloc[: len(development)].reset_index(drop=True)
    validation_components = joint_components.iloc[len(development) :].reset_index(drop=True)
    validation_clusters = set(validation_components.astype(str))
    before = len(development)
    development = development.loc[
        ~development_components.astype(str).isin(validation_clusters).to_numpy()
    ].reset_index(drop=True)
    validation = validation.reset_index(drop=True)
    validation["joint_peptide_cluster"] = validation_components.to_numpy()

    combined = pd.concat((development, validation), ignore_index=True)
    scored = evaluator.score_models(
        combined,
        [Path(item) for item in args.models.split(",") if item],
    )
    n_development = len(development)
    train_y = (development["outcome"] == "Positive").to_numpy(dtype=int)
    test_y = (validation["outcome"] == "Positive").to_numpy(dtype=int)
    train_mhc, mhc_names = _continuous_mhc_features(development)
    test_mhc, _ = _continuous_mhc_features(validation)
    train_mhc_tcr, mhc_tcr_names = _continuous_mhc_features(
        development, include_tcr_shape=True
    )
    test_mhc_tcr, _ = _continuous_mhc_features(validation, include_tcr_shape=True)
    groups = validation["joint_peptide_cluster"].astype(str).to_numpy()

    direction_audit = {}
    discovery = (
        registered.loc[registered["role"].eq("discovery")]
        if "role" in registered
        else registered.iloc[0:0]
    )
    for label, column, reverse in (
        ("presentation", "mhcflurry_presentation_score", False),
        ("affinity", "mhcflurry_affinity", True),
        ("processing", "mhcflurry_processing_score", False),
    ):
        direction_audit[label] = {
            "discovery_auc": _auc_if_defined(discovery, column, reverse=reverse),
            "validation_auc": _auc_if_defined(validation, column, reverse=reverse),
            "broad_development_auc": _auc_if_defined(
                development, column, reverse=reverse
            ),
        }

    results = []
    for model_dir in [Path(item) for item in args.models.split(",") if item]:
        config = json.loads((model_dir / "config.json").read_text())
        architecture = config.get("architecture", "composition_mlp")
        for content_label in (
            "virus_foreign_like",
            "p_viruses",
            "one_minus_mammals",
        ):
            content_name = f"{architecture}__{content_label}"
            content = scored[content_name].to_numpy(dtype=float)
            train_content = content[:n_development]
            test_content = content[n_development:]
            raw_auc = float(roc_auc_score(test_y, test_content))
            for base_label, train_base, test_base, base_names in (
                ("mhc_continuous", train_mhc, test_mhc, mhc_names),
                ("mhc_continuous_plus_tcr_shape", train_mhc_tcr, test_mhc_tcr, mhc_tcr_names),
            ):
                baseline_prediction, _ = _fit_predict(train_base, train_y, test_base)
                train_x = np.column_stack((train_base, train_content))
                test_x = np.column_stack((test_base, test_content))
                prediction, coefficients = _fit_predict(train_x, train_y, test_x)
                name = f"{base_label}_plus_{content_label}"
                baseline_auc = float(roc_auc_score(test_y, baseline_prediction))
                augmented_auc = float(roc_auc_score(test_y, prediction))
                result = {
                    "architecture": architecture,
                    "feature_set": name,
                    "content_signal": content_label,
                    "development_n": len(development),
                    "development_positive": int(train_y.sum()),
                    "validation_n": len(validation),
                    "validation_positive": int(test_y.sum()),
                    "raw_foreignness_auc": raw_auc,
                    "baseline_auc": baseline_auc,
                    "augmented_auc": augmented_auc,
                    "delta_auc": augmented_auc - baseline_auc,
                    "baseline_log_loss": float(log_loss(test_y, baseline_prediction, labels=[0, 1])),
                    "augmented_log_loss": float(log_loss(test_y, prediction, labels=[0, 1])),
                    "delta_log_loss": float(
                        log_loss(test_y, prediction, labels=[0, 1])
                        - log_loss(test_y, baseline_prediction, labels=[0, 1])
                    ),
                    "standardized_coefficients": dict(
                        zip(base_names + [content_label], coefficients)
                    ),
                    "cluster_bootstrap": _cluster_bootstrap(
                        test_y,
                        baseline_prediction,
                        prediction,
                        groups,
                        args.bootstrap_draws,
                    ),
                    "matched_presentation_pair": {
                        "caliper_percentile_points": 0.1,
                        "raw_content": _matched_pair_auc(
                            validation, test_content
                        ),
                        "baseline_prediction": _matched_pair_auc(
                            validation, baseline_prediction
                        ),
                        "augmented_prediction": _matched_pair_auc(
                            validation, prediction
                        ),
                    },
                }
                result["by_cohort"] = {
                    str(cohort): {
                        "n": int(len(part)),
                        "baseline_auc": float(
                            roc_auc_score(part["outcome"] == "Positive", part["baseline"])
                        ),
                        "augmented_auc": float(
                            roc_auc_score(part["outcome"] == "Positive", part["prediction"])
                        ),
                    }
                    for cohort, part in pd.DataFrame(
                        {
                            "cohort": validation["cohort"].to_numpy(),
                            "outcome": validation["outcome"].to_numpy(),
                            "baseline": baseline_prediction,
                            "prediction": prediction,
                        }
                    ).groupby("cohort")
                    if part["outcome"].nunique() > 1
                }
                results.append(result)

    report = {
        "development": str(args.development),
        "validation": str(args.validation),
        "excluded_development_rows": before - len(development),
        "excluded_validation_clusters": len(validation_clusters),
        "models": args.models,
        "protocol": "fit on broad development after joint exact/7-mer/Hamming peptide-component exclusion; score registered validation once",
        "direction_audit": direction_audit,
        "results": results,
    }
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    scored.iloc[:n_development].to_csv(output / "development-scored.csv", index=False)
    scored.iloc[n_development:].to_csv(output / "validation-scored.csv", index=False)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--development", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--data-script", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
