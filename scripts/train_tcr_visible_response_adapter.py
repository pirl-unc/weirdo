#!/usr/bin/env python3
"""Fit deployable domain adapters for the selected TCR-face/HLA model.

The fitted layer consumes a peptide, restricting HLA, MHCflurry 2.3
presentation score, and NetMHCpan BA rank. It deliberately excludes PRIME and
self-neighborhood features: the held-out feature-family sweep found the exact
HLA/TCR-face model to be the strongest specification with positive point
estimates in every maintained cohort.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_tcr_visible_nonself import (  # noqa: E402
    COHORTS,
    base_numeric,
    sequence_features,
)


MODEL_VERSION = "tcr-visible-response-adapter-v1"
SEQUENCE_REPRESENTATION = "allele_face_composition"
REGULARIZATION_C = 0.003


def design_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    scaler: StandardScaler | None = None,
    vectorizer: DictVectorizer | None = None,
) -> tuple[object, object, StandardScaler, DictVectorizer]:
    """Build the numeric MHC and sparse exact-HLA/TCR-face matrices."""
    train_numeric = base_numeric(train, include_prime=False)
    test_numeric = base_numeric(test, include_prime=False)
    if scaler is None:
        scaler = StandardScaler().fit(train_numeric)
    train_matrix = csr_matrix(scaler.transform(train_numeric))
    test_matrix = csr_matrix(scaler.transform(test_numeric))
    train_rows = [
        sequence_features(peptide, hla, SEQUENCE_REPRESENTATION)
        for peptide, hla in zip(train["peptide"], train["hla"])
    ]
    test_rows = [
        sequence_features(peptide, hla, SEQUENCE_REPRESENTATION)
        for peptide, hla in zip(test["peptide"], test["hla"])
    ]
    if vectorizer is None:
        vectorizer = DictVectorizer().fit(train_rows)
    return (
        hstack((train_matrix, vectorizer.transform(train_rows))).tocsr(),
        hstack((test_matrix, vectorizer.transform(test_rows))).tocsr(),
        scaler,
        vectorizer,
    )


def fit_adapter(frame: pd.DataFrame, domain: str) -> tuple[dict, dict]:
    """Fit one domain-specific response adapter and return its manifest."""
    domain_cohorts = [
        cohort for cohort, (_, candidate_domain) in COHORTS.items() if candidate_domain == domain
    ]
    training = frame.loc[frame["cohort"].isin(domain_cohorts)].copy().reset_index(
        drop=True
    )
    if training.empty or training["outcome"].nunique() != 2:
        raise ValueError(f"Domain {domain!r} does not contain both response classes")
    matrix, _, scaler, vectorizer = design_matrices(training, training)
    model = LogisticRegression(
        C=REGULARIZATION_C,
        class_weight="balanced",
        max_iter=4_000,
        random_state=84_342,
        solver="liblinear",
    ).fit(matrix, training["outcome"].to_numpy(int))
    prediction = model.predict_proba(matrix)[:, 1]
    bundle = {
        "model_version": MODEL_VERSION,
        "domain": domain,
        "sequence_representation": SEQUENCE_REPRESENTATION,
        "regularization_C": REGULARIZATION_C,
        "numeric_features": (
            "logit_mhcflurry_2_3_0_presentation_score",
            "negative_log10_netmhcpan_ba_rank",
        ),
        "scaler": scaler,
        "vectorizer": vectorizer,
        "model": model,
    }
    manifest = {
        "domain": domain,
        "cohorts": domain_cohorts,
        "n_rows": len(training),
        "n_positive": int(training["outcome"].sum()),
        "n_unique_peptides": int(training["peptide"].nunique()),
        "n_hla_alleles": int(training["hla"].nunique()),
        "training_roc_auc": float(roc_auc_score(training["outcome"], prediction)),
        "training_average_precision": float(
            average_precision_score(training["outcome"], prediction)
        ),
        "training_log_loss": float(log_loss(training["outcome"], prediction)),
        "note": (
            "Training metrics are descriptive only. Use the held-out cohort report "
            "for performance estimates."
        ),
    }
    return bundle, manifest


def predict_adapter(bundle: dict, frame: pd.DataFrame) -> np.ndarray:
    """Predict T-cell response probabilities with a fitted adapter bundle."""
    required = {
        "peptide",
        "hla",
        "mhcflurry_2_3_0_presentation_score",
        "netmhcpan_ba_rank",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Adapter input is missing columns: {missing}")
    numeric = bundle["scaler"].transform(base_numeric(frame, include_prime=False))
    rows = [
        sequence_features(peptide, hla, bundle["sequence_representation"])
        for peptide, hla in zip(frame["peptide"], frame["hla"])
    ]
    matrix = hstack(
        (csr_matrix(numeric), bundle["vectorizer"].transform(rows))
    ).tocsr()
    prediction = bundle["model"].predict_proba(matrix)[:, 1]
    if len(prediction) != len(frame) or not np.isfinite(prediction).all():
        raise ValueError("Adapter returned invalid predictions")
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.scores, low_memory=False)
    required = {
        "cohort",
        "outcome",
        "peptide",
        "hla",
        "mhcflurry_2_3_0_presentation_score",
        "netmhcpan_ba_rank",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Training scores are missing columns: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifests = []
    for domain in ("tumor", "viral"):
        bundle, manifest = fit_adapter(frame, domain)
        path = args.output / f"{domain}-adapter.joblib"
        joblib.dump(bundle, path)
        loaded = joblib.load(path)
        domain_frame = frame.loc[
            frame["cohort"].isin(manifest["cohorts"])
        ].reset_index(drop=True)
        prediction = predict_adapter(loaded, domain_frame)
        if not np.isfinite(prediction).all():
            raise ValueError(f"Reloaded {domain} adapter failed validation")
        manifest["artifact"] = path.name
        manifest["artifact_bytes"] = path.stat().st_size
        manifests.append(manifest)
        print(f"Wrote {domain} adapter to {path}", flush=True)
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "model_version": MODEL_VERSION,
                "required_input_columns": sorted(
                    required - {"cohort", "outcome"}
                ),
                "sequence_representation": SEQUENCE_REPRESENTATION,
                "regularization_C": REGULARIZATION_C,
                "adapters": manifests,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
