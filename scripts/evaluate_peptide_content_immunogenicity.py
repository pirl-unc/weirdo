#!/usr/bin/env python3
"""Test peptide-content scores for signal beyond MHC presentation.

The taxonomy models never see T-cell assay outcomes.  This script scores a
pre-existing HLA-resolved assay table and compares grouped out-of-fold response
models with and without each content score.  Peptide-overlap clusters stay in a
single fold, while HLA and cohort fixed effects accompany MHCflurry presentation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from weirdo.peptide_content import summarize_peptide_content


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_peptide_content_models.py"


def _load_training_helpers(path: str | Path):
    """Import the trainer once so model architecture matches its split builder."""
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(
        f"weirdo_peptide_content_training_{path.stat().st_mtime_ns}", path
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import training helpers from {path}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


training = _load_training_helpers(TRAIN_SCRIPT)

CATEGORIES = tuple(training.CATEGORIES)
SCORE_NAMES = (
    "microbial_foreign_like",
    "microbial_like",
    "virus_foreign_like",
    "one_minus_mammals",
)


def set_data_script(path: str | Path) -> None:
    """Match category dimensions and derived scores to one split builder."""
    global CATEGORIES
    training.set_data_module(training.load_data_module(path))
    CATEGORIES = tuple(training.CATEGORIES)
    if "viruses" not in CATEGORIES or "mammals" not in CATEGORIES:
        raise ValueError("Evaluation requires independent 'viruses' and 'mammals' heads.")


def derive_scores(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    """Return the named summaries used in the immunogenicity comparison."""
    probabilities = np.asarray(probabilities, dtype=float)
    indices = {name: index for index, name in enumerate(CATEGORIES)}
    microbial_categories = tuple(
        category
        for category in ("viruses", "bacteria", "archaea", "fungi", "protists")
        if category in indices
    )
    summary = summarize_peptide_content(
        probabilities, CATEGORIES, microbial_categories=microbial_categories
    )
    microbial = summary["microbial_like"]
    mammals = probabilities[:, indices["mammals"]]
    viruses = probabilities[:, indices["viruses"]]
    return {
        "microbial_foreign_like": summary["microbial_foreign_like"],
        "microbial_like": microbial,
        "virus_foreign_like": viruses * (1.0 - mammals),
        "one_minus_mammals": 1.0 - mammals,
    }


def _load_predictor(model_dir: Path):
    config = json.loads((model_dir / "config.json").read_text())
    model_categories = tuple(config.get("categories", CATEGORIES))
    if model_categories != CATEGORIES:
        raise ValueError(
            f"{model_dir} categories do not match --data-script: "
            f"{model_categories} != {CATEGORIES}"
        )
    architecture = config.get("architecture")
    if architecture is None and config.get("scorer_type") == "MLPScorer":
        architecture = "composition_mlp"
    if architecture == "composition_mlp":
        from weirdo.scorers import MLPScorer

        scorer = MLPScorer.load(model_dir)
        return architecture, scorer.predict_proba

    import torch

    if architecture.startswith("esm"):
        from transformers import AutoConfig, AutoTokenizer

        model_name = config["esm_model"]
        tokenizer_path = model_dir / "tokenizer"
        encoder_config_path = model_dir / "encoder_config"
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path.is_dir() else model_name
        )
        encoder_config = (
            AutoConfig.from_pretrained(encoder_config_path)
            if encoder_config_path.is_dir()
            else None
        )
        model = training.build_esm_model(
            len(CATEGORIES),
            model_name,
            freeze=architecture == "esm_frozen",
            encoder_config=encoder_config,
        )
    else:
        tokenizer = None
        model = training.build_torch_model(architecture, len(CATEGORIES))
    model.load_state_dict(torch.load(model_dir / "weights.pt", map_location="cpu"))
    if architecture.startswith("esm"):
        if not tokenizer_path.is_dir():
            tokenizer.save_pretrained(tokenizer_path)
        if not encoder_config_path.is_dir():
            model.encoder.config.save_pretrained(encoder_config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def predict(peptides):
        labels = np.zeros((len(peptides), len(CATEGORIES)), dtype=np.float32)
        batch_size = 1024 if architecture.startswith("esm") else 4096
        loader = training.make_loader(
            # This module loads the training helpers dynamically, so spawned
            # workers cannot import their synthetic module name reliably.
            list(peptides), labels, architecture, tokenizer, batch_size, workers=0
        )
        probabilities, _, _ = training.predict_torch(model, loader, device)
        return probabilities

    return architecture, predict


def score_models(frame: pd.DataFrame, models: list[Path]) -> pd.DataFrame:
    peptides = sorted(frame["peptide"].dropna().unique())
    peptide_index = {peptide: index for index, peptide in enumerate(peptides)}
    rows = np.asarray([peptide_index[peptide] for peptide in frame["peptide"]])
    result = frame.copy()
    for model_dir in models:
        architecture, predict = _load_predictor(model_dir)
        probabilities = np.asarray(predict(peptides), dtype=float)
        if probabilities.shape != (len(peptides), len(CATEGORIES)):
            raise ValueError(
                f"{architecture} returned {probabilities.shape}; expected "
                f"{(len(peptides), len(CATEGORIES))}"
            )
        for index, category in enumerate(CATEGORIES):
            result[f"{architecture}__p_{category}"] = probabilities[rows, index]
        for name, values in derive_scores(probabilities).items():
            result[f"{architecture}__{name}"] = values[rows]
    return result


def _pool_rare(values: pd.Series, minimum: int = 20) -> pd.Series:
    text = values.fillna("__missing__").astype(str)
    counts = text.value_counts()
    return text.where(text.map(counts) >= minimum, "__pooled__")


def design_matrix(
    frame: pd.DataFrame,
    include_source_species: bool,
    mhc_feature: str = "presentation",
) -> np.ndarray:
    if mhc_feature == "presentation":
        presentation = np.clip(
            frame["mhcflurry_presentation_score"].to_numpy(dtype=float),
            1e-6,
            1 - 1e-6,
        )
        mhc_value = np.log(presentation / (1.0 - presentation))
    elif mhc_feature == "affinity":
        affinity = np.clip(
            frame["mhcflurry_affinity"].to_numpy(dtype=float), 1e-3, None
        )
        mhc_value = -np.log10(affinity)
    else:
        raise ValueError("mhc_feature must be 'presentation' or 'affinity'")
    columns = [mhc_value[:, None]]
    categorical = ["hla", "cohort"]
    if include_source_species and "source_species" in frame:
        categorical.append("source_species")
    for name in categorical:
        dummies = pd.get_dummies(_pool_rare(frame[name]), drop_first=True, dtype=float)
        if dummies.shape[1]:
            columns.append(dummies.to_numpy())
    return np.hstack(columns)


def _positive_probability(model, matrix: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(matrix)
    position = int(np.flatnonzero(model[-1].classes_ == 1)[0])
    return probabilities[:, position]


def _oof_predictions(
    matrix: np.ndarray,
    outcome: np.ndarray,
    groups: np.ndarray,
    *,
    seeds: tuple[int, ...],
    last_feature: bool,
) -> tuple[np.ndarray, list[float]]:
    predictions = np.zeros((len(seeds), len(outcome)), dtype=float)
    coefficients: list[float] = []
    for repeat, seed in enumerate(seeds):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        seen = np.zeros(len(outcome), dtype=bool)
        for train, test in splitter.split(matrix, outcome, groups):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.25, max_iter=1000, random_state=seed),
            )
            model.fit(matrix[train], outcome[train])
            predictions[repeat, test] = _positive_probability(model, matrix[test])
            seen[test] = True
            if last_feature:
                coefficients.append(float(model[-1].coef_[0, -1]))
        if not seen.all():
            raise RuntimeError("Cross-validation did not score every assay")
    return predictions.mean(axis=0), coefficients


def _cluster_bootstrap(
    outcome: np.ndarray,
    baseline: np.ndarray,
    augmented: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> dict:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.RandomState(seed)
    delta_auc = []
    delta_loss = []
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        selected = np.concatenate([members[group] for group in sampled])
        y = outcome[selected]
        if len(np.unique(y)) < 2:
            continue
        delta_auc.append(
            roc_auc_score(y, augmented[selected]) - roc_auc_score(y, baseline[selected])
        )
        delta_loss.append(
            log_loss(y, augmented[selected], labels=[0, 1])
            - log_loss(y, baseline[selected], labels=[0, 1])
        )

    def interval(values):
        values = np.asarray(values, dtype=float)
        return [float(x) for x in np.quantile(values, [0.025, 0.975])]

    return {
        "draws": len(delta_auc),
        "delta_auc_95ci": interval(delta_auc),
        "delta_log_loss_95ci": interval(delta_loss),
        "probability_delta_auc_positive": float(np.mean(np.asarray(delta_auc) > 0)),
        "probability_delta_log_loss_negative": float(
            np.mean(np.asarray(delta_loss) < 0)
        ),
    }


def evaluate_incremental(
    frame: pd.DataFrame,
    architectures: list[str],
    *,
    ablation_architecture: str | None = "esm_finetune",
    mhc_feature: str = "presentation",
    seeds: tuple[int, ...] = (11, 29, 47, 71, 101),
    bootstrap_draws: int = 1000,
) -> list[dict]:
    results = []
    domains = sorted(frame["domain"].dropna().unique()) if "domain" in frame else ["all"]
    for domain in domains:
        domain_frame = (
            frame.loc[frame["domain"] == domain].reset_index(drop=True)
            if domain != "all"
            else frame.reset_index(drop=True)
        )
        analysis_sets = [("all", domain_frame)]
        if "role" in domain_frame and domain_frame["role"].nunique() > 1:
            analysis_sets.extend(
                (str(role), part.reset_index(drop=True))
                for role, part in domain_frame.groupby("role")
            )
        for analysis_set, subset in analysis_sets:
            outcome = (subset["outcome"] == "Positive").to_numpy(dtype=int)
            if outcome.sum() < 10 or (1 - outcome).sum() < 10:
                continue
            groups = (
                subset.get("peptide_cluster", subset["peptide"])
                .fillna(subset["peptide"])
                .astype(str)
                .to_numpy()
            )
            variants = [False]
            if domain in {"viral", "all"} and "source_species" in subset:
                variants.append(True)
            for include_source_species in variants:
                results.extend(
                    _evaluate_design(
                        subset,
                        outcome,
                        groups,
                        architectures,
                        domain=domain,
                        analysis_set=analysis_set,
                        include_source_species=include_source_species,
                        ablation_architecture=ablation_architecture,
                        mhc_feature=mhc_feature,
                        seeds=seeds,
                        bootstrap_draws=bootstrap_draws,
                    )
                )
    return results


def _evaluate_design(
    subset: pd.DataFrame,
    outcome: np.ndarray,
    groups: np.ndarray,
    architectures: list[str],
    *,
    domain: str,
    analysis_set: str,
    include_source_species: bool,
    ablation_architecture: str | None,
    mhc_feature: str,
    seeds: tuple[int, ...],
    bootstrap_draws: int,
) -> list[dict]:
    results = []
    baseline_matrix = design_matrix(subset, include_source_species, mhc_feature)
    baseline, _ = _oof_predictions(
        baseline_matrix, outcome, groups, seeds=seeds, last_feature=False
    )
    baseline_auc = float(roc_auc_score(outcome, baseline))
    baseline_loss = float(log_loss(outcome, baseline, labels=[0, 1]))
    for architecture in architectures:
        score_names = (
            SCORE_NAMES
            if architecture == ablation_architecture
            else ("microbial_foreign_like",)
        )
        for score_name in score_names:
            column = f"{architecture}__{score_name}"
            values = subset[column].to_numpy(dtype=float)[:, None]
            augmented, coefficients = _oof_predictions(
                np.hstack((baseline_matrix, values)),
                outcome,
                groups,
                seeds=seeds,
                last_feature=True,
            )
            augmented_auc = float(roc_auc_score(outcome, augmented))
            augmented_loss = float(log_loss(outcome, augmented, labels=[0, 1]))
            bootstrap = _cluster_bootstrap(
                outcome,
                baseline,
                augmented,
                groups,
                draws=bootstrap_draws,
                seed=seeds[0],
            )
            results.append(
                {
                    "domain": domain,
                    "analysis_set": analysis_set,
                    "baseline": (
                        f"MHCflurry {mhc_feature} + HLA + cohort + source species"
                        if include_source_species
                        else f"MHCflurry {mhc_feature} + HLA + cohort"
                    ),
                    "architecture": architecture,
                    "score": score_name,
                    "n_assays": len(subset),
                    "n_positive": int(outcome.sum()),
                    "n_clusters": int(len(np.unique(groups))),
                    "cv_repeats": len(seeds),
                    "baseline_auc": baseline_auc,
                    "augmented_auc": augmented_auc,
                    "delta_auc": augmented_auc - baseline_auc,
                    "baseline_log_loss": baseline_loss,
                    "augmented_log_loss": augmented_loss,
                    "delta_log_loss": augmented_loss - baseline_loss,
                    "mean_standardized_score_coefficient": float(
                        np.mean(coefficients)
                    ),
                    "positive_coefficient_folds": int(
                        np.sum(np.asarray(coefficients) > 0)
                    ),
                    "n_coefficient_folds": len(coefficients),
                    "cluster_bootstrap": bootstrap,
                }
            )
    return results


def _transfer_predictions(
    train_matrix: np.ndarray,
    train_outcome: np.ndarray,
    test_matrix: np.ndarray,
    *,
    last_feature: bool,
) -> tuple[np.ndarray, float | None]:
    """Fit only discovery outcomes and return untouched validation predictions."""
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.25, max_iter=1000, random_state=42)
    )
    model.fit(train_matrix, train_outcome)
    coefficient = float(model[-1].coef_[0, -1]) if last_feature else None
    return _positive_probability(model, test_matrix), coefficient


def evaluate_registered_transfer(
    frame: pd.DataFrame,
    architectures: list[str],
    *,
    ablation_architecture: str | None = "esm_finetune",
    mhc_feature: str = "presentation",
    bootstrap_draws: int = 1000,
) -> list[dict]:
    """Use discovery outcomes once, then evaluate on the registered viral holdout.

    This is intentionally separate from the grouped-CV analysis.  It answers
    the prospective question: does a taxonomy score learned without outcomes
    improve an MHC baseline when its response coefficient is fitted on the
    discovery cohorts and frozen before scoring the validation cohorts?
    """
    if "role" not in frame or not {"discovery", "validation"} <= set(frame["role"]):
        return []
    discovery = frame.loc[frame["role"] == "discovery"].reset_index(drop=True)
    validation = frame.loc[frame["role"] == "validation"].reset_index(drop=True)
    # A response model must not receive the same peptide in discovery and
    # validation, even if the source CSV later acquires a duplicate row.
    seen_peptides = set(discovery["peptide"].astype(str))
    validation = validation.loc[
        ~validation["peptide"].astype(str).isin(seen_peptides)
    ].reset_index(drop=True)
    train_outcome = (discovery["outcome"] == "Positive").to_numpy(dtype=int)
    outcome = (validation["outcome"] == "Positive").to_numpy(dtype=int)
    if min(train_outcome.sum(), len(train_outcome) - train_outcome.sum()) < 10:
        return []
    if min(outcome.sum(), len(outcome) - outcome.sum()) < 10:
        return []

    combined = pd.concat((discovery, validation), ignore_index=True)
    baseline_all = design_matrix(combined, False, mhc_feature)
    baseline_train = baseline_all[: len(discovery)]
    baseline_test = baseline_all[len(discovery) :]
    baseline, _ = _transfer_predictions(
        baseline_train, train_outcome, baseline_test, last_feature=False
    )
    groups = (
        validation.get("peptide_cluster", validation["peptide"])
        .fillna(validation["peptide"])
        .astype(str)
        .to_numpy()
    )
    results = []
    for architecture in architectures:
        score_names = (
            SCORE_NAMES
            if architecture == ablation_architecture
            else ("microbial_foreign_like",)
        )
        for score_name in score_names:
            column = f"{architecture}__{score_name}"
            values = combined[column].to_numpy(dtype=float)[:, None]
            augmented, coefficient = _transfer_predictions(
                np.hstack((baseline_train, values[: len(discovery)])),
                train_outcome,
                np.hstack((baseline_test, values[len(discovery) :])),
                last_feature=True,
            )
            bootstrap = _cluster_bootstrap(
                outcome,
                baseline,
                augmented,
                groups,
                draws=bootstrap_draws,
                seed=42,
            )
            results.append(
                {
                    "domain": "viral",
                    "analysis_set": "validation",
                    "evaluation_protocol": "discovery_to_registered_validation",
                    "baseline": f"MHCflurry {mhc_feature} + HLA + cohort",
                    "architecture": architecture,
                    "score": score_name,
                    "n_discovery_assays": len(discovery),
                    "n_discovery_positive": int(train_outcome.sum()),
                    "n_assays": len(validation),
                    "n_positive": int(outcome.sum()),
                    "n_clusters": int(len(np.unique(groups))),
                    "dropped_validation_peptide_overlap": int(
                        (frame["role"] == "validation").sum() - len(validation)
                    ),
                    "baseline_auc": float(roc_auc_score(outcome, baseline)),
                    "augmented_auc": float(roc_auc_score(outcome, augmented)),
                    "delta_auc": float(roc_auc_score(outcome, augmented) - roc_auc_score(outcome, baseline)),
                    "baseline_log_loss": float(log_loss(outcome, baseline, labels=[0, 1])),
                    "augmented_log_loss": float(log_loss(outcome, augmented, labels=[0, 1])),
                    "delta_log_loss": float(
                        log_loss(outcome, augmented, labels=[0, 1])
                        - log_loss(outcome, baseline, labels=[0, 1])
                    ),
                    "standardized_discovery_score_coefficient": coefficient,
                    "cluster_bootstrap": bootstrap,
                }
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", required=True)
    parser.add_argument("--models", required=True, help="comma-separated model dirs")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--data-script",
        default=str(training.DATA_SCRIPT),
        help="split-builder module that defines the trained model's categories",
    )
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument(
        "--mhc-feature", choices=("presentation", "affinity"), default="presentation"
    )
    parser.add_argument(
        "--ablation-architecture",
        default="esm_finetune",
        help="architecture for which to evaluate score components as well as the primary score",
    )
    args = parser.parse_args()

    set_data_script(args.data_script)

    frame = pd.read_csv(args.input)
    mhc_column = (
        "mhcflurry_presentation_score"
        if args.mhc_feature == "presentation"
        else "mhcflurry_affinity"
    )
    frame = frame.loc[
        frame["peptide"].astype(str).str.len().between(8, 15)
        & frame[mhc_column].notna()
        & frame["outcome"].isin(("Positive", "Negative"))
    ].reset_index(drop=True)
    model_dirs = [Path(item) for item in args.models.split(",") if item]
    scored = score_models(frame, model_dirs)
    architectures = [
        json.loads((path / "config.json").read_text()).get("architecture", "composition_mlp")
        for path in model_dirs
    ]
    results = evaluate_incremental(
        scored,
        architectures,
        ablation_architecture=args.ablation_architecture,
        mhc_feature=args.mhc_feature,
        bootstrap_draws=args.bootstrap_draws,
    )
    registered_transfer = evaluate_registered_transfer(
        scored,
        architectures,
        ablation_architecture=args.ablation_architecture,
        mhc_feature=args.mhc_feature,
        bootstrap_draws=args.bootstrap_draws,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output / "scored-assays.csv", index=False)
    primary = [
        row
        for row in results
        if row["score"] == "microbial_foreign_like"
        and "source species" not in row["baseline"]
    ]
    report = {
        "input": str(args.input),
        "models": [str(path) for path in model_dirs],
        "mhc_feature": args.mhc_feature,
        "score_definitions": {
            "microbial_like": "max of available microbial-category heads",
            "microbial_foreign_like": "microbial_like * (1 - p_mammals)",
            "virus_foreign_like": "p_viruses * (1 - p_mammals)",
            "one_minus_mammals": "1 - p_mammals",
        },
        "guardrails": [
            "taxonomy models were trained without T-cell assay outcomes",
            "peptide-overlap clusters never cross response-model folds",
            "registered transfer fits the response coefficient on discovery only and scores validation once",
            "architecture selection must use taxonomy metrics, not this report",
            "viral source organisms may occur in SwissProt training; source-species-adjusted results are reported as a sensitivity analysis",
        ],
        "primary": primary,
        "all_results": results,
        "registered_validation_transfer": registered_transfer,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"primary": primary}, indent=2))


if __name__ == "__main__":
    main()
