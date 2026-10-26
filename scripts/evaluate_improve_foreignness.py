#!/usr/bin/env python3
"""Evaluate peptide-content foreignness scores on the IMPROVE neoepitope cohort.

The response model is evaluated out of fold with patient-grouped splits.  Any
training row whose exact mutant peptide appears in the test fold is purged, so
neither a patient nor an exact peptide can leak into its fold's training data.
Taxonomy models are frozen and never see the IMPROVE response labels.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from weirdo.peptide_content import summarize_peptide_content


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "scripts" / "evaluate_peptide_content_immunogenicity.py"
CANONICAL_PEPTIDE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{8,15}$")
DEFAULT_EXISTING_TABLES = (
    ROOT / "data" / "viral-tcell-benchmark" / "scored-assays.csv",
    ROOT / "data" / "foreignness-choices" / "dev-scored-assays.csv",
    ROOT / "data" / "foreignness-choices" / "dev2-scored-assays.csv",
    ROOT / "data" / "foreignness-choices" / "dev3-scored-assays.csv",
)
DEFAULT_EXISTING_JSON = (
    ROOT / "data" / "viral-tcell-benchmark" / "iedb-31511069-page-000.json",
    ROOT / "data" / "viral-tcell-benchmark" / "iedb-33038342-page-000.json",
)
DEFAULT_PUBLISHED_PREDICTIONS = (
    ROOT / "data" / "improve-validation" / "neoepitopes_predictions.tsv"
)

PUBLISHED_COLUMN_SPECS = (
    ("published_iedb_calis", "IEDB immunogenicity", 1.0),
    ("published_deepnetbim_binding", "DeepNetBim binding", 1.0),
    ("published_deepnetbim_immunogenicity", "DeepNetBim immunogenicity", 1.0),
    (
        "published_deepnetbim_probability",
        "DeepNetBim immunogenicity probability",
        1.0,
    ),
    ("published_deepimmuno", "DeepImmuno", 1.0),
    ("published_deephlapan_binding", "DeepHLApan binding score", 1.0),
    (
        "published_deephlapan_immunogenicity",
        "DeepHLApan immunogenic score",
        1.0,
    ),
    ("published_prime_rank", "PRIME rank", -1.0),
    ("published_prime_score", "PRIME score", 1.0),
    ("published_mixmhcpred", "MixMHCpred", -1.0),
    ("published_ittca_rf", "iTTCA-RF", 1.0),
    ("published_ittca_rf_probability", "iTTCA-RF probability", 1.0),
    ("published_mhcflurry_affinity", "MHCflurry MHC affinity", -1.0),
    ("published_mhcflurry_processing", "MHCflurry processing", 1.0),
    ("published_mhcflurry_presentation", "MHCflurry presentation", 1.0),
    ("published_hlaathena_msic", "HLAthena MSiC", -1.0),
    ("published_hlaathena_msie", "HLAthena MSiE", -1.0),
    ("published_hlaathena_msice", "HLAthena MSiCE", -1.0),
)
PUBLISHED_COMPLETE_SCORES = tuple(
    name
    for name, source, _ in PUBLISHED_COLUMN_SPECS
    if not source.startswith("DeepNetBim") and source != "HLAthena MSiCE"
)
PUBLISHED_DEEPNETBIM_SCORES = tuple(
    name for name, source, _ in PUBLISHED_COLUMN_SPECS if source.startswith("DeepNetBim")
)


def _load_evaluator():
    spec = importlib.util.spec_from_file_location(
        "improve_peptide_content_evaluator", EVALUATOR
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical(value: object) -> str | None:
    text = str(value or "").strip().upper()
    return text if CANONICAL_PEPTIDE.fullmatch(text) else None


def load_existing_peptides() -> set[str]:
    """Return peptides already exposed in the project's response benchmarks."""
    peptides: set[str] = set()
    for path in DEFAULT_EXISTING_TABLES:
        if not path.is_file():
            continue
        frame = pd.read_csv(path, usecols=["peptide"])
        peptides.update(
            peptide
            for peptide in (_canonical(value) for value in frame["peptide"])
            if peptide is not None
        )
    for path in DEFAULT_EXISTING_JSON:
        if not path.is_file():
            continue
        for row in json.loads(path.read_text()):
            peptide = _canonical(row.get("assay_antigen__name"))
            if peptide is not None:
                peptides.add(peptide)
    return peptides


def load_improve(path: str | Path) -> pd.DataFrame:
    """Load and validate the unmodified IMPROVE feature table."""
    frame = pd.read_csv(path, sep="\t", low_memory=False)
    required = {
        "Patient",
        "HLA_allele",
        "Norm_peptide",
        "Mut_peptide",
        "RankEL_4.1",
        "RankBA_4.1",
        "PeptLen",
        "HydroCore",
        "PropAro",
        "cohort",
        "response",
        "SelfSim",
        "Foreigness",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"IMPROVE input is missing columns: {missing}")
    frame = frame.copy()
    frame["Mut_peptide"] = frame["Mut_peptide"].map(_canonical)
    frame["Norm_peptide"] = frame["Norm_peptide"].map(_canonical)
    if frame["Mut_peptide"].isna().any():
        raise ValueError("IMPROVE input contains a non-canonical mutant peptide")
    frame["response"] = frame["response"].replace({"yes": 1, "no": 0})
    frame["response"] = pd.to_numeric(frame["response"], errors="raise").astype(int)
    if not set(frame["response"]) <= {0, 1}:
        raise ValueError("IMPROVE response must be binary")
    if frame.duplicated(["Patient", "HLA_allele", "Mut_peptide"]).any():
        raise ValueError("IMPROVE input contains duplicate patient/HLA/peptide rows")
    return frame


def attach_published_predictions(
    frame: pd.DataFrame, path: str | Path
) -> pd.DataFrame:
    """Join the IMPROVE authors' frozen public tool predictions to source rows."""
    published = pd.read_csv(path, sep="\t", low_memory=False)
    required = {
        "HLA_allele",
        "Norm_peptide",
        "Mut_peptide",
        "Expression",
        "response",
        *(source for _, source, _ in PUBLISHED_COLUMN_SPECS),
    }
    missing = sorted(required - set(published.columns))
    if missing:
        raise ValueError(f"Published prediction table is missing columns: {missing}")
    published_names = [name for name, _, _ in PUBLISHED_COLUMN_SPECS]
    result = frame.drop(
        columns=[name for name in published_names if name in frame], errors="ignore"
    ).copy()
    if "Expression" not in result:
        raise ValueError("Expression is required to match published predictions")
    result["_expression_3"] = pd.to_numeric(
        result["Expression"], errors="raise"
    ).round(3)
    published["_expression_3"] = pd.to_numeric(
        published["Expression"], errors="raise"
    ).round(3)
    keys = [
        "HLA_allele",
        "Norm_peptide",
        "Mut_peptide",
        "response",
        "_expression_3",
    ]
    selected = published[keys].copy()
    for name, source, direction in PUBLISHED_COLUMN_SPECS:
        selected[name] = pd.to_numeric(published[source], errors="coerce") * direction
    if selected.duplicated(keys).any():
        raise ValueError("Published prediction join keys are not unique")
    result["_source_order"] = np.arange(len(result))
    result = result.merge(selected, on=keys, how="left", validate="one_to_one")
    result = result.sort_values("_source_order").drop(
        columns=["_source_order", "_expression_3"]
    )
    if result[list(PUBLISHED_COMPLETE_SCORES)].isna().any().any():
        raise ValueError("A full-coverage published predictor failed to match")
    return result.reset_index(drop=True)


def patient_groups(frame: pd.DataFrame) -> np.ndarray:
    """Return stable integer group labels for patient-level validation."""
    labels, _ = pd.factorize(frame["Patient"].astype(str), sort=True)
    return labels


def purge_shared_test_peptides(
    train: np.ndarray, test: np.ndarray, peptides: np.ndarray
) -> np.ndarray:
    """Remove training rows whose exact peptide occurs in the held-out fold."""
    test_peptides = set(peptides[test])
    keep = np.fromiter(
        (peptide not in test_peptides for peptide in peptides[train]),
        dtype=bool,
        count=len(train),
    )
    return train[keep]


def _content_scores(
    probabilities: np.ndarray, categories: tuple[str, ...]
) -> dict[str, np.ndarray]:
    indices = {name: index for index, name in enumerate(categories)}
    required = {"viruses", "bacteria", "mammals", "human"}
    missing = sorted(required - set(indices))
    if missing:
        raise ValueError(f"Taxonomy model is missing heads: {missing}")
    microbial_categories = tuple(
        category
        for category in ("viruses", "bacteria", "archaea", "fungi", "protists")
        if category in indices
    )
    summary = summarize_peptide_content(
        probabilities, categories, microbial_categories=microbial_categories
    )
    viruses = probabilities[:, indices["viruses"]]
    bacteria = probabilities[:, indices["bacteria"]]
    mammals = probabilities[:, indices["mammals"]]
    human = probabilities[:, indices["human"]]
    virus_bacteria_max = np.maximum(viruses, bacteria)
    scores = {
        "p_viruses": viruses,
        "p_bacteria": bacteria,
        "virus_bacteria_max": virus_bacteria_max,
        "microbial_like": summary["microbial_like"],
        "one_minus_mammals": 1.0 - mammals,
        "one_minus_human": 1.0 - human,
        "virus_vs_mammals": viruses * (1.0 - mammals),
        "virus_vs_human": viruses * (1.0 - human),
        "bacteria_vs_mammals": bacteria * (1.0 - mammals),
        "bacteria_vs_human": bacteria * (1.0 - human),
        "virus_bacteria_vs_mammals": virus_bacteria_max * (1.0 - mammals),
        "virus_bacteria_vs_human": virus_bacteria_max * (1.0 - human),
        "microbial_foreign_like": summary["microbial_foreign_like"],
    }
    host_heads = {"mammal_host_viruses", "other_host_viruses"}
    if host_heads <= set(indices):
        mammal_host = probabilities[:, indices["mammal_host_viruses"]]
        other_host = probabilities[:, indices["other_host_viruses"]]
        scores.update(
            {
                "p_mammal_host_viruses": mammal_host,
                "p_other_host_viruses": other_host,
                "host_virus_max": np.maximum(mammal_host, other_host),
                "mammal_host_virus_margin": mammal_host - other_host,
                "mammal_host_virus_specific": mammal_host * (1.0 - other_host),
                "mammal_host_virus_vs_mammals": mammal_host * (1.0 - mammals),
                "mammal_host_virus_vs_human": mammal_host * (1.0 - human),
            }
        )
    return scores


def score_taxonomy(
    frame: pd.DataFrame, model_dir: str | Path, evaluator
) -> tuple[pd.DataFrame, str]:
    """Score unique mutant and available WT peptides with one frozen model."""
    mutant = frame["Mut_peptide"].astype(str)
    wild_type = frame["Norm_peptide"].dropna().astype(str)
    peptides = sorted(set(mutant) | set(wild_type))
    architecture, predict = evaluator._load_predictor(Path(model_dir))
    probabilities = np.asarray(predict(peptides), dtype=float)
    categories = tuple(evaluator.CATEGORIES)
    if probabilities.shape != (len(peptides), len(categories)):
        raise ValueError(
            f"{architecture} returned {probabilities.shape}; expected "
            f"{(len(peptides), len(categories))}"
        )
    scores = _content_scores(probabilities, categories)
    peptide_index = {peptide: index for index, peptide in enumerate(peptides)}
    result = frame.copy()
    mutant_rows = mutant.map(peptide_index).to_numpy(int)
    wild_rows = frame["Norm_peptide"].map(peptide_index)
    for name, values in scores.items():
        result[name] = values[mutant_rows]
        wt_values = np.full(len(result), np.nan)
        available = wild_rows.notna().to_numpy()
        wt_values[available] = values[wild_rows[available].to_numpy(int)]
        result[f"wt_{name}"] = wt_values
        result[f"delta_{name}"] = result[name] - wt_values
    result["improve_foreignness"] = pd.to_numeric(
        result["Foreigness"], errors="raise"
    )
    result["inverse_self_similarity"] = -pd.to_numeric(
        result["SelfSim"], errors="raise"
    )
    return result, architecture


def design_matrix(frame: pd.DataFrame, *, include_tcr_chemistry: bool) -> np.ndarray:
    """Build a count-free MHC baseline comparable to the prior viral audit."""
    rank_el = np.clip(frame["RankEL_4.1"].to_numpy(float), 1e-5, None)
    rank_ba = np.clip(frame["RankBA_4.1"].to_numpy(float), 1e-5, None)
    columns = [
        -np.log10(rank_el)[:, None],
        -np.log10(rank_ba)[:, None],
        frame["PeptLen"].to_numpy(float)[:, None],
    ]
    if include_tcr_chemistry:
        columns.extend(
            (
                frame["HydroCore"].to_numpy(float)[:, None],
                frame["PropAro"].to_numpy(float)[:, None],
            )
        )
    for name in ("HLA_allele", "cohort"):
        dummies = pd.get_dummies(frame[name].astype(str), drop_first=True, dtype=float)
        if dummies.shape[1]:
            columns.append(dummies.to_numpy())
    return np.hstack(columns)


def _positive_probability(model, matrix: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(matrix)
    position = int(np.flatnonzero(model[-1].classes_ == 1)[0])
    return probabilities[:, position]


def make_purged_splits(
    outcome: np.ndarray,
    groups: np.ndarray,
    peptides: np.ndarray,
    seeds: tuple[int, ...],
) -> list[tuple[int, list[tuple[np.ndarray, np.ndarray]]]]:
    """Create each repeated patient split and purge peptide overlaps once."""
    repeats = []
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = []
        for train, test in splitter.split(peptides, outcome, groups):
            folds.append((purge_shared_test_peptides(train, test, peptides), test))
        repeats.append((seed, folds))
    return repeats


def oof_predictions(
    matrix: np.ndarray,
    outcome: np.ndarray,
    splits: list[tuple[int, list[tuple[np.ndarray, np.ndarray]]]],
    *,
    last_feature: bool,
) -> tuple[np.ndarray, list[float]]:
    predictions = np.zeros((len(splits), len(outcome)), dtype=float)
    coefficients: list[float] = []
    for repeat, (seed, folds) in enumerate(splits):
        seen = np.zeros(len(outcome), dtype=bool)
        for train, test in folds:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=0.25, max_iter=2000, random_state=seed),
            )
            model.fit(matrix[train], outcome[train])
            predictions[repeat, test] = _positive_probability(model, matrix[test])
            seen[test] = True
            if last_feature:
                coefficients.append(float(model[-1].coef_[0, -1]))
        if not seen.all():
            raise RuntimeError("Cross-validation did not score every IMPROVE row")
    return predictions.mean(axis=0), coefficients


def _metrics(outcome: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(outcome, prediction)),
        "pr_auc": float(average_precision_score(outcome, prediction)),
        "log_loss": float(log_loss(outcome, prediction, labels=[0, 1])),
    }


def _ranking_metrics(outcome: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Metrics that are valid for an arbitrary, uncalibrated ranking score."""
    return {
        "roc_auc": float(roc_auc_score(outcome, score)),
        "pr_auc": float(average_precision_score(outcome, score)),
    }


def bootstrap_auc_reference(
    outcome: np.ndarray,
    baseline: np.ndarray,
    groups: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    unique = np.unique(groups)
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.RandomState(seed)
    samples = []
    baseline_auc = []
    for _ in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        selected = np.concatenate([members[group] for group in sampled])
        y = outcome[selected]
        if len(np.unique(y)) < 2:
            continue
        samples.append(selected)
        baseline_auc.append(roc_auc_score(y, baseline[selected]))
    return samples, np.asarray(baseline_auc, dtype=float)


def cluster_bootstrap_auc(
    outcome: np.ndarray,
    augmented: np.ndarray,
    samples: list[np.ndarray],
    baseline_auc: np.ndarray,
) -> dict:
    augmented_auc = np.asarray(
        [roc_auc_score(outcome[index], augmented[index]) for index in samples]
    )
    delta = augmented_auc - baseline_auc

    return {
        "draws": int(len(delta)),
        "delta_roc_auc_95ci": [
            float(value) for value in np.quantile(delta, [0.025, 0.975])
        ],
        "probability_delta_roc_auc_positive": float(np.mean(delta > 0)),
    }


def evaluate_score_set(
    frame: pd.DataFrame,
    score_names: tuple[str, ...],
    *,
    analysis_set: str,
    include_tcr_chemistry: bool,
    seeds: tuple[int, ...],
    bootstrap_draws: int,
) -> list[dict]:
    outcome = frame["response"].to_numpy(int)
    groups = frame["cv_patient_group"].to_numpy(int)
    peptides = frame["Mut_peptide"].astype(str).to_numpy()
    splits = make_purged_splits(outcome, groups, peptides, seeds)
    baseline_matrix = design_matrix(frame, include_tcr_chemistry=include_tcr_chemistry)
    baseline, _ = oof_predictions(
        baseline_matrix,
        outcome,
        splits,
        last_feature=False,
    )
    baseline_metrics = _metrics(outcome, baseline)
    bootstrap_samples, bootstrap_baseline_auc = bootstrap_auc_reference(
        outcome,
        baseline,
        groups,
        draws=bootstrap_draws,
        seed=seeds[0],
    )
    results = []
    for score_name in score_names:
        score = frame[score_name].to_numpy(float)
        augmented, coefficients = oof_predictions(
            np.column_stack((baseline_matrix, score)),
            outcome,
            splits,
            last_feature=True,
        )
        augmented_metrics = _metrics(outcome, augmented)
        raw_metrics = _ranking_metrics(outcome, score)
        bootstrap = cluster_bootstrap_auc(
            outcome,
            augmented,
            bootstrap_samples,
            bootstrap_baseline_auc,
        )
        results.append(
            {
                "analysis_set": analysis_set,
                "baseline": (
                    "binding ranks + length + HLA + cohort + TCR chemistry"
                    if include_tcr_chemistry
                    else "binding ranks + length + HLA + cohort"
                ),
                "score": score_name,
                "n": int(len(frame)),
                "n_positive": int(outcome.sum()),
                "n_patients": int(frame["Patient"].nunique()),
                "n_cv_patient_groups": int(len(np.unique(groups))),
                "cv_repeats": int(len(seeds)),
                "raw_score_roc_auc": raw_metrics["roc_auc"],
                "raw_score_pr_auc": raw_metrics["pr_auc"],
                "baseline_roc_auc": baseline_metrics["roc_auc"],
                "augmented_roc_auc": augmented_metrics["roc_auc"],
                "delta_roc_auc": augmented_metrics["roc_auc"]
                - baseline_metrics["roc_auc"],
                "baseline_pr_auc": baseline_metrics["pr_auc"],
                "augmented_pr_auc": augmented_metrics["pr_auc"],
                "delta_pr_auc": augmented_metrics["pr_auc"]
                - baseline_metrics["pr_auc"],
                "baseline_log_loss": baseline_metrics["log_loss"],
                "augmented_log_loss": augmented_metrics["log_loss"],
                "delta_log_loss": augmented_metrics["log_loss"]
                - baseline_metrics["log_loss"],
                "mean_standardized_score_coefficient": float(np.mean(coefficients)),
                "positive_coefficient_folds": int(np.sum(np.asarray(coefficients) > 0)),
                "n_coefficient_folds": int(len(coefficients)),
                "cluster_bootstrap": bootstrap,
            }
        )
    return results


def _write_markdown(report: dict, path: Path) -> None:
    rows = sorted(
        report["results"],
        key=lambda row: (
            row["analysis_set"],
            row["baseline"],
            -row["delta_roc_auc"],
        ),
    )
    lines = [
        "# IMPROVE foreignness evaluation",
        "",
        f"Frozen taxonomy model: `{report['model']['architecture']}`.",
        f"Rows after overlap exclusion: {report['data']['analysis_rows']:,} "
        f"({report['data']['analysis_positive']:,} positive).",
        "",
        "| Analysis set | Baseline | Added score | ΔROC-AUC | 95% CI | ΔPR-AUC | Δlog loss |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        ci = row["cluster_bootstrap"]["delta_roc_auc_95ci"]
        lines.append(
            f"| {row['analysis_set']} | {row['baseline']} | `{row['score']}` | "
            f"{row['delta_roc_auc']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{row['delta_pr_auc']:+.4f} | {row['delta_log_loss']:+.5f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict:
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    source = load_improve(args.input)
    existing = load_existing_peptides() if args.exclude_existing else set()
    source["overlaps_existing_benchmark"] = source["Mut_peptide"].isin(existing)
    frame = source.loc[~source["overlaps_existing_benchmark"]].reset_index(drop=True)
    cache_path = output / "scored-neoepitopes.csv"
    required_cached_scores = {
        "p_bacteria",
        "virus_bacteria_max",
        "bacteria_vs_mammals",
        "bacteria_vs_human",
        "virus_bacteria_vs_mammals",
        "virus_bacteria_vs_human",
    }
    can_reuse = args.reuse_scored and cache_path.is_file()
    if can_reuse:
        scored = pd.read_csv(cache_path, low_memory=False)
        can_reuse = required_cached_scores <= set(scored.columns)
    if can_reuse:
        expected = frame[["Patient", "HLA_allele", "Mut_peptide"]].astype(str)
        observed = scored[["Patient", "HLA_allele", "Mut_peptide"]].astype(str)
        if len(scored) != len(frame) or not observed.reset_index(drop=True).equals(
            expected.reset_index(drop=True)
        ):
            raise ValueError("Existing scored cache does not match the filtered input")
        architecture = Path(args.model).name
    else:
        evaluator = _load_evaluator()
        evaluator.set_data_script(args.data_script)
        scored, architecture = score_taxonomy(frame, args.model, evaluator)
    if "Expression" not in scored:
        scored["Expression"] = frame["Expression"].to_numpy()
    scored = attach_published_predictions(scored, args.published_predictions)
    scored["cv_patient_group"] = patient_groups(scored)

    host_scores = tuple(
        name
        for name in (
            "p_mammal_host_viruses",
            "p_other_host_viruses",
            "host_virus_max",
            "mammal_host_virus_margin",
            "mammal_host_virus_specific",
            "mammal_host_virus_vs_mammals",
            "mammal_host_virus_vs_human",
        )
        if name in scored
    )
    all_scores = (
        "p_viruses",
        "p_bacteria",
        "virus_bacteria_max",
        "microbial_like",
        "one_minus_mammals",
        "one_minus_human",
        "virus_vs_mammals",
        "virus_vs_human",
        "bacteria_vs_mammals",
        "bacteria_vs_human",
        "virus_bacteria_vs_mammals",
        "virus_bacteria_vs_human",
        "microbial_foreign_like",
        "improve_foreignness",
        "inverse_self_similarity",
        *host_scores,
        *PUBLISHED_COMPLETE_SCORES,
    )
    paired_scores = (
        "one_minus_mammals",
        "one_minus_human",
        "delta_one_minus_mammals",
        "delta_one_minus_human",
        "delta_microbial_foreign_like",
        "delta_virus_vs_mammals",
        "delta_p_bacteria",
        "delta_virus_bacteria_max",
        "delta_bacteria_vs_mammals",
        "delta_bacteria_vs_human",
        "delta_virus_bacteria_vs_mammals",
        "delta_virus_bacteria_vs_human",
        *(f"delta_{name}" for name in host_scores),
    )
    seeds = tuple(int(value) for value in args.seeds.split(","))
    results = []
    for include_tcr_chemistry in (False, True):
        results.extend(
            evaluate_score_set(
                scored,
                all_scores,
                analysis_set="all mutant peptides",
                include_tcr_chemistry=include_tcr_chemistry,
                seeds=seeds,
                bootstrap_draws=args.bootstrap_draws,
            )
        )
        sparse_references = (
            "p_viruses",
            "p_bacteria",
            "virus_bacteria_max",
            "published_iedb_calis",
            "published_prime_score",
        )
        for analysis_set, sparse_scores in (
            (
                "DeepNetBim-covered peptides",
                (*sparse_references, *PUBLISHED_DEEPNETBIM_SCORES),
            ),
            (
                "HLAthena-MSiCE-covered peptides",
                (*sparse_references, "published_hlaathena_msice"),
            ),
        ):
            covered = scored.loc[scored[list(sparse_scores)].notna().all(axis=1)].reset_index(
                drop=True
            )
            results.extend(
                evaluate_score_set(
                    covered,
                    sparse_scores,
                    analysis_set=analysis_set,
                    include_tcr_chemistry=include_tcr_chemistry,
                    seeds=seeds,
                    bootstrap_draws=args.bootstrap_draws,
                )
            )
        paired = scored.loc[scored["Norm_peptide"].notna()].reset_index(drop=True)
        results.extend(
            evaluate_score_set(
                paired,
                paired_scores,
                analysis_set="mutant-WT pairs",
                include_tcr_chemistry=include_tcr_chemistry,
                seeds=seeds,
                bootstrap_draws=args.bootstrap_draws,
            )
        )

    keep = [
        "Patient",
        "HLA_allele",
        "Norm_peptide",
        "Mut_peptide",
        "cohort",
        "response",
        "Expression",
        "RankEL_4.1",
        "RankBA_4.1",
        "PeptLen",
        "HydroCore",
        "PropAro",
        "SelfSim",
        "Foreigness",
        "cv_patient_group",
        *all_scores,
        "wt_one_minus_mammals",
        "wt_one_minus_human",
        "delta_one_minus_mammals",
        "delta_one_minus_human",
        "delta_microbial_foreign_like",
        "delta_virus_vs_mammals",
        "delta_p_bacteria",
        "delta_virus_bacteria_max",
        "delta_bacteria_vs_mammals",
        "delta_bacteria_vs_human",
        "delta_virus_bacteria_vs_mammals",
        "delta_virus_bacteria_vs_human",
        *(f"wt_{name}" for name in host_scores),
        *(f"delta_{name}" for name in host_scores),
        *PUBLISHED_DEEPNETBIM_SCORES,
        "published_hlaathena_msice",
    ]
    keep = list(dict.fromkeys(keep))
    scored[keep].to_csv(output / "scored-neoepitopes.csv", index=False)
    leaderboard = pd.DataFrame(results).drop(columns=["cluster_bootstrap"])
    leaderboard.sort_values(
        ["analysis_set", "baseline", "delta_roc_auc"],
        ascending=[True, True, False],
    ).to_csv(output / "leaderboard.csv", index=False)
    report = {
        "protocol": (
            "Repeated five-fold stratified patient-group CV; exact mutant peptides "
            "in each test fold purged from its training rows; frozen taxonomy scores; "
            "patient-cluster bootstrap"
        ),
        "input": str(args.input),
        "model": {"path": str(args.model), "architecture": architecture},
        "published_predictions": str(args.published_predictions),
        "data": {
            "source_rows": int(len(source)),
            "source_positive": int(source["response"].sum()),
            "overlap_rows_excluded": int(source["overlaps_existing_benchmark"].sum()),
            "overlap_positive_excluded": int(
                source.loc[source["overlaps_existing_benchmark"], "response"].sum()
            ),
            "analysis_rows": int(len(scored)),
            "analysis_positive": int(scored["response"].sum()),
            "patients": int(scored["Patient"].nunique()),
            "hla_alleles": int(scored["HLA_allele"].nunique()),
            "cv_patient_groups": int(scored["cv_patient_group"].nunique()),
            "paired_rows": int(scored["Norm_peptide"].notna().sum()),
        },
        "seeds": list(seeds),
        "bootstrap_draws": args.bootstrap_draws,
        "results": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    _write_markdown(report, output / "report.md")
    print(json.dumps(report["data"], indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input",
        default=ROOT
        / "data"
        / "improve-validation"
        / "02_2_Validated_neoepitopes_calculated_features_TME.txt",
        type=Path,
    )
    parser.add_argument(
        "--model",
        default=ROOT / "data" / "peptide-content-v1" / "models" / "esm_finetune",
        type=Path,
    )
    parser.add_argument(
        "--data-script",
        default=ROOT / "scripts" / "peptide_content_data.py",
        type=Path,
        help="Split-builder module defining the model's category schema",
    )
    parser.add_argument(
        "--published-predictions",
        default=DEFAULT_PUBLISHED_PREDICTIONS,
        type=Path,
    )
    parser.add_argument("--out", default=ROOT / "tmp" / "improve-foreignness", type=Path)
    parser.add_argument("--seeds", default="11,29,47,71,101")
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument(
        "--reuse-scored",
        action="store_true",
        help="Reuse a matching scored-neoepitopes.csv in the output directory",
    )
    parser.add_argument(
        "--exclude-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
