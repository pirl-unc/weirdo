#!/usr/bin/env python3
"""Evaluate 20 frozen taxonomy variants on internal and external responses.

Taxonomy checkpoints are never trained or selected using response labels. The
20 variants are fixed below before reading TESLA or Braun outcomes. Each
variant is evaluated only as one extra numeric feature beyond MHCflurry 2.3
presentation and NetMHCpan 4.2c BA rank.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/weirdo-taxonomy-response-20")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


COHORTS = {
    "bladder": ("IMPROVE bladder", "tumor"),
    "melanoma": ("IMPROVE melanoma", "tumor"),
    "Basket": ("IMPROVE basket", "tumor"),
    "yellow_fever_17d": ("Yellow fever 17D", "viral"),
    "multivirus_concise_set": ("Multi-virus panel", "viral"),
}
BLUE = "#2D6FBB"
ORANGE = "#D3871C"
GREEN = "#2E8B62"
INK = "#17263D"
GRID = "#D7E1EA"


@dataclass(frozen=True)
class Experiment:
    """One preregistered frozen taxonomy output supplied to the response adapter."""

    experiment_id: str
    family: str
    description: str
    column: str


EXPERIMENTS = (
    Experiment("01", "scale", "broad-nine sigmoid ESM-2, 10k", "broad9__esm_10k__viruses"),
    Experiment("02", "scale", "broad-nine sigmoid ESM-2, 20k", "broad9__esm_20k__viruses"),
    Experiment("03", "scale", "broad-nine sigmoid ESM-2, 40k", "broad9__esm_40k__viruses"),
    Experiment("04", "scale", "broad-nine sigmoid ESM-2, 80k", "broad9__esm_80k__viruses"),
    Experiment("05", "scale", "broad-nine sigmoid ESM-2, 160k", "broad9__esm_160k__viruses"),
    Experiment("06", "architecture", "broad-nine sigmoid composition MLP, 160k", "broad9__composition_mlp_160k__viruses"),
    Experiment("07", "architecture", "broad-nine sigmoid residue CNN, 160k", "broad9__cnn_160k__viruses"),
    Experiment("08", "transform", "ESM-2 160k, 1 minus mammal occurrence", "broad9__esm_160k__one_minus_mammals"),
    Experiment("09", "transform", "ESM-2 160k, virus × non-mammal geometric mean", "broad9__esm_160k__virus_foreign_like"),
    Experiment("10", "transform", "ESM-2 160k, microbial × non-mammal geometric mean", "broad9__esm_160k__microbial_foreign_like"),
    Experiment("11", "softmax", "unique broad-nine softmax, 10k P(virus)", "ablation__unique_softmax_10k__viruses"),
    Experiment("12", "softmax", "unique broad-nine softmax, 40k P(virus)", "ablation__unique_softmax_40k__viruses"),
    Experiment("13", "softmax", "unique broad-nine softmax, 120k P(virus)", "ablation__unique_softmax_120k__viruses"),
    Experiment("14", "softmax transform", "unique softmax 40k, virus × non-mammal", "ablation__unique_softmax_40k__virus_and_not_mammal"),
    Experiment("15", "softmax transform", "unique softmax 40k, microbial-versus-mammal contrast", "ablation__unique_softmax_40k__microbial_vs_mammal"),
    Experiment("16", "host-aware softmax", "mammal-host-virus softmax, 10k", "ablation__mammal_host_virus_mammal_softmax_10k__mammal_host_viruses"),
    Experiment("17", "host-aware softmax", "mammal-host-virus softmax, 40k", "ablation__mammal_host_virus_mammal_softmax_40k__mammal_host_viruses"),
    Experiment("18", "host-aware softmax", "mammal-host-virus softmax, 120k", "ablation__mammal_host_virus_mammal_softmax_120k__mammal_host_viruses"),
    Experiment("19", "legacy output", "deployed legacy sigmoid P(virus)", "weirdo_p_viruses"),
    Experiment("20", "legacy transform", "deployed legacy virus × non-mammal geometric mean", "weirdo_virus_foreign_like"),
)


def sha256(path: Path) -> str:
    """Return a stable provenance hash."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logit(values: np.ndarray | pd.Series) -> np.ndarray:
    """Clip bounded scores before the linear response adapter."""
    values = np.clip(np.asarray(values, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(values / (1.0 - values))


def numeric_matrix(frame: pd.DataFrame, taxonomy_column: str | None = None) -> np.ndarray:
    """Construct MHC baseline features, optionally with one frozen taxonomy score."""
    columns = [
        logit(frame["mhcflurry_2_3_0_presentation_score"]),
        -np.log10(np.clip(frame["netmhcpan_ba_rank"].to_numpy(float), 1e-5, 100.0)),
    ]
    if taxonomy_column is not None:
        columns.append(logit(frame[taxonomy_column]))
    return np.column_stack(columns)


def fit_response_adapter(
    train: pd.DataFrame, test: pd.DataFrame, taxonomy_column: str | None
) -> np.ndarray:
    """Fit a fixed-regularization MHC response adapter and return probabilities."""
    if train["outcome"].nunique() != 2:
        raise ValueError("Response-adapter training must contain both classes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=2_000, random_state=8434),
    )
    model.fit(numeric_matrix(train, taxonomy_column), train["outcome"].to_numpy(int))
    return model.predict_proba(numeric_matrix(test, taxonomy_column))[:, 1]


def metric_dict(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """Return response metrics with numerical clipping for likelihoods."""
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
        "log_loss": float(log_loss(y, np.clip(score, 1e-7, 1 - 1e-7), labels=[0, 1])),
    }


def group_bootstrap_delta(
    y: np.ndarray,
    baseline: np.ndarray,
    augmented: np.ndarray,
    groups: np.ndarray,
    draws: int,
    seed: int,
) -> tuple[float, float, int]:
    """Return a group-resampled 95% interval for the paired AUC change."""
    unique = pd.unique(groups.astype(str))
    members = [np.flatnonzero(groups.astype(str) == group) for group in unique]
    random = np.random.default_rng(seed)
    values = np.full(draws, np.nan)
    for draw in range(draws):
        selected = random.integers(0, len(members), len(members))
        indices = np.concatenate([members[index] for index in selected])
        if np.unique(y[indices]).size == 2:
            values[draw] = roc_auc_score(y[indices], augmented[indices]) - roc_auc_score(
                y[indices], baseline[indices]
            )
    valid = values[np.isfinite(values)]
    if not len(valid):
        return np.nan, np.nan, 0
    bounds = np.quantile(valid, (0.025, 0.975))
    return float(bounds[0]), float(bounds[1]), len(valid)


def merge_internal(broad: pd.DataFrame, objective: pd.DataFrame) -> pd.DataFrame:
    """Merge independently scored frozen outputs after validating assay identity."""
    keys = ["source_dataset", "source_row"]
    identity = ["cohort", "outcome", "peptide", "hla", "group"]
    missing = [column for column in (*keys, *identity) if column not in broad or column not in objective]
    if missing:
        raise ValueError(f"Internal score tables are missing identity columns: {missing}")
    if broad.duplicated(keys).any() or objective.duplicated(keys).any():
        raise ValueError("Internal score tables have duplicate source rows")
    objective_columns = [column for column in objective if column.startswith("ablation__")]
    merged = broad.merge(
        objective[[*keys, *identity, *objective_columns]],
        on=keys,
        suffixes=("", "_objective"),
        validate="one_to_one",
    )
    if len(merged) != len(broad):
        raise ValueError("Internal taxonomy score tables do not cover the same rows")
    for column in identity:
        other = f"{column}_objective"
        if not merged[column].astype(str).eq(merged[other].astype(str)).all():
            raise ValueError(f"Internal score tables disagree on {column}")
        merged = merged.drop(columns=other)
    return merged


def check_experiment_columns(frame: pd.DataFrame) -> None:
    """Ensure every preregistered score is finite and bounded before fitting."""
    missing = [experiment.column for experiment in EXPERIMENTS if experiment.column not in frame]
    if missing:
        raise ValueError(f"Missing experiment columns: {missing}")
    for experiment in EXPERIMENTS:
        values = frame[experiment.column].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Experiment {experiment.experiment_id} has invalid scores")


def internal_evaluation(frame: pd.DataFrame) -> pd.DataFrame:
    """Leave one same-domain response cohort out for all 20 frozen variants."""
    rows = []
    for heldout, (_, domain) in COHORTS.items():
        test = frame.loc[frame["cohort"].eq(heldout)].reset_index(drop=True)
        train_cohorts = [
            cohort for cohort, (_, candidate_domain) in COHORTS.items()
            if cohort != heldout and candidate_domain == domain
        ]
        train = frame.loc[frame["cohort"].isin(train_cohorts)].copy()
        before = len(train)
        train = train.loc[~train["peptide"].isin(set(test["peptide"]))].reset_index(drop=True)
        y = test["outcome"].to_numpy(int)
        baseline = fit_response_adapter(train, test, None)
        base_metrics = metric_dict(y, baseline)
        for experiment in EXPERIMENTS:
            augmented = fit_response_adapter(train, test, experiment.column)
            metrics = metric_dict(y, augmented)
            rows.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "family": experiment.family,
                    "description": experiment.description,
                    "heldout_cohort": heldout,
                    "domain": domain,
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_positive": int(y.sum()),
                    "n_exact_peptide_train_rows_purged": before - len(train),
                    "baseline_auc": base_metrics["roc_auc"],
                    "augmented_auc": metrics["roc_auc"],
                    "delta_auc": metrics["roc_auc"] - base_metrics["roc_auc"],
                    "baseline_ap": base_metrics["average_precision"],
                    "augmented_ap": metrics["average_precision"],
                    "delta_ap": metrics["average_precision"] - base_metrics["average_precision"],
                    "delta_log_loss": metrics["log_loss"] - base_metrics["log_loss"],
                }
            )
    return pd.DataFrame(rows)


def external_evaluation(train: pd.DataFrame, external: pd.DataFrame, draws: int) -> pd.DataFrame:
    """Transfer tumor response adapters to TESLA and Braun without refitting on them."""
    rows = []
    subsets = {
        "tesla_public_2022": external.loc[external["benchmark"].eq("tesla_public_2022")],
        "braun_rcc_2025_all": external.loc[external["benchmark"].eq("braun_rcc_2025")],
        "braun_rcc_2025_frameshift": external.loc[
            external["benchmark"].eq("braun_rcc_2025")
            & external["mutation_class"].eq("frameshift")
        ],
    }
    for subset_index, (name, test) in enumerate(subsets.items()):
        test = test.reset_index(drop=True)
        if test.empty or test["outcome"].nunique() != 2:
            raise ValueError(f"External benchmark {name} requires both response classes")
        purged_train = train.loc[~train["peptide"].isin(set(test["peptide"]))].reset_index(drop=True)
        y = test["outcome"].to_numpy(int)
        baseline = fit_response_adapter(purged_train, test, None)
        base_metrics = metric_dict(y, baseline)
        for experiment_index, experiment in enumerate(EXPERIMENTS):
            augmented = fit_response_adapter(purged_train, test, experiment.column)
            metrics = metric_dict(y, augmented)
            low, high, valid = group_bootstrap_delta(
                y,
                baseline,
                augmented,
                test["group"].astype(str).to_numpy(),
                draws,
                seed=20261000 + subset_index * 100 + experiment_index,
            )
            rows.append(
                {
                    "experiment_id": experiment.experiment_id,
                    "family": experiment.family,
                    "description": experiment.description,
                    "benchmark": name,
                    "n_train": len(purged_train),
                    "n_test": len(test),
                    "n_positive": int(y.sum()),
                    "n_groups": int(test["group"].nunique()),
                    "n_exact_peptide_train_rows_purged": len(train) - len(purged_train),
                    "raw_taxonomy_auc": float(roc_auc_score(y, test[experiment.column])),
                    "baseline_auc": base_metrics["roc_auc"],
                    "augmented_auc": metrics["roc_auc"],
                    "delta_auc": metrics["roc_auc"] - base_metrics["roc_auc"],
                    "delta_auc_low": low,
                    "delta_auc_high": high,
                    "bootstrap_valid": valid,
                    "baseline_ap": base_metrics["average_precision"],
                    "augmented_ap": metrics["average_precision"],
                    "delta_ap": metrics["average_precision"] - base_metrics["average_precision"],
                    "delta_log_loss": metrics["log_loss"] - base_metrics["log_loss"],
                }
            )
    return pd.DataFrame(rows)


def plot_internal(results: pd.DataFrame, path: Path) -> None:
    """Draw the response-screen result only for the neoantigen/tumor cohorts."""
    values = (
        results.loc[results["domain"].eq("tumor")]
        .groupby(["experiment_id", "description"], as_index=False)["delta_auc"]
        .mean()
        .sort_values("delta_auc")
    )
    figure, axis = plt.subplots(figsize=(9, 7))
    colors = [GREEN if value > 0 else ORANGE for value in values["delta_auc"]]
    axis.barh(values["experiment_id"] + "  " + values["description"], values["delta_auc"], color=colors)
    axis.axvline(0, color=INK, linewidth=0.8)
    axis.set_xlabel("Mean leave-one-tumor-cohort Δ ROC AUC")
    axis.set_title("Twenty frozen taxonomy variants over MHCflurry + NetMHCpan", color=INK, weight="bold")
    axis.grid(axis="x", color=GRID, linewidth=0.7)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_external(results: pd.DataFrame, path: Path) -> None:
    """Plot raw external AUC changes without selecting a winner."""
    benchmarks = tuple(results["benchmark"].drop_duplicates())
    figure, axes = plt.subplots(1, len(benchmarks), figsize=(16, 7), sharey=True)
    for axis, benchmark in zip(np.atleast_1d(axes), benchmarks):
        part = results.loc[results["benchmark"].eq(benchmark)].sort_values("delta_auc")
        error = np.vstack(
            (
                part["delta_auc"] - part["delta_auc_low"],
                part["delta_auc_high"] - part["delta_auc"],
            )
        )
        axis.errorbar(
            part["delta_auc"],
            np.arange(len(part)),
            xerr=error,
            fmt="o",
            color=BLUE,
            ecolor=INK,
            capsize=2,
            markersize=4,
        )
        axis.axvline(0, color=INK, linewidth=0.8)
        axis.set_yticks(np.arange(len(part)), part["experiment_id"])
        axis.set_title(benchmark.replace("_", " "), color=INK, weight="bold")
        axis.grid(axis="x", color=GRID, linewidth=0.7)
    figure.supxlabel("External Δ ROC AUC over MHC baseline")
    figure.suptitle("All 20 fixed taxonomy variants; intervals are group bootstraps", color=INK, weight="bold")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_analysis(
    output: Path, internal: pd.DataFrame, external: pd.DataFrame, braun_unit: str
) -> None:
    """Write a concise result summary while preserving the multiplicity boundary."""
    tumor = internal.loc[internal["domain"].eq("tumor")]
    internal_mean = tumor.groupby("experiment_id", as_index=False)["delta_auc"].mean()
    external_mean = external.loc[external["benchmark"].eq("tesla_public_2022")]
    best_internal = internal_mean.loc[internal_mean["delta_auc"].idxmax()]
    best_tesla = external_mean.loc[external_mean["delta_auc"].idxmax()]
    description = {experiment.experiment_id: experiment.description for experiment in EXPERIMENTS}
    tesla = external.loc[external["benchmark"].eq("tesla_public_2022")].iloc[0]
    braun = external.loc[external["benchmark"].eq("braun_rcc_2025_all")].iloc[0]
    frameshift = external.loc[external["benchmark"].eq("braun_rcc_2025_frameshift")].iloc[0]
    braun_unit_description = {
        "minimal_epitope": "collapsed patient/mutation/minimal-epitope tests",
        "vaccine_construct": (
            "individual vaccine-construct tests, each scored at the authors' reported "
            "best short epitope"
        ),
    }[braun_unit]
    text = f"""# Taxonomy response screen: 20 frozen variants

## Design

All 20 taxonomy variants in [`tables/experiments.csv`](tables/experiments.csv)
were fixed before any TESLA or Braun label was read. They vary taxonomy-training
scale, network architecture, independent-sigmoid versus softmax output space,
and non-self transforms. Taxonomy training contains no neoantigen or response
labels. Each response model is an identical class-balanced logistic adapter
over logit(MHCflurry 2.3 presentation) and -log10(NetMHCpan 4.2c BA rank),
with exactly one frozen taxonomy feature added.

Internal estimates leave one response cohort out within domain and remove exact
held-out peptide sequences from training. External adapters are fit only on all
three IMPROVE tumour cohorts after the same exact-peptide purge; neither TESLA
nor Braun labels are used for fitting or model selection.

## External data

The [public CC-BY TESLA derivative](https://data.mendeley.com/datasets/6x87nx8jtc/1)
contributes {int(tesla['n_test']):,} deduplicated peptide--HLA tests
({int(tesla['n_positive'])} positive). It is a public derivative whose count
differs from the original TESLA paper's 608/37 table, so both source and
resulting count are recorded rather than conflated. [Braun et al.
(*Nature*, 2025)](https://doi.org/10.1038/s41586-024-08507-5) contributes
{int(braun['n_test'])} {braun_unit_description}
({int(braun['n_positive'])} positive).
Its frameshift subset is only {int(frameshift['n_test'])} tests with
{int(frameshift['n_positive'])} positives, so it is an imprecise stress test,
not a decisive small-effect validation.

## Result boundary

The largest mean internal tumour gain is experiment {best_internal['experiment_id']}
({description[best_internal['experiment_id']]}) at
{best_internal['delta_auc']:+.4f}. The largest TESLA point gain is experiment
{best_tesla['experiment_id']} ({description[best_tesla['experiment_id']]}) at
{best_tesla['delta_auc']:+.4f}. Those maxima are descriptions of a 20-way
screen, not confirmatory findings. Use the complete external table and its
group-bootstrap intervals rather than selecting a post-hoc winner.

The report therefore answers the narrow question—whether frozen peptide
taxonomy features add beyond two standard MHC predictors—without relabeling a
response model as a taxonomy model or treating small external subsets as
conclusive.

## Outputs

- [`tables/internal_heldout_results.csv`](tables/internal_heldout_results.csv)
- [`tables/external_results.csv`](tables/external_results.csv)
- [`raw/external-neoantigen-benchmarks-mhc.csv.gz`](raw/external-neoantigen-benchmarks-mhc.csv.gz)
- [`raw/external-broad-taxonomy-scores.csv.gz`](raw/external-broad-taxonomy-scores.csv.gz)
- [`raw/external-taxonomy-objective-scores.csv.gz`](raw/external-taxonomy-objective-scores.csv.gz)

![Internal response screen](figures/01_internal_tumor_delta_auc.png)

![External response screen](figures/02_external_delta_auc.png)
"""
    (output / "analysis.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--broad-scores", type=Path, required=True)
    parser.add_argument("--objective-scores", type=Path, required=True)
    parser.add_argument("--external-broad-scores", type=Path, required=True)
    parser.add_argument("--external-objective-scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=1_000)
    parser.add_argument(
        "--braun-unit",
        choices=("minimal_epitope", "vaccine_construct"),
        default="minimal_epitope",
        help="Report the Braun assay unit used in the external input.",
    )
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be at least 100")

    broad = pd.read_csv(args.broad_scores, low_memory=False)
    objective = pd.read_csv(args.objective_scores, low_memory=False)
    internal = merge_internal(broad, objective)
    external = pd.read_csv(args.external_broad_scores, low_memory=False).merge(
        pd.read_csv(args.external_objective_scores, low_memory=False),
        on="record_id",
        validate="one_to_one",
    )
    check_experiment_columns(internal)
    check_experiment_columns(external)
    internal_results = internal_evaluation(internal)
    tumor_train = internal.loc[internal["cohort"].isin(
        [cohort for cohort, (_, domain) in COHORTS.items() if domain == "tumor"]
    )].reset_index(drop=True)
    external_results = external_evaluation(tumor_train, external, args.bootstrap_draws)

    args.output.mkdir(parents=True, exist_ok=True)
    tables = args.output / "tables"
    figures = args.output / "figures"
    tables.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    pd.DataFrame([asdict(experiment) for experiment in EXPERIMENTS]).to_csv(
        tables / "experiments.csv", index=False
    )
    internal_results.to_csv(tables / "internal_heldout_results.csv", index=False)
    external_results.to_csv(tables / "external_results.csv", index=False)
    plot_internal(internal_results, figures / "01_internal_tumor_delta_auc.png")
    plot_external(external_results, figures / "02_external_delta_auc.png")
    write_analysis(args.output, internal_results, external_results, args.braun_unit)
    provenance = {
        "schema": "taxonomy-response-20-v1",
        "experiments": [asdict(experiment) for experiment in EXPERIMENTS],
        "inputs": {
            str(path): {"sha256": sha256(path), "bytes": path.stat().st_size}
            for path in (
                args.broad_scores,
                args.objective_scores,
                args.external_broad_scores,
                args.external_objective_scores,
            )
        },
        "bootstrap_draws": args.bootstrap_draws,
        "braun_unit": args.braun_unit,
        "response_adapter": {
            "C": 1.0,
            "class_weight": "balanced",
            "baseline": ["MHCflurry_2.3_presentation", "NetMHCpan_4.2c_BA_rank"],
        },
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    print(external_results.to_string(index=False, float_format=lambda value: f"{value:+.4f}"))
    print(f"Wrote 20-variant taxonomy response report to {args.output}")


if __name__ == "__main__":
    main()
