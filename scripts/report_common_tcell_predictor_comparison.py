#!/usr/bin/env python3
"""Compare one common predictor stack on the same CD8 assay rows per cohort."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/weirdo-common-report-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.report_tcell_predictor_comparison import cluster_metrics, logit  # noqa: E402
from weirdo.peptide_content_models import geometric_mean_scores  # noqa: E402


PREDICTIONS = ROOT / "output/cd8-predictor-comparison/common-tcell-predictions.csv.gz"
PREDICTION_PROVENANCE = ROOT / "output/cd8-predictor-comparison/common-tcell-predictions-provenance.json"
COHORTS = {
    "improve_bladder": ("bladder", "IMPROVE bladder"),
    "improve_melanoma": ("melanoma", "IMPROVE melanoma"),
    "improve_basket": ("Basket", "IMPROVE basket"),
    "viral_yellow_fever": ("yellow_fever_17d", "Yellow fever 17D"),
    "viral_multivirus": ("multivirus_concise_set", "Multi-virus panel"),
}
POOLED = {
    "improve_pooled": ("IMPROVE", "IMPROVE pooled"),
    "viral_pooled": ("registered_viral_validation", "Viral validation pooled"),
}
RAW_SCORES = {
    "netmhcpan_ba": ("NetMHCpan 4.2c BA rank", lambda d: -d.netmhcpan_ba_rank.to_numpy(float)),
    "netmhcpan_el": ("NetMHCpan 4.2c EL rank", lambda d: -d.netmhcpan_el_rank.to_numpy(float)),
    "mhcflurry_2_2_0_presentation": ("MHCflurry release 2.2.0 presentation", lambda d: d.mhcflurry_2_2_0_presentation_score.to_numpy(float)),
    "mhcflurry_2_2_0_affinity": ("MHCflurry release 2.2.0 affinity rank", lambda d: -d.mhcflurry_2_2_0_affinity_percentile.to_numpy(float)),
    "mhcflurry_2_2_0_processing": ("MHCflurry release 2.2.0 processing", lambda d: d.mhcflurry_2_2_0_processing_score.to_numpy(float)),
    "mhcflurry_2_3_0_presentation": ("MHCflurry release 2.3.0 presentation", lambda d: d.mhcflurry_2_3_0_presentation_score.to_numpy(float)),
    "mhcflurry_2_3_0_affinity": ("MHCflurry release 2.3.0 affinity rank", lambda d: -d.mhcflurry_2_3_0_affinity_percentile.to_numpy(float)),
    "mhcflurry_2_3_0_processing": ("MHCflurry release 2.3.0 processing", lambda d: d.mhcflurry_2_3_0_processing_score.to_numpy(float)),
    "prime_2_1": ("PRIME 2.1 immunogenicity rank", lambda d: -d.prime_2_1_rank.to_numpy(float)),
    "weirdo_p_viruses": ("Weirdo P(virus)", lambda d: d.weirdo_p_viruses.to_numpy(float)),
    "weirdo_p_bacteria": ("Weirdo P(bacteria)", lambda d: d.weirdo_p_bacteria.to_numpy(float)),
    "weirdo_microbial_like": ("Weirdo microbial-like", lambda d: d.weirdo_microbial_like.to_numpy(float)),
    "weirdo_p_mammals": ("Weirdo P(mammals; raw)", lambda d: d.weirdo_p_mammals.to_numpy(float)),
    "weirdo_one_minus_mammals": ("Weirdo 1 − P(mammals)", lambda d: d.weirdo_one_minus_mammals.to_numpy(float)),
    "weirdo_one_minus_human": ("Weirdo 1 − P(human)", lambda d: d.weirdo_one_minus_human.to_numpy(float)),
    "weirdo_virus_mammal_like": ("Weirdo geomean[P(virus), P(mammals)]", lambda d: d.weirdo_virus_mammal_like.to_numpy(float)),
    "weirdo_virus_foreign_like": ("Weirdo geomean[P(virus), 1 − P(mammals)]", lambda d: d.weirdo_virus_foreign_like.to_numpy(float)),
    "weirdo_microbial_foreign_like": ("Weirdo geomean[microbial-like, 1 − P(mammals)]", lambda d: d.weirdo_microbial_foreign_like.to_numpy(float)),
}
WEIRDO_SCORES = tuple(name for name in RAW_SCORES if name.startswith("weirdo_"))
MODEL_LABELS = {name: label for name, (label, _) in RAW_SCORES.items()}
COLORS = ["#275d86", "#d68d28", "#258b79", "#753d8c", "#bf5368"]
BASELINE_RAW_SCORES = (
    "netmhcpan_ba",
    "netmhcpan_el",
    "mhcflurry_2_2_0_presentation",
    "mhcflurry_2_2_0_affinity",
    "mhcflurry_2_2_0_processing",
    "mhcflurry_2_3_0_presentation",
    "mhcflurry_2_3_0_affinity",
    "mhcflurry_2_3_0_processing",
    "prime_2_1",
)
HEADLINE_SCORE_COLUMNS = (
    "NetMHCpan BA",
    "NetMHCpan EL",
    "MHCflurry 2.2 pres.",
    "MHCflurry 2.3 pres.",
    "PRIME 2.1",
    "Weirdo P(virus)",
    "Weirdo P(mammals)",
    "Weirdo virus/mammal geomean",
    "MHCflurry 2.3 + P(virus)",
    "MHCflurry 2.3 + P(mammals)",
    "MHCflurry 2.3 + virus/mammal",
)
PRESENTATION_BASELINES = (
    "mhcflurry_2_2_0_presentation",
    "mhcflurry_2_3_0_presentation",
)
PRIME_OVERLAP_NOTE = (
    "PRIME training-overlap audit: the published PRIME2.0 training table contains "
    "10 exact peptide-HLA matches among the 18,738 benchmark assay rows (2 bladder, "
    "7 melanoma, 1 multi-virus, 0 basket and 0 yellow fever). At peptide-sequence "
    "level, 32 assay rows representing 22 unique peptides overlap. PRIME2.1 changes "
    "model weights and expands allele/rank assets, but its public release does not "
    "provide a separate row-level immunogenicity training manifest. PRIME should "
    "therefore be treated as a partially training-contaminated comparator, not a "
    "clean external-validation result. Audit source: PRIME2.0 Table S4, "
    "doi:10.1016/j.cels.2022.12.002."
)
PRIME_SENSITIVITY_NOTE = (
    "Removing the 10 exact peptide-HLA matches and recomputing PRIME AUC changes it "
    "by +0.0000 in bladder, +0.0002 in melanoma, +0.0000 in basket, -0.0008 in the "
    "multi-virus panel and +0.0000 in yellow fever. The exact matches therefore do "
    "not explain PRIME's observed ranking signal, although broader source-study and "
    "predictor-training overlap remains possible."
)
PRIME_FEATURE_NOTE = (
    "PRIME2.1 takes a peptide sequence and restricting MHC-I allele. Its shipped "
    "neural network has 28 inputs: -log(MixMHCpred percentile rank), the frequencies "
    "of all 20 amino acids over allele- and length-specific peptide positions chosen "
    "to minimize direct MHC-binding influence, and seven one-hot peptide-length "
    "indicators for lengths 8 through 14. Those inputs feed five sigmoid hidden "
    "units and a sigmoid output. PRIME percentile rank is then referenced to random "
    "8--14mers from the human proteome. It does not use transcript abundance, source "
    "protein abundance, cleavage, or an explicit TCR-sequence feature."
)
PRIME_OVERLAP_ROWS = (
    ("IMPROVE bladder", 2, 7, 5),
    ("IMPROVE melanoma", 7, 16, 8),
    ("IMPROVE basket", 0, 8, 8),
    ("Yellow fever 17D", 0, 0, 0),
    ("Multi-virus panel", 1, 1, 1),
    ("Total", 10, 32, 22),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mhc_matrices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    presentation = logit(frame.mhcflurry_2_3_0_presentation_score)[:, None]
    full = np.column_stack(
        [
            presentation,
            -np.log10(frame.netmhcpan_ba_rank.clip(lower=1e-5)),
            -np.log10(frame.netmhcpan_el_rank.clip(lower=1e-5)),
            -np.log10(frame.mhcflurry_2_3_0_affinity_percentile.clip(lower=1e-5)),
            logit(frame.mhcflurry_2_3_0_processing_score),
        ]
    )
    return presentation, full


def presentation_matrix(frame: pd.DataFrame, release: str) -> np.ndarray:
    column = f"mhcflurry_{release.replace('.', '_')}_presentation_score"
    return logit(frame[column])[:, None]


def dataset_frames(frame: pd.DataFrame) -> dict[str, tuple[str, pd.DataFrame]]:
    result = {
        name: (label, frame.loc[frame.cohort.eq(cohort)].reset_index(drop=True))
        for name, (cohort, label) in COHORTS.items()
    }
    result.update(
        {
            name: (label, frame.loc[frame.source_dataset.eq(source)].reset_index(drop=True))
            for name, (source, label) in POOLED.items()
        }
    )
    if sum(len(result[name][1]) for name in COHORTS) != len(frame):
        raise ValueError("Cohort definitions must partition all assay rows exactly once")
    return result


def score_dataset(name: str, label: str, frame: pd.DataFrame):
    y = frame.outcome.to_numpy(int)
    groups = frame.group.astype(str).to_numpy()
    if len(np.unique(y)) != 2:
        raise ValueError(f"{name} has only one outcome class")
    scores = {score: function(frame) for score, (_, function) in RAW_SCORES.items()}
    pairs = {}
    for baseline in PRESENTATION_BASELINES:
        for signal in WEIRDO_SCORES:
            model = f"{baseline}+{signal}"
            scores[model] = geometric_mean_scores(
                scores[baseline], scores[signal]
            )
            MODEL_LABELS[model] = (
                f"geomean[{MODEL_LABELS[baseline]}, {MODEL_LABELS[signal]}]"
            )
            pairs[model] = baseline
    return y, groups, scores, pairs, {
        "label": label,
        "n": len(frame),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "n_groups": len(set(groups)),
        "combination": "deterministic geometric mean; no response fitting",
    }


def evaluate_dataset(name, label, frame, draws):
    print(f"Scoring {label}: n={len(frame):,}, positive={frame.outcome.sum():,}", flush=True)
    y, groups, scores, pairs, audit = score_dataset(name, label, frame)
    rows, samples = cluster_metrics(y, scores, groups, draws)
    lookup = {row["model"]: row for row in rows}
    for row in rows:
        row.update(
            dataset=name,
            dataset_label=label,
            n=len(frame),
            n_positive=int(y.sum()),
            prevalence=float(y.mean()),
            n_groups=len(set(groups)),
            label=MODEL_LABELS[row["model"]],
            evaluation=("deterministic geometric mean" if row["model"] in pairs else "frozen score"),
        )
    deltas = []
    for model, baseline in pairs.items():
        distribution = samples[model] - samples[baseline]
        bounds = np.nanquantile(distribution, [0.025, 0.975], axis=0)
        deltas.append(
            {
                "dataset": name,
                "dataset_label": label,
                "model": model,
                "baseline": baseline,
                "signal": model.split("+", 1)[1],
                "signal_label": MODEL_LABELS[model.split("+", 1)[1]],
                "n": len(frame),
                "n_positive": int(y.sum()),
                "delta_auc": lookup[model]["roc_auc"] - lookup[baseline]["roc_auc"],
                "delta_low": bounds[0, 0],
                "delta_high": bounds[1, 0],
                "delta_ap": lookup[model]["average_precision"] - lookup[baseline]["average_precision"],
                "delta_ap_low": bounds[0, 1],
                "delta_ap_high": bounds[1, 1],
                "bootstrap_valid": int(np.isfinite(distribution[:, 0]).sum()),
            }
        )
    prediction_frame = pd.DataFrame(
        {
            "dataset": name,
            "source_dataset": frame.source_dataset,
            "source_row": frame.source_row,
            "cohort": frame.cohort,
            "outcome": y,
            **scores,
        }
    )
    return rows, deltas, audit, prediction_frame, (y, scores)


def evaluate_presentation_filtered(name, label, frame, release, draws):
    rank_column = f"mhcflurry_{release.replace('.', '_')}_presentation_percentile"
    passing = frame[rank_column].le(1.0).to_numpy()
    y = frame.outcome.to_numpy(int)
    groups = frame.group.astype(str).to_numpy()
    scores = {
        score: np.where(passing, function(frame), 0.0)
        for score, (_, function) in RAW_SCORES.items()
        if score in WEIRDO_SCORES
    }
    rows, _ = cluster_metrics(y, scores, groups, draws)
    for row in rows:
        row.update(
            dataset=name,
            dataset_label=label,
            model_release=release,
            filter=f"Weirdo score if MHCflurry {release} presentation percentile <= 1; otherwise 0",
            n=len(frame),
            n_positive=int(y.sum()),
            prevalence=float(y.mean()),
            n_groups=len(set(groups)),
            n_passing=int(passing.sum()),
            n_positive_passing=int(y[passing].sum()),
            label=MODEL_LABELS[row["model"]],
            evaluation="frozen Weirdo score with failed presentation gate set to zero",
        )
    return rows


def save_figure(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", dpi=180, facecolor="white", bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def heatmap(ax, table, row_labels, column_labels, title, vmin, vmax, cmap="RdYlBu"):
    image = ax.imshow(table, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    ax.set_xticks(range(len(column_labels)), column_labels, fontsize=9)
    for row in range(table.shape[0]):
        for column in range(table.shape[1]):
            value = table[row, column]
            ax.text(column, row, f"{value:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(value - (vmin + vmax) / 2) > (vmax - vmin) * 0.32 else "#17212b")
    ax.set_title(title, pad=12)
    return image


def make_figures(out, metrics, deltas, filtered_metrics, curve_data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "svg.fonttype": "none"})
    datasets = list(COHORTS)
    dataset_labels = [COHORTS[name][1].replace("IMPROVE ", "").replace("Yellow fever 17D", "YF 17D") for name in datasets]
    raw = [*BASELINE_RAW_SCORES, *WEIRDO_SCORES]
    row_labels = [MODEL_LABELS[name] for name in raw]
    auc = metrics.pivot(index="model", columns="dataset", values="roc_auc").loc[raw, datasets].to_numpy()
    ap = metrics.pivot(index="model", columns="dataset", values="average_precision").loc[raw, datasets].to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(15, 8.8), gridspec_kw={"width_ratios": [1, 1]})
    heatmap(axes[0], auc, row_labels, dataset_labels, "ROC AUC", 0.4, 0.8)
    heatmap(axes[1], ap, row_labels, dataset_labels, "Average precision", 0.0, max(0.7, float(ap.max())))
    fig.suptitle("Same frozen scores, same assay rows within each CD8 cohort", fontsize=17, x=0.03, ha="left")
    fig.text(0.03, 0.01, "Every cell in a column uses the identical labeled rows. AUC chance = 0.5; AP chance = that cohort's positive prevalence.", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95], w_pad=2)
    save_figure(fig, out, "01_common_raw_scores")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    for ax, release in zip(axes, ("2.2.0", "2.3.0")):
        selected = filtered_metrics[filtered_metrics.model_release.eq(release)]
        table = (
            selected.pivot(index="model", columns="dataset", values="roc_auc")
            .loc[list(WEIRDO_SCORES), datasets]
            .to_numpy()
        )
        heatmap(
            ax,
            table,
            [MODEL_LABELS[name] for name in WEIRDO_SCORES],
            dataset_labels,
            f"MHCflurry {release} rank <= 1; failed rows score 0",
            0.4,
            0.8,
        )
    fig.suptitle("Presentation-gated Weirdo outputs | ROC AUC", fontsize=17, x=0.03, ha="left")
    fig.text(0.03, 0.01, "Every cohort row remains in the evaluation. Rows with presentation percentile rank >1 receive score 0; passing rows retain their Weirdo score.", fontsize=9)
    fig.tight_layout(rect=[0, 0.04, 1, 0.94], w_pad=2)
    save_figure(fig, out, "02_presentation_filtered_weirdo")

    for baseline, title, filename in [
        ("mhcflurry_2_2_0_presentation", "Geometric mean with MHCflurry 2.2.0 presentation", "03_geomean_2_2_0"),
        ("mhcflurry_2_3_0_presentation", "Geometric mean with MHCflurry 2.3.0 presentation", "04_geomean_2_3_0"),
    ]:
        selected = deltas[(deltas.baseline == baseline) & deltas.dataset.isin(datasets)]
        delta_table = selected.pivot(index="signal", columns="dataset", values="delta_auc").loc[list(WEIRDO_SCORES), datasets].to_numpy()
        combined_models = [f"{baseline}+{signal}" for signal in WEIRDO_SCORES]
        absolute_table = (
            metrics.pivot(index="model", columns="dataset", values="roc_auc")
            .loc[combined_models, datasets]
            .to_numpy()
        )
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
        heatmap(
            axes[0], absolute_table,
            [MODEL_LABELS[name] for name in WEIRDO_SCORES], dataset_labels,
            "Presentation/Weirdo geomean ROC AUC", 0.45, 0.85,
        )
        heatmap(
            axes[1], delta_table,
            [MODEL_LABELS[name] for name in WEIRDO_SCORES], dataset_labels,
            "Change versus presentation alone", -0.06, 0.06, cmap="PiYG",
        )
        fig.suptitle(title, fontsize=17, x=0.03, ha="left")
        fig.text(0.03, 0.01, "Both inputs are frozen. The geometric mean is deterministic and uses no T-cell outcomes or fitted parameters.", fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 0.94], w_pad=2)
        save_figure(fig, out, filename)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.8), squeeze=False)
    selected_models = [
        "netmhcpan_ba",
        "mhcflurry_2_2_0_presentation",
        "mhcflurry_2_3_0_presentation",
        "prime_2_1",
        "weirdo_virus_foreign_like",
    ]
    for column, dataset in enumerate(datasets):
        y, scores = curve_data[dataset]
        for index, model in enumerate(selected_models):
            fpr, tpr, _ = roc_curve(y, scores[model])
            precision, recall, _ = precision_recall_curve(y, scores[model])
            axes[0, column].plot(fpr, tpr, color=COLORS[index], lw=1.5, label=MODEL_LABELS[model])
            axes[1, column].plot(recall, precision, color=COLORS[index], lw=1.5)
        axes[0, column].plot([0, 1], [0, 1], "--", color="#adb5bd", lw=1)
        axes[1, column].axhline(y.mean(), color="#adb5bd", ls="--", lw=1)
        axes[0, column].set_title(COHORTS[dataset][1] + f"\n{int(y.sum()):,}/{len(y):,} positive", fontsize=10)
        axes[0, column].set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
        axes[1, column].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("Raw predictor curves on identical cohort rows", fontsize=17, x=0.03, ha="left")
    fig.tight_layout(rect=[0, 0.1, 1, 0.94])
    save_figure(fig, out, "05_common_roc_pr_curves")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    signal = "weirdo_virus_foreign_like"
    for ax, baseline, title in zip(
        axes,
        PRESENTATION_BASELINES,
        ("Beyond MHCflurry 2.2.0 presentation", "Beyond MHCflurry 2.3.0 presentation"),
    ):
        part = deltas[(deltas.baseline == baseline) & deltas.signal.eq(signal)].set_index("dataset").loc[datasets]
        ax.axvline(0, color="#879099", lw=1)
        ax.errorbar(part.delta_auc, np.arange(len(part)),
                    xerr=[part.delta_auc - part.delta_low, part.delta_high - part.delta_auc],
                    fmt="o", color="#753d8c", capsize=3)
        ax.set_yticks(np.arange(len(part)), [COHORTS[name][1] for name in datasets], fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Change in ROC AUC")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.15)
    fig.suptitle("Fixed Weirdo geomean | paired group-bootstrap intervals", fontsize=17, x=0.03, ha="left")
    fig.text(0.03, 0.01, "Both component scores and their geometric mean are fixed; intervals are not adjusted for multiple score definitions.", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    save_figure(fig, out, "06_geomean_score_intervals")


def headline_table(metrics):
    rows = []
    for dataset, (_, label) in COHORTS.items():
        part = metrics[metrics.dataset.eq(dataset)].set_index("model")
        rows.append(
            {
                "Dataset": label,
                "n (+)": f"{int(part.n.iloc[0]):,} ({int(part.n_positive.iloc[0]):,})",
                "NetMHCpan BA": part.loc["netmhcpan_ba", "roc_auc"],
                "NetMHCpan EL": part.loc["netmhcpan_el", "roc_auc"],
                "MHCflurry 2.2 pres.": part.loc["mhcflurry_2_2_0_presentation", "roc_auc"],
                "MHCflurry 2.3 pres.": part.loc["mhcflurry_2_3_0_presentation", "roc_auc"],
                "PRIME 2.1": part.loc["prime_2_1", "roc_auc"],
                "Weirdo P(virus)": part.loc["weirdo_p_viruses", "roc_auc"],
                "Weirdo P(mammals)": part.loc["weirdo_p_mammals", "roc_auc"],
                "Weirdo virus/mammal geomean": part.loc["weirdo_virus_mammal_like", "roc_auc"],
                "MHCflurry 2.3 + P(virus)": part.loc[
                    "mhcflurry_2_3_0_presentation+weirdo_p_viruses", "roc_auc"
                ],
                "MHCflurry 2.3 + P(mammals)": part.loc[
                    "mhcflurry_2_3_0_presentation+weirdo_p_mammals", "roc_auc"
                ],
                "MHCflurry 2.3 + virus/mammal": part.loc[
                    "mhcflurry_2_3_0_presentation+weirdo_virus_mammal_like",
                    "roc_auc",
                ],
            }
        )
    return pd.DataFrame(rows)


def headline_html(table: pd.DataFrame) -> str:
    """Render the compact table with best and near-best AUCs bolded per row."""
    parts = ["<table><thead><tr>"]
    parts.extend(f"<th>{html.escape(str(column))}</th>" for column in table.columns)
    parts.append("</tr></thead><tbody>")
    for _, row in table.iterrows():
        best = max(float(row[column]) for column in HEADLINE_SCORE_COLUMNS)
        parts.append("<tr>")
        for column in table.columns:
            value = row[column]
            if column in HEADLINE_SCORE_COLUMNS:
                formatted = f"{float(value):.4f}"
                if best - float(value) < 0.01:
                    formatted = f"<strong>{formatted}</strong>"
            else:
                formatted = html.escape(str(value))
            parts.append(f"<td>{formatted}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def raw_auc_table(metrics, models):
    datasets = list(COHORTS)
    values = (
        metrics[metrics.model.isin(models)]
        .pivot(index="model", columns="dataset", values="roc_auc")
        .loc[list(models), datasets]
    )
    values.index = [MODEL_LABELS[model] for model in models]
    values.columns = [
        "bladder",
        "melanoma",
        "basket",
        "YF 17D",
        "multi-virus",
    ]
    return values.rename_axis("Individual frozen predictor").reset_index()


def selection_table(provenance):
    records = provenance["predictor_provenance"]["predictors"]["weirdo"][
        "selection_policy"
    ]["candidates"]
    labels = {
        "composition_mlp": "Composition MLP",
        "cnn": "Residue CNN",
        "transformer": "Small Transformer",
        "esm_frozen": "Frozen ESM-2 8M",
        "esm_finetune": "Fine-tuned ESM-2 8M",
    }
    rows = [
        {
            "Taxonomy architecture": labels.get(record["architecture"], record["architecture"]),
            "Held-out origin AUC": record["value"],
            "Benchmark model": "selected" if record.get("selected") else "",
        }
        for record in records
        if record.get("eligible")
    ]
    return pd.DataFrame(rows).sort_values("Held-out origin AUC", ascending=False)


def selected_taxonomy_summary(provenance):
    model = provenance["predictor_provenance"]["predictors"]["weirdo"]["model"]
    evaluation = json.loads((ROOT / model / "evaluation.json").read_text())
    return evaluation["summary"]


METHODS = [
    "Scope: five human MHC class-I/CD8 response cohorts only. Bladder, melanoma and basket are IMPROVE cancer neoepitope cohorts. Yellow fever 17D is a 363-row vaccine multimer/tetramer cohort (131 positive, 232 negative; 250 unique 8--11mer peptides across 43 HLA alleles) assembled from IEDB records for PMIDs 28386132 and 32983097. Its negatives are peptide-HLA combinations tested without a positive response, not random all-proteome negatives. The multi-virus set measures T-cell activation. Their endpoints and prevalences differ, so cohort-specific results are primary and pooled results are supplemental.",
    "Common-row rule: within each cohort, NetMHCpan, MHCflurry, PRIME and Weirdo are evaluated on every one of the same peptide-HLA assay rows. There is no method-specific row filtering or imputation. The scoring run contains 18,738 assays, 18,576 unique peptide-HLA pairs and 16,323 unique peptides, with no missing predictions.",
    "Predictors: NetMHCpan 4.2c BA and EL percentile ranks; MHCflurry presentation release 2.2.0 (2020 weights) and release 2.3.0 (2026 weights), both executed by package 2.3.16; PRIME 2.1 immunogenicity percentile rank, run through mhctools with MixMHCpred 3.0; and the selected Weirdo peptide-content-v1 fine-tuned ESM-2 8M model (facebook/esm2_t6_8M_UR50D). MHCflurry affinity-percentile and processing outputs are also shown individually for each release. " + PRIME_FEATURE_NOTE,
    "Weirdo training rows are balanced samples, not every overlapping 8--15mer. For each split, taxonomy stratum and length, the builder samples proteins (caps: 10,000 train and 2,000 validation/test) and takes one random window per selected protein. It then scans every source protein for those candidates to assign multi-label membership and remove cross-fold occurrences. The selected SwissProt artifact has 799,357 training peptides.",
    "Non-circular Weirdo selection: architecture selection used only microbial-foreign-like AUC on a held-out peptide-origin taxonomy split, never these T-cell outcomes. The 0.6730 value distinguishes microbial-only from mammal-only peptides and is not a T-cell-response AUC. The frozen outputs are P(virus), P(bacteria), microbial-like, P(mammals), 1-P(mammals), 1-P(human), and three explicit geometric means.",
    "Why not the 0.7662 model: that newer complete-reference-proteome fine-tune has virus and mammal-family heads but no bacteria, archaea, fungi or protist heads. It therefore cannot supply P(bacteria) or the requested five-way microbial-like maximum. The 0.6730 fine-tuned ESM-2 is the best eligible single artifact that supplies every output in this uniform benchmark.",
    "Raw scores: lower NetMHCpan, MHCflurry affinity and PRIME percentile ranks are oriented as stronger; higher scores are stronger for every other output. No score direction is selected from response labels. All raw AUC and AP values use frozen predictor outputs without response-model fitting.",
    PRIME_OVERLAP_NOTE,
    "Presentation-gated mode: every assay row stays in the evaluation. A row keeps its frozen Weirdo score when the selected MHCflurry release's presentation percentile rank is <=1 and receives score 0 otherwise. This uses no T-cell outcome. It preserves Weirdo ranking among predicted presenters, treats all failed rows as tied at zero, and discards continuous presentation strength within both groups.",
    "Combined mode: for every row and every Weirdo output, the combined score is sqrt(MHCflurry presentation probability * Weirdo score). It retains continuous presentation information but has no learned weight or intercept: both inputs are frozen and no T-cell outcome is used to construct the score.",
    "Metrics and uncertainty: ROC AUC and stepwise average precision are reported; threshold accuracy is omitted because no decision threshold was specified. Paired percentile bootstrap resamples patient groups or peptide clusters while holding deterministic scores fixed. Intervals are exploratory, omit multiplicity correction, and should not be used to select a different Weirdo definition per cohort.",
    "Interpretation: these datasets were already available during method development and cannot provide new prospective confirmation. Cohorts differ in biology and assay design. AUC differences across cohorts should not be attributed solely to predictor quality.",
]


def write_reports(out, pdf_path, metrics, deltas, filtered_metrics, audit, provenance):
    from PIL import Image as PILImage
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    table = headline_table(metrics)
    baseline_individual = raw_auc_table(metrics, BASELINE_RAW_SCORES)
    weirdo_individual = raw_auc_table(metrics, WEIRDO_SCORES)
    architectures = selection_table(provenance)
    taxonomy = selected_taxonomy_summary(provenance)
    prime_overlap = pd.DataFrame(
        PRIME_OVERLAP_ROWS,
        columns=(
            "Dataset",
            "Exact peptide-HLA rows",
            "Sequence-overlap rows",
            "Unique overlapping peptides",
        ),
    )
    figure_names = ["01_common_raw_scores", "02_presentation_filtered_weirdo", "03_geomean_2_2_0", "04_geomean_2_3_0", "05_common_roc_pr_curves", "06_geomean_score_intervals"]
    figure_titles = ["Individual frozen scores", "Presentation-gated Weirdo scores", "Geometric means with MHCflurry 2.2.0", "Geometric means with MHCflurry 2.3.0", "Cohort ROC and PR curves", "Geometric-mean uncertainty"]
    captions = [
        "Every method in a cohort column uses the same rows and outcomes. This directly compares frozen ranking scores; it does not compare calibration or decision thresholds.",
        "Every row is retained. A row keeps its frozen Weirdo score when that MHCflurry release's presentation percentile rank is <=1 and receives score 0 otherwise; no T-cell outcome is used.",
        "Left: absolute AUC of sqrt(MHCflurry 2.2.0 presentation * Weirdo). Right: change versus presentation alone. The combination is deterministic and unfitted.",
        "Left: absolute AUC of sqrt(MHCflurry 2.3.0 presentation * Weirdo). Right: change versus presentation alone. The combination is deterministic and unfitted.",
        "Curves show five frozen outputs on identical rows. PR baselines differ with cohort prevalence; this is why AP should be read within a cohort.",
        "Paired group-bootstrap intervals for the deterministic geometric mean of presentation and the fixed Weirdo virus/foreign score.",
    ]
    body = "<h1>Weirdo and T-cell predictor baselines on common CD8 assay datasets</h1><p class='deck'>One predictor stack; identical rows per cohort; dataset-specific evaluation</p>"
    body += "<h2>Headline comparison: selected standalone and combined scores</h2>" + headline_html(table)
    body += "<p>The three standalone Weirdo columns are P(virus), P(mammals), and geomean[P(virus), P(mammals)]. In the next three headers, '+' means the deterministic geometric mean with MHCflurry 2.3.0 presentation; it is not an additive or fitted model. Bold marks the best ROC AUC in each cohort and every score less than 0.01 AUC below it; this is a descriptive tie band, not a significance test.</p>"
    body += "<h2>What is fitted to the T-cell data?</h2><p>Nothing. The MHC scores, Weirdo outputs, presentation-rank filters, and presentation/Weirdo geometric means are frozen or deterministic. T-cell outcomes are used only to evaluate the scores.</p>"
    body += "<h2>Is PRIME independent of these test rows?</h2><p>" + html.escape(PRIME_OVERLAP_NOTE) + "</p>"
    body += prime_overlap.to_html(index=False, border=0)
    body += "<p>" + html.escape(PRIME_SENSITIVITY_NOTE) + "</p>"
    body += "<p>" + html.escape(PRIME_FEATURE_NOTE) + "</p>"
    body += "<h2>What the 0.673 model-selection AUC means</h2>"
    body += f"<p>This is not a T-cell-response AUC. On the shared held-out taxonomy split, the score geomean[microbial-like, 1-P(mammals)] distinguished {taxonomy['n_microbial_only']:,} microbial-only peptides from {taxonomy['n_mammal_only']:,} mammal-only peptides; {taxonomy['n_shared_excluded']:,} peptides found in both groups were excluded. No HLA or T-cell outcome entered this selection. The square-root transform is monotone, so its ROC AUC is identical to the former raw product.</p>"
    body += architectures.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)
    body += "<p>The newer 0.7662 complete-reference-proteome fine-tune is not an eligible replacement for this uniform score family: it has virus and mammal-family heads, but no bacteria, archaea, fungi, or protist heads. It cannot produce P(bacteria) or the requested five-way microbial-like maximum.</p>"
    body += "<h2>Individual frozen baseline scores: ROC AUC</h2>" + baseline_individual.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)
    body += "<h2>Individual frozen Weirdo outputs: ROC AUC</h2>" + weirdo_individual.to_html(index=False, float_format=lambda x: f"{x:.4f}", border=0)
    body += "<p>These are raw ranking scores. The later figures show the same nine Weirdo modes after applying the presentation gate and in release-specific presentation geometric means. Under the gate, every row remains: rank <=1 rows keep their Weirdo score and all other rows score 0. The geometric mean also keeps every row but retains continuous presentation information, with no learned parameters.</p>"
    for name, title, caption in zip(figure_names, figure_titles, captions):
        body += f"<section><h2>{title}</h2><img src='{name}.png' alt='{title}'><p class='caption'>{caption}</p><a href='{name}.svg'>Vector SVG</a></section>"
    body += "<h2>Methods and limits</h2>" + "".join(f"<p>{html.escape(item)}</p>" for item in METHODS)
    body += "<p>Full numeric results: aggregate_metrics.csv, presentation_filtered_metrics.csv and incremental_metrics.csv. Reader-friendly breakouts: individual_predictor_auc.csv and presentation_geomean_delta_auc.csv. Deterministic row scores: scored_rows.csv.gz. Frozen pair scores and predictor provenance are linked from provenance.json.</p>"
    style = "body{font:16px/1.55 system-ui,sans-serif;color:#223044;max-width:1360px;margin:40px auto;padding:0 30px}h1{font-size:38px}h2{margin-top:40px;color:#275d86}p{max-width:1120px}.deck,.caption{color:#647184}img{width:100%;height:auto}table{font-size:12px;border-collapse:collapse;width:100%}th,td{padding:7px;text-align:left;border-bottom:1px solid #dfe4e9}th{background:#edf2f6}section{page-break-inside:avoid}a{color:#275d86}"
    (out / "report.html").write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Common CD8 predictor comparison</title><style>{style}</style></head><body>{body}</body></html>")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    page_w, page_h = landscape(A4)
    doc = SimpleDocTemplate(str(pdf_path), pagesize=(page_w, page_h), leftMargin=38, rightMargin=38, topMargin=35, bottomMargin=34)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportBody", fontName="Helvetica", fontSize=9.4, leading=13, spaceAfter=8, textColor=colors.HexColor("#223044")))
    styles.add(ParagraphStyle(name="ReportSmall", fontName="Helvetica", fontSize=8.2, leading=10.5, spaceAfter=6, textColor=colors.HexColor("#526174")))
    styles.add(ParagraphStyle(name="ReportTableHeader", fontName="Helvetica-Bold", fontSize=6.7, leading=7.4, textColor=colors.HexColor("#223044")))

    def pdf_dataframe(frame, widths, font_size=7.4, bold_best_columns=()):
        headers = [
            Paragraph(html.escape(str(column)), styles["ReportTableHeader"])
            for column in frame.columns
        ]
        values = [headers]
        for row in frame.itertuples(index=False, name=None):
            values.append(
                [
                    f"{value:.4f}" if isinstance(value, float) else str(value).replace("−", "-").replace("×", "*")
                    for value in row
                ]
            )
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfeaf1")),
            ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#223044")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 1.3),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#c8d0d8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f7f9")]),
        ]
        if bold_best_columns:
            for row_index, (_, row) in enumerate(frame.iterrows(), start=1):
                best = max(float(row[column]) for column in bold_best_columns)
                for column in bold_best_columns:
                    if best - float(row[column]) < 0.01:
                        column_index = frame.columns.get_loc(column)
                        table_style.append(
                            ("FONTNAME", (column_index, row_index), (column_index, row_index), "Helvetica-Bold")
                        )
        result = Table(values, colWidths=widths, repeatRows=1)
        result.setStyle(TableStyle(table_style))
        return result

    story = [Paragraph("WEIRDO / COMMON CD8 DATASETS", styles["Title"]),
             Paragraph("NetMHCpan 4.2c, MHCflurry 2.2.0 and 2.3.0, PRIME 2.1, and fine-tuned ESM-2 Weirdo", styles["ReportBody"]), Spacer(1, 8)]
    story += [
        pdf_dataframe(
            table,
            [68, 42, 44, 44, 52, 52, 44, 48, 52, 62, 72, 77, 80],
            font_size=6.2,
            bold_best_columns=HEADLINE_SCORE_COLUMNS,
        ),
        Spacer(1, 12),
        Paragraph(
            "Every number above is a frozen or deterministic-score ROC AUC on the "
            "identical assay rows within that cohort. Bold marks the best and every value less than "
            "0.01 AUC below it; this is a descriptive tie band, not a significance test. "
            "In the three rightmost headers, '+' means geometric mean with MHCflurry "
            "2.3.0 presentation, not addition or a fitted model. Confidence intervals "
            "appear only in the later detailed analysis.",
            styles["ReportBody"],
        ),
    ]

    story += [
        PageBreak(),
        Paragraph("What is fitted, and what 0.673 means", styles["Heading1"]),
        Paragraph(
            "<b>The individual MHC scores, individual Weirdo scores, and presentation "
            "filters are frozen.</b> The combined score is the deterministic geometric "
            "mean of MHCflurry presentation and one frozen Weirdo output. Nothing is "
            "fitted to T-cell outcomes.",
            styles["ReportBody"],
        ),
        Paragraph(
            f"<b>The 0.673 value is not a T-cell-response AUC.</b> It is held-out "
            f"peptide-origin AUC for geomean[microbial-like, 1 - P(mammals)]: "
            f"{taxonomy['n_microbial_only']:,} microbial-only peptides versus "
            f"{taxonomy['n_mammal_only']:,} mammal-only peptides, with "
            f"{taxonomy['n_shared_excluded']:,} shared peptides excluded. No HLA or "
            "T-cell response label enters model selection.",
            styles["ReportBody"],
        ),
        Paragraph(
            "<b>The 0.7662 complete-reference model is a different artifact.</b> "
            "It has virus and mammal-family heads but lacks bacteria, archaea, "
            "fungi and protist heads, so it cannot produce the requested uniform "
            "P(bacteria) and five-way microbial-like outputs. The 0.6730 model is "
            "the best eligible single artifact with every benchmark head.",
            styles["ReportBody"],
        ),
        pdf_dataframe(architectures, [180, 110, 90], font_size=8),
        Spacer(1, 10),
        Paragraph(
            "The first-page table contains exactly three standalone Weirdo views: "
            "P(virus), P(mammals), and geomean[P(virus), P(mammals)]. It also "
            "contains the geometric mean of MHCflurry 2.3.0 presentation with each "
            "of those three scores. All nine Weirdo outputs remain broken out in "
            "the detailed table below; no per-cohort Weirdo winner is selected.",
            styles["ReportSmall"],
        ),
    ]

    story += [
        PageBreak(),
        Paragraph("PRIME features and training overlap", styles["Heading1"]),
        Paragraph(
            "<b>PRIME is frozen here, but it is not fully independent of the benchmark.</b> "
            + html.escape(PRIME_OVERLAP_NOTE),
            styles["ReportBody"],
        ),
        pdf_dataframe(prime_overlap, [170, 120, 120, 135], font_size=8),
        Spacer(1, 12),
        Paragraph(html.escape(PRIME_SENSITIVITY_NOTE), styles["ReportBody"]),
        Paragraph(html.escape(PRIME_FEATURE_NOTE), styles["ReportBody"]),
        Paragraph(
            "The exact-match audit used the published PRIME2.0 Table S4 training "
            "manifest. Sequence-overlap counts include matches under a different HLA; "
            "exact peptide-HLA counts are the clearest direct leakage signal. We retain "
            "the rows to preserve the common-row comparison, but interpret PRIME with "
            "this limitation.",
            styles["ReportSmall"],
        ),
    ]

    story += [
        PageBreak(),
        Paragraph("Individual frozen predictors", styles["Heading1"]),
        Paragraph(
            "Raw ROC AUC on identical T-cell assay rows. These values do not use a "
            "response-model fit. AUC 0.5 is chance ranking.",
            styles["ReportBody"],
        ),
        Paragraph("MHC and T-cell predictor baselines", styles["Heading2"]),
        pdf_dataframe(baseline_individual, [220, 72, 72, 72, 72, 72], font_size=7.2),
        Spacer(1, 12),
        Paragraph("Weirdo outputs", styles["Heading2"]),
        pdf_dataframe(weirdo_individual, [245, 72, 72, 72, 72, 72], font_size=7.5),
        Spacer(1, 8),
        Paragraph(
            "Why the presentation-rank gate is not the geometric mean: rank <=1 rows "
            "keep their Weirdo score and every failed row is tied at 0. The gate "
            "therefore discards continuous presentation strength within both groups. "
            "The geometric mean preserves continuous presentation strength without "
            "learning a weight from T-cell outcomes.",
            styles["ReportSmall"],
        ),
    ]
    for name, title, caption in zip(figure_names, figure_titles, captions):
        story += [PageBreak(), Paragraph(title, styles["Heading1"])]
        path = out / f"{name}.png"
        with PILImage.open(path) as image:
            width, height = image.size
        scale = min((page_w - 76) / width, 415 / height)
        story += [Image(str(path), width=width * scale, height=height * scale), Paragraph(caption, styles["ReportSmall"])]
    for offset, title in ((0, "Methods"), (6, "Uncertainty and limits")):
        story += [PageBreak(), Paragraph(title, styles["Heading1"])]
        for item in METHODS[offset:offset + 6]:
            story.append(Paragraph(html.escape(item), styles["ReportBody"]))
    story += [PageBreak(), Paragraph("Reproducibility", styles["Heading1"]),
              Paragraph(f"Created {provenance['created_utc']}. {provenance['bootstrap_draws']} group-bootstrap draws per dataset. Common frozen predictions SHA-256: {provenance['common_predictions_sha256']}. All full-row and presentation-gated analyses use identical cohort rows; gated analyses also report release-specific passing counts.", styles["ReportBody"]),
              Paragraph("Reproduce predictor scores: python scripts/score_common_tcell_predictors.py", styles["ReportSmall"]),
              Paragraph("Reproduce report: python scripts/report_common_tcell_predictor_comparison.py --bootstrap-draws 1000", styles["ReportSmall"])]
    for dataset, item in audit.items():
        story.append(Paragraph(f"{dataset}: n={item['n']}, positive={item['n_positive']}, groups={item['n_groups']}; deterministic geometric-mean combination with no fitted folds.", styles["ReportSmall"]))
    def footer(canvas, document):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#647184"))
        canvas.drawString(38, 20, "Weirdo | common CD8 assay comparison | " + provenance["created_utc"][:10])
        canvas.drawRightString(page_w - 38, 20, str(document.page))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "output/cd8-common-predictor-comparison")
    parser.add_argument("--pdf", type=Path, default=ROOT / "output/pdf/weirdo-cd8-common-dataset-comparison.pdf")
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap_draws < 20:
        parser.error("Use at least 20 bootstrap draws")
    frame = pd.read_csv(PREDICTIONS, dtype={"group": str, "patient": str})
    score_columns = [
        column
        for column in frame
        if column.startswith(("netmhcpan_", "mhcflurry_", "prime_", "weirdo_"))
    ]
    if frame[score_columns].isna().any().any():
        raise ValueError("Common prediction table has missing scores")
    datasets = dataset_frames(frame)
    args.out.mkdir(parents=True, exist_ok=True)
    metric_rows, delta_rows, filtered_rows, audits, prediction_frames, curve_data = [], [], [], {}, [], {}
    for name, (label, part) in datasets.items():
        rows, deltas, audit, predictions, curves = evaluate_dataset(name, label, part, args.bootstrap_draws)
        metric_rows.extend(rows)
        delta_rows.extend(deltas)
        audits[name] = audit
        prediction_frames.append(predictions)
        curve_data[name] = curves
        if name in COHORTS:
            for release in ("2.2.0", "2.3.0"):
                filtered_rows.extend(
                    evaluate_presentation_filtered(
                        name, label, part, release, args.bootstrap_draws
                    )
                )
    metrics = pd.DataFrame(metric_rows)
    deltas = pd.DataFrame(delta_rows)
    filtered_metrics = pd.DataFrame(filtered_rows)
    metrics.to_csv(args.out / "aggregate_metrics.csv", index=False)
    deltas.to_csv(args.out / "incremental_metrics.csv", index=False)
    filtered_metrics.to_csv(args.out / "presentation_filtered_metrics.csv", index=False)
    pd.concat(
        [
            raw_auc_table(metrics, BASELINE_RAW_SCORES).assign(family="baseline"),
            raw_auc_table(metrics, WEIRDO_SCORES).assign(family="Weirdo"),
        ],
        ignore_index=True,
    ).to_csv(args.out / "individual_predictor_auc.csv", index=False)
    deltas[deltas.baseline.isin(PRESENTATION_BASELINES)].to_csv(
        args.out / "presentation_geomean_delta_auc.csv", index=False
    )
    pd.concat(prediction_frames, ignore_index=True).to_csv(args.out / "scored_rows.csv.gz", index=False)
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_draws": args.bootstrap_draws,
        "common_predictions": str(PREDICTIONS.relative_to(ROOT)),
        "common_predictions_sha256": sha256(PREDICTIONS),
        "predictor_provenance": json.loads(PREDICTION_PROVENANCE.read_text()),
        "script_sha256": sha256(Path(__file__)),
        "common_row_policy": "Every frozen predictor has a finite score for every assay row; combined scores are deterministic geometric means and use no response-model fit.",
    }
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    make_figures(args.out, metrics, deltas, filtered_metrics, curve_data)
    write_reports(
        args.out, args.pdf, metrics, deltas, filtered_metrics, audits, provenance
    )
    print(
        headline_table(metrics).to_string(index=False),
        flush=True,
    )
    print(f"Report: {args.out / 'report.html'}\nPDF: {args.pdf}", flush=True)


if __name__ == "__main__":
    main()
