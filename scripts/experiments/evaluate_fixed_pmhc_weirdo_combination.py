#!/usr/bin/env python3
"""Leave-one-family-out search for a fixed Weirdo + MHCflurry + NetMHCpan score.

Nothing here is fitted to response labels. Each score is rank-normalized against a
pooled, label-free reference (empirical CDFs of the scored peptides), so the
pMHC predictors enter on the same scale. A candidate combination is

    core = mean(rank(MHCflurry presentation), rank(-NetMHCpan BA rank))
    additive:        core + lam * (rank(weirdo) - 0.5)
    multiplicative:  core * rank(weirdo) ** alpha

Datasets are grouped into families that give distinct looks at immunogenicity:
IMPROVE tumour cohorts, viral CD8 panels, TESLA validated neoepitopes, and Braun
RCC vaccine constructs. For each held-out family the combination is chosen on the
other families only (maximizing the worst-then-mean family AUC) and then scored once.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

PRESENTATION = "mhcflurry_2_3_0_presentation_score"
BA_RANK = "netmhcpan_ba_rank"
FAMILIES = {
    "improve_tumour": ("bladder", "melanoma", "Basket"),
    "viral": ("yellow_fever_17d", "multivirus_concise_set"),
    "tesla": ("tesla",),
    "braun": ("braun",),
}
WEIRDO_SCORES = (
    "weirdo_one_minus_mammals",
    "weirdo_one_minus_human",
    "weirdo_p_viruses",
    "weirdo_virus_foreign_like",
    "weirdo_microbial_foreign_like",
    "broad9__esm_160k__one_minus_mammals",
    "broad9__esm_160k__virus_foreign_like",
    "broad9__esm_160k__microbial_foreign_like",
    "broad9__esm_40k__virus_foreign_like",
)
LAMBDAS = (0.05, 0.1, 0.2, 0.33)
ALPHAS = (0.1, 0.25, 0.5)


def load(internal: Path, external: Path) -> pd.DataFrame:
    """Stack internal cohorts and the TESLA / Braun-construct external sets."""
    inner = pd.read_csv(internal, low_memory=False).rename(columns={"cohort": "dataset"})
    outer = pd.read_csv(external, low_memory=False)
    outer["dataset"] = outer["benchmark"].map(
        {"tesla_public_2022": "tesla", "braun_rcc_2025": "braun"}
    )
    columns = ["dataset", "group", "outcome", PRESENTATION, BA_RANK, *WEIRDO_SCORES]
    frame = pd.concat([inner[columns], outer[columns]], ignore_index=True)
    frame["group"] = frame["dataset"] + ":" + frame["group"].astype(str)
    if frame[columns].isna().any().any():
        raise ValueError("Scored table has missing values")
    return frame


def ecdf(values: np.ndarray) -> np.ndarray:
    """Label-free pooled rank in (0, 1]."""
    return rankdata(values, method="average") / len(values)


def candidates(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """Every fixed candidate score, keyed by a readable name."""
    presentation = ecdf(frame[PRESENTATION].to_numpy(float))
    binding = ecdf(-frame[BA_RANK].to_numpy(float))
    core = (presentation + binding) / 2
    scores = {
        "mhcflurry_presentation": presentation,
        "netmhcpan_ba": binding,
        "pmhc_rank_mean": core,
    }
    for name in WEIRDO_SCORES:
        weirdo = ecdf(frame[name].to_numpy(float))
        for lam in LAMBDAS:
            scores[f"add|{name}|{lam}"] = core + lam * (weirdo - 0.5)
        for alpha in ALPHAS:
            scores[f"mult|{name}|{alpha}"] = core * weirdo**alpha
    return scores


def dataset_auc(frame: pd.DataFrame, scores: dict[str, np.ndarray]) -> pd.DataFrame:
    """AUC of every candidate on every dataset."""
    rows = {}
    for dataset, index in frame.groupby("dataset").indices.items():
        outcome = frame["outcome"].to_numpy(int)[index]
        rows[dataset] = {
            name: roc_auc_score(outcome, values[index]) for name, values in scores.items()
        }
    return pd.DataFrame(rows)


def family_auc(table: pd.DataFrame) -> pd.DataFrame:
    """Mean AUC per family, so IMPROVE's three cohorts count once."""
    return pd.DataFrame(
        {family: table[list(members)].mean(axis=1) for family, members in FAMILIES.items()}
    )


def leave_one_family_out(by_family: pd.DataFrame) -> pd.DataFrame:
    """Choose on the other families (worst, then mean AUC), report the held-out one."""
    rows = []
    for held_out in by_family.columns:
        rest = by_family.drop(columns=held_out)
        ranking = pd.DataFrame({"worst": rest.min(axis=1), "mean": rest.mean(axis=1)})
        chosen = ranking.sort_values(["worst", "mean"], ascending=False).index[0]
        rows.append(
            {
                "held_out_family": held_out,
                "chosen": chosen,
                "chosen_auc": by_family.loc[chosen, held_out],
                "pmhc_rank_mean_auc": by_family.loc["pmhc_rank_mean", held_out],
                "mhcflurry_auc": by_family.loc["mhcflurry_presentation", held_out],
                "ba_auc": by_family.loc["netmhcpan_ba", held_out],
            }
        )
    return pd.DataFrame(rows)


def bootstrap_delta(
    frame: pd.DataFrame, a: np.ndarray, b: np.ndarray, draws: int, seed: int = 84_342
) -> pd.DataFrame:
    """Per-dataset group-bootstrap interval for AUC(a) - AUC(b)."""
    rng = np.random.default_rng(seed)
    rows = []
    outcome_all = frame["outcome"].to_numpy(int)
    for dataset, index in frame.groupby("dataset").indices.items():
        groups = frame["group"].to_numpy()[index]
        clusters = [index[groups == g] for g in np.unique(groups)]
        outcome = outcome_all[index]
        deltas = []
        for _ in range(draws):
            pick = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
            if len(np.unique(outcome_all[pick])) < 2:
                continue
            deltas.append(
                roc_auc_score(outcome_all[pick], a[pick]) - roc_auc_score(outcome_all[pick], b[pick])
            )
        low, high = np.percentile(deltas, [2.5, 97.5])
        rows.append(
            {
                "dataset": dataset,
                "delta_auc": roc_auc_score(outcome, a[index]) - roc_auc_score(outcome, b[index]),
                "low": low,
                "high": high,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    args = parser.parse_args()
    frame = load(args.internal, args.external)
    scores = candidates(frame)
    table = dataset_auc(frame, scores)
    by_family = family_auc(table)
    by_family["worst"] = by_family.min(axis=1)
    by_family["macro"] = by_family[list(FAMILIES)].mean(axis=1)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "dataset_auc_grid.csv")
    by_family.to_csv(args.output / "family_auc_grid.csv")
    selection = leave_one_family_out(by_family[list(FAMILIES)])
    selection.to_csv(args.output / "leave_one_family_out.csv", index=False)
    best_weirdo = (
        by_family.loc[~by_family.index.isin(["mhcflurry_presentation", "netmhcpan_ba", "pmhc_rank_mean"])]
        .sort_values(["worst", "macro"], ascending=False)
        .index[0]
    )
    for label, name in (("best_weirdo_in_sample", best_weirdo),):
        bootstrap_delta(
            frame, scores[name], scores["pmhc_rank_mean"], args.bootstrap_draws
        ).assign(candidate=name).to_csv(args.output / f"{label}_vs_core.csv", index=False)
    for name in ("mhcflurry_presentation", "netmhcpan_ba"):
        bootstrap_delta(
            frame, scores["pmhc_rank_mean"], scores[name], args.bootstrap_draws
        ).assign(candidate=f"pmhc_rank_mean vs {name}").to_csv(
            args.output / f"core_vs_{name}.csv", index=False
        )
    pd.set_option("display.width", 220)
    print(table.loc[["mhcflurry_presentation", "netmhcpan_ba", "pmhc_rank_mean"]].round(3))
    print(by_family.loc[["mhcflurry_presentation", "netmhcpan_ba", "pmhc_rank_mean"]].round(3))
    print(by_family.sort_values(["worst", "macro"], ascending=False).head(8).round(3))
    print(selection.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
