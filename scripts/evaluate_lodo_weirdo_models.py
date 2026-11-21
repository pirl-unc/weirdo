#!/usr/bin/env python3
"""Leave-one-family-out models that combine the pMHC core with shipped Weirdo heads.

Declared hypotheses (nothing outside this list was tried for this report):

* ``core``       rank-mean of MHCflurry presentation and -NetMHCpan BA rank
* ``fixed``      core + 0.1 * (rank(virus_foreign_like) - 0.5)   (no fitting)
* ``lin_*``      L2 logistic on [core, Weirdo head ranks]
* ``gate_*``     L2 logistic on [core, core * head ranks]  (taxonomy modulates pMHC)
* ``heads_only`` L2 logistic on Weirdo heads alone (diagnostic)

Fitted models never see the held-out family; datasets are weighted equally and
classes are balanced within each dataset. Features are pooled label-free ranks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

PRESENTATION = "mhcflurry_2_3_0_presentation_score"
BA_RANK = "netmhcpan_ba_rank"
FAMILIES = {
    "improve_tumour": ("bladder", "melanoma", "Basket"),
    "viral": ("yellow_fever_17d", "multivirus_concise_set"),
    "tesla": ("tesla",),
    "braun": ("braun",),
}
SUMMARY_HEADS = ("one_minus_mammals", "virus_foreign_like", "microbial_foreign_like", "p_viruses")
NON_HEADS = {"peptide"}
REGULARIZATION = (0.01, 0.1, 1.0)


def ecdf(values: np.ndarray) -> np.ndarray:
    """Label-free pooled rank in (0, 1]."""
    return rankdata(values, method="average") / len(values)


def load(internal: Path, external: Path, weirdo: Path) -> tuple[pd.DataFrame, list[str]]:
    """Stack datasets, attach shipped Weirdo heads, and rank-normalize every feature."""
    inner = pd.read_csv(internal, low_memory=False).rename(columns={"cohort": "dataset"})
    outer = pd.read_csv(external, low_memory=False)
    outer["dataset"] = outer["benchmark"].map(
        {"tesla_public_2022": "tesla", "braun_rcc_2025": "braun"}
    )
    columns = ["dataset", "group", "peptide", "hla", "outcome", PRESENTATION, BA_RANK]
    frame = pd.concat([inner[columns], outer[columns]], ignore_index=True)
    frame["group"] = frame["dataset"] + ":" + frame["group"].astype(str)
    scores = pd.read_csv(weirdo)
    heads = [c for c in scores.columns if c not in NON_HEADS]
    frame = frame.merge(scores, on="peptide", how="left", validate="many_to_one")
    if frame[[PRESENTATION, BA_RANK, *heads]].isna().any().any():
        raise ValueError("Scored table has missing values")
    frame["core"] = (
        ecdf(frame[PRESENTATION].to_numpy(float)) + ecdf(-frame[BA_RANK].to_numpy(float))
    ) / 2
    for head in heads:
        frame[f"r_{head}"] = ecdf(frame[head].to_numpy(float))
    return frame, heads


def design(frame: pd.DataFrame, kind: str, heads: list[str]) -> np.ndarray:
    """Feature matrix for one hypothesis kind."""
    ranks = frame[[f"r_{h}" for h in heads]].to_numpy(float) - 0.5
    core = frame[["core"]].to_numpy(float) - 0.5
    if kind == "lin":
        return np.hstack([core, ranks])
    if kind == "gate":
        return np.hstack([core, core * ranks])
    if kind == "heads_only":
        return ranks
    raise ValueError(kind)


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal total weight per dataset and per class within each dataset."""
    weights = np.zeros(len(frame))
    for _, index in frame.groupby("dataset").indices.items():
        outcome = frame["outcome"].to_numpy(int)[index]
        for label in (0, 1):
            mask = outcome == label
            if mask.any():
                weights[index[mask]] = 1.0 / mask.sum()
    return weights


def hypotheses(heads: list[str]) -> dict[str, tuple[str, list[str], float]]:
    """name -> (kind, head list, C)."""
    result = {}
    for kind in ("lin", "gate", "heads_only"):
        for label, subset in (("all", heads), ("summary", list(SUMMARY_HEADS))):
            for c in REGULARIZATION:
                result[f"{kind}_{label}_C{c}"] = (kind, subset, c)
    return result


def run(frame: pd.DataFrame, heads: list[str]) -> pd.DataFrame:
    """Held-out score of every hypothesis, one column each, aligned to ``frame``."""
    out = pd.DataFrame(index=frame.index)
    out["core"] = frame["core"]
    out["fixed"] = frame["core"] + 0.1 * (frame["r_virus_foreign_like"] - 0.5)
    weights = sample_weights(frame)
    family_of = {d: f for f, members in FAMILIES.items() for d in members}
    family = frame["dataset"].map(family_of)
    outcome = frame["outcome"].to_numpy(int)
    for name, (kind, subset, c) in hypotheses(heads).items():
        matrix = design(frame, kind, subset)
        score = np.zeros(len(frame))
        for held_out in FAMILIES:
            test = (family == held_out).to_numpy()
            model = LogisticRegression(C=c, max_iter=2_000, solver="lbfgs")
            model.fit(matrix[~test], outcome[~test], sample_weight=weights[~test] * (~test).sum() / weights[~test].sum())
            score[test] = model.decision_function(matrix[test])
        out[name] = score
    return out


def cluster_delta(frame: pd.DataFrame, a: np.ndarray, b: np.ndarray, draws: int, seed: int = 84_342) -> tuple[float, float, float]:
    """Group-bootstrap AUC(a) - AUC(b) within one dataset."""
    outcome = frame["outcome"].to_numpy(int)
    groups = frame["group"].to_numpy()
    clusters = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        pick = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        if len(np.unique(outcome[pick])) < 2:
            continue
        deltas.append(roc_auc_score(outcome[pick], a[pick]) - roc_auc_score(outcome[pick], b[pick]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(roc_auc_score(outcome, a) - roc_auc_score(outcome, b)), float(low), float(high)


def summarize(frame: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset AUC, then family means and the macro delta against the core."""
    rows = []
    for name in scores.columns:
        per_dataset = {
            d: roc_auc_score(frame["outcome"].to_numpy(int)[i], scores[name].to_numpy()[i])
            for d, i in frame.groupby("dataset").indices.items()
        }
        per_family = {f: np.mean([per_dataset[d] for d in members]) for f, members in FAMILIES.items()}
        rows.append({"hypothesis": name, **per_family, "macro": np.mean(list(per_family.values())), "worst": min(per_family.values())})
    table = pd.DataFrame(rows).set_index("hypothesis")
    for family in FAMILIES:
        table[f"d_{family}"] = table[family] - table.loc["core", family]
    table["d_macro"] = table["macro"] - table.loc["core", "macro"]
    table["families_improved"] = (table[[f"d_{f}" for f in FAMILIES]] > 0).sum(axis=1)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--weirdo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    args = parser.parse_args()
    frame, heads = load(args.internal, args.external, args.weirdo)
    scores = run(frame, heads)
    table = summarize(frame, scores)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "hypotheses.csv")
    pd.set_option("display.width", 250)
    print(table.sort_values("d_macro", ascending=False).round(3).to_string())
    best = table.drop(index=["core"]).sort_values("d_macro", ascending=False).index[0]
    rows = []
    for dataset, index in frame.groupby("dataset").indices.items():
        sub = frame.iloc[index]
        point, low, high = cluster_delta(sub, scores[best].to_numpy()[index], scores["core"].to_numpy()[index], args.bootstrap_draws)
        rows.append({"dataset": dataset, "best": best, "delta_auc": point, "low": low, "high": high})
    pd.DataFrame(rows).to_csv(args.output / "best_vs_core_by_dataset.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
