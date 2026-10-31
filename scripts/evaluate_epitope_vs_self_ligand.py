#!/usr/bin/env python3
"""T-cell epitopes versus presented self ligands, matched on HLA, length and pMHC.

This is a *different task* from ranking peptides inside a T-cell study: every
candidate is presented, but one side is a human self ligand (HLA Ligand Atlas,
tolerated) and the other an IEDB peptide tested for T-cell reactivity. Each
IEDB peptide is matched without replacement to a self ligand of the same HLA and
length, nearest in (log10 MHCflurry presentation, log10 NetMHCpan BA rank) within
a caliper, so the pMHC core is near chance by construction.

The control that decides what the result means: the same matching is run for IEDB
peptides that were tested and found *non-reactive*. A score that separates both
positives and assay-negatives from self ligands equally well detects foreignness
or source, not T-cell epitope-ness; only a positive-specific excess would be
epitope signal. Both AUCs and their difference are reported.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import weirdo  # noqa: E402
from weirdo.pmhc_score import face_surprise  # noqa: E402

CALIPER = 0.5
NEIGHBORS = 25
HEADS = ("virus_foreign_like", "microbial_foreign_like", "one_minus_mammals", "one_minus_human", "p_viruses")


def pmhc_space(frame: pd.DataFrame) -> np.ndarray:
    """(log10 presentation, log10 BA rank) used for matching."""
    return np.column_stack(
        [
            np.log10(frame["mhcflurry_2_3_0_presentation_score"].clip(1e-6, 1)),
            np.log10(frame["netmhcpan_ba_rank"].clip(1e-3, None)),
        ]
    )


def match(cases: pd.DataFrame, controls: pd.DataFrame, scale: np.ndarray, caliper: float = CALIPER) -> pd.DataFrame:
    """Greedy nearest-neighbour matching without replacement within (hla, length)."""
    pairs = []
    cases = cases.assign(_length=cases["peptide"].str.len())
    controls = controls.assign(_length=controls["peptide"].str.len())
    for key, group in cases.groupby(["hla", "_length"]):
        pool = controls.loc[(controls["hla"] == key[0]) & (controls["_length"] == key[1])]
        if pool.empty:
            continue
        tree = cKDTree(pmhc_space(pool) / scale)
        used: set[int] = set()
        k = min(NEIGHBORS, len(pool))
        distances, neighbors = tree.query(pmhc_space(group) / scale, k=k)
        distances = np.asarray(distances).reshape(len(group), -1)
        neighbors = np.asarray(neighbors).reshape(len(group), -1)
        for row in range(len(group)):
            for dist, near in zip(distances[row], neighbors[row]):
                if dist > caliper:
                    break
                if int(near) not in used:
                    used.add(int(near))
                    pairs.append((group.index[row], pool.index[int(near)]))
                    break
    return pd.DataFrame(pairs, columns=["case", "control"])


def score_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Candidate scores to evaluate (higher = more epitope-like)."""
    args = (frame["peptide"], frame["mhcflurry_2_3_0_presentation_score"], frame["netmhcpan_ba_rank"])
    out = pd.DataFrame(index=frame.index)
    out["core"] = weirdo.pmhc_weirdo_score(*args, weirdo_weight=0.0, face_weight=0.0)["score"].to_numpy()
    out["presentation"] = frame["mhcflurry_2_3_0_presentation_score"].to_numpy()
    out["ba"] = -frame["netmhcpan_ba_rank"].to_numpy()
    for head in HEADS:
        out[head] = frame[head].to_numpy()
    out["face_surprise"] = [face_surprise(p) for p in frame["peptide"]]
    out["full"] = weirdo.pmhc_weirdo_score(*args, weirdo_scores=frame["virus_foreign_like"])["score"].to_numpy()
    return out


def auc_with_ci(y: np.ndarray, score: np.ndarray, clusters: np.ndarray, draws: int, seed: int) -> tuple[float, float, float]:
    index = {c: np.flatnonzero(clusters == c) for c in np.unique(clusters)}
    keys = list(index)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        pick = np.concatenate([index[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        if len(np.unique(y[pick])) == 2:
            boot.append(roc_auc_score(y[pick], score[pick]))
    return float(roc_auc_score(y, score)), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def evaluate(
    cases: pd.DataFrame, controls: pd.DataFrame, scores_cases: pd.DataFrame, scores_controls: pd.DataFrame,
    draws: int, caliper: float = CALIPER,
) -> pd.DataFrame:
    scale = pmhc_space(pd.concat([cases, controls])).std(axis=0)
    paired = match(cases, controls, scale, caliper)
    rows = []
    case_ids, control_ids = paired["case"].to_numpy(), paired["control"].to_numpy()
    y = np.r_[np.ones(len(paired)), np.zeros(len(paired))].astype(int)
    clusters = np.r_[cases.loc[case_ids, "pmid"].to_numpy(), cases.loc[case_ids, "pmid"].to_numpy()]
    for name in scores_cases.columns:
        score = np.r_[scores_cases.loc[case_ids, name].to_numpy(), scores_controls.loc[control_ids, name].to_numpy()]
        auc, low, high = auc_with_ci(y, score, clusters, draws, 84_342)
        rows.append({"score": name, "auc": auc, "low": low, "high": high, "pairs": len(paired)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--self-ligands", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=300)
    parser.add_argument("--caliper", type=float, default=CALIPER, help="matching caliper in pMHC SD units")
    args = parser.parse_args()
    pool = pd.read_csv(args.pool, low_memory=False)
    pool["pmid"] = pool["pmid"].astype(str)
    pool = pool.loc[pool["peptide"].str.len().between(8, 11)].drop_duplicates(["peptide", "hla"]).reset_index(drop=True)
    self_ligands = pd.read_csv(args.self_ligands, low_memory=False)
    scores_self = score_table(self_ligands)
    scores_pool = score_table(pool)
    results = {}
    for label, mask in (("positive", pool["outcome"].eq(1)), ("assay_negative", pool["outcome"].eq(0))):
        cases = pool.loc[mask]
        results[label] = evaluate(cases, self_ligands, scores_pool.loc[cases.index], scores_self, args.bootstrap_draws, args.caliper).set_index("score")
    table = results["positive"].join(results["assay_negative"], lsuffix="_pos", rsuffix="_neg")
    table["pos_minus_neg"] = table["auc_pos"] - table["auc_neg"]
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / f"epitope_vs_self_ligand_caliper{args.caliper:g}.csv")
    pd.set_option("display.width", 220)
    print(table[["pairs_pos", "auc_pos", "low_pos", "high_pos", "pairs_neg", "auc_neg", "low_neg", "high_neg", "pos_minus_neg"]].round(3).to_string())


if __name__ == "__main__":
    main()
