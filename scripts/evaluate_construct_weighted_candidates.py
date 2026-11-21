#!/usr/bin/env python3
"""Rank-weighted candidate aggregation with a Weirdo term, per Braun construct.

Every distinct reference-absent peptide-HLA window that spans a construct's
mutated region is scored by a fixed, label-free candidate evidence

    e = core + lam * (rank(weirdo) - 0.5),
    core = mean(rank(MHCflurry presentation), rank(-NetMHCpan BA rank)),

then the construct score is a rank-weighted sum with steep dropoff,

    S = sum_k rho**(k-1) * e_(k)**gamma     (e sorted descending),

so rho -> 0 is "best candidate only" and rho = 1 is an unweighted sum. Ranks are
pooled-empirical CDFs over all candidates (labels are never used). The Weirdo
score is computed by the shipped ``peptide-content-v1-ensemble`` model.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PRESENTATION = "mhcflurry_2_3_0_presentation_score"
BA_RANK = "netmhcpan_ba_rank"
WEIRDO_MODEL = "peptide-content-v1-ensemble"
WEIRDO_SCORES = ("one_minus_mammals", "virus_foreign_like", "p_viruses", "microbial_foreign_like")
RHOS = (0.0, 0.1, 0.25, 0.5, 1.0)
GAMMAS = (1.0, 3.0, 6.0)
LAMBDAS = (0.0, 0.1, 0.2, 0.33)


def ecdf(values: np.ndarray) -> np.ndarray:
    """Label-free pooled rank in (0, 1]."""
    return rankdata(values, method="average") / len(values)


def score_weirdo(peptides: list[str]) -> pd.DataFrame:
    """Score unique peptides with the bundled taxonomy ensemble."""
    import weirdo

    scored = weirdo.predict_taxonomy_scores(peptides, model=WEIRDO_MODEL)
    scored = pd.DataFrame(scored).reset_index(drop=True)
    if scored["peptide"].tolist() != list(peptides):
        raise ValueError("Weirdo output order does not match the input peptides")
    return scored[["peptide", *WEIRDO_SCORES]].rename(
        columns={name: f"weirdo_{name}" for name in WEIRDO_SCORES}
    )


def prepare(candidates: pd.DataFrame, weirdo_scores: pd.DataFrame) -> pd.DataFrame:
    """Join Weirdo scores, drop duplicate windows, and compute the pMHC core."""
    frame = candidates.drop_duplicates(["record_id", "peptide", "hla"]).merge(
        weirdo_scores, on="peptide", how="left", validate="many_to_one"
    )
    columns = [PRESENTATION, BA_RANK, *(f"weirdo_{name}" for name in WEIRDO_SCORES)]
    if frame[columns].isna().any().any():
        raise ValueError("Candidate table has missing scores")
    presentation = ecdf(frame[PRESENTATION].to_numpy(float))
    binding = ecdf(-frame[BA_RANK].to_numpy(float))
    frame["core"] = (presentation + binding) / 2
    for name in WEIRDO_SCORES:
        frame[f"rank_{name}"] = ecdf(frame[f"weirdo_{name}"].to_numpy(float))
    return frame


def construct_scores(
    frame: pd.DataFrame, weirdo: str | None, lam: float, rho: float, gamma: float
) -> pd.Series:
    """Rank-weighted evidence sum per construct for one fixed configuration."""
    evidence = frame["core"].to_numpy(float)
    if weirdo is not None and lam > 0:
        evidence = evidence + lam * (frame[f"rank_{weirdo}"].to_numpy(float) - 0.5)
    evidence = np.clip(evidence, 1e-9, None) ** gamma
    scored = frame[["record_id"]].assign(evidence=evidence)
    result = {}
    for record, values in scored.groupby("record_id")["evidence"]:
        ordered = np.sort(values.to_numpy())[::-1]
        weights = rho ** np.arange(len(ordered)) if rho > 0 else np.eye(1, len(ordered))[0]
        result[record] = float(weights @ ordered)
    return pd.Series(result)


def configurations() -> list[tuple[str | None, float, float, float]]:
    """The full grid; (None, 0, rho, gamma) is the pMHC-only control."""
    grid = [(None, 0.0, rho, gamma) for rho in RHOS for gamma in GAMMAS]
    for name in WEIRDO_SCORES:
        for lam in LAMBDAS[1:]:
            grid.extend((name, lam, rho, gamma) for rho in RHOS for gamma in GAMMAS)
    return grid


def evaluate_grid(frame: pd.DataFrame, constructs: pd.DataFrame) -> pd.DataFrame:
    """AUC of every configuration on all constructs and on the frameshift subset."""
    outcome = constructs.set_index("record_id")["outcome"]
    frameshift = constructs.set_index("record_id")["mutation_class"].eq("frameshift")
    rows = []
    for name, lam, rho, gamma in configurations():
        score = construct_scores(frame, name, lam, rho, gamma).reindex(outcome.index).fillna(0.0)
        rows.append(
            {
                "weirdo": name or "none",
                "lam": lam,
                "rho": rho,
                "gamma": gamma,
                "auc_all": roc_auc_score(outcome, score),
                "auc_frameshift": roc_auc_score(outcome[frameshift], score[frameshift]),
            }
        )
    return pd.DataFrame(rows)


def patient_bootstrap_delta(
    constructs: pd.DataFrame, a: pd.Series, b: pd.Series, draws: int, seed: int = 84_342
) -> tuple[float, float, float]:
    """Patient-clustered interval for AUC(a) - AUC(b)."""
    constructs = constructs.set_index("record_id")
    a, b = a.reindex(constructs.index).fillna(0.0), b.reindex(constructs.index).fillna(0.0)
    patients = constructs["patient"].to_numpy()
    clusters = [np.flatnonzero(patients == p) for p in np.unique(patients)]
    outcome = constructs["outcome"].to_numpy(int)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        pick = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])
        if len(np.unique(outcome[pick])) < 2:
            continue
        deltas.append(
            roc_auc_score(outcome[pick], a.to_numpy()[pick])
            - roc_auc_score(outcome[pick], b.to_numpy()[pick])
        )
    low, high = np.percentile(deltas, [2.5, 97.5])
    point = roc_auc_score(outcome, a.to_numpy()) - roc_auc_score(outcome, b.to_numpy())
    return float(point), float(low), float(high)


def leave_one_patient_out(frame: pd.DataFrame, constructs: pd.DataFrame) -> pd.DataFrame:
    """Choose a configuration on the other patients, score the held-out patient's constructs."""
    outcome = constructs.set_index("record_id")["outcome"]
    patient = constructs.set_index("record_id")["patient"]
    cache = {
        config: construct_scores(frame, *config).reindex(outcome.index).fillna(0.0)
        for config in configurations()
    }
    rows = []
    for held_out in patient.unique():
        train = patient.index[patient != held_out]
        best = max(cache, key=lambda c: roc_auc_score(outcome[train], cache[c][train]))
        test = patient.index[patient == held_out]
        for record in test:
            rows.append({"record_id": record, "patient": held_out, "config": str(best), "score": cache[best][record]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--constructs", type=Path, required=True, help="construct-aggregates.csv.gz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=1_000)
    args = parser.parse_args()
    candidates = pd.read_csv(args.candidates)
    constructs = pd.read_csv(args.constructs)[["record_id", "patient", "mutation_class", "outcome"]]
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "weirdo-candidate-scores.csv"
    if cache.exists():
        weirdo_scores = pd.read_csv(cache)
    else:
        weirdo_scores = score_weirdo(sorted(candidates["peptide"].unique()))
        weirdo_scores.to_csv(cache, index=False)
    frame = prepare(candidates, weirdo_scores)
    grid = evaluate_grid(frame, constructs)
    grid.to_csv(args.output / "configuration_grid.csv", index=False)
    held_out = leave_one_patient_out(frame, constructs)
    held_out.to_csv(args.output / "leave_one_patient_out.csv", index=False)
    merged = constructs.merge(held_out[["record_id", "score"]], on="record_id")
    print("LOPO pooled AUC", round(roc_auc_score(merged["outcome"], merged["score"]), 4))
    pd.set_option("display.width", 200)
    print(grid.sort_values("auc_all", ascending=False).head(12).round(3).to_string(index=False))
    print(grid.loc[grid.weirdo.eq("none")].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
