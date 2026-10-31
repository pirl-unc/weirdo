#!/usr/bin/env python3
"""Dose-response and permutation null for an additive Weirdo term on the pMHC core.

For every shipped head h and weight lam, ``score = core + lam * (rank(h) - 0.5)``;
the statistic is the macro (family-mean) AUC gain over the core. The null shuffles
each head's values within dataset, which preserves its marginal distribution and
breaks any link to the outcome, then takes the maximum gain over the same
head x lam grid -- so the observed best cell is judged against the multiplicity
of the whole search, not against zero.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_lodo_weirdo_models import FAMILIES, load  # noqa: E402

LAMBDAS = (0.05, 0.1, 0.2, 0.33)


def fast_auc(outcome: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC with average tie handling."""
    ranks = rankdata(score)
    n_pos = outcome.sum()
    n_neg = len(outcome) - n_pos
    return float((ranks[outcome == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def macro_gain(indices: dict, outcome: np.ndarray, core: np.ndarray, base: dict, head_ranks: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """Macro AUC gain and per-family gains for one head column and weight."""
    score = core + lam * (head_ranks - 0.5)
    per_dataset = {d: fast_auc(outcome[i], score[i]) - base[d] for d, i in indices.items()}
    per_family = np.array([np.mean([per_dataset[d] for d in members]) for members in FAMILIES.values()])
    return float(per_family.mean()), per_family


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--weirdo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=200)
    args = parser.parse_args()
    frame, heads = load(args.internal, args.external, args.weirdo)
    indices = frame.groupby("dataset").indices
    outcome = frame["outcome"].to_numpy(int)
    core = frame["core"].to_numpy(float)
    base = {d: fast_auc(outcome[i], core[i]) for d, i in indices.items()}
    rows = []
    for head in heads:
        ranks = frame[f"r_{head}"].to_numpy(float)
        for lam in LAMBDAS:
            gain, per_family = macro_gain(indices, outcome, core, base, ranks, lam)
            rows.append({"head": head, "lam": lam, "d_macro": gain, "families_improved": int((per_family > 0).sum()), **dict(zip(FAMILIES, per_family))})
    table = pd.DataFrame(rows).sort_values("d_macro", ascending=False)
    rng = np.random.default_rng(84_342)
    null_max = []
    columns = np.column_stack([frame[f"r_{h}"].to_numpy(float) for h in heads])
    for _ in range(args.permutations):
        shuffled = columns.copy()
        for i in indices.values():
            shuffled[i] = columns[rng.permutation(i)]
        best = max(
            macro_gain(indices, outcome, core, base, shuffled[:, j], lam)[0]
            for j in range(len(heads))
            for lam in LAMBDAS
        )
        null_max.append(best)
    null_max = np.array(null_max)
    observed = table["d_macro"].iloc[0]
    table["max_null_p"] = [(null_max >= g).mean() for g in table["d_macro"]]
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "weirdo_term_dose_response.csv", index=False)
    pd.Series(null_max, name="max_gain_under_null").to_csv(args.output / "weirdo_term_null_max.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.head(12).round(4).to_string(index=False))
    print("null max gain: median %.4f, 95th %.4f, 99th %.4f; observed best %.4f" % (
        np.median(null_max), np.percentile(null_max, 95), np.percentile(null_max, 99), observed))
    vfl = table[table["head"].eq("virus_foreign_like")].sort_values("lam")
    print(vfl.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
