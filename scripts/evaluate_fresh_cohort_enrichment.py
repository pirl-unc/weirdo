#!/usr/bin/env python3
"""Short-list view of the frozen score on a fresh cohort (secondary, post hoc).

For each eligible study, R-precision is the fraction of positives among the top-k
ranked pairs, where k is that study's number of positives (so a random ranking
scores the study's prevalence). Reported as the mean within-study gain over the
core with a study-clustered bootstrap interval. This is a post-hoc view added after
the primary AUC result and cannot rescue it.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_fresh_cohort import eligible, score_variants  # noqa: E402


def r_precision(outcome: np.ndarray, score: np.ndarray) -> float:
    """Positives among the top-k by score, k = number of positives (ties broken by order)."""
    k = int(outcome.sum())
    top = np.argsort(-score, kind="stable")[:k]
    return float(outcome[top].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2_000)
    args = parser.parse_args()
    frame = pd.read_csv(args.scored, low_memory=False)
    frame["pmid"] = frame["pmid"].astype(str)
    scores = score_variants(frame)
    studies = eligible(frame)
    y = frame["outcome"].to_numpy(int)
    rows = {}
    for study in studies:
        m = frame["pmid"].eq(study).to_numpy()
        rows[study] = {name: r_precision(y[m], scores[name].to_numpy()[m]) for name in scores.columns}
        rows[study]["prevalence"] = y[m].mean()
    table = pd.DataFrame(rows).T
    picks = np.random.default_rng(84_342).integers(0, len(studies), (args.draws, len(studies)))
    out = []
    for name in ("presentation", "ba", "core_vfl", "core_face", "full"):
        gap = (table[name] - table["core"]).to_numpy()
        boot = gap[picks].mean(axis=1)
        out.append({"score": name, "mean_r_precision": table[name].mean(), "gain_vs_core": gap.mean(),
                    "low": np.percentile(boot, 2.5), "high": np.percentile(boot, 97.5),
                    "studies_improved": int((gap > 0).sum()), "studies": len(studies)})
    result = pd.DataFrame(out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print("mean prevalence (random R-precision):", round(table["prevalence"].mean(), 4), "core:", round(table["core"].mean(), 4))
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
