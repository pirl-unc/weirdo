#!/usr/bin/env python3
"""Score the frozen tumor TCR-face/HLA adapter once on external neoantigen sets.

The adapter was fitted only on the IMPROVE tumor cohorts, so TESLA and Braun are
untouched confirmation data. Nothing here refits or tunes: it loads the packaged
adapter, scores each external endpoint, and reports paired group-bootstrap
AUC differences against MHCflurry presentation alone and NetMHCpan BA alone.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_tcr_visible_response_adapter import predict_adapter  # noqa: E402

BASELINES = {
    "mhcflurry_presentation": "mhcflurry_2_3_0_presentation_score",
    "netmhcpan_ba_rank": "netmhcpan_ba_rank",
}


def endpoints(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split the external table into its three evaluation endpoints."""
    braun = frame["benchmark"].eq("braun_rcc_2025")
    return {
        "tesla_public_2022": frame.loc[frame["benchmark"].eq("tesla_public_2022")],
        "braun_rcc_2025_all": frame.loc[braun],
        "braun_rcc_2025_frameshift": frame.loc[braun & frame["mutation_class"].eq("frameshift")],
    }


def orient(frame: pd.DataFrame, column: str) -> np.ndarray:
    """Return a higher-is-more-likely-responder version of a baseline score."""
    values = frame[column].to_numpy(float)
    return -values if column == "netmhcpan_ba_rank" else values


def paired_interval(
    frame: pd.DataFrame, candidate: np.ndarray, baseline: np.ndarray, draws: int, seed: int
) -> tuple[float, float]:
    """Group-clustered bootstrap interval for AUC(candidate) - AUC(baseline)."""
    outcome = frame["outcome"].to_numpy(int)
    index_by_group = [
        np.flatnonzero(mask)
        for mask in (frame["group"].to_numpy() == group for group in frame["group"].unique())
    ]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(draws):
        rows = np.concatenate(
            [index_by_group[i] for i in rng.integers(0, len(index_by_group), len(index_by_group))]
        )
        if len(np.unique(outcome[rows])) < 2:
            continue
        deltas.append(
            roc_auc_score(outcome[rows], candidate[rows]) - roc_auc_score(outcome[rows], baseline[rows])
        )
    low, high = np.percentile(deltas, [2.5, 97.5])
    return float(low), float(high)


def evaluate(frame: pd.DataFrame, bundle: dict, draws: int = 1_000, seed: int = 84_342) -> pd.DataFrame:
    """Score the frozen adapter on every endpoint and compare with baselines."""
    scored = frame.copy()
    scored["adapter"] = predict_adapter(bundle, scored)
    rows = []
    for name, subset in endpoints(scored).items():
        outcome = subset["outcome"].to_numpy(int)
        adapter = subset["adapter"].to_numpy(float)
        for baseline_name, column in BASELINES.items():
            baseline = orient(subset, column)
            low, high = paired_interval(subset, adapter, baseline, draws, seed)
            rows.append(
                {
                    "endpoint": name,
                    "n": len(subset),
                    "n_positive": int(outcome.sum()),
                    "adapter_auc": roc_auc_score(outcome, adapter),
                    "adapter_ap": average_precision_score(outcome, adapter),
                    "baseline": baseline_name,
                    "baseline_auc": roc_auc_score(outcome, baseline),
                    "baseline_ap": average_precision_score(outcome, baseline),
                    "delta_auc": roc_auc_score(outcome, adapter) - roc_auc_score(outcome, baseline),
                    "delta_auc_low": low,
                    "delta_auc_high": high,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", type=Path, required=True, help="External MHC-scored CSV(.gz)")
    parser.add_argument("--adapter", type=Path, required=True, help="Frozen tumor-adapter.joblib")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=1_000)
    args = parser.parse_args()
    results = evaluate(pd.read_csv(args.external), joblib.load(args.adapter), args.bootstrap_draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
