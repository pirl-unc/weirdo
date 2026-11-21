#!/usr/bin/env python3
"""Compatibility entry point and rank-metric helpers for the CD8 report.

There is one maintained benchmark implementation:
``report_common_tcell_predictor_comparison.py``. Running this historical entry
point delegates to it, so every generated report uses the same common rows,
standalone scores, presentation filters, and deterministic geometric means.
No response adapter is fitted.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def logit(values):
    """Stable logit used to display legacy continuous MHC score matrices."""
    values = np.clip(np.asarray(values, float), 1e-5, 1 - 1e-5)
    return np.log(values / (1 - values))


def aligned_scores(base, extra, keys, columns):
    """Validate keys, labels, and order before attaching frozen predictions."""
    if base.duplicated(keys).any() or extra.duplicated(keys).any():
        raise ValueError(f"Non-unique join keys: {keys}")
    if not base[keys].reset_index(drop=True).equals(
        extra[keys].reset_index(drop=True)
    ):
        raise ValueError("Saved scoring tables have different row identities or order")
    for label in ("response", "outcome"):
        if label in base and not base[label].equals(extra[label]):
            raise ValueError("Saved scoring tables disagree on response labels")
    return {name: extra[column].to_numpy(float) for name, column in columns.items()}


def ranking_plan(y, score):
    """Pre-sort ties once for exact weighted ROC AUC and average precision."""
    order = np.argsort(-np.asarray(score), kind="stable")
    sorted_score = np.asarray(score)[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_score)), len(order) - 1]
    return order, np.asarray(y, float)[order], ends


def weighted_rank_metrics(plan, weights):
    """ROC AUC and stepwise AP under integer or non-negative row weights."""
    order, y, ends = plan
    weight = np.asarray(weights)[order]
    tp = np.cumsum(weight * y)[ends]
    fp = np.cumsum(weight * (1 - y))[ends]
    if tp[-1] == 0 or fp[-1] == 0:
        return np.nan, np.nan
    tpr = np.r_[0, tp / tp[-1]]
    fpr = np.r_[0, fp / fp[-1]]
    auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=tp + fp > 0)
    ap = np.sum(np.diff(np.r_[0, tp]) / tp[-1] * precision)
    return float(auc), float(ap)


def cluster_metrics(y, scores, groups, draws, seed=20261026):
    """Paired bootstrap of whole groups for fixed, already-computed scores."""
    y = np.asarray(y, int)
    names = list(scores)
    plans = [ranking_plan(y, scores[name]) for name in names]
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    count = int(inverse.max()) + 1
    rng = np.random.default_rng(seed)
    sampled = np.full((draws, len(names), 2), np.nan)
    for draw in range(draws):
        multiplicity = np.bincount(
            rng.integers(count, size=count), minlength=count
        )
        weight = multiplicity[inverse]
        for index, plan in enumerate(plans):
            sampled[draw, index] = weighted_rank_metrics(plan, weight)
    metrics = []
    for index, name in enumerate(names):
        point = weighted_rank_metrics(plans[index], np.ones(len(y)))
        if not np.isclose(point[0], roc_auc_score(y, scores[name]), atol=1e-12):
            raise AssertionError("weighted ROC AUC does not match scikit-learn")
        if not np.isclose(
            point[1], average_precision_score(y, scores[name]), atol=1e-12
        ):
            raise AssertionError("weighted AP does not match scikit-learn")
        bounds = np.nanquantile(sampled[:, index], [0.025, 0.975], axis=0)
        metrics.append(
            {
                "model": name,
                "roc_auc": point[0],
                "average_precision": point[1],
                "roc_low": bounds[0, 0],
                "roc_high": bounds[1, 0],
                "ap_low": bounds[0, 1],
                "ap_high": bounds[1, 1],
                "bootstrap_valid": int(
                    np.isfinite(sampled[:, index, 0]).sum()
                ),
            }
        )
    return metrics, {name: sampled[:, index] for index, name in enumerate(names)}


def main():
    """Run the single maintained common-row benchmark."""
    from scripts.report_common_tcell_predictor_comparison import main as common_main

    common_main()


if __name__ == "__main__":
    main()
