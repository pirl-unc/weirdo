#!/usr/bin/env python3
"""Evaluation helpers for multi-label species-category probabilities.

All functions take a probability matrix ``P`` and a 0/1 label matrix ``Y`` of shape
``(n_peptides, n_categories)`` with columns in :data:`category_data.CATEGORIES`
order. Rank metrics (AUC, average precision) say how well the outputs order
peptides; log loss, calibration slope/intercept and expected calibration error say
whether the numbers can be read as probabilities.
"""

import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from category_data import CATEGORIES  # noqa: E402

PATHOGEN_CATEGORIES = ("bacteria", "viruses")
SELF_CATEGORIES = ("human", "rodents", "mammals")
EPSILON = 1e-7


def _clipped(P: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(P, dtype=np.float64), EPSILON, 1.0 - EPSILON)


def log_loss_per_category(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Mean binary cross-entropy of each column."""
    P = _clipped(P)
    return -(Y * np.log(P) + (1.0 - Y) * np.log1p(-P)).mean(axis=0)


def prevalence_log_loss(Y: np.ndarray, prevalence: np.ndarray) -> np.ndarray:
    """Log loss of always predicting each category's training prevalence."""
    return log_loss_per_category(np.broadcast_to(prevalence, Y.shape), Y)


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Mean absolute gap between predicted and observed frequency over equal-width bins."""
    p = np.asarray(p, dtype=np.float64)
    index = np.clip(np.digitize(p, np.linspace(0.0, 1.0, bins + 1)[1:-1]), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        members = index == b
        if members.any():
            error += members.mean() * abs(p[members].mean() - y[members].mean())
    return float(error)


def calibration_line(p: np.ndarray, y: np.ndarray, iterations: int = 50) -> Tuple[float, float]:
    """Intercept and slope of ``logit P(y=1) = a + b * logit(p)`` (calibrated: 0 and 1).

    A slope below 1 means the outputs are over-confident, above 1 under-confident.
    Constant outputs carry no information: the slope is reported as 0 and the
    intercept is the logit of the observed frequency.
    """
    p = _clipped(p)
    z = np.log(p) - np.log1p(-p)
    if z.std() < 1e-6:
        frequency = float(np.clip(np.mean(y), EPSILON, 1.0 - EPSILON))
        return float(np.log(frequency) - np.log1p(-frequency)), 0.0
    a, b = 0.0, 1.0
    for _ in range(iterations):
        mu = 0.5 * (1.0 + np.tanh(0.5 * (a + b * z)))
        w = mu * (1.0 - mu) + 1e-12
        gradient = np.array([np.sum(y - mu), np.sum((y - mu) * z)])
        hessian = np.array([[np.sum(w), np.sum(w * z)], [np.sum(w * z), np.sum(w * z * z)]])
        step = np.linalg.solve(hessian + 1e-9 * np.eye(2), gradient)
        a, b = a + step[0], b + step[1]
        if np.abs(step).max() < 1e-8:
            break
    return float(a), float(b)


def auc_per_category(P: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """ROC AUC of each column (nan when a column has a single class)."""
    values = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        if 0 < Y[:, j].sum() < len(Y):
            values[j] = roc_auc_score(Y[:, j], P[:, j])
    return values


def evaluate_categories(
    P: np.ndarray,
    Y: np.ndarray,
    prevalence: np.ndarray,
    categories: Sequence[str] = CATEGORIES,
) -> Dict[str, Dict[str, float]]:
    """Per-category metrics plus their mean, as nested dicts of floats."""
    P = np.asarray(P, dtype=np.float64)
    loss = log_loss_per_category(P, Y)
    baseline = prevalence_log_loss(Y, prevalence)
    auc = auc_per_category(P, Y)
    result: Dict[str, Dict[str, float]] = {}
    for j, name in enumerate(categories):
        intercept, slope = calibration_line(P[:, j], Y[:, j])
        result[name] = {
            "prevalence": float(Y[:, j].mean()),
            "auc": float(auc[j]),
            "average_precision": float(average_precision_score(Y[:, j], P[:, j])),
            "log_loss": float(loss[j]),
            "log_loss_prevalence": float(baseline[j]),
            "log_loss_skill": float(1.0 - loss[j] / baseline[j]),
            "ece": expected_calibration_error(P[:, j], Y[:, j]),
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "mean_probability": float(P[:, j].mean()),
        }
    keys = next(iter(result.values())).keys()
    result["mean"] = {key: float(np.mean([result[c][key] for c in categories])) for key in keys}
    return result


def group_indices(
    categories: Sequence[str],
    pathogen: Sequence[str] = PATHOGEN_CATEGORIES,
    self_group: Sequence[str] = SELF_CATEGORIES,
) -> Tuple[List[int], List[int]]:
    names = list(categories)
    return [names.index(c) for c in pathogen], [names.index(c) for c in self_group]


def summary_scores(
    P: np.ndarray,
    categories: Sequence[str] = CATEGORIES,
    pathogen: Sequence[str] = PATHOGEN_CATEGORIES,
    self_group: Sequence[str] = SELF_CATEGORIES,
) -> Dict[str, np.ndarray]:
    """Pathogen-versus-self summaries of a probability matrix.

    ``sum_ratio`` is ``sum(pathogen) / (sum(pathogen) + sum(self))``, ``max_ratio``
    the same with maxima, and ``sum_difference`` is ``sum(pathogen) - sum(self)``.
    """
    pathogen_idx, self_idx = group_indices(categories, pathogen, self_group)
    P = np.asarray(P, dtype=np.float64)
    path_sum, self_sum = P[:, pathogen_idx].sum(axis=1), P[:, self_idx].sum(axis=1)
    path_max, self_max = P[:, pathogen_idx].max(axis=1), P[:, self_idx].max(axis=1)
    return {
        "sum_ratio": path_sum / (path_sum + self_sum),
        "max_ratio": path_max / (path_max + self_max),
        "sum_difference": path_sum - self_sum,
    }


def summary_masks(
    Y: np.ndarray,
    categories: Sequence[str] = CATEGORIES,
    pathogen: Sequence[str] = PATHOGEN_CATEGORIES,
    self_group: Sequence[str] = SELF_CATEGORIES,
) -> Tuple[np.ndarray, np.ndarray]:
    """Masks of peptides found only in pathogen categories / only in self categories."""
    pathogen_idx, self_idx = group_indices(categories, pathogen, self_group)
    in_pathogen = Y[:, pathogen_idx].max(axis=1) > 0
    in_self = Y[:, self_idx].max(axis=1) > 0
    return in_pathogen & ~in_self, in_self & ~in_pathogen


def summary_auc(
    P: np.ndarray,
    Y: np.ndarray,
    categories: Sequence[str] = CATEGORIES,
    pathogen: Sequence[str] = PATHOGEN_CATEGORIES,
    self_group: Sequence[str] = SELF_CATEGORIES,
) -> Dict[str, float]:
    """AUC of each summary score for pathogen-only versus self-only peptides."""
    pathogen_only, self_only = summary_masks(Y, categories, pathogen, self_group)
    keep = pathogen_only | self_only
    result = {"n_pathogen_only": float(pathogen_only.sum()), "n_self_only": float(self_only.sum())}
    if pathogen_only.sum() == 0 or self_only.sum() == 0:
        return result
    scores = summary_scores(P[keep], categories, pathogen, self_group)
    truth = pathogen_only[keep].astype(int)
    for name, values in scores.items():
        result[name] = float(roc_auc_score(truth, values))
    return result


def summary_quantiles(
    P: np.ndarray,
    Y: np.ndarray,
    categories: Sequence[str] = CATEGORIES,
    pathogen: Sequence[str] = PATHOGEN_CATEGORIES,
    self_group: Sequence[str] = SELF_CATEGORIES,
    quantiles: Sequence[float] = (5.0, 50.0, 95.0),
    scores: Sequence[str] = ("max_ratio", "sum_ratio"),
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Quantiles of the ratio scores for k-mers found only in pathogen categories, only in
    self categories, and for all k-mers, keyed by score name and then by k-mer set.

    ``max_ratio`` is the default foreignness score and ``sum_ratio`` the ``combine='sum'``
    alternative. Because the base rate of the pathogen group (bacteria is in 40% of 8-mers)
    exceeds the base rate of the self group, a typical peptide scores well above 0.5;
    compare a score with these reference distributions rather than with 0.5.
    """
    all_scores = summary_scores(P, categories, pathogen, self_group)
    pathogen_only, self_only = summary_masks(Y, categories, pathogen, self_group)
    groups = {"pathogen_only": pathogen_only, "self_only": self_only,
              "all": np.ones(len(P), dtype=bool)}
    result: Dict[str, Dict[str, Dict[str, float]]] = {}
    for score_name in scores:
        score = all_scores[score_name]
        result[score_name] = {}
        for name, mask in groups.items():
            if mask.sum() == 0:
                continue
            values = np.percentile(score[mask], list(quantiles))
            result[score_name][name] = {
                "n": float(mask.sum()),
                **{f"p{int(q)}": float(v) for q, v in zip(quantiles, values)},
            }
    return result


def bucket_members(groups: np.ndarray) -> List[np.ndarray]:
    """Row indices of each bucket id present in ``groups`` (ascending id order)."""
    order = np.argsort(groups, kind="stable")
    ids, starts = np.unique(groups[order], return_index=True)
    return [order[a:b] for a, b in zip(starts, list(starts[1:]) + [len(order)])]


def cluster_bootstrap(
    statistic: Callable[[np.ndarray], np.ndarray],
    groups: np.ndarray,
    n_boot: int = 100,
    seed: int = 0,
    level: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """Percentile interval of ``statistic(row_indices)`` resampling whole buckets.

    Resampling buckets of protein families (rather than single k-mers) keeps the
    correlation among k-mers of one family, so the interval reflects how much the
    result depends on which families are in the test fold.
    """
    members = bucket_members(groups)
    rng = np.random.RandomState(seed)
    draws = []
    for _ in range(n_boot):
        chosen = rng.randint(0, len(members), size=len(members))
        draws.append(np.asarray(statistic(np.concatenate([members[g] for g in chosen])), dtype=float))
    draws_array = np.vstack(draws)
    tail = 100.0 * (1.0 - level) / 2.0
    return (np.nanpercentile(draws_array, tail, axis=0),
            np.nanpercentile(draws_array, 100.0 - tail, axis=0))


def bootstrap_report(
    P: np.ndarray,
    Y: np.ndarray,
    groups: np.ndarray,
    prevalence: np.ndarray,
    categories: Sequence[str] = CATEGORIES,
    n_boot: int = 100,
    seed: int = 0,
    legacy_P: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """95% cluster-bootstrap intervals for AUC, log-loss skill and the summary AUCs."""
    n = len(categories)
    summary_names = ["sum_ratio", "max_ratio", "sum_difference"]

    def statistic(index: np.ndarray) -> np.ndarray:
        p, y = P[index], Y[index]
        loss = log_loss_per_category(p, y)
        skill = 1.0 - loss / prevalence_log_loss(y, prevalence)
        auc = auc_per_category(p, y)
        summary = summary_auc(p, y, categories)
        values = [auc, skill, [auc.mean(), skill.mean()],
                  [summary.get(name, np.nan) for name in summary_names]]
        if legacy_P is not None:
            legacy = summary_auc(legacy_P[index], y, categories)
            values.append([legacy.get("max_ratio", np.nan)])
        return np.concatenate([np.ravel(v) for v in values])

    low, high = cluster_bootstrap(statistic, groups, n_boot=n_boot, seed=seed)

    def pack(offset: int, count: int) -> List[List[float]]:
        return [[float(low[offset + i]), float(high[offset + i])] for i in range(count)]

    report: Dict[str, Dict[str, List[float]]] = {
        "auc": dict(zip(categories, pack(0, n))),
        "log_loss_skill": dict(zip(categories, pack(n, n))),
        "mean": dict(zip(["auc", "log_loss_skill"], pack(2 * n, 2))),
        "summary_auc": dict(zip(summary_names, pack(2 * n + 2, 3))),
    }
    if legacy_P is not None:
        report["summary_auc"]["legacy_max_ratio"] = pack(2 * n + 5, 1)[0]
    return report
