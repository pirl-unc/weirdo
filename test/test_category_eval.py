"""Tests for the category-probability evaluation helpers (scripts/category_eval.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import category_eval as ce  # noqa: E402
from category_data import CATEGORIES  # noqa: E402


def calibrated_data(n=40_000, seed=0, sharpen=1.0):
    """Labels drawn from p, outputs reported as the sharpened logit of p."""
    rng = np.random.RandomState(seed)
    p = rng.beta(0.6, 1.5, size=n)
    y = (rng.rand(n) < p).astype(float)
    z = np.log(p) - np.log1p(-p)
    reported = 1.0 / (1.0 + np.exp(-sharpen * z))
    return reported, y


class TestCalibration:
    def test_perfectly_calibrated_probabilities(self):
        p, y = calibrated_data()
        intercept, slope = ce.calibration_line(p, y)
        assert slope == pytest.approx(1.0, abs=0.05)
        assert intercept == pytest.approx(0.0, abs=0.05)
        assert ce.expected_calibration_error(p, y) < 0.01

    def test_overconfident_outputs_have_slope_below_one(self):
        p, y = calibrated_data(sharpen=2.0)
        _, slope = ce.calibration_line(p, y)
        assert slope == pytest.approx(0.5, abs=0.05)
        assert ce.expected_calibration_error(p, y) > 0.05

    def test_constant_outputs_have_slope_zero_and_the_base_rate_intercept(self):
        _, y = calibrated_data()
        intercept, slope = ce.calibration_line(np.full_like(y, 0.3), y)
        assert slope == 0.0
        frequency = y.mean()
        assert intercept == pytest.approx(np.log(frequency / (1 - frequency)))

    def test_ece_of_a_biased_constant(self):
        y = np.zeros(1000)
        y[:100] = 1.0
        assert ce.expected_calibration_error(np.full(1000, 0.5), y) == pytest.approx(0.4)


class TestLogLoss:
    def test_matches_the_definition_and_the_prevalence_baseline(self):
        Y = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        P = np.array([[0.9, 0.2], [0.1, 0.1], [0.8, 0.6], [0.3, 0.7]])
        expected = -(Y * np.log(P) + (1 - Y) * np.log(1 - P)).mean(axis=0)
        np.testing.assert_allclose(ce.log_loss_per_category(P, Y), expected)
        prevalence = np.array([0.5, 0.5])
        np.testing.assert_allclose(ce.prevalence_log_loss(Y, prevalence), np.log(2.0) * np.ones(2))

    def test_extreme_probabilities_do_not_overflow(self):
        Y = np.array([[1.0], [0.0]])
        assert np.isfinite(ce.log_loss_per_category(np.array([[0.0], [1.0]]), Y)).all()


class TestEvaluateCategories:
    def test_informative_outputs_beat_the_prevalence_baseline(self):
        rng = np.random.RandomState(1)
        Y = (rng.rand(5000, len(CATEGORIES)) < 0.2).astype(float)
        P = np.clip(0.2 + 0.5 * (Y - 0.2) + 0.1 * rng.randn(*Y.shape), 0.01, 0.99)
        result = ce.evaluate_categories(P, Y, np.full(len(CATEGORIES), 0.2))
        assert set(result) == set(CATEGORIES) | {"mean"}
        for name in CATEGORIES:
            assert result[name]["auc"] == pytest.approx(roc_auc_score(Y[:, CATEGORIES.index(name)], P[:, CATEGORIES.index(name)]))
            assert result[name]["log_loss_skill"] > 0.0
        assert result["mean"]["auc"] > 0.9

    def test_single_class_column_gives_nan_auc(self):
        Y = np.zeros((10, 2))
        Y[:5, 0] = 1.0
        assert np.isnan(ce.auc_per_category(np.random.rand(10, 2), Y)[1])


class TestSummary:
    @staticmethod
    def _matrix(rows):
        P = np.full((len(rows), len(CATEGORIES)), 0.01)
        for i, row in enumerate(rows):
            for name, value in row.items():
                P[i, CATEGORIES.index(name)] = value
        return P

    def test_sum_and_max_ratios(self):
        P = self._matrix([{"bacteria": 0.9, "viruses": 0.3, "human": 0.2, "rodents": 0.1, "mammals": 0.4}])
        scores = ce.summary_scores(P)
        pathogen, mammalian = 0.9 + 0.3, 0.2 + 0.1 + 0.4
        assert scores["sum_ratio"][0] == pytest.approx(pathogen / (pathogen + mammalian))
        assert scores["max_ratio"][0] == pytest.approx(0.9 / (0.9 + 0.4))
        assert scores["sum_difference"][0] == pytest.approx(pathogen - mammalian)

    def test_masks_exclude_peptides_found_in_both_groups(self):
        Y = np.zeros((4, len(CATEGORIES)))
        Y[0, CATEGORIES.index("viruses")] = 1                                    # pathogen only
        Y[1, CATEGORIES.index("human")] = Y[1, CATEGORIES.index("mammals")] = 1  # self only
        Y[2, CATEGORIES.index("bacteria")] = Y[2, CATEGORIES.index("human")] = 1  # both
        Y[3, CATEGORIES.index("plants")] = 1                                     # neither
        pathogen_only, self_only = ce.summary_masks(Y)
        assert pathogen_only.tolist() == [True, False, False, False]
        assert self_only.tolist() == [False, True, False, False]

    def test_summary_auc_separates_the_two_groups(self):
        rng = np.random.RandomState(2)
        n = 400
        is_pathogen = rng.rand(n) < 0.5
        Y = np.zeros((n, len(CATEGORIES)))
        Y[is_pathogen, CATEGORIES.index("bacteria")] = 1
        Y[~is_pathogen, CATEGORIES.index("human")] = 1
        rows = [{"bacteria": 0.7, "human": 0.2} if f else {"bacteria": 0.2, "human": 0.7} for f in is_pathogen]
        P = np.clip(self._matrix(rows) + 0.05 * rng.randn(n, len(CATEGORIES)), 0.001, 0.999)
        result = ce.summary_auc(P, Y)
        assert result["n_pathogen_only"] == is_pathogen.sum()
        assert result["sum_ratio"] > 0.95 and result["max_ratio"] > 0.95

    def test_summary_auc_without_both_groups_reports_counts_only(self):
        Y = np.zeros((5, len(CATEGORIES)))
        Y[:, CATEGORIES.index("viruses")] = 1
        result = ce.summary_auc(np.full((5, len(CATEGORIES)), 0.3), Y)
        assert result == {"n_pathogen_only": 5.0, "n_self_only": 0.0}


    def test_summary_quantiles_describe_each_group(self):
        rng = np.random.RandomState(5)
        n = 600
        is_pathogen = rng.rand(n) < 0.5
        Y = np.zeros((n, len(CATEGORIES)))
        Y[is_pathogen, CATEGORIES.index("viruses")] = 1
        Y[~is_pathogen, CATEGORIES.index("human")] = 1
        P = np.full((n, len(CATEGORIES)), 0.05)
        P[is_pathogen, CATEGORIES.index("viruses")] = 0.8
        P[~is_pathogen, CATEGORIES.index("human")] = 0.8
        result = ce.summary_quantiles(P, Y)
        assert set(result) == {"max_ratio", "sum_ratio"}
        for score in result.values():
            assert set(score) == {"pathogen_only", "self_only", "all"}
            assert score["pathogen_only"]["n"] == is_pathogen.sum()
            assert score["pathogen_only"]["p50"] > 0.7 > 0.3 > score["self_only"]["p50"]
            assert score["all"]["p5"] <= score["all"]["p50"] <= score["all"]["p95"]
        # one pathogen output at 0.8 against a self maximum of 0.05: 0.8 / 0.85
        assert result["max_ratio"]["pathogen_only"]["p50"] == pytest.approx(0.8 / 0.85)
        assert ce.summary_quantiles(P, np.zeros_like(Y))["max_ratio"]["all"]["n"] == n  # groups may be empty
        assert set(ce.summary_quantiles(P, Y, scores=("sum_ratio",))) == {"sum_ratio"}

    def test_summary_quantiles_skip_empty_groups(self):
        Y = np.zeros((10, len(CATEGORIES)))
        Y[:, CATEGORIES.index("viruses")] = 1
        result = ce.summary_quantiles(np.full((10, len(CATEGORIES)), 0.2), Y)
        assert set(result["max_ratio"]) == {"pathogen_only", "all"}


class TestBootstrap:
    def test_bucket_members_partition_the_rows(self):
        groups = np.array([2, 0, 2, 1, 0, 5])
        members = ce.bucket_members(groups)
        assert [m.tolist() for m in members] == [[1, 4], [3], [0, 2], [5]]

    def test_interval_covers_the_estimate_and_widens_with_fewer_clusters(self):
        rng = np.random.RandomState(3)

        def interval_width(n_groups):
            groups = rng.randint(0, n_groups, size=6000).astype(np.uint8)
            shift = rng.randn(n_groups)[groups]  # cluster-level effect
            values = rng.randn(6000) + shift
            low, high = ce.cluster_bootstrap(lambda idx: np.array([values[idx].mean()]), groups, n_boot=300, seed=1)
            assert low[0] <= values.mean() <= high[0]
            return high[0] - low[0]

        assert interval_width(4) > interval_width(40)

    def test_bootstrap_report_has_an_interval_for_every_metric(self):
        rng = np.random.RandomState(4)
        n = 3000
        Y = (rng.rand(n, len(CATEGORIES)) < 0.3).astype(float)
        P = np.clip(0.3 + 0.4 * (Y - 0.3) + 0.15 * rng.randn(n, len(CATEGORIES)), 0.01, 0.99)
        groups = rng.randint(0, 10, size=n).astype(np.uint8)
        report = ce.bootstrap_report(P, Y, groups, np.full(len(CATEGORIES), 0.3), n_boot=20, seed=0, legacy_P=P)
        assert set(report["auc"]) == set(CATEGORIES)
        assert set(report["mean"]) == {"auc", "log_loss_skill"}
        assert {"sum_ratio", "max_ratio", "sum_difference", "legacy_max_ratio"} <= set(report["summary_auc"])
        for low, high in report["auc"].values():
            assert low <= high
