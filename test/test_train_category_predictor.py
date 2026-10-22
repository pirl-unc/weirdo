"""Tests for scripts/train_category_predictor.py, including a tiny end-to-end run."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import category_data as cd  # noqa: E402
import category_eval as ce  # noqa: E402
import train_category_predictor as tcp  # noqa: E402

from weirdo.scorers import MLPScorer  # noqa: E402

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
BIASED = {"bacteria": "AGRL", "viruses": "KNDI", "human": "EQSP", "plants": "CWYF"}


def synthetic_corpus(path, per_fold=150, length=90, seed=0):
    """Random proteins whose composition depends on the category, spread over all folds."""
    rng = np.random.RandomState(seed)
    rows = []
    taken = set()
    for fold in range(3):
        for _ in range(per_fold):
            index = 0
            while True:
                key = f"F{fold}{rng.randint(10**6)}"
                if key not in taken and cd.split_of_family(key) == fold:
                    taken.add(key)
                    break
                index += 1
            category = list(BIASED)[rng.randint(len(BIASED))]
            pool = AMINO_ACIDS + list(BIASED[category]) * 4
            rows.append((f"{key}_SPEC", "".join(rng.choice(pool, length)), category))
    pd.DataFrame(rows, columns=["identifier", "sequence", "category"]).to_csv(path)


def make_report(with_legacy=True, with_lengths=True):
    rng = np.random.RandomState(1)
    n, groups = 600, rng.randint(0, 5, size=600).astype(np.uint8)
    Y = (rng.rand(n, len(cd.CATEGORIES)) < 0.3).astype(float)
    P = np.clip(0.3 + 0.3 * (Y - 0.3) + 0.1 * rng.randn(*Y.shape), 0.02, 0.98)
    prevalence = np.full(len(cd.CATEGORIES), 0.3)
    entry = {"n": n, "metrics": ce.evaluate_categories(P, Y, prevalence), "summary_auc": ce.summary_auc(P, Y),
             "summary_quantiles": ce.summary_quantiles(P, Y)}
    report = {
        "protocol": {"bootstrap_resamples": 5, "family_buckets": 5},
        "test": entry,
        "validation": entry,
        "train_subset": entry,
        "within_family_holdout": entry,
        "test_intervals": ce.bootstrap_report(P, Y, groups, prevalence, n_boot=5, legacy_P=P),
        "lengths": {"9": entry} if with_lengths else {},
    }
    if with_legacy:
        report["legacy_baseline"] = entry
    return report


class TestHelpers:
    def test_parse_length_splits(self):
        assert tcp.parse_length_splits(None) == {}
        assert tcp.parse_length_splits("9=a/b,11=c") == {9: "a/b", 11: "c"}

    def test_markdown_has_every_section(self):
        text = tcp.render_markdown(make_report())
        for heading in ("### Held-out families", "### Pathogen versus self", "### Previous squared-loss head",
                        "### How much the split protocol matters", "### Longer peptides"):
            assert heading in text
        for name in cd.CATEGORIES:
            assert f"| {name} |" in text
        assert "**mean**" in text and "*(default, `combine='max'`)*" in text
        assert "`combine='sum'`" in text
        assert "reference distributions" in text and "only in bacteria/viruses" in text

    def test_markdown_omits_optional_sections(self):
        text = tcp.render_markdown(make_report(with_legacy=False, with_lengths=False))
        assert "### Previous squared-loss head" not in text
        assert "### Longer peptides" not in text
        assert "previous model (squared loss)" not in text


# The toy corpus has four categories, so six have no positives: their metrics are NaN by design.
@pytest.mark.filterwarnings("ignore:No positive class found:UserWarning")
@pytest.mark.filterwarnings("ignore:All-NaN slice encountered:RuntimeWarning")
def test_end_to_end_on_a_tiny_corpus(tmp_path, monkeypatch, capsys):
    corpus = tmp_path / "proteins.csv"
    synthetic_corpus(corpus)
    for k, out in ((8, "k8"), (9, "k9")):
        cd.build_and_cache(argparse.Namespace(
            sequences=str(corpus), out=str(tmp_path / out), k=k, salt=cd.DEFAULT_SALT, seed=0,
            n_train=3000 if k == 8 else 0, n_val=600 if k == 8 else 0, n_test=800, n_all=0,
            n_train_heldout=500 if k == 8 else 0, verify_table=None,
        ))
    model_dir, report_path, markdown_path = tmp_path / "model", tmp_path / "report.json", tmp_path / "report.md"
    monkeypatch.setattr(sys, "argv", [
        "train_category_predictor.py", "--splits", str(tmp_path / "k8"), "--out", str(model_dir),
        "--report", str(report_path), "--markdown", str(markdown_path),
        "--length-splits", f"9={tmp_path / 'k9'},10={tmp_path / 'missing'}",
        "--legacy-baseline", "--train-n", "3000", "--val-n", "600", "--test-n", "800",
        "--heldout-n", "500", "--train-eval-n", "500", "--hidden", "16", "--epochs", "4",
        "--bootstrap", "4", "--batch-size", "128",
    ])
    tcp.main()

    report = json.loads(report_path.read_text())
    assert report["test"]["n"] == 800
    assert set(report["test"]["metrics"]) == set(cd.CATEGORIES) | {"mean"}
    assert set(report["lengths"]) == {"9"}  # the missing 10-mer split is skipped, not fatal
    assert report["within_family_holdout"]["n"] == 500
    assert report["legacy_baseline"]["metrics"]["mean"]["ece"] > report["test"]["metrics"]["mean"]["ece"]
    assert report["model"]["n_features"] == 592
    assert "### Held-out families" in markdown_path.read_text()

    # the saved model reloads, ships its evaluation and gives probabilities
    loaded = MLPScorer.load(model_dir)
    assert loaded.target_categories == cd.CATEGORIES
    metadata = json.loads((model_dir / "metadata.json").read_text())
    assert metadata["evaluation"]["test"]["n"] == 800
    assert metadata["categories"] == cd.CATEGORIES
    probabilities = loaded.predict_proba(["ACDEFGHIKL", "KLMNPQRSTV"])
    assert probabilities.shape == (2, len(cd.CATEGORIES))
    assert ((probabilities > 0) & (probabilities < 1)).all()

    # refuses to clobber an existing model directory
    monkeypatch.setattr(sys, "argv", ["train_category_predictor.py", "--splits", str(tmp_path / "k8"),
                                      "--out", str(model_dir)])
    with pytest.raises(SystemExit, match="--overwrite"):
        tcp.main()
