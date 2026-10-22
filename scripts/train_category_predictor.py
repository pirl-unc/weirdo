#!/usr/bin/env python3
"""Train and evaluate the WEIRDO species-category predictor.

Trains :class:`weirdo.scorers.MLPScorer` with one independent sigmoid output per
SwissProt division (binary cross-entropy) on the family-aware splits built by
``category_data.py``, then evaluates it on protein families it never saw:

* the test fold (primary result, with family-bucket bootstrap intervals);
* the validation fold used for early stopping;
* random k-mers from the *training* families (shows how much a naive random split
  would overstate performance);
* the training sample itself;
* longer peptides (9-, 10-, 11-mers) scored by averaging 8-mer windows;
* optionally the previous squared-loss model under the same protocol.

The test fold is scored once, after training has finished.

Example::

    python scripts/category_data.py build --sequences data/swissprot-sequences.csv \\
        --out data/category-splits --n-all 0 --n-train-heldout 300000
    python scripts/category_data.py build --sequences data/swissprot-sequences.csv \\
        --out data/category-splits-k9 --k 9 --n-train 0 --n-val 0 --n-all 0 --n-test 400000
    python scripts/train_category_predictor.py --splits data/category-splits \\
        --length-splits 9=data/category-splits-k9 --legacy-baseline \\
        --out weirdo/pretrained/swissprot-categories --markdown data/report.md
"""

import argparse
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import category_eval as ce  # noqa: E402
from category_data import CATEGORIES, load_split  # noqa: E402

DESCRIPTION = (
    "Multi-label sigmoid predictor of the ten SwissProt taxonomic divisions (archaea, "
    "bacteria, fungi, human, invertebrates, mammals, plants, rodents, vertebrates, viruses) "
    "for 8-mer windows. Output c is the probability that the peptide occurs in SwissProt "
    "proteins of division c; outputs are independent and need not sum to one."
)
DATA_SOURCE = "UniProtKB/Swiss-Prot taxonomic-division files, https://www.uniprot.org (CC BY 4.0)"
LABEL_DEFINITION = (
    "A k-mer is positive for a category when it occurs in at least one SwissProt protein of "
    "that UniProt taxonomic division. Train, validation and test k-mers come from disjoint "
    "protein families (entry-name mnemonic prefixes) and no k-mer occurs in two folds."
)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def git_describe() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "describe", "--always", "--dirty"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except Exception:
        return None


def parse_length_splits(text: Optional[str]) -> Dict[int, str]:
    if not text:
        return {}
    result = {}
    for item in text.split(","):
        length, directory = item.split("=", 1)
        result[int(length)] = directory
    return result


def evaluate_fold(
    scorer, peptides: List[str], Y: np.ndarray, prevalence: np.ndarray
) -> Dict[str, Any]:
    probabilities = scorer.predict_proba(peptides)
    return {
        "n": len(peptides),
        "metrics": ce.evaluate_categories(probabilities, Y, prevalence),
        "summary_auc": ce.summary_auc(probabilities, Y),
    }, probabilities


def build_model(args: argparse.Namespace, loss: str):
    from weirdo.scorers import MLPScorer

    return MLPScorer(
        k=8,
        hidden_layer_sizes=tuple(int(x) for x in args.hidden.split(",")),
        activation=args.activation,
        alpha=args.alpha,
        learning_rate_init=args.lr,
        batch_size=args.batch_size,
        random_state=args.seed,
        early_stopping=True,
        loss=loss,
    )


def train_model(args, loss, train, val):
    scorer = build_model(args, loss)
    scorer.train(
        train[0], train[1], val_peptides=val[0], val_labels=val[1],
        target_categories=CATEGORIES, epochs=args.epochs, verbose=True,
        cache_features=True, n_iter_no_change=args.patience,
    )
    return scorer


def format_interval(interval) -> str:
    return f"{interval[0]:.3f}–{interval[1]:.3f}"


def render_markdown(report: Dict[str, Any]) -> str:
    test = report["test"]
    metrics, intervals = test["metrics"], report["test_intervals"]
    lines: List[str] = []
    n_boot = report["protocol"]["bootstrap_resamples"]
    lines.append(f"### Held-out families: test fold ({test['n']:,} 8-mers)\n")
    lines.append(f"Intervals are 95% percentile intervals from {n_boot} bootstrap resamples of "
                 f"{report['protocol']['family_buckets']} family buckets.\n")
    lines.append("| Category | Prevalence | AUC (95% interval) | Average precision | "
                 "Log-loss skill (95% interval) | ECE | Calibration slope |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in CATEGORIES:
        m = metrics[name]
        lines.append(
            f"| {name} | {m['prevalence']:.3f} | {m['auc']:.3f} ({format_interval(intervals['auc'][name])}) | "
            f"{m['average_precision']:.3f} | {m['log_loss_skill']:.3f} "
            f"({format_interval(intervals['log_loss_skill'][name])}) | {m['ece']:.4f} | "
            f"{m['calibration_slope']:.2f} |"
        )
    m = metrics["mean"]
    lines.append(
        f"| **mean** | {m['prevalence']:.3f} | **{m['auc']:.3f}** ({format_interval(intervals['mean']['auc'])}) | "
        f"{m['average_precision']:.3f} | **{m['log_loss_skill']:.3f}** "
        f"({format_interval(intervals['mean']['log_loss_skill'])}) | {m['ece']:.4f} | "
        f"{m['calibration_slope']:.2f} |"
    )
    lines.append("\nLog-loss skill is `1 − log loss / log loss of always predicting the training "
                 "prevalence`; 0 means no better than the base rate. ECE is the expected "
                 "calibration error over 10 bins; a calibration slope of 1 is perfect.\n")

    summary = test["summary_auc"]
    ci = intervals["summary_auc"]
    lines.append("### Pathogen versus self\n")
    lines.append(f"AUC for telling {int(summary['n_pathogen_only']):,} k-mers found only in "
                 f"bacteria/viruses from {int(summary['n_self_only']):,} found only in "
                 "human/rodents/mammals.\n")
    lines.append("| Score | AUC (95% interval) |")
    lines.append("| --- | ---: |")
    labels = {
        "max_ratio": "max(pathogen) / (max(pathogen) + max(self))  *(default, `combine='max'`)*",
        "sum_ratio": "sum(pathogen) / (sum(pathogen) + sum(self))  *(`combine='sum'`)*",
        "sum_difference": "sum(pathogen) − sum(self)",
    }
    for key, label in labels.items():
        lines.append(f"| {label} | {summary[key]:.3f} ({format_interval(ci[key])}) |")
    if "legacy_baseline" in report:
        legacy = report["legacy_baseline"]
        lines.append(
            f"| previous model (squared loss), max ratio | {legacy['summary_auc']['max_ratio']:.3f} "
            f"({format_interval(ci['legacy_max_ratio'])}) |"
        )

    quantiles = test.get("summary_quantiles")
    if quantiles:
        lines.append("\nA typical peptide scores above 0.5, because the bacteria output has a base rate "
                     "near 0.4 against much lower base rates for the mammalian outputs. Read a "
                     "`foreignness` value against these reference distributions on the test fold, not "
                     "against 0.5:\n")
        lines.append("| Score | k-mers found | Number | 5th percentile | Median | 95th percentile |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        names = {"pathogen_only": "only in bacteria/viruses", "self_only": "only in human/rodents/mammals",
                 "all": "all test k-mers"}
        score_labels = {"max_ratio": "`combine='max'` (default)", "sum_ratio": "`combine='sum'`"}
        for score_key, score_label in score_labels.items():
            for key, label in names.items():
                if key in quantiles.get(score_key, {}):
                    q = quantiles[score_key][key]
                    lines.append(f"| {score_label} | {label} | {int(q['n']):,} | {q['p5']:.3f} | "
                                 f"{q['p50']:.3f} | {q['p95']:.3f} |")

    if "legacy_baseline" in report:
        legacy = report["legacy_baseline"]["metrics"]["mean"]
        lines.append("\n### Previous squared-loss head versus the sigmoid head (same data, same test fold)\n")
        lines.append("| Model | Mean AUC | Mean log loss | Log loss of the base rate | Mean ECE |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        lines.append(f"| previous (squared loss + sigmoid) | {legacy['auc']:.3f} | {legacy['log_loss']:.3f} | "
                     f"{legacy['log_loss_prevalence']:.3f} | {legacy['ece']:.3f} |")
        lines.append(f"| sigmoid + cross-entropy | {m['auc']:.3f} | {m['log_loss']:.3f} | "
                     f"{m['log_loss_prevalence']:.3f} | {m['ece']:.4f} |")

    lines.append("\n### How much the split protocol matters\n")
    lines.append("| Evaluation set | 8-mers | Mean AUC | Mean log-loss skill | Mean ECE |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    rows = [("training sample (seen during training)", report.get("train_subset")),
            ("random k-mers from the training families (not seen)", report.get("within_family_holdout")),
            ("validation fold (new families; used for early stopping)", report["validation"]),
            ("**test fold (new families)**", test)]
    for label, entry in rows:
        if entry is None:
            continue
        e = entry["metrics"]["mean"]
        lines.append(f"| {label} | {entry['n']:,} | {e['auc']:.3f} | {e['log_loss_skill']:.3f} | {e['ece']:.4f} |")

    if report.get("lengths"):
        lines.append("\n### Longer peptides (8-mer windows averaged)\n")
        lines.append("The model is trained on 8-mers; a longer peptide is scored as the mean of its "
                     "8-mer windows. Labels are presence of the whole peptide in a category, and the "
                     "base rate used for the skill is that set's own prevalence.\n")
        lines.append("| Length | Peptides | Mean AUC | Mean log-loss skill | Mean calibration slope | "
                     "Pathogen-vs-self AUC (max ratio) | Pathogen-vs-self AUC (sum ratio) |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for length, entry in sorted(report["lengths"].items(), key=lambda kv: int(kv[0])):
            e = entry["metrics"]["mean"]

            def cell(key):
                value = entry["summary_auc"].get(key)
                return "n/a" if value is None else format(value, ".3f")

            lines.append(
                f"| {length} | {entry['n']:,} | {e['auc']:.3f} | {e['log_loss_skill']:.3f} | "
                f"{e['calibration_slope']:.2f} | {cell('max_ratio')} | {cell('sum_ratio')} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--splits", required=True, help="Directory built by category_data.py (k=8)")
    parser.add_argument("--out", default=None, help="Model directory to write (omit to skip saving)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--report", default=None, help="Write the full evaluation as JSON")
    parser.add_argument("--markdown", default=None, help="Write the result tables as Markdown")
    parser.add_argument("--length-splits", default=None,
                        help="Comma-separated LENGTH=DIRECTORY pairs, e.g. 9=data/category-splits-k9")
    parser.add_argument("--legacy-baseline", action="store_true",
                        help="Also train the previous squared-loss model on the same data")
    parser.add_argument("--train-n", type=int, default=3_000_000)
    parser.add_argument("--val-n", type=int, default=300_000)
    parser.add_argument("--test-n", type=int, default=600_000)
    parser.add_argument("--heldout-n", type=int, default=300_000)
    parser.add_argument("--train-eval-n", type=int, default=100_000)
    parser.add_argument("--hidden", default="256,128,64")
    parser.add_argument("--activation", default="relu")
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else None
    if out_dir is not None and out_dir.exists() and not args.overwrite:
        raise SystemExit(f"{out_dir} exists; pass --overwrite to replace it")

    started = time.time()
    split_meta = json.loads((Path(args.splits) / "splits.json").read_text())
    train = load_split(args.splits, "train", args.train_n, seed=args.seed)
    val = load_split(args.splits, "val", args.val_n, seed=args.seed)
    test_peptides, Y_test, test_groups = load_split(
        args.splits, "test", args.test_n, seed=args.seed, return_groups=True
    )
    prevalence = train[1].mean(axis=0)
    log(f"train {len(train[0]):,}  val {len(val[0]):,}  test {len(test_peptides):,}")

    # ---- train the sigmoid / cross-entropy model -------------------------------------------
    scorer = train_model(args, "bce", train, val)
    if not scorer._is_logistic:
        raise RuntimeError("expected a sigmoid-head model")
    log(f"trained; best epoch {scorer._metadata.get('best_epoch')} of {scorer._metadata['n_epochs']}")

    # ---- evaluate (test fold only after training is finished) ---------------------------------------
    log("scoring the test fold")
    test_result, P_test = evaluate_fold(scorer, test_peptides, Y_test, prevalence)
    test_result["summary_quantiles"] = ce.summary_quantiles(P_test, Y_test)
    report: Dict[str, Any] = {
        "created": time.strftime("%Y-%m-%d"),
        "code_version": git_describe(),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "protocol": {
            "family_salt": split_meta["salt"], "k": split_meta["k"],
            "sequences": split_meta["sequences"], "proteins": split_meta["proteins"],
            "universe_sizes": split_meta["universe_sizes"],
            "k_mers_shared_across_folds_dropped": split_meta["shared_across_folds"],
            "bootstrap_resamples": args.bootstrap, "family_buckets": int(test_groups.max()) + 1,
            "train_n": len(train[0]), "val_n": len(val[0]), "test_n": len(test_peptides),
            "label_definition": LABEL_DEFINITION,
            "data_source": DATA_SOURCE,
        },
        "model": {
            "hidden_layer_sizes": [int(x) for x in args.hidden.split(",")],
            "activation": args.activation, "alpha": args.alpha, "learning_rate": args.lr,
            "batch_size": args.batch_size, "seed": args.seed,
            "n_features": int(scorer._metadata["n_features"]),
            "n_parameters": int(sum(w.size for w in scorer._model.coefs_)
                                + sum(b.size for b in scorer._model.intercepts_)),
            "epochs_run": int(scorer._metadata["n_epochs"]),
            "best_epoch": scorer._metadata.get("best_epoch"),
        },
        "history": scorer.training_history,
        "train_prevalence": dict(zip(CATEGORIES, [float(p) for p in prevalence])),
        "test": test_result,
    }

    legacy_P = None
    if args.legacy_baseline:
        log("training the previous squared-loss model for comparison")
        legacy = train_model(args, "squared", train, val)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            legacy_P = legacy.predict_proba(test_peptides)
        report["legacy_baseline"] = {
            "n": len(test_peptides),
            "metrics": ce.evaluate_categories(legacy_P, Y_test, prevalence),
            "summary_auc": ce.summary_auc(legacy_P, Y_test),
        }
        del legacy

    log(f"bootstrap intervals ({args.bootstrap} resamples)")
    report["test_intervals"] = ce.bootstrap_report(
        P_test, Y_test, test_groups, prevalence, n_boot=args.bootstrap, seed=args.seed, legacy_P=legacy_P
    )

    log("scoring validation, training sample and within-family holdout")
    val_result, _ = evaluate_fold(scorer, val[0], val[1], prevalence)
    report["validation"] = val_result
    train_eval = load_split(args.splits, "train", args.train_eval_n, seed=args.seed)
    report["train_subset"], _ = evaluate_fold(scorer, train_eval[0], train_eval[1], prevalence)
    report["within_family_holdout"] = None
    try:
        held = load_split(args.splits, "train_heldout", args.heldout_n, seed=args.seed)
    except KeyError:
        held = None
    if held is None or len(held[0]) == 0:
        log("no train_heldout k-mers in the splits; skipping the within-family comparison")
    else:
        report["within_family_holdout"], _ = evaluate_fold(scorer, held[0], held[1], prevalence)

    report["lengths"] = {}
    for length, directory in parse_length_splits(args.length_splits).items():
        log(f"scoring {length}-mers")
        try:
            peptides, Y, _ = load_split(directory, "test", None, seed=args.seed, return_groups=True)
        except (FileNotFoundError, KeyError) as error:
            log(f"skipping {length}-mers: {error}")
            continue
        result, _ = evaluate_fold(scorer, peptides, Y, Y.mean(axis=0))
        report["lengths"][str(length)] = result

    # ---- write outputs -------------------------------------------------------------------------------
    report["seconds"] = round(time.time() - started)
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=1))
    markdown = render_markdown(report)
    if args.markdown:
        Path(args.markdown).write_text(markdown)
    if out_dir is not None:
        scorer._metadata.update({
            "description": DESCRIPTION,
            "categories": CATEGORIES,
            "label_definition": LABEL_DEFINITION,
            "code_version": report["code_version"],
            "environment": report["environment"],
            "protocol": report["protocol"],
            "evaluation": {key: report[key] for key in (
                "test", "test_intervals", "validation", "train_subset", "within_family_holdout", "lengths"
            )},
        })
        if "legacy_baseline" in report:
            scorer._metadata["evaluation"]["legacy_baseline"] = report["legacy_baseline"]
        if out_dir.exists():
            for leftover in out_dir.iterdir():
                leftover.unlink()
        scorer.save(out_dir)
        log(f"saved model to {out_dir}")
    print(markdown)
    log(f"done in {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
