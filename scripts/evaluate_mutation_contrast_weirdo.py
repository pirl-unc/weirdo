#!/usr/bin/env python3
"""Does the Weirdo mutant-minus-wild-type contrast add to the pMHC core?

Only the IMPROVE cohorts ship wild-type peptides. For each mutant window the
shipped ``peptide-content-v1-ensemble`` is scored on the mutant and its wild-type
counterpart; ``delta_h = h(mutant) - h(wild type)`` says how far the substitution
moved the peptide along head ``h`` (for example away from mammalian-like).

Hypotheses, all declared up front: an additive ``core + lam * (rank(delta_h) - 0.5)``
over every head and lam (judged against a within-cohort permutation null that takes
the maximum over the same grid), and leave-one-cohort-out logistic fits on
[core, delta summary heads]. Windows whose wild type equals the mutant (delta = 0)
stay in the evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_lodo_weirdo_models import SUMMARY_HEADS, ecdf, load  # noqa: E402
from scripts.evaluate_weirdo_term_null import fast_auc  # noqa: E402

COHORTS = ("bladder", "melanoma", "Basket")
LAMBDAS = (0.1, 0.2, 0.33)
REGULARIZATION = (0.01, 0.1, 1.0)
MODEL = "peptide-content-v1-ensemble"


def wild_type_scores(wild_types: list[str], cache: Path) -> pd.DataFrame:
    """Shipped-ensemble heads for every distinct wild-type peptide (cached)."""
    if cache.exists():
        return pd.read_csv(cache)
    import weirdo

    scored = pd.DataFrame(weirdo.predict_taxonomy_scores(wild_types, model=MODEL)).reset_index(drop=True)
    if scored["peptide"].tolist() != wild_types:
        raise ValueError("Weirdo output order does not match the input peptides")
    scored.to_csv(cache, index=False)
    return scored


def build(frame: pd.DataFrame, heads: list[str], raw: Path, cache: Path) -> pd.DataFrame:
    """IMPROVE rows with a wild-type partner and rank-normalized contrasts."""
    table = pd.read_csv(raw, sep="\t", low_memory=False)
    table["dataset"] = table["cohort"]
    table["peptide"] = table["Mut_peptide"]
    # raw alleles look like HLA-A01:01; the scored tables use HLA-A*01:01
    table["hla"] = table["HLA_allele"].str.replace(r"^(HLA-[A-Z]+)(\d)", r"\1*\2", regex=True)
    keys = ["dataset", "peptide", "hla"]
    ambiguous = table.groupby(keys)["Norm_peptide"].nunique()
    table = table.loc[~table.set_index(keys).index.isin(ambiguous[ambiguous > 1].index)]
    table = table.drop_duplicates(keys)[[*keys, "Norm_peptide"]].rename(columns={"Norm_peptide": "wild_type"})
    keep = frame.loc[frame["dataset"].isin(COHORTS)].merge(table, on=keys, how="inner", validate="many_to_one")
    usable = keep["wild_type"].notna() & keep["wild_type"].str.len().between(8, 15)
    print("rows:", len(keep), "usable wild-type contrasts:", int(usable.sum()))
    keep = keep.loc[usable].copy()
    wild_scores = wild_type_scores(sorted(keep["wild_type"].unique()), cache).rename(columns={"peptide": "wild_type"})
    keep = keep.merge(wild_scores, on="wild_type", how="left", suffixes=("", "_wt"), validate="many_to_one")
    for head in heads:
        keep[f"delta_{head}"] = keep[head] - keep[f"{head}_wt"]
        keep[f"rd_{head}"] = ecdf(keep[f"delta_{head}"].to_numpy(float))
    keep["identical"] = keep["wild_type"].eq(keep["peptide"])
    return keep.reset_index(drop=True)


def macro(frame: pd.DataFrame, score: np.ndarray, base: dict) -> float:
    """Mean per-cohort AUC gain over the core."""
    outcome = frame["outcome"].to_numpy(int)
    return float(np.mean([fast_auc(outcome[i], score[i]) - base[d] for d, i in frame.groupby("dataset").indices.items()]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--weirdo", type=Path, required=True)
    parser.add_argument("--raw-improve", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=300)
    args = parser.parse_args()
    frame, heads = load(args.internal, args.external, args.weirdo)
    args.output.mkdir(parents=True, exist_ok=True)
    data = build(frame, heads, args.raw_improve, args.output / "wild-type-weirdo-scores.csv")
    print("rows", len(data), "identical wt", int(data["identical"].sum()), "positives", int(data["outcome"].sum()))
    outcome = data["outcome"].to_numpy(int)
    core = data["core"].to_numpy(float)
    indices = data.groupby("dataset").indices
    base = {d: fast_auc(outcome[i], core[i]) for d, i in indices.items()}
    rows = []
    for head in heads:
        ranks = data[f"rd_{head}"].to_numpy(float)
        for lam in LAMBDAS:
            score = core + lam * (ranks - 0.5)
            rows.append({"head": head, "lam": lam, "d_macro": macro(data, score, base),
                         **{d: fast_auc(outcome[i], score[i]) - base[d] for d, i in indices.items()}})
    table = pd.DataFrame(rows).sort_values("d_macro", ascending=False)
    rng = np.random.default_rng(84_342)
    columns = np.column_stack([data[f"rd_{h}"].to_numpy(float) for h in heads])
    null_max = []
    for _ in range(args.permutations):
        shuffled = columns.copy()
        for i in indices.values():
            shuffled[i] = columns[rng.permutation(i)]
        null_max.append(max(
            macro(data, core + lam * (shuffled[:, j] - 0.5), base) for j in range(len(heads)) for lam in LAMBDAS
        ))
    null_max = np.array(null_max)
    table["max_null_p"] = [(null_max >= g).mean() for g in table["d_macro"]]
    table.to_csv(args.output / "mutation_contrast_additive.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.head(10).round(4).to_string(index=False))
    print("null max: median %.4f 95th %.4f; observed %.4f" % (np.median(null_max), np.percentile(null_max, 95), table["d_macro"].iloc[0]))
    # leave-one-cohort-out fitted contrast model
    summary = [f"rd_{h}" for h in SUMMARY_HEADS]
    matrix = np.hstack([core[:, None] - 0.5, data[summary].to_numpy(float) - 0.5])
    fitted = []
    for c in REGULARIZATION:
        score = np.zeros(len(data))
        for held in COHORTS:
            test = (data["dataset"] == held).to_numpy()
            model = LogisticRegression(C=c, class_weight="balanced", max_iter=2_000).fit(matrix[~test], outcome[~test])
            score[test] = model.decision_function(matrix[test])
        fitted.append({"C": c, "d_macro": macro(data, score, base),
                       **{d: fast_auc(outcome[i], score[i]) - base[d] for d, i in indices.items()}})
    print(pd.DataFrame(fitted).round(4).to_string(index=False))
    pd.DataFrame(fitted).to_csv(args.output / "mutation_contrast_loco.csv", index=False)


if __name__ == "__main__":
    main()
