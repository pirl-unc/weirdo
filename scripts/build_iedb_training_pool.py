#!/usr/bin/env python3
"""Pool every labelled human class I IEDB peptide-HLA pair for study-grouped learning.

Same row filters as the fresh cohorts (canonical 8-14mers, exact class I HLA,
positive/negative outcome, (peptide, HLA) positive if any assay is positive) but no
study or peptide exclusions: the study id is the grouping unit for cross-validation,
and peptides shared with the older evaluation datasets are purged at modelling time.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_fresh_iedb_cohort import CANONICAL, EXACT_HLA_I, positive  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assays", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = pd.read_csv(args.assays, low_memory=False)
    raw = raw.loc[raw["reference__pmid"].notna()].copy()
    raw["pmid"] = raw["reference__pmid"].astype(float).astype(int).astype(str)
    raw["peptide"] = raw["assay_antigen__name"].fillna(raw["epitope__name"]).astype(str).str.upper()
    raw["outcome"] = raw["assay__qualitative_measurement"].map(positive)
    keep = (
        raw["outcome"].notna()
        & raw["peptide"].str.fullmatch(CANONICAL.pattern)
        & raw["mhc_restriction__name"].fillna("").str.fullmatch(EXACT_HLA_I.pattern)
        & raw["host__name"].eq("Homo sapiens (human)")
        & raw["mhc_restriction__class"].eq("I")
    )
    raw = raw.loc[keep]
    pool = (
        raw.groupby(["pmid", "peptide", "mhc_restriction__name"], as_index=False)
        .agg(outcome=("outcome", "max"), source_organism=("epitope__source_organism", "first"))
        .rename(columns={"mhc_restriction__name": "hla"})
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(args.output, index=False)
    per_study = pool.groupby("pmid")["outcome"].agg(["size", "sum"])
    both = (per_study["sum"] > 0) & (per_study["sum"] < per_study["size"])
    print(len(pool), "pairs;", int(pool["outcome"].sum()), "positive;", pool["pmid"].nunique(), "studies;",
          int(both.sum()), "with both classes;", pool[["peptide", "hla"]].drop_duplicates().shape[0], "unique pairs")


if __name__ == "__main__":
    main()
