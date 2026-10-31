#!/usr/bin/env python3
"""Sample presented human self ligands from the HLA Ligand Atlas.

Keeps class I-only peptides of length 8-11 with exactly one atlas-predicted *strong*
binding class I allele among the donor alleles, so each peptide gets an unambiguous
restricting HLA. Peptides that also appear in the IEDB T-cell pool are excluded (their
self/epitope label would be ambiguous). A fixed-seed sample, balanced across lengths.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

ALLELE = re.compile(r"^s/([ABC])\*(\d+):(\d+)$")


def strong_class_i_alleles(donor_alleles: str) -> list[str]:
    """Exact HLA names of the strong class I alleles listed for a peptide."""
    alleles = []
    for item in donor_alleles.split(","):
        match = ALLELE.match(item.strip())
        if match:
            alleles.append(f"HLA-{match.group(1)}*{match.group(2)}:{match.group(3)}")
    return sorted(set(alleles))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregated", type=Path, required=True, help="HLA_aggregated.tsv")
    parser.add_argument("--pool", type=Path, required=True, help="IEDB pool CSV (peptides to exclude)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-length", type=int, default=3_000)
    parser.add_argument("--seed", type=int, default=84_342)
    args = parser.parse_args()
    atlas = pd.read_csv(args.aggregated, sep="\t", usecols=["peptide_sequence", "hla_class", "donor_alleles"])
    atlas = atlas.loc[
        atlas["hla_class"].eq("HLA-I") & atlas["peptide_sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{8,11}")
    ].copy()
    strong = atlas["donor_alleles"].map(strong_class_i_alleles)
    atlas = atlas.loc[strong.map(len).eq(1)].copy()
    atlas["hla"] = strong.loc[atlas.index].str[0]
    seen = set(pd.read_csv(args.pool, usecols=["peptide"])["peptide"])
    atlas = atlas.loc[~atlas["peptide_sequence"].isin(seen)].drop_duplicates("peptide_sequence")
    atlas["length"] = atlas["peptide_sequence"].str.len()
    parts = [
        group.sample(min(len(group), args.per_length), random_state=args.seed)
        for _, group in atlas.groupby("length")
    ]
    sample = pd.concat(parts, ignore_index=True)
    cohort = pd.DataFrame(
        {
            "pmid": "hla-ligand-atlas",
            "peptide": sample["peptide_sequence"],
            "hla": sample["hla"],
            "outcome": 0,
            "source_organism": "Homo sapiens",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False)
    print(len(cohort), "self ligands;", cohort["hla"].nunique(), "alleles;", cohort["peptide"].str.len().value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
