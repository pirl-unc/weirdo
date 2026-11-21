#!/usr/bin/env python3
"""Score the fresh IEDB cohort with MHCflurry 2.3, NetMHCpan and shipped Weirdo."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mhctools.unsupported_allele import UnsupportedAllele  # noqa: E402
from scripts.score_common_tcell_predictors import score_mhcflurry, score_netmhcpan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--process-limit", type=int, default=4)
    args = parser.parse_args()
    import weirdo

    cohort = pd.read_csv(args.cohort)
    pairs = cohort[["peptide", "hla"]].drop_duplicates().reset_index(drop=True)
    dropped_alleles: list[str] = []
    while True:  # rule: both predictors must support the allele; drop pairs otherwise
        try:
            netmhc, _ = score_netmhcpan(pairs, args.process_limit)
            break
        except UnsupportedAllele as error:
            bad = re.findall(r"HLA-[ABC]\*\d+:\d+", str(error))
            if not bad:
                raise
            dropped_alleles.extend(bad)
            pairs = pairs.loc[~pairs["hla"].isin(bad)].reset_index(drop=True)
    cohort = cohort.loc[~cohort["hla"].isin(dropped_alleles)].reset_index(drop=True)
    print("dropped unsupported alleles:", sorted(set(dropped_alleles)), "rows left", len(cohort))
    flurry, _ = score_mhcflurry(pairs, "2.3.0")
    heads = pd.DataFrame(
        weirdo.predict_taxonomy_scores(sorted(pairs["peptide"].unique()), model="peptide-content-v1-ensemble")
    )
    scored = (
        cohort.merge(netmhc, on=["peptide", "hla"], validate="many_to_one")
        .merge(flurry, on=["peptide", "hla"], validate="many_to_one")
        .merge(heads, on="peptide", validate="many_to_one")
    )
    if len(scored) != len(cohort):
        raise ValueError(f"Scoring changed the row count: {len(cohort)} -> {len(scored)}")
    missing = [c for c in scored.columns[scored.isna().any()] if c != "source_organism"]
    if missing:
        raise ValueError(f"Scoring left missing values in {missing}: {int(scored[missing].isna().any(axis=1).sum())} rows")
    scored.to_csv(args.output, index=False)
    print(len(scored), "scored rows")


if __name__ == "__main__":
    main()
