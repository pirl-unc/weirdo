#!/usr/bin/env python3
"""Weirdo taxonomy profile of each peptide's source protein.

Every 9-mer window of each source protein is scored with the dependency-free bundled
``peptide-content-v1`` model. Per protein: the mean and standard deviation of four
heads over its windows (how foreign the whole protein reads). Per mapped pair: the
protein means plus the peptide's own head minus the protein mean, in protein-SD units
(how unusual this peptide is for its own protein). No outcome information is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HEADS = ("virus_foreign_like", "microbial_foreign_like", "one_minus_mammals", "p_viruses")
WINDOW = 9


def protein_profiles(sequences: pd.DataFrame) -> pd.DataFrame:
    """Mean/SD of each head over the 9-mer windows of every protein."""
    import weirdo

    rows = []
    for accession, sequence in sequences.itertuples(index=False):
        windows = [sequence[i:i + WINDOW] for i in range(len(sequence) - WINDOW + 1)]
        windows = [w for w in windows if set(w) <= set("ACDEFGHIKLMNPQRSTVWY")]
        if len(windows) < 20:
            continue
        scored = weirdo.predict_taxonomy_scores(windows)
        row = {"accession": accession, "n_windows": len(windows)}
        for head in HEADS:
            row[f"prot_{head}_mean"] = float(scored[head].mean())
            row[f"prot_{head}_sd"] = float(scored[head].std())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True, help="source-context.csv (needs accession)")
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import weirdo

    context = pd.read_csv(args.context, dtype={"pmid": str})
    profiles = protein_profiles(pd.read_csv(args.sequences))
    own = pd.DataFrame(weirdo.predict_taxonomy_scores(context["peptide"].tolist()))[list(HEADS)].add_prefix("own_")
    table = pd.concat([context[["pmid", "peptide", "hla", "accession"]], own], axis=1).merge(profiles, on="accession", how="left")
    for head in HEADS:
        sd = table[f"prot_{head}_sd"].replace(0, np.nan)
        table[f"prot_{head}_local_z"] = (table[f"own_{head}"] - table[f"prot_{head}_mean"]) / sd
    keep = ["pmid", "peptide", "hla"] + [c for c in table.columns if c.startswith("prot_") and not c.endswith("_sd")]
    table[keep].to_csv(args.output, index=False)
    print(len(table), "pairs;", f"{table['prot_virus_foreign_like_mean'].notna().mean():.1%}", "with a protein profile")


if __name__ == "__main__":
    main()
