#!/usr/bin/env python3
"""Score every evaluation peptide with the shipped peptide-content ensemble.

Writes one row per unique peptide with all taxonomy heads the model emits, so a
combination search sees exactly what ``weirdo`` ships rather than research-only
checkpoints.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL = "peptide-content-v1-ensemble"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import weirdo

    peptides = sorted(
        {p for table in args.tables for p in pd.read_csv(table, usecols=["peptide"], low_memory=False)["peptide"]}
    )
    scored = pd.DataFrame(weirdo.predict_taxonomy_scores(peptides, model=MODEL)).reset_index(drop=True)
    if scored["peptide"].tolist() != peptides:
        raise ValueError("Weirdo output order does not match the input peptides")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    print(len(scored), "peptides,", len(scored.columns) - 1, "heads")


if __name__ == "__main__":
    main()
