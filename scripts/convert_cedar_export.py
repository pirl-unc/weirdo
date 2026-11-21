#!/usr/bin/env python3
"""Flatten the CEDAR full T-cell export to the IEDB-pull column schema.

CEDAR (the cancer sister database of IEDB) ships a two-row header CSV. Only the
columns needed to build peptide-HLA outcome tables are read, then renamed to the
names ``build_fresh_iedb_cohort.py --assays`` expects. The archive is untrusted
input: read it with ``python -I`` from a different directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

RENAME = {
    ("Reference", "PMID"): "reference__pmid",
    ("Assay Antigen", "Name"): "assay_antigen__name",
    ("Epitope", "Name"): "epitope__name",
    ("Epitope", "Source Organism"): "epitope__source_organism",
    ("MHC Restriction", "Name"): "mhc_restriction__name",
    ("MHC Restriction", "Class"): "mhc_restriction__class",
    ("Host", "Name"): "host__name",
    ("Assay", "Qualitative Measurement"): "assay__qualitative_measurement",
    ("Epitope", "Mutation"): "epitope__mutation",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    header = pd.read_csv(args.export, nrows=0, header=[0, 1])
    positions = {pair: i for i, pair in enumerate(header.columns)}
    missing = [pair for pair in RENAME if pair not in positions]
    if missing:
        raise ValueError(f"CEDAR export is missing expected columns: {missing}")
    order = sorted(RENAME, key=lambda pair: positions[pair])
    frame = pd.read_csv(
        args.export, header=None, skiprows=2, usecols=[positions[pair] for pair in order], low_memory=False
    )
    frame.columns = [RENAME[pair] for pair in order]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(len(frame), "rows;", frame["reference__pmid"].nunique(), "studies")


if __name__ == "__main__":
    main()
