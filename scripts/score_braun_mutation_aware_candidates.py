#!/usr/bin/env python3
"""Checkpoint MHCflurry and NetMHCpan scoring for mutation-aware candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from scripts.score_common_tcell_predictors import score_mhcflurry, score_netmhcpan


def score_batch(candidates: pd.DataFrame, batch_size: int, batch_index: int) -> tuple[pd.DataFrame, dict]:
    """Score one deterministic peptide--HLA batch and preserve all row metadata."""
    required = {"peptide", "hla", "reference_absent_candidate"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise ValueError(f"Candidate table is missing columns: {missing}")
    filtered = candidates.loc[candidates["reference_absent_candidate"]].copy()
    pairs = filtered[["peptide", "hla"]].drop_duplicates().sort_values(["hla", "peptide"]).reset_index(drop=True)
    start = batch_index * batch_size
    batch = pairs.iloc[start : start + batch_size].reset_index(drop=True)
    if batch.empty:
        raise ValueError(f"Batch {batch_index} is empty; {len(pairs)} pairs at size {batch_size}")
    netmhc, netmhc_info = score_netmhcpan(batch, process_limit=4)
    mhcflurry, mhcflurry_info = score_mhcflurry(batch, "2.3.0")
    scored = batch.merge(netmhc, on=["peptide", "hla"], validate="one_to_one").merge(
        mhcflurry, on=["peptide", "hla"], validate="one_to_one"
    )
    rows = filtered.merge(scored, on=["peptide", "hla"], how="inner", validate="many_to_one")
    return rows, {"pair_count": len(pairs), "batch_pair_count": len(batch), "netmhcpan": netmhc_info, "mhcflurry": mhcflurry_info}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--batch-index", type=int, required=True)
    args = parser.parse_args()
    candidates = pd.read_csv(args.candidates, low_memory=False)
    rows, provenance = score_batch(candidates, args.batch_size, args.batch_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"batch-{args.batch_index:03d}.csv.gz"
    rows.to_csv(output, index=False, compression="gzip")
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Wrote {len(rows):,} rows for {provenance['batch_pair_count']:,} pMHC pairs to {output}")


if __name__ == "__main__":
    main()
