#!/usr/bin/env python3
"""Source-protein context features for IEDB peptide-HLA pairs.

Each pair is mapped into its curated UniProt parent protein by exact substring
match (the IEDB start position only breaks ties between repeats). From the mapped
location come:

* ``ctx_mapped``                 1 if the peptide was found in its parent protein
* ``ctx_protein_length``, ``ctx_relative_position``, ``ctx_from_n``, ``ctx_from_c``
* ``ctx_flank_presentation``     MHCflurry 2.3 presentation score with 10-residue
                                 flanks (processing-aware); NaN when unmapped

Flanks are properties of the candidate in its source protein and are known at
prediction time in epitope discovery; no outcome information is used.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FLANK = 10


def locate(sequence: str, peptide: str, hint: float) -> int | None:
    """0-based start of ``peptide`` in ``sequence`` (nearest the 1-based hint if repeated)."""
    starts, at = [], sequence.find(peptide)
    while at != -1:
        starts.append(at)
        at = sequence.find(peptide, at + 1)
    if not starts:
        return None
    if len(starts) == 1 or pd.isna(hint):
        return starts[0]
    return min(starts, key=lambda s: abs(s + 1 - hint))


def build(pool: pd.DataFrame, context: pd.DataFrame, sequences: dict[str, str]) -> pd.DataFrame:
    """Map pairs into proteins and derive position and flank columns."""
    context = context.copy()
    context["pmid"] = context["reference__pmid"].astype(float).astype("Int64").astype(str)
    context["peptide"] = context["assay_antigen__name"].fillna(context["epitope__name"]).astype(str).str.upper()
    context["hla"] = context["mhc_restriction__name"]
    context["accession"] = context["epitope__molecule_parent_iri"].str.extract(r"uniprot/([A-Z0-9]+)")[0]
    first = (
        context.dropna(subset=["accession"])
        .drop_duplicates(["pmid", "peptide", "hla"])
        [["pmid", "peptide", "hla", "accession", "epitope__starting_position"]]
    )
    table = pool[["pmid", "peptide", "hla"]].merge(first, on=["pmid", "peptide", "hla"], how="left")
    rows = []
    for pmid, peptide, hla, accession, hint in table.itertuples(index=False):
        sequence = sequences.get(accession) if isinstance(accession, str) else None
        start = locate(sequence, peptide, hint) if sequence else None
        if start is None:
            rows.append((pmid, peptide, hla, accession if isinstance(accession, str) else None, 0, np.nan, np.nan, np.nan, np.nan, None, None))
            continue
        end = start + len(peptide)
        rows.append(
            (pmid, peptide, hla, accession, 1, len(sequence), start / len(sequence), start, len(sequence) - end,
             sequence[max(0, start - FLANK):start], sequence[end:end + FLANK])
        )
    return pd.DataFrame(
        rows,
        columns=["pmid", "peptide", "hla", "accession", "ctx_mapped", "ctx_protein_length", "ctx_relative_position",
                 "ctx_from_n", "ctx_from_c", "n_flank", "c_flank"],
    )


def flank_presentation(frame: pd.DataFrame) -> np.ndarray:
    """MHCflurry 2.3 flank-aware presentation score for the mapped rows."""
    from mhcflurry import Class1PresentationPredictor
    from mhcflurry.downloads import get_default_class1_presentation_models_dir  # noqa: F401
    from scripts.score_common_tcell_predictors import get_model_release_dir

    predictor = Class1PresentationPredictor.load(get_model_release_dir("2.3.0"))
    mapped = frame["ctx_mapped"].eq(1).to_numpy()
    out = np.full(len(frame), np.nan)
    sub = frame.loc[mapped]
    scored = predictor.predict(
        peptides=sub["peptide"].tolist(),
        n_flanks=sub["n_flank"].fillna("").tolist(),
        c_flanks=sub["c_flank"].fillna("").tolist(),
        alleles={allele: [allele] for allele in sub["hla"].unique()},
        sample_names=sub["hla"].tolist(),
        verbose=0,
    ).sort_values("peptide_num")
    out[np.flatnonzero(mapped)] = scored["presentation_score"].to_numpy(float)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pool = pd.read_csv(args.pool, dtype={"pmid": str})
    sequences = dict(pd.read_csv(args.sequences).itertuples(index=False))
    features = build(pool, pd.read_csv(args.context, low_memory=False), sequences)
    features["ctx_flank_presentation"] = flank_presentation(features)
    features.drop(columns=["n_flank", "c_flank"]).to_csv(args.output, index=False)
    print(len(features), "pairs;", f"{features['ctx_mapped'].mean():.1%}", "mapped into a source protein")


if __name__ == "__main__":
    main()
