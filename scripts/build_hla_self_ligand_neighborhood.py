#!/usr/bin/env python3
"""Same-HLA self-ligand neighbourhood features from the HLA Ligand Atlas.

For each peptide-HLA pair the nearest *presented* human self ligands of the same
allele and length are found (atlas-predicted strong binders of that allele), and
summarised on the TCR-facing positions P3..P(L-1):

* ``selfnbr_face_min_mismatch``  fewest face mismatches to any other same-HLA ligand
* ``selfnbr_face_le1`` / ``_le2``  log1p count of ligands within 1 / 2 face mismatches
* ``selfnbr_face_blosum_max``    best BLOSUM62 sum over the face against any ligand
* ``selfnbr_exact_hit``          the whole peptide is itself an atlas ligand of that HLA
* ``selfnbr_n_ligands``          size of the same-HLA/length reference set

Identical peptides are excluded from the distance statistics. Alleles with no atlas
ligands get NaN (no information), not zeros.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
INDEX = {a: i for i, a in enumerate(ALPHABET)}
BLOSUM62 = {
    "A": [4, 0, -2, -1, -2, 0, -2, -1, -1, -1, -1, -2, -1, -1, -1, 1, 0, 0, -3, -2],
    "C": [0, 9, -3, -4, -2, -3, -3, -1, -3, -1, -1, -3, -3, -3, -3, -1, -1, -1, -2, -2],
    "D": [-2, -3, 6, 2, -3, -1, -1, -3, -1, -4, -3, 1, -1, 0, -2, 0, -1, -3, -4, -3],
    "E": [-1, -4, 2, 5, -3, -2, 0, -3, 1, -3, -2, 0, -1, 2, 0, 0, -1, -2, -3, -2],
    "F": [-2, -2, -3, -3, 6, -3, -1, 0, -3, 0, 0, -3, -4, -3, -3, -2, -2, -1, 1, 3],
    "G": [0, -3, -1, -2, -3, 6, -2, -4, -2, -4, -3, 0, -2, -2, -2, 0, -2, -3, -2, -3],
    "H": [-2, -3, -1, 0, -1, -2, 8, -3, -1, -3, -2, 1, -2, 0, 0, -1, -2, -3, -2, 2],
    "I": [-1, -1, -3, -3, 0, -4, -3, 4, -3, 2, 1, -3, -3, -3, -3, -2, -1, 3, -3, -1],
    "K": [-1, -3, -1, 1, -3, -2, -1, -3, 5, -2, -1, 0, -1, 1, 2, 0, -1, -2, -3, -2],
    "L": [-1, -1, -4, -3, 0, -4, -3, 2, -2, 4, 2, -3, -3, -2, -2, -2, -1, 1, -2, -1],
    "M": [-1, -1, -3, -2, 0, -3, -2, 1, -1, 2, 5, -2, -2, 0, -1, -1, -1, 1, -1, -1],
    "N": [-2, -3, 1, 0, -3, 0, 1, -3, 0, -3, -2, 6, -2, 0, 0, 1, 0, -3, -4, -2],
    "P": [-1, -3, -1, -1, -4, -2, -2, -3, -1, -3, -2, -2, 7, -1, -2, -1, -1, -2, -4, -3],
    "Q": [-1, -3, 0, 2, -3, -2, 0, -3, 1, -2, 0, 0, -1, 5, 1, 0, -1, -2, -2, -1],
    "R": [-1, -3, -2, 0, -3, -2, 0, -3, 2, -2, -1, 0, -2, 1, 5, -1, -1, -3, -3, -2],
    "S": [1, -1, 0, 0, -2, 0, -1, -2, 0, -2, -1, 1, -1, 0, -1, 4, 1, -2, -3, -2],
    "T": [0, -1, -1, -1, -2, -2, -2, -1, -1, -1, -1, 0, -1, -1, -1, 1, 5, 0, -2, -2],
    "V": [0, -1, -3, -2, -1, -3, -3, 3, -2, 1, 1, -3, -2, -2, -3, -2, 0, 4, -3, -1],
    "W": [-3, -2, -4, -3, 1, -2, -2, -3, -3, -2, -1, -4, -4, -2, -3, -3, -2, -3, 11, 2],
    "Y": [-2, -2, -3, -2, 3, -3, 2, -1, -2, -1, -1, -2, -3, -1, -2, -2, -2, -1, 2, 7],
}
BLOSUM = np.array([BLOSUM62[a] for a in ALPHABET], dtype=np.int8)
CHUNK = 64


def encode(peptides) -> np.ndarray:
    return np.array([[INDEX[a] for a in p] for p in peptides], dtype=np.uint8)


def ligands_by_allele(aggregated: Path) -> dict[tuple[str, int], np.ndarray]:
    """(allele, length) -> encoded same-HLA atlas ligands (atlas strong binders)."""
    atlas = pd.read_csv(aggregated, sep="\t", usecols=["peptide_sequence", "hla_class", "donor_alleles"])
    atlas = atlas.loc[atlas["hla_class"].eq("HLA-I") & atlas["peptide_sequence"].str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{8,11}")]
    buckets: dict[tuple[str, int], list[str]] = {}
    for peptide, donors in zip(atlas["peptide_sequence"], atlas["donor_alleles"]):
        for item in donors.split(","):
            if item.startswith("s/") and item[2] in "ABC":
                gene, rest = item[2], item[4:]
                buckets.setdefault((f"HLA-{gene}*{rest}", len(peptide)), []).append(peptide)
    return {key: (np.array(sorted(set(v))), encode(sorted(set(v)))) for key, v in buckets.items()}


def features_for(query: np.ndarray, reference: np.ndarray, query_peptides: np.ndarray, ref_peptides: np.ndarray) -> np.ndarray:
    """Per-query statistics against one same-HLA, same-length reference set."""
    length = query.shape[1]
    face = slice(2, length - 1)
    out = np.full((len(query), 5), np.nan)
    for start in range(0, len(query), CHUNK):
        q = query[start:start + CHUNK]
        face_mismatch = (q[:, None, face] != reference[None, :, face]).sum(-1)
        whole_mismatch = (q[:, None, :] != reference[None, :, :]).sum(-1)
        identical = whole_mismatch == 0
        blosum = BLOSUM[q[:, None, face], reference[None, :, face]].astype(np.int16).sum(-1)
        face_mismatch = np.where(identical, 99, face_mismatch)
        blosum = np.where(identical, -999, blosum)
        out[start:start + CHUNK, 0] = face_mismatch.min(1)
        out[start:start + CHUNK, 1] = np.log1p((face_mismatch <= 1).sum(1))
        out[start:start + CHUNK, 2] = np.log1p((face_mismatch <= 2).sum(1))
        out[start:start + CHUNK, 3] = blosum.max(1)
        out[start:start + CHUNK, 4] = identical.any(1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregated", type=Path, required=True, help="HLA_aggregated.tsv")
    parser.add_argument("--pairs", type=Path, required=True, help="CSV with peptide and hla columns")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pairs = pd.read_csv(args.pairs, usecols=["peptide", "hla"]).drop_duplicates().reset_index(drop=True)
    pairs = pairs.loc[pairs["peptide"].str.len().between(8, 11)].reset_index(drop=True)
    reference = ligands_by_allele(args.aggregated)
    columns = ["selfnbr_face_min_mismatch", "selfnbr_face_le1", "selfnbr_face_le2", "selfnbr_face_blosum_max", "selfnbr_exact_hit"]
    result = pd.DataFrame(np.nan, index=pairs.index, columns=columns)
    result["selfnbr_n_ligands"] = np.nan
    for (allele, length), group in pairs.groupby([pairs["hla"], pairs["peptide"].str.len()]):
        if (allele, length) not in reference:
            continue
        ref_peptides, ref = reference[(allele, length)]
        stats = features_for(encode(group["peptide"]), ref, group["peptide"].to_numpy(), ref_peptides)
        result.loc[group.index, columns] = stats
        result.loc[group.index, "selfnbr_n_ligands"] = len(ref)
    out = pd.concat([pairs, result], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    covered = out["selfnbr_n_ligands"].notna().mean()
    print(len(out), "pairs;", f"{covered:.1%}", "with a same-HLA ligand reference;",
          "exact self-ligand hits:", int(out["selfnbr_exact_hit"].fillna(0).sum()))


if __name__ == "__main__":
    main()
