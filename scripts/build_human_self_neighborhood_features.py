#!/usr/bin/env python3
"""Build TCR-facing nearest-human-peptide features for response benchmarks.

The nearest-neighbor search is performed on the non-anchor peptide face
(P3 through P-1) with a binary one-hot Hamming index. Retrieved candidates are
then rescored using the complete peptide, explicit P2/P-omega anchors, and
BLOSUM62.  This keeps the retrieval step close to the part of a class-I peptide
normally exposed to a TCR without pretending that sequence source taxonomy is
a model of thymic selection.

The output is peptide-level and can be joined to any response table.  A second
long-form table retains the best human candidate peptides for a later
same-HLA presentation pass.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {amino_acid: index for index, amino_acid in enumerate(AMINO_ACIDS)}
CANONICAL = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]+$")


def open_text(path: str | Path):
    """Open a plain-text or gzip-compressed UniProt flat file."""
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open()


def load_uniprot_sequences(path: str | Path) -> list[str]:
    """Extract canonical primary sequences from a UniProt ``.dat`` file."""
    result: list[str] = []
    sequence_lines: list[str] = []
    in_sequence = False
    with open_text(path) as stream:
        for line in stream:
            if line.startswith("SQ "):
                in_sequence = True
                sequence_lines = []
            elif in_sequence and line.startswith("  "):
                sequence_lines.append("".join(line.split()))
            elif in_sequence and line.startswith("//"):
                sequence = "".join(sequence_lines)
                if sequence and CANONICAL.fullmatch(sequence):
                    result.append(sequence)
                in_sequence = False
    return result


def encode_sequence(sequence: str) -> np.ndarray:
    """Encode a canonical amino-acid sequence as small integer indices."""
    try:
        return np.fromiter(
            (AA_TO_INDEX[amino_acid] for amino_acid in sequence),
            dtype=np.uint8,
            count=len(sequence),
        )
    except KeyError as error:
        raise ValueError(f"Non-canonical amino acid in {sequence!r}") from error


def peptide_windows(sequences: Iterable[np.ndarray], length: int) -> np.ndarray:
    """Return all within-protein windows of one peptide length."""
    arrays = [
        np.lib.stride_tricks.sliding_window_view(sequence, length)
        for sequence in sequences
        if len(sequence) >= length
    ]
    if not arrays:
        return np.empty((0, length), dtype=np.uint8)
    return np.concatenate(arrays).astype(np.uint8, copy=False)


def tcr_face(values: np.ndarray) -> np.ndarray:
    """Return P3 through P-1, excluding P1, P2, and P-omega."""
    if values.shape[-1] < 4:
        raise ValueError("A class-I peptide must contain at least four residues")
    return values[..., 2:-1]


def binary_one_hot(values: np.ndarray) -> tuple[np.ndarray, int]:
    """Pack positional 20-way one-hot residue encodings for FAISS binary search."""
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D residue matrix, got {values.shape}")
    unpadded_bits = values.shape[1] * len(AMINO_ACIDS)
    dimensions = ((unpadded_bits + 7) // 8) * 8
    result = np.zeros((len(values), dimensions // 8), dtype=np.uint8)
    rows = np.arange(len(values))
    for position in range(values.shape[1]):
        bit = position * len(AMINO_ACIDS) + values[:, position].astype(np.int64)
        result[rows, bit // 8] |= (1 << (bit % 8)).astype(np.uint8)
    return result, dimensions


def blosum62_array() -> np.ndarray:
    """Return BLOSUM62 in the canonical residue order used by this script."""
    source = substitution_matrices.load("BLOSUM62")
    return np.asarray(
        [
            [float(source[left, right]) for right in AMINO_ACIDS]
            for left in AMINO_ACIDS
        ],
        dtype=np.float32,
    )


def candidate_statistics(
    queries: np.ndarray,
    candidates: np.ndarray,
    *,
    blosum: np.ndarray | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Rescore candidate self peptides and return features plus ranking arrays.

    ``candidates`` has shape ``(n_queries, n_candidates, peptide_length)``.
    The two returned score arrays are the anchor-aware distance (lower is
    closer) and weighted BLOSUM similarity (higher is closer).
    """
    if candidates.ndim != 3 or queries.ndim != 2:
        raise ValueError("Queries and candidates must be 2-D and 3-D arrays")
    if candidates.shape[0] != len(queries) or candidates.shape[2] != queries.shape[1]:
        raise ValueError("Candidate shape does not match queries")
    if blosum is None:
        blosum = blosum62_array()

    mismatches = candidates != queries[:, None, :]
    face_mismatches = tcr_face(mismatches).sum(axis=2).astype(np.float32)
    full_mismatches = mismatches.sum(axis=2).astype(np.float32)
    anchor_match = ~mismatches[:, :, 1] & ~mismatches[:, :, -1]
    weighted_distance = (
        face_mismatches
        + 0.50 * mismatches[:, :, 0]
        + 0.25 * mismatches[:, :, 1]
        + 0.25 * mismatches[:, :, -1]
    ).astype(np.float32)

    position_weights = np.ones(queries.shape[1], dtype=np.float32)
    position_weights[0] = 0.50
    position_weights[1] = 0.25
    position_weights[-1] = 0.25
    similarity = np.zeros(candidates.shape[:2], dtype=np.float32)
    for position, weight in enumerate(position_weights):
        similarity += weight * blosum[
            queries[:, None, position], candidates[:, :, position]
        ]
    similarity /= position_weights.sum()

    anchor_face_distance = np.where(anchor_match, face_mismatches, np.inf)
    anchor_similarity = np.where(anchor_match, similarity, -np.inf)
    no_anchor_candidate = ~anchor_match.any(axis=1)
    minimum_anchor_face = anchor_face_distance.min(axis=1)
    maximum_anchor_similarity = anchor_similarity.max(axis=1)
    minimum_anchor_face[no_anchor_candidate] = np.nan
    maximum_anchor_similarity[no_anchor_candidate] = np.nan

    features = pd.DataFrame(
        {
            "human_self_face_mismatch": face_mismatches.min(axis=1),
            "human_self_full_mismatch": full_mismatches.min(axis=1),
            "human_self_weighted_mismatch": weighted_distance.min(axis=1),
            "human_self_anchor_matched_face_mismatch": minimum_anchor_face,
            "human_self_weighted_blosum": similarity.max(axis=1),
            "human_self_anchor_matched_blosum": maximum_anchor_similarity,
            "human_self_face_exact_log_density": np.log1p(
                (face_mismatches == 0).sum(axis=1)
            ),
            "human_self_face_oneoff_log_density": np.log1p(
                (face_mismatches <= 1).sum(axis=1)
            ),
            "human_self_anchor_face_exact_log_density": np.log1p(
                ((face_mismatches == 0) & anchor_match).sum(axis=1)
            ),
            "human_self_anchor_face_oneoff_log_density": np.log1p(
                ((face_mismatches <= 1) & anchor_match).sum(axis=1)
            ),
        }
    )
    return features, weighted_distance, similarity


def best_candidate_indices(
    weighted_distance: np.ndarray,
    similarity: np.ndarray,
    candidates: np.ndarray,
    queries: np.ndarray,
    keep: int,
) -> list[np.ndarray]:
    """Retain a diverse union of distance-, similarity-, and anchor-ranked hits."""
    anchor_match = (candidates[:, :, 1] == queries[:, None, 1]) & (
        candidates[:, :, -1] == queries[:, None, -1]
    )
    result = []
    per_ranking = max(1, keep // 3)
    for query_index in range(len(queries)):
        by_distance = np.argsort(weighted_distance[query_index])[:per_ranking]
        by_similarity = np.argsort(-similarity[query_index])[:per_ranking]
        anchor_score = np.where(
            anchor_match[query_index], weighted_distance[query_index], np.inf
        )
        by_anchor = np.argsort(anchor_score)[:per_ranking]
        by_anchor = by_anchor[np.isfinite(anchor_score[by_anchor])]
        chosen = np.unique(np.concatenate((by_distance, by_similarity, by_anchor)))
        if len(chosen) < keep:
            fill = np.argsort(weighted_distance[query_index])
            chosen = np.unique(np.concatenate((chosen, fill[: keep - len(chosen)])))
        result.append(chosen[:keep])
    return result


def build_length_features(
    human_sequences: list[np.ndarray],
    peptide_strings: list[str],
    *,
    nlist: int,
    nprobe: int,
    search_candidates: int,
    retained_candidates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build peptide and candidate features for a single peptide length."""
    import faiss

    length = len(peptide_strings[0])
    if any(len(peptide) != length for peptide in peptide_strings):
        raise ValueError("All query peptides must have the same length")
    human_windows = peptide_windows(human_sequences, length)
    query_values = np.vstack([encode_sequence(peptide) for peptide in peptide_strings])
    human_codes, dimensions = binary_one_hot(tcr_face(human_windows))
    query_codes, query_dimensions = binary_one_hot(tcr_face(query_values))
    if query_dimensions != dimensions:
        raise AssertionError("Human and query encodings have inconsistent dimensions")

    effective_nlist = min(nlist, max(1, len(human_codes) // 40))
    quantizer = faiss.IndexBinaryFlat(dimensions)
    index = faiss.IndexBinaryIVF(quantizer, dimensions, effective_nlist)
    random = np.random.default_rng(seed + length)
    training_rows = random.choice(
        len(human_codes), min(250_000, len(human_codes)), replace=False
    )
    index.train(human_codes[training_rows])
    index.add(human_codes)
    index.nprobe = min(nprobe, effective_nlist)
    k = min(search_candidates, len(human_windows))
    binary_distances, neighbors = index.search(query_codes, k)
    if (neighbors < 0).any():
        raise ValueError("FAISS failed to return a full candidate set")
    candidates = human_windows[neighbors]
    features, weighted_distance, similarity = candidate_statistics(
        query_values, candidates
    )
    features.insert(0, "peptide", peptide_strings)
    features["peptide_length"] = length
    features["human_self_retrieval_last_face_mismatch"] = (
        binary_distances[:, -1] / 2.0
    )

    selected = best_candidate_indices(
        weighted_distance,
        similarity,
        candidates,
        query_values,
        retained_candidates,
    )
    candidate_rows = []
    for query_index, indices in enumerate(selected):
        for rank, candidate_index in enumerate(indices):
            encoded = candidates[query_index, candidate_index]
            candidate_rows.append(
                {
                    "peptide": peptide_strings[query_index],
                    "candidate_rank": rank,
                    "human_peptide": "".join(AMINO_ACIDS[value] for value in encoded),
                    "weighted_mismatch": float(
                        weighted_distance[query_index, candidate_index]
                    ),
                    "weighted_blosum": float(similarity[query_index, candidate_index]),
                    "anchor_match": bool(
                        encoded[1] == query_values[query_index, 1]
                        and encoded[-1] == query_values[query_index, -1]
                    ),
                }
            )
    diagnostics = {
        "length": length,
        "query_peptides": len(peptide_strings),
        "human_windows": len(human_windows),
        "binary_dimensions": dimensions,
        "nlist": effective_nlist,
        "nprobe": index.nprobe,
        "search_candidates": k,
        "retained_candidates": retained_candidates,
    }
    return features, pd.DataFrame(candidate_rows), diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--human-uniprot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--nlist", type=int, default=4_096)
    parser.add_argument("--nprobe", type=int, default=96)
    parser.add_argument("--search-candidates", type=int, default=512)
    parser.add_argument("--retained-candidates", type=int, default=24)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=84_342)
    args = parser.parse_args()

    scores = pd.read_csv(args.scores, usecols=["peptide"])
    peptides = sorted(set(scores["peptide"].astype(str)))
    invalid = [peptide for peptide in peptides if not CANONICAL.fullmatch(peptide)]
    if invalid:
        raise ValueError(f"Non-canonical query peptides: {invalid[:5]}")
    human = load_uniprot_sequences(args.human_uniprot)
    if not human:
        raise ValueError("No canonical human sequences were found")
    encoded_human = [encode_sequence(sequence) for sequence in human]
    import faiss

    faiss.omp_set_num_threads(args.threads)

    feature_parts = []
    candidate_parts = []
    diagnostics = []
    for length in sorted({len(peptide) for peptide in peptides}):
        queries = [peptide for peptide in peptides if len(peptide) == length]
        print(
            f"Length {length}: {len(queries):,} queries; building human neighborhood",
            flush=True,
        )
        features, candidates, info = build_length_features(
            encoded_human,
            queries,
            nlist=args.nlist,
            nprobe=args.nprobe,
            search_candidates=args.search_candidates,
            retained_candidates=args.retained_candidates,
            seed=args.seed,
        )
        feature_parts.append(features)
        candidate_parts.append(candidates)
        diagnostics.append(info)
        print(
            f"Length {length}: retained {len(candidates):,} query/self candidates",
            flush=True,
        )

    result = pd.concat(feature_parts, ignore_index=True).sort_values("peptide")
    candidate_result = pd.concat(candidate_parts, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    candidate_result.to_csv(args.candidate_output, index=False)
    metadata_path = args.metadata_output or args.output.with_suffix(".metadata.json")
    metadata_path.write_text(
        json.dumps(
            {
                "scores": str(args.scores),
                "human_uniprot": str(args.human_uniprot),
                "human_sequences": len(human),
                "human_residues": sum(map(len, human)),
                "unique_query_peptides": len(peptides),
                "search": diagnostics,
                "tcr_face": "P3 through P-1",
                "anchor_positions": ["P2", "P-omega"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"Wrote {len(result):,} peptide features to {args.output}", flush=True)
    print(
        f"Wrote {len(candidate_result):,} retained self candidates to "
        f"{args.candidate_output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
