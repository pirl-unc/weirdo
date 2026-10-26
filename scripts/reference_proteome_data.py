#!/usr/bin/env python3
"""Build exact 8--15mer labels from a frozen complete-reference manifest.

Unlike the SwissProt builder, this script consumes only the FASTAs selected by
``reference-proteomes-v1``.  Mammalian labels are nested (for example, dog is
both ``carnivora`` and ``mammals``), and virus labels are nested by realm.
Identical peptides retain every membership label. Exact duplicate peptides seen
in more than one split are excluded before model fitting.

First fetch the corpus (safely resumable):

    weirdo data reference-proteome --manifest reference-proteomes-v1

Then build splits:

    python scripts/reference_proteome_data.py build \
        --reference-dir ~/.weirdo/downloads/reference-proteomes \
        --out ~/.weirdo/reference-proteome-splits-v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from weirdo.reference_proteomes import (
    iter_fasta_sequences,
    load_reference_proteome_manifest,
    reference_proteome_filename,
)


TAXONOMY_CATEGORIES = (
    "viruses",
    "riboviria",
    "duplodnaviria",
    "varidnaviria",
    "monodnaviria",
    "mammals",
    "primates",
    "glires",
    "carnivora",
    "other_mammals",
)
SPECIES_CATEGORIES = ("human", "mouse", "dog", "cat", "cattle")
CATEGORIES = TAXONOMY_CATEGORIES + SPECIES_CATEGORIES
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SALT = "reference-proteomes-v1"
CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Protein:
    """One sequence and all frozen-manifest labels that apply to it."""

    identifier: str
    sequence: str
    proteome_id: str
    labels: tuple[str, ...]
    split: int


def split_of_identifier(
    proteome_id: str,
    identifier: str,
    salt: str = DEFAULT_SALT,
    train_percent: int = 70,
    val_percent: int = 15,
) -> int:
    """Give one protein identifier a stable, reproducible split assignment."""
    bucket = zlib.crc32(f"{salt}:{proteome_id}:{identifier}".encode("utf-8")) % 100
    if bucket < train_percent:
        return 0
    if bucket < train_percent + val_percent:
        return 1
    return 2


def _canonical(sequence: str) -> bool:
    return bool(sequence) and set(sequence) <= CANONICAL


def load_proteins(
    reference_dir: str | Path,
    manifest: str = "reference-proteomes-v1",
    progress=print,
) -> list[Protein]:
    """Read every downloaded FASTA selected in a frozen manifest."""
    reference_dir = Path(reference_dir)
    selection = load_reference_proteome_manifest(manifest)
    proteins: list[Protein] = []
    missing = []
    for entry in selection.entries:
        path = reference_dir / reference_proteome_filename(entry.proteome)
        if not path.is_file():
            missing.append(entry.proteome.proteome_id)
            continue
        start = len(proteins)
        for identifier, sequence in iter_fasta_sequences(path):
            if not _canonical(sequence):
                continue
            proteins.append(
                Protein(
                    identifier=identifier,
                    sequence=sequence,
                    proteome_id=entry.proteome.proteome_id,
                    labels=entry.labels,
                    split=split_of_identifier(entry.proteome.proteome_id, identifier),
                )
            )
        progress(
            f"{entry.proteome.proteome_id}: {len(proteins) - start:,} canonical proteins"
        )
    if missing:
        example = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing):,} FASTAs from manifest {manifest!r} are missing "
            f"in {reference_dir} (for example: {example}). Run: "
            f"weirdo data reference-proteome --manifest {manifest}"
        )
    return proteins


def label_mask(protein: Protein, category_index: dict[str, int]) -> int:
    """Encode all nested membership labels in one compact integer mask."""
    mask = 0
    for label in protein.labels:
        # A manifest also carries convenient source aliases (for example, rat
        # and cattle).  They are useful to select and audit references but are
        # not necessarily learned heads.  Parent labels such as ``glires`` and
        # ``mammals`` retain the source membership in that case.
        index = category_index.get(label)
        if index is not None:
            mask |= 1 << index
    return mask


def summarize(proteins: Iterable[Protein]) -> dict:
    """Report protein and residue totals for every independent model head."""
    rows = list(proteins)
    counts = Counter()
    residues = Counter()
    proteomes = defaultdict(set)
    for protein in rows:
        for label in protein.labels:
            if label not in CATEGORIES:
                continue
            counts[label] += 1
            residues[label] += len(protein.sequence)
            proteomes[label].add(protein.proteome_id)
    return {
        "proteins": len(rows),
        "amino_acids": int(sum(len(protein.sequence) for protein in rows)),
        "categories": {
            name: {
                "proteomes": len(proteomes[name]),
                "proteins": counts[name],
                "amino_acids": residues[name],
            }
            for name in CATEGORIES
        },
        "fold_proteins": {
            SPLIT_NAMES[fold]: sum(protein.split == fold for protein in rows)
            for fold in range(3)
        },
    }


def load_excluded_peptides(
    path: str | Path | None,
    *,
    column: str = "peptide",
    role: str | None = None,
) -> frozenset[str]:
    """Read a reproducible exact-peptide exclusion list from a CSV file.

    The T-cell benchmark is never used as a taxonomy label.  Still, excluding
    its registered validation sequences from candidate sampling prevents a
    trained taxonomy model from seeing an evaluation peptide verbatim.
    """
    if path is None:
        return frozenset()
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if column not in (reader.fieldnames or ()):
            raise ValueError(f"{path} has no {column!r} column")
        if role is not None and "role" not in (reader.fieldnames or ()):
            raise ValueError(f"{path} has no 'role' column for --exclude-role")
        peptides = {
            row[column].strip().upper()
            for row in reader
            if (role is None or row.get("role") == role)
            and _canonical(row.get(column, "").strip().upper())
        }
    return frozenset(peptides)


def sample_candidates(
    proteins: Sequence[Protein],
    lengths: Sequence[int],
    per_stratum: dict[str, int],
    seed: int,
    excluded_peptides: frozenset[str] = frozenset(),
) -> tuple[list[str], np.ndarray, dict]:
    """Sample candidate peptides evenly by split, nested label, and length."""
    members: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, protein in enumerate(proteins):
        for label in protein.labels:
            if label in CATEGORIES:
                members[(protein.split, label)].append(index)

    candidate_to_index: dict[str, int] = {}
    peptides: list[str] = []
    sampled_fold_masks: list[int] = []
    sampled_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    rng = np.random.RandomState(seed)
    for fold, split_name in enumerate(SPLIT_NAMES):
        for stratum in CATEGORIES:
            available = members[(fold, stratum)]
            limit = int(per_stratum[split_name])
            for length in lengths:
                eligible = [index for index in available if len(proteins[index].sequence) >= length]
                if len(eligible) > limit:
                    chosen = rng.choice(eligible, size=limit, replace=False).tolist()
                else:
                    chosen = eligible
                accepted = 0
                for protein_index in chosen:
                    sequence = proteins[protein_index].sequence
                    start = int(rng.randint(0, len(sequence) - length + 1))
                    peptide = sequence[start:start + length]
                    if peptide in excluded_peptides:
                        continue
                    index = candidate_to_index.get(peptide)
                    if index is None:
                        index = len(peptides)
                        candidate_to_index[peptide] = index
                        peptides.append(peptide)
                        sampled_fold_masks.append(0)
                    sampled_fold_masks[index] |= 1 << fold
                    accepted += 1
                sampled_counts[split_name][stratum][str(length)] = accepted
    return peptides, np.asarray(sampled_fold_masks, dtype=np.uint8), sampled_counts


def exact_membership(
    proteins: Sequence[Protein], peptides: Sequence[str], progress=print
) -> tuple[np.ndarray, np.ndarray]:
    """Find every sampled peptide in every selected reference sequence."""
    try:
        import ahocorasick
    except ImportError as error:  # pragma: no cover - depends on optional builder extra
        raise RuntimeError(
            "Exact label construction needs pyahocorasick; install it with "
            "`python -m pip install pyahocorasick`."
        ) from error

    automaton = ahocorasick.Automaton()
    for index, peptide in enumerate(peptides):
        automaton.add_word(peptide, index)
    automaton.make_automaton()
    observed_folds = np.zeros(len(peptides), dtype=np.uint8)
    fold_labels = np.zeros((3, len(peptides)), dtype=np.uint16)
    category_index = {name: index for index, name in enumerate(CATEGORIES)}
    started = time.time()
    for protein_number, protein in enumerate(proteins, 1):
        protein_label = label_mask(protein, category_index)
        fold_bit = 1 << protein.split
        for _, peptide_index in automaton.iter(protein.sequence):
            observed_folds[peptide_index] |= fold_bit
            fold_labels[protein.split, peptide_index] |= protein_label
        if protein_number % 50_000 == 0:
            rate = protein_number / max(time.time() - started, 1e-9)
            progress(f"scanned {protein_number:,}/{len(proteins):,} proteins ({rate:,.0f}/s)")
    return observed_folds, fold_labels


def build(args: argparse.Namespace) -> dict:
    """Build the reusable leakage-filtered split archive."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proteins = load_proteins(args.reference_dir, args.manifest)
    excluded_peptides = load_excluded_peptides(
        args.exclude_peptide_csv,
        column=args.exclude_peptide_column,
        role=args.exclude_role or None,
    )
    lengths = tuple(range(args.min_length, args.max_length + 1))
    per_stratum = {
        "train": args.train_per_stratum_length,
        "val": args.val_per_stratum_length,
        "test": args.test_per_stratum_length,
    }
    peptides, sampled_folds, sampled_counts = sample_candidates(
        proteins, lengths, per_stratum, args.seed, excluded_peptides
    )
    print(f"{len(peptides):,} unique sampled candidate peptides", flush=True)
    observed_folds, fold_labels = exact_membership(proteins, peptides)
    arrays: dict[str, np.ndarray] = {}
    metadata = {
        "version": 1,
        "manifest": args.manifest,
        "categories": list(CATEGORIES),
        "taxonomy_categories": list(TAXONOMY_CATEGORIES),
        "species_categories": list(SPECIES_CATEGORIES),
        "lengths": list(lengths),
        "seed": args.seed,
        "salt": DEFAULT_SALT,
        "reference_dir": str(args.reference_dir),
        "excluded_peptides": {
            "source": str(args.exclude_peptide_csv) if args.exclude_peptide_csv else None,
            "column": args.exclude_peptide_column,
            "role": args.exclude_role or None,
            "n": len(excluded_peptides),
            "sha256": hashlib.sha256(
                "\n".join(sorted(excluded_peptides)).encode("ascii")
            ).hexdigest(),
        },
        "taxonomy": summarize(proteins),
        "sampling": sampled_counts,
        "candidate_peptides": len(peptides),
        "splits": {},
    }
    peptide_array = np.asarray(peptides, dtype=f"U{args.max_length}")
    for fold, split_name in enumerate(SPLIT_NAMES):
        fold_bit = 1 << fold
        keep = (observed_folds == fold_bit) & ((sampled_folds & fold_bit) != 0)
        chosen = np.flatnonzero(keep)
        rng = np.random.RandomState(args.seed + 100 + fold)
        rng.shuffle(chosen)
        split_peptides = peptide_array[chosen]
        split_masks = fold_labels[fold, chosen]
        arrays[f"{split_name}_peptides"] = split_peptides
        arrays[f"{split_name}_masks"] = split_masks
        metadata["splits"][split_name] = {
            "n": int(len(chosen)),
            "lengths": dict(Counter(map(len, split_peptides.tolist()))),
            "prevalence": {
                name: float(((split_masks >> bit) & 1).mean()) if len(split_masks) else None
                for bit, name in enumerate(CATEGORIES)
            },
        }
    metadata["cross_fold_candidates_dropped"] = int(
        np.sum(~np.isin(observed_folds, np.asarray((1, 2, 4), dtype=np.uint8)))
    )
    np.savez_compressed(out / "splits.npz", **arrays)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata["splits"], indent=2), flush=True)
    return metadata


def load_split(directory: str | Path, name: str) -> tuple[list[str], np.ndarray]:
    """Load one of the generated train/validation/test data splits."""
    archive = np.load(Path(directory) / "splits.npz")
    peptides = archive[f"{name}_peptides"].tolist()
    masks = archive[f"{name}_masks"]
    labels = ((masks[:, None] >> np.arange(len(CATEGORIES))) & 1).astype(np.float32)
    return peptides, labels


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("summarize", "build"):
        command = subparsers.add_parser(name)
        command.add_argument("--reference-dir", required=True)
        command.add_argument("--manifest", default="reference-proteomes-v1")
        if name == "build":
            command.add_argument("--out", required=True)
            command.add_argument("--min-length", type=int, default=8)
            command.add_argument("--max-length", type=int, default=15)
            command.add_argument("--train-per-stratum-length", type=int, default=10_000)
            command.add_argument("--val-per-stratum-length", type=int, default=2_000)
            command.add_argument("--test-per-stratum-length", type=int, default=2_000)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument(
                "--exclude-peptide-csv",
                help="CSV containing exact peptides that must not enter any split",
            )
            command.add_argument("--exclude-peptide-column", default="peptide")
            command.add_argument(
                "--exclude-role",
                help="optional value in a CSV 'role' column to exclude",
            )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "build":
        build(args)
    else:
        print(json.dumps(summarize(load_proteins(args.reference_dir, args.manifest)), indent=2))


if __name__ == "__main__":
    main()
