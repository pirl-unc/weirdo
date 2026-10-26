#!/usr/bin/env python3
"""Build leakage-controlled 8--15mer taxonomy-membership training data.

The model target is *membership*, not a mutually exclusive source label.  If an
identical peptide occurs in a mammal and a bacterium, both outputs are positive.
Candidate peptides are sampled evenly across taxonomic strata and lengths, then
looked up against every protein in the corresponding protein-family fold with an
Aho--Corasick automaton.  Candidates occurring in more than one fold are dropped.

The ten public taxonomy heads are deliberately coarse::

    viruses, bacteria, archaea, fungi, plants, protists, invertebrates,
    fish, non_mammalian_tetrapods, mammals

Human, mouse and dog are additional diagnostic/self heads nested under mammals.
They do not replace the mammal head.  ``protists`` means the non-animal entries
in UniProt's historical ``invertebrates`` division; ``fish`` means non-tetrapod
vertebrates; and ``non_mammalian_tetrapods`` contains amphibians, reptiles and
birds.

Building the exact membership labels requires ``pyahocorasick``::

    python -m pip install pyahocorasick
    python scripts/peptide_content_data.py build \
        --dat-dir data/swissprot-dat-files \
        --out data/peptide-content-splits
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np


TAXONOMY_CATEGORIES = (
    "viruses",
    "bacteria",
    "archaea",
    "fungi",
    "plants",
    "protists",
    "invertebrates",
    "fish",
    "non_mammalian_tetrapods",
    "mammals",
)
SPECIES_CATEGORIES = ("human", "mouse", "dog")
CATEGORIES = TAXONOMY_CATEGORIES + SPECIES_CATEGORIES
SPECIES_TAXA = {9606: "human", 10090: "mouse", 9615: "dog"}
DIVISIONS = (
    "archaea", "bacteria", "fungi", "human", "invertebrates",
    "mammals", "plants", "rodents", "vertebrates", "viruses",
)
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SALT = "peptide-content-v1"
CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")

_TETRAPOD_LINEAGES = frozenset(
    (
        "Amphibia",
        "Aves",
        "Archelosauria",
        "Crocodylia",
        "Lepidosauria",
        "Squamata",
        "Testudines",
    )
)


@dataclass(frozen=True)
class Protein:
    identifier: str
    sequence: str
    division: str
    taxon_id: int | None
    category: str
    split: int
    host_taxon_ids: tuple[int, ...] = ()
    extra_categories: tuple[str, ...] = ()


def family_key(identifier: str) -> str:
    """UniProt mnemonic prefix used as a conservative ortholog-family proxy."""
    return identifier.split("_", 1)[0]


def split_of_family(
    key: str,
    salt: str = DEFAULT_SALT,
    train_percent: int = 70,
    val_percent: int = 15,
) -> int:
    bucket = zlib.crc32((salt + ":" + key).encode("utf-8")) % 100
    if bucket < train_percent:
        return 0
    if bucket < train_percent + val_percent:
        return 1
    return 2


def coarse_category(division: str, lineage: Sequence[str]) -> str:
    """Map one SwissProt record to one mutually exclusive coarse taxon leaf."""
    if division in {"viruses", "bacteria", "archaea", "fungi", "plants"}:
        return division
    if division in {"human", "mammals", "rodents"}:
        return "mammals"
    lineage_set = set(lineage)
    if division == "invertebrates":
        return "invertebrates" if "Metazoa" in lineage_set else "protists"
    if division == "vertebrates":
        return (
            "non_mammalian_tetrapods"
            if lineage_set.intersection(_TETRAPOD_LINEAGES)
            else "fish"
        )
    raise ValueError(f"Unknown SwissProt division: {division!r}")


def _parse_taxon(lines: Sequence[str]) -> int | None:
    match = re.search(r"NCBI_TaxID=(\d+)", " ".join(lines))
    return int(match.group(1)) if match else None


def iter_swissprot_records(path: Path, division: str) -> Iterator[Protein]:
    """Yield taxonomy-annotated proteins from one ``.dat`` or ``.dat.gz`` file."""
    opener = gzip.open if path.suffix == ".gz" else open
    identifier: str | None = None
    expected: int | None = None
    taxon_lines: list[str] = []
    host_taxon_ids: list[int] = []
    lineage_chunks: list[str] = []
    sequence_chunks: list[str] = []
    with opener(path, "rt", encoding="ascii", errors="replace") as stream:
        for line in stream:
            tag = line[:2]
            if tag == "ID":
                identifier = line.split(None, 2)[1]
            elif tag == "OX":
                taxon_lines.append(line[5:].strip())
            elif tag == "OC":
                lineage_chunks.append(line[5:].strip())
            elif tag == "OH":
                host_taxon_ids.extend(
                    int(value) for value in re.findall(r"NCBI_TaxID=(\d+)", line)
                )
            elif tag == "SQ":
                expected = int(line.split()[2])
                sequence_chunks = []
            elif tag == "//":
                if identifier is not None and expected is not None:
                    sequence = "".join(sequence_chunks).replace(" ", "")
                    if len(sequence) != expected:
                        raise ValueError(
                            f"{path}: {identifier} states {expected} residues, found {len(sequence)}"
                        )
                    lineage = tuple(
                        item.strip().rstrip(".")
                        for item in " ".join(lineage_chunks).split(";")
                        if item.strip()
                    )
                    yield Protein(
                        identifier=identifier,
                        sequence=sequence,
                        division=division,
                        taxon_id=_parse_taxon(taxon_lines),
                        category=coarse_category(division, lineage),
                        split=split_of_family(family_key(identifier)),
                        host_taxon_ids=tuple(host_taxon_ids),
                    )
                identifier = None
                expected = None
                taxon_lines = []
                lineage_chunks = []
                host_taxon_ids = []
                sequence_chunks = []
            elif expected is not None and line.startswith(" "):
                sequence_chunks.append("".join(line.split()))


def division_path(dat_dir: Path, division: str) -> Path:
    compressed = dat_dir / f"uniprot_sprot_{division}.dat.gz"
    plain = dat_dir / f"uniprot_sprot_{division}.dat"
    if compressed.is_file():
        return compressed
    if plain.is_file():
        return plain
    raise FileNotFoundError(f"No SwissProt {division!r} division in {dat_dir}")


def load_proteins(dat_dir: Path, progress=print) -> list[Protein]:
    proteins: list[Protein] = []
    for division in DIVISIONS:
        start = len(proteins)
        proteins.extend(iter_swissprot_records(division_path(dat_dir, division), division))
        progress(f"{division}: {len(proteins) - start:,} proteins")
    return proteins


def label_mask(protein: Protein, category_index: dict[str, int]) -> int:
    mask = 1 << category_index[protein.category]
    species = SPECIES_TAXA.get(protein.taxon_id)
    if species is not None:
        mask |= 1 << category_index[species]
    for category in protein.extra_categories:
        mask |= 1 << category_index[category]
    return mask


def summarize(proteins: Iterable[Protein]) -> dict:
    rows = list(proteins)
    by_category = Counter(p.category for p in rows)
    residues = Counter()
    taxa_by_category: dict[str, set[int]] = defaultdict(set)
    for protein in rows:
        residues[protein.category] += len(protein.sequence)
        if protein.taxon_id is not None:
            taxa_by_category[protein.category].add(protein.taxon_id)
    by_species = Counter(SPECIES_TAXA[p.taxon_id] for p in rows if p.taxon_id in SPECIES_TAXA)
    species_residues = Counter()
    for protein in rows:
        species = SPECIES_TAXA.get(protein.taxon_id)
        if species:
            species_residues[species] += len(protein.sequence)
    return {
        "proteins": len(rows),
        "amino_acids": int(sum(len(p.sequence) for p in rows)),
        "categories": {
            name: {
                "taxa": len(taxa_by_category[name]),
                "proteins": by_category[name],
                "amino_acids": residues[name],
            }
            for name in TAXONOMY_CATEGORIES
        },
        "species_heads": {
            name: {
                "taxa": 1 if by_species[name] else 0,
                "proteins": by_species[name],
                "amino_acids": species_residues[name],
            }
            for name in SPECIES_CATEGORIES
        },
        "fold_proteins": {
            SPLIT_NAMES[fold]: sum(p.split == fold for p in rows) for fold in range(3)
        },
    }


def _canonical(sequence: str) -> bool:
    return bool(sequence) and set(sequence) <= CANONICAL


def sample_candidates(
    proteins: Sequence[Protein],
    lengths: Sequence[int],
    per_stratum: dict[str, int],
    seed: int,
) -> tuple[list[str], np.ndarray, dict]:
    """Sample peptides by fold, length and source stratum.

    A peptide may be selected by several strata or folds.  It is represented once
    and its sampling-fold bit mask is retained for the later leakage check.
    """
    strata = CATEGORIES
    members: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, protein in enumerate(proteins):
        members[(protein.split, protein.category)].append(index)
        species = SPECIES_TAXA.get(protein.taxon_id)
        if species is not None:
            members[(protein.split, species)].append(index)
        for category in protein.extra_categories:
            members[(protein.split, category)].append(index)

    candidate_to_index: dict[str, int] = {}
    peptides: list[str] = []
    sampled_fold_masks: list[int] = []
    sampled_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    rng = np.random.RandomState(seed)
    for fold, split_name in enumerate(SPLIT_NAMES):
        for stratum in strata:
            available = members[(fold, stratum)]
            limit = int(per_stratum[split_name])
            for length in lengths:
                eligible = [i for i in available if len(proteins[i].sequence) >= length]
                if len(eligible) > limit:
                    chosen = rng.choice(eligible, size=limit, replace=False).tolist()
                else:
                    chosen = eligible
                accepted = 0
                for protein_index in chosen:
                    sequence = proteins[protein_index].sequence
                    start = int(rng.randint(0, len(sequence) - length + 1))
                    peptide = sequence[start:start + length]
                    if not _canonical(peptide):
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
    """Scan every protein for every sampled peptide and return fold/category masks."""
    try:
        import ahocorasick
    except ImportError as error:  # pragma: no cover - exercised by the real builder
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
    category_index = {name: i for i, name in enumerate(CATEGORIES)}
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
    dat_dir = Path(args.dat_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proteins = load_proteins(dat_dir)
    lengths = tuple(range(args.min_length, args.max_length + 1))
    per_stratum = {
        "train": args.train_per_stratum_length,
        "val": args.val_per_stratum_length,
        "test": args.test_per_stratum_length,
    }
    peptides, sampled_folds, sampled_counts = sample_candidates(
        proteins, lengths, per_stratum, args.seed
    )
    print(f"{len(peptides):,} unique sampled candidate peptides", flush=True)
    observed_folds, fold_labels = exact_membership(proteins, peptides)

    arrays: dict[str, np.ndarray] = {}
    metadata = {
        "version": 1,
        "categories": list(CATEGORIES),
        "taxonomy_categories": list(TAXONOMY_CATEGORIES),
        "species_categories": list(SPECIES_CATEGORIES),
        "lengths": list(lengths),
        "seed": args.seed,
        "salt": DEFAULT_SALT,
        "dat_dir": str(dat_dir),
        "taxonomy": summarize(proteins),
        "sampling": sampled_counts,
        "candidate_peptides": len(peptides),
        "splits": {},
    }
    peptide_array = np.asarray(peptides, dtype=f"U{args.max_length}")
    # A kept peptide occurs in exactly one protein-family fold, and was sampled
    # from that fold.  This also removes candidates independently sampled in two
    # folds before their exact occurrence was known.
    for fold, name in enumerate(SPLIT_NAMES):
        fold_bit = 1 << fold
        keep = (observed_folds == fold_bit) & ((sampled_folds & fold_bit) != 0)
        chosen = np.flatnonzero(keep)
        rng = np.random.RandomState(args.seed + 100 + fold)
        rng.shuffle(chosen)
        split_peptides = peptide_array[chosen]
        split_masks = fold_labels[fold, chosen]
        arrays[f"{name}_peptides"] = split_peptides
        arrays[f"{name}_masks"] = split_masks
        prevalence = [
            float(((split_masks >> bit) & 1).mean()) if len(split_masks) else None
            for bit in range(len(CATEGORIES))
        ]
        metadata["splits"][name] = {
            "n": int(len(chosen)),
            "lengths": dict(Counter(map(len, split_peptides.tolist()))),
            "prevalence": dict(zip(CATEGORIES, prevalence)),
        }
    metadata["cross_fold_candidates_dropped"] = int(
        np.sum(~np.isin(observed_folds, np.asarray((1, 2, 4), dtype=np.uint8)))
    )
    np.savez_compressed(out / "splits.npz", **arrays)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps(metadata["splits"], indent=2), flush=True)
    return metadata


def load_split(directory: str | Path, name: str) -> tuple[list[str], np.ndarray]:
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
        command.add_argument("--dat-dir", required=True)
        if name == "build":
            command.add_argument("--out", required=True)
            command.add_argument("--min-length", type=int, default=8)
            command.add_argument("--max-length", type=int, default=15)
            command.add_argument("--train-per-stratum-length", type=int, default=10_000)
            command.add_argument("--val-per-stratum-length", type=int, default=2_000)
            command.add_argument("--test-per-stratum-length", type=int, default=2_000)
            command.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if args.command == "build":
        build(args)
    else:
        result = summarize(load_proteins(Path(args.dat_dir)))
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
