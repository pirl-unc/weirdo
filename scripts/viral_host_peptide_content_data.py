#!/usr/bin/env python3
"""Build peptide-content splits with nested mammalian-virus host heads.

The original coarse ``viruses`` head is retained. Viral proteins additionally
receive exactly one of ``mammal_host_viruses`` or ``other_host_viruses`` when
their Swiss-Prot ``OH`` annotations support a host assignment. Peptides remain
multi-label: a peptide occurring in both viral host groups is positive for both.

Host NCBI TaxIDs are aggregated by viral TaxID before assignment. Missing or
taxonomically unresolved hosts are not forced into the negative class.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

import peptide_content_data as base


MAMMALIA_TAXON_ID = 40674
HOST_CATEGORIES = ("mammal_host_viruses", "other_host_viruses")
CATEGORIES = tuple(base.CATEGORIES) + HOST_CATEGORIES
TAXONOMY_CATEGORIES = tuple(base.TAXONOMY_CATEGORIES)
SPECIES_CATEGORIES = tuple(base.SPECIES_CATEGORIES)
SPLIT_NAMES = tuple(base.SPLIT_NAMES)

# UniProt 2023 annotates these avian viruses only at the broad Vertebrata host
# node. Their organism identities resolve the otherwise ambiguous annotation.
NON_MAMMAL_VIRUS_EXCEPTIONS = frozenset((10261, 928301))

# The reusable builder reads its category schema dynamically at call time.
base.CATEGORIES = CATEGORIES


class Taxonomy:
    def __init__(self, directory: str | Path):
        directory = Path(directory)
        self.parents = self._pairs(directory / "nodes.dmp")
        self.merged = self._pairs(directory / "merged.dmp")
        self._mammal_cache: dict[int, bool] = {}
        self._mammal_ancestors = frozenset(self.lineage(MAMMALIA_TAXON_ID))

    @staticmethod
    def _pairs(path: Path) -> dict[int, int]:
        if not path.is_file():
            raise FileNotFoundError(path)
        result = {}
        with path.open() as stream:
            for line in stream:
                fields = [field.strip() for field in line.split("|")]
                result[int(fields[0])] = int(fields[1])
        return result

    def resolve(self, taxon_id: int) -> int:
        seen = set()
        while taxon_id in self.merged and taxon_id not in seen:
            seen.add(taxon_id)
            taxon_id = self.merged[taxon_id]
        return taxon_id

    def lineage(self, taxon_id: int) -> tuple[int, ...]:
        taxon_id = self.resolve(taxon_id)
        result = []
        seen = set()
        while taxon_id in self.parents and taxon_id not in seen:
            result.append(taxon_id)
            seen.add(taxon_id)
            parent = self.parents[taxon_id]
            if parent == taxon_id:
                break
            taxon_id = parent
        return tuple(result)

    def is_mammal(self, taxon_id: int) -> bool:
        taxon_id = self.resolve(taxon_id)
        if taxon_id not in self._mammal_cache:
            self._mammal_cache[taxon_id] = MAMMALIA_TAXON_ID in self.lineage(taxon_id)
        return self._mammal_cache[taxon_id]

    def contains_mammals(self, taxon_id: int) -> bool:
        """Whether a broad host node is an ancestor of Mammalia."""
        return self.resolve(taxon_id) in self._mammal_ancestors


def viral_host_assignments(
    proteins: Sequence[base.Protein], taxonomy: Taxonomy
) -> tuple[dict[int, str | None], dict]:
    hosts_by_virus: dict[int, set[int]] = defaultdict(set)
    for protein in proteins:
        if protein.category == "viruses" and protein.taxon_id is not None:
            hosts_by_virus[protein.taxon_id].update(protein.host_taxon_ids)

    assignments: dict[int, str | None] = {}
    reasons = Counter()
    for virus_taxon, raw_hosts in hosts_by_virus.items():
        hosts = {taxonomy.resolve(host) for host in raw_hosts}
        known = {host for host in hosts if host in taxonomy.parents}
        if any(taxonomy.is_mammal(host) for host in known):
            assignment = "mammal_host_viruses"
            reason = "explicit_mammal"
        elif virus_taxon in NON_MAMMAL_VIRUS_EXCEPTIONS:
            assignment = "other_host_viruses"
            reason = "curated_non_mammal_exception"
        elif any(taxonomy.contains_mammals(host) for host in known):
            assignment = None
            reason = "broad_host_contains_mammals"
        elif known:
            assignment = "other_host_viruses"
            reason = "explicit_non_mammal"
        elif hosts:
            assignment = None
            reason = "unresolved_host_taxonomy"
        else:
            assignment = None
            reason = "missing_host"
        assignments[virus_taxon] = assignment
        reasons[reason] += 1
    return assignments, dict(reasons)


def load_proteins(
    dat_dir: str | Path, taxonomy_dir: str | Path, progress=print
) -> tuple[list[base.Protein], dict]:
    proteins = base.load_proteins(Path(dat_dir), progress=progress)
    taxonomy = Taxonomy(taxonomy_dir)
    assignments, reasons = viral_host_assignments(proteins, taxonomy)
    labeled = [
        replace(
            protein,
            extra_categories=(assignments[protein.taxon_id],)
            if protein.category == "viruses"
            and protein.taxon_id in assignments
            and assignments[protein.taxon_id] is not None
            else (),
        )
        for protein in proteins
    ]
    return labeled, reasons


def summarize(proteins: Iterable[base.Protein], reasons: dict | None = None) -> dict:
    proteins = list(proteins)
    result = base.summarize(proteins)
    host_heads = {}
    for category in HOST_CATEGORIES:
        selected = [p for p in proteins if category in p.extra_categories]
        host_heads[category] = {
            "taxa": len({p.taxon_id for p in selected if p.taxon_id is not None}),
            "proteins": len(selected),
            "amino_acids": sum(len(p.sequence) for p in selected),
        }
    unknown = [
        p for p in proteins if p.category == "viruses" and not p.extra_categories
    ]
    result["viral_host_heads"] = host_heads
    result["viral_host_unknown"] = {
        "taxa": len({p.taxon_id for p in unknown if p.taxon_id is not None}),
        "proteins": len(unknown),
        "amino_acids": sum(len(p.sequence) for p in unknown),
    }
    result["viral_host_assignment_reasons"] = reasons or {}
    return result


def build(args: argparse.Namespace) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    proteins, reasons = load_proteins(args.dat_dir, args.taxonomy_dir)
    lengths = tuple(range(args.min_length, args.max_length + 1))
    per_stratum = {
        "train": args.train_per_stratum_length,
        "val": args.val_per_stratum_length,
        "test": args.test_per_stratum_length,
    }
    peptides, sampled_folds, sampled_counts = base.sample_candidates(
        proteins, lengths, per_stratum, args.seed
    )
    print(f"{len(peptides):,} unique sampled candidate peptides", flush=True)
    observed_folds, fold_labels = base.exact_membership(proteins, peptides)

    arrays: dict[str, np.ndarray] = {}
    metadata = {
        "version": 1,
        "categories": list(CATEGORIES),
        "taxonomy_categories": list(TAXONOMY_CATEGORIES),
        "species_categories": list(SPECIES_CATEGORIES),
        "viral_host_categories": list(HOST_CATEGORIES),
        "lengths": list(lengths),
        "seed": args.seed,
        "salt": base.DEFAULT_SALT,
        "dat_dir": str(args.dat_dir),
        "taxonomy_dir": str(args.taxonomy_dir),
        "taxonomy": summarize(proteins, reasons),
        "sampling": sampled_counts,
        "candidate_peptides": len(peptides),
        "splits": {},
    }
    peptide_array = np.asarray(peptides, dtype=f"U{args.max_length}")
    for fold, name in enumerate(SPLIT_NAMES):
        fold_bit = 1 << fold
        keep = (observed_folds == fold_bit) & ((sampled_folds & fold_bit) != 0)
        chosen = np.flatnonzero(keep)
        np.random.RandomState(args.seed + 100 + fold).shuffle(chosen)
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
        command.add_argument("--taxonomy-dir", required=True)
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
        proteins, reasons = load_proteins(args.dat_dir, args.taxonomy_dir)
        print(json.dumps(summarize(proteins, reasons), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
