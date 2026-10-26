#!/usr/bin/env python3
"""Count SwissProt lineage clades to design balanced taxonomy leaves."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from pathlib import Path


DIVISIONS = (
    "archaea",
    "bacteria",
    "fungi",
    "human",
    "invertebrates",
    "mammals",
    "plants",
    "rodents",
    "vertebrates",
    "viruses",
)
MAMMAL_DIVISIONS = frozenset(("human", "mammals", "rodents"))
GENERIC_LINEAGES = frozenset(
    (
        "cellular organisms",
        "Bacteria",
        "Eukaryota",
        "Opisthokonta",
        "Metazoa",
        "Eumetazoa",
        "Bilateria",
        "Deuterostomia",
        "Chordata",
        "Craniata",
        "Vertebrata",
        "Euteleostomi",
        "Mammalia",
        "Theria",
    )
)


def proposed_bacterial_bucket(lineage: tuple[str, ...]) -> str:
    """Mutually exclusive bacterial leaf with roughly <=20M corpus residues."""
    clades = set(lineage)
    if "Enterobacterales" in clades:
        return "enterobacterales"
    if "Gammaproteobacteria" in clades:
        return "other_gammaproteobacteria"
    if "Alphaproteobacteria" in clades:
        return "alphaproteobacteria"
    if clades.intersection(("Pseudomonadota", "Proteobacteria")):
        return "other_pseudomonadota"
    if "Bacilli" in clades:
        return "bacilli"
    if clades.intersection(("Actinomycetota", "Actinobacteria")):
        return "actinomycetota"
    return "other_bacteria"


def proposed_mammal_bucket(lineage: tuple[str, ...]) -> str:
    """Mutually exclusive mammalian leaf proposed for the next model."""
    clades = set(lineage)
    if "Primates" in clades:
        return "primates"
    if "Glires" in clades:
        return "glires"
    if "Carnivora" in clades:
        return "carnivora"
    return "other_mammals"


def division_path(dat_dir: Path, division: str) -> Path:
    for suffix in (".dat.gz", ".dat"):
        candidate = dat_dir / f"uniprot_sprot_{division}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No SwissProt {division!r} division in {dat_dir}")


def iter_records(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    taxon_id = None
    lineage_chunks: list[str] = []
    length = None
    with opener(path, "rt", encoding="ascii", errors="replace") as stream:
        for line in stream:
            tag = line[:2]
            if tag == "ID":
                taxon_id = None
                lineage_chunks = []
                length = None
            elif tag == "OX":
                match = re.search(r"NCBI_TaxID=(\d+)", line)
                if match:
                    taxon_id = int(match.group(1))
            elif tag == "OC":
                lineage_chunks.append(line[5:].strip())
            elif tag == "SQ":
                length = int(line.split()[2])
            elif tag == "//" and length is not None:
                lineage = tuple(
                    item.strip().rstrip(".")
                    for item in " ".join(lineage_chunks).split(";")
                    if item.strip()
                )
                yield taxon_id, lineage, length


def lineage_counts(dat_dir: Path, divisions: frozenset[str]) -> dict[str, dict]:
    proteins = defaultdict(int)
    amino_acids = defaultdict(int)
    taxa = defaultdict(set)
    total_proteins = 0
    total_amino_acids = 0
    total_taxa = set()
    for division in DIVISIONS:
        if division not in divisions:
            continue
        for taxon_id, lineage, length in iter_records(division_path(dat_dir, division)):
            total_proteins += 1
            total_amino_acids += length
            if taxon_id is not None:
                total_taxa.add(taxon_id)
            for clade in set(lineage):
                proteins[clade] += 1
                amino_acids[clade] += length
                if taxon_id is not None:
                    taxa[clade].add(taxon_id)
    rows = {
        clade: {
            "taxa": len(taxa[clade]),
            "proteins": proteins[clade],
            "amino_acids": amino_acids[clade],
        }
        for clade in proteins
    }
    return {
        "total": {
            "taxa": len(total_taxa),
            "proteins": total_proteins,
            "amino_acids": total_amino_acids,
        },
        "lineages": rows,
    }


def partition_counts(dat_dir: Path, divisions: frozenset[str], classify) -> dict:
    proteins = defaultdict(int)
    amino_acids = defaultdict(int)
    taxa = defaultdict(set)
    for division in DIVISIONS:
        if division not in divisions:
            continue
        for taxon_id, lineage, length in iter_records(division_path(dat_dir, division)):
            bucket = classify(lineage)
            proteins[bucket] += 1
            amino_acids[bucket] += length
            if taxon_id is not None:
                taxa[bucket].add(taxon_id)
    return {
        bucket: {
            "taxa": len(taxa[bucket]),
            "proteins": proteins[bucket],
            "amino_acids": amino_acids[bucket],
        }
        for bucket in proteins
    }


def top_rows(summary: dict, limit: int) -> list[dict]:
    rows = [
        {"clade": clade, **counts}
        for clade, counts in summary["lineages"].items()
        if clade not in GENERIC_LINEAGES
    ]
    return sorted(rows, key=lambda row: row["amino_acids"], reverse=True)[:limit]


def analyze(dat_dir: Path, limit: int = 100) -> dict:
    bacteria = lineage_counts(dat_dir, frozenset(("bacteria",)))
    mammals = lineage_counts(dat_dir, MAMMAL_DIVISIONS)
    return {
        "bacteria_total": bacteria["total"],
        "bacteria_top_lineages": top_rows(bacteria, limit),
        "proposed_bacterial_buckets": partition_counts(
            dat_dir, frozenset(("bacteria",)), proposed_bacterial_bucket
        ),
        "mammals_total": mammals["total"],
        "mammals_top_lineages": top_rows(mammals, limit),
        "proposed_mammal_buckets": partition_counts(
            dat_dir, MAMMAL_DIVISIONS, proposed_mammal_bucket
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dat-dir", required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(analyze(Path(args.dat_dir), args.limit), indent=2))


if __name__ == "__main__":
    main()
