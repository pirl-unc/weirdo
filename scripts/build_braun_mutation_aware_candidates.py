#!/usr/bin/env python3
"""Enumerate mutation-aware candidate pMHCs from Braun vaccine constructs.

Candidates are every 8--11mer from a vaccinated long construct that is absent
from the supplied human reference proteome.  This is a conservative operational
definition, not proof of the mutation coordinate: a reference-absent peptide
can arise from an unrepresented isoform, while a mutant peptide can happen to
occur elsewhere in human.  The output retains both the filter status and every
construct coordinate so those edge cases remain auditable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from scripts import peptide_content_data as content
from scripts.prepare_external_neoantigen_benchmarks import braun_annotated_constructs


def windows(sequence: str, minimum: int = 8, maximum: int = 11) -> list[tuple[str, int]]:
    """Return each contiguous peptide and zero-based construct coordinate."""
    return [
        (sequence[start : start + length], start)
        for length in range(minimum, maximum + 1)
        for start in range(max(0, len(sequence) - length + 1))
    ]


def reference_present(candidates: set[str], reference_sequences: list[str]) -> set[str]:
    """Find candidate strings occurring anywhere in the reference proteome."""
    try:
        import ahocorasick
    except ImportError as error:  # pragma: no cover - environment dependency
        raise RuntimeError("Install pyahocorasick to scan the reference proteome") from error
    automaton = ahocorasick.Automaton()
    for candidate in candidates:
        automaton.add_word(candidate, candidate)
    automaton.make_automaton()
    present: set[str] = set()
    for sequence in reference_sequences:
        for _, candidate in automaton.iter(sequence):
            present.add(candidate)
    return present


def build_candidates(constructs: pd.DataFrame, reference_sequences: list[str]) -> pd.DataFrame:
    """Expand each construct into annotated candidate/HLA pairs."""
    rows = []
    for item in constructs.itertuples(index=False):
        for peptide, start in windows(item.vaccine_peptide):
            for hla in item.alleles:
                rows.append(
                    {
                        "record_id": item.record_id,
                        "patient": item.patient,
                        "mutation": item.mutation,
                        "mutation_class": item.mutation_class,
                        "construct_id": item.construct_id,
                        "vaccine_peptide": item.vaccine_peptide,
                        "outcome": item.outcome,
                        "peptide": peptide,
                        "hla": hla,
                        "construct_start": start,
                        "is_nominated_short_epitope": peptide == item.peptide,
                    }
                )
    result = pd.DataFrame(rows)
    present = reference_present(set(result["peptide"]), reference_sequences)
    result["present_in_human_reference"] = result["peptide"].isin(present)
    result["reference_absent_candidate"] = ~result["present_in_human_reference"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--braun", type=Path, required=True)
    parser.add_argument("--human-dat-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    constructs = braun_annotated_constructs(args.braun)
    constructs.insert(0, "record_id", "braun_construct:" + constructs["construct_id"].astype(str))
    proteins = list(content.iter_swissprot_records(content.division_path(args.human_dat_dir, "human"), "human"))
    candidates = build_candidates(constructs, [protein.sequence for protein in proteins])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(args.output, index=False, compression="gzip" if args.output.suffix == ".gz" else None)


if __name__ == "__main__":
    main()
