#!/usr/bin/env python3
"""Fetch UniProt sequences for the parent accessions referenced by IEDB epitopes."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import urllib.parse
import urllib.request

import pandas as pd

BATCH = 200


def fetch_batch(accessions: list[str]) -> str:
    url = "https://rest.uniprot.org/uniprotkb/accessions?" + urllib.parse.urlencode(
        {"accessions": ",".join(accessions), "format": "fasta"}
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read().decode()
        except Exception:  # noqa: BLE001 - network flakiness, retried with backoff
            time.sleep(2 ** attempt)
    return ""


def parse_fasta(text: str) -> list[tuple[str, str]]:
    records, accession, parts = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if accession:
                records.append((accession, "".join(parts)))
            accession, parts = line[1:].split("|")[1], []
        else:
            parts.append(line.strip())
    if accession:
        records.append((accession, "".join(parts)))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True, help="CSV from pull_iedb_human_class_i.py --context")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = pd.read_csv(args.context, usecols=["epitope__molecule_parent_iri"], low_memory=False)
    accessions = sorted(set(context["epitope__molecule_parent_iri"].dropna().str.extract(r"uniprot/([A-Z0-9]+)")[0].dropna()))
    rows = []
    for start in range(0, len(accessions), BATCH):
        rows.extend(parse_fasta(fetch_batch(accessions[start:start + BATCH])))
    table = pd.DataFrame(rows, columns=["accession", "sequence"]).drop_duplicates("accession")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(len(table), "of", len(accessions), "accessions fetched")


if __name__ == "__main__":
    main()
