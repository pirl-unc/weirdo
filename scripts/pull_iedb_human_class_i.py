#!/usr/bin/env python3
"""Pull every human-host class I T-cell assay row from the IEDB query API.

Only the fields needed to build peptide-HLA outcome tables are requested; rows are
paged in 1,000s and written as one CSV under the git-ignored data directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

import pandas as pd

BASE = "https://query-api.iedb.org/tcell_export"
FIELDS = (
    "assay_id,reference__pmid,assay_antigen__name,epitope__name,epitope__source_organism,"
    "epitope__source_molecule,mhc_restriction__name,mhc_restriction__class,host__name,"
    "assay__qualitative_measurement,assay__method,epitope__object_type"
)
CONTEXT_FIELDS = (
    "assay_id,reference__pmid,epitope__name,assay_antigen__name,mhc_restriction__name,"
    "epitope__molecule_parent,epitope__molecule_parent_iri,epitope__starting_position,"
    "epitope__ending_position,epitope__source_organism_iri"
)
FILTERS = {"host__name": "eq.Homo sapiens (human)", "mhc_restriction__class": "eq.I"}
PAGE = 1_000


def fetch(offset: int, fields: str = FIELDS) -> list[dict]:
    query = urllib.parse.urlencode({"select": fields, "order": "assay_id", "limit": PAGE, "offset": offset, **FILTERS})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{BASE}?{query}", timeout=120) as response:
                return json.load(response)
        except Exception:  # noqa: BLE001 - network flakiness, retried with backoff
            time.sleep(2 ** attempt)
    raise RuntimeError(f"IEDB page at offset {offset} failed after retries")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", action="store_true", help="pull the source-protein context fields instead")
    args = parser.parse_args()
    rows: list[dict] = []
    offset = 0
    while True:
        page = fetch(offset, CONTEXT_FIELDS if args.context else FIELDS)
        rows.extend(page)
        if len(page) < PAGE:
            break
        offset += PAGE
    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(len(frame), "rows,", frame["reference__pmid"].nunique(), "studies")


if __name__ == "__main__":
    main()
