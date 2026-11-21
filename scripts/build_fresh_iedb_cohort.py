#!/usr/bin/env python3
"""Build a fresh T-cell confirmation cohort from IEDB studies never used before.

Protocol (frozen before any scoring):

* source: the local IEDB T-cell assay pages under ``data/viral-tcell-benchmark``,
  or, with ``--assays``, the full human class I API pull from
  ``pull_iedb_human_class_i.py``;
* keep human host, class I, exact ``HLA-A/B/C*xx:xx`` restriction, canonical
  8-14mers and a known positive/negative qualitative outcome;
* a (peptide, HLA) pair is positive if any assay for it is positive;
* keep only PMIDs that contributed neither to development nor validation
  (``used``) and that have at least one positive and one negative pair;
* purge every exact peptide that appears in any dataset used for development,
  screening or validation (IMPROVE, registered viral, TESLA, Braun), so the
  cohort shares no peptide with anything scored before;
* clusters for bootstrap are PMIDs.

Weights of the evaluated score are fixed beforehand (shipped defaults).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re

import pandas as pd

CANONICAL = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY]{8,14}$")
EXACT_HLA_I = re.compile(r"^HLA-[ABC]\*\d{2,3}:\d{2,3}$")


def positive(value: object) -> int | None:
    text = str(value or "").lower()
    if text.startswith("positive"):
        return 1
    if text.startswith("negative"):
        return 0
    return None


def used_pmids(root: Path, choices: Path) -> set[str]:
    """PMIDs used in any earlier development, screening or validation report."""
    used: set[str] = set()
    for report in choices.glob("*-report.json"):
        payload = json.loads(report.read_text())
        for row in payload.get("data_funnel", {}).get("audit", []):
            if row.get("included"):
                used.add(str(row["cohort"]))
    used.update(pd.read_csv(root / "scored-assays.csv", usecols=["cohort"])["cohort"].astype(str))
    return used


def used_peptides(root: Path, improve: Path, tesla: Path, external: Path) -> set[str]:
    """Every peptide sequence already seen in a scored dataset."""
    peptides = set(pd.read_csv(root / "scored-assays.csv", usecols=["peptide"])["peptide"])
    peptides |= set(pd.read_csv(improve, sep="\t", usecols=["Mut_peptide", "Norm_peptide"]).stack().dropna())
    peptides |= set(pd.read_excel(tesla)["peptide"])
    peptides |= set(pd.read_csv(external, usecols=["peptide"])["peptide"])
    return peptides


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iedb-root", type=Path, default=Path("data/viral-tcell-benchmark"))
    parser.add_argument("--choices", type=Path, default=Path("data/foreignness-choices"))
    parser.add_argument("--improve", type=Path, required=True)
    parser.add_argument("--tesla", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--assays", type=Path, help="CSV from pull_iedb_human_class_i.py")
    parser.add_argument("--exclude-cohort", type=Path, nargs="*", default=[],
                        help="earlier fresh cohorts whose studies and peptides are also spent")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pairs: dict[tuple[str, str, str], list[int]] = {}
    meta: dict[str, str] = {}

    def add(row: dict) -> None:
        raw_pmid = row.get("reference__pmid")
        pmid = "" if pd.isna(raw_pmid) else str(raw_pmid).removesuffix(".0")
        peptide = str(row.get("assay_antigen__name") or row.get("epitope__name") or "").upper()
        hla = str(row.get("mhc_restriction__name") or "")
        outcome = positive(row.get("assay__qualitative_measurement"))
        if not pmid or outcome is None or not CANONICAL.fullmatch(peptide) or not EXACT_HLA_I.fullmatch(hla):
            return
        if row.get("host__name") != "Homo sapiens (human)" or row.get("mhc_restriction__class") != "I":
            return
        pairs.setdefault((pmid, peptide, hla), []).append(outcome)
        organism = row.get("epitope__source_organism")
        meta[pmid] = organism if isinstance(organism, str) else ""

    if args.assays:
        for row in pd.read_csv(args.assays, low_memory=False).to_dict("records"):
            add(row)
    else:
        for path in sorted(glob.glob(str(args.iedb_root / "iedb-*-page-*.json"))):
            for row in json.loads(Path(path).read_text()):
                add(row)
    table = pd.DataFrame(
        [{"pmid": p, "peptide": s, "hla": h, "outcome": int(any(v)), "source_organism": meta[p]}
         for (p, s, h), v in pairs.items()]
    )
    seen_pmids = used_pmids(args.iedb_root, args.choices)
    seen_peptides = used_peptides(args.iedb_root, args.improve, args.tesla, args.external)
    for earlier in args.exclude_cohort:
        spent = pd.read_csv(earlier, dtype={"pmid": str})
        seen_pmids |= set(spent["pmid"])
        seen_peptides |= set(spent["peptide"])
    funnel = {"pairs": len(table), "pmids": int(table["pmid"].nunique())}
    table = table.loc[~table["pmid"].isin(seen_pmids)]
    funnel["after_unused_pmids"] = len(table)
    table = table.loc[~table["peptide"].isin(seen_peptides)]
    funnel["after_peptide_purge"] = len(table)
    balanced = table.groupby("pmid")["outcome"].agg(["min", "max"])
    keep = balanced.index[(balanced["min"] == 0) & (balanced["max"] == 1)]
    table = table.loc[table["pmid"].isin(keep)].reset_index(drop=True)
    funnel.update(
        final_pairs=len(table), final_pmids=int(table["pmid"].nunique()),
        positives=int(table["outcome"].sum()), unique_peptides=int(table["peptide"].nunique()),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    args.output.with_suffix(".funnel.json").write_text(json.dumps(funnel, indent=2))
    print(json.dumps(funnel, indent=2))
    print(table.groupby("pmid").agg(n=("outcome", "size"), pos=("outcome", "sum"), organism=("source_organism", "first")).sort_values("n", ascending=False).head(15))


if __name__ == "__main__":
    main()
