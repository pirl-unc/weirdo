#!/usr/bin/env python3
"""Prepare public neoantigen-response benchmarks with uniform MHC features.

The input studies have intentionally different assay units. TESLA rows are
peptide--HLA tests. Braun et al. reports minimal epitopes nested in vaccine
peptides, so repeated vaccine constructs are collapsed to one
patient/mutation/minimal-epitope test before scoring. No taxonomy or response
model is fit here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from mhctools.allele_normalization import normalize_allele_name

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.score_common_tcell_predictors import (  # noqa: E402
    score_mhcflurry,
    score_netmhcpan,
)


CANONICAL_PEPTIDE = re.compile(r"[ACDEFGHIKLMNPQRSTVWY]{8,15}")
BRAUN_ALLELE = re.compile(r"(?:HLA-)?([ABC]\d{4})")


def sha256(path: Path) -> str:
    """Return a stable content hash for a source or output file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_tesla(path: Path) -> pd.DataFrame:
    """Load the public TESLA derivative and remove exact duplicate pMHC rows."""
    frame = pd.read_excel(path, sheet_name="TESLA_dataset")
    required = {"peptide", "target_value", "allele"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"TESLA workbook is missing columns: {missing}")
    result = pd.DataFrame(
        {
            "benchmark": "tesla_public_2022",
            "peptide": frame["peptide"].astype(str).str.strip(),
            "hla": frame["allele"].map(normalize_allele_name),
            "outcome": pd.to_numeric(frame["target_value"], errors="raise").astype(int),
        }
    )
    if not result["peptide"].map(lambda value: bool(CANONICAL_PEPTIDE.fullmatch(value))).all():
        raise ValueError("TESLA has non-canonical peptide sequences")
    if not result["outcome"].isin((0, 1)).all():
        raise ValueError("TESLA target values must be binary")
    conflicts = result.groupby(["peptide", "hla"])["outcome"].nunique()
    if conflicts.gt(1).any():
        raise ValueError("TESLA has contradictory duplicate peptide-HLA labels")
    result = result.drop_duplicates(["peptide", "hla"]).reset_index(drop=True)
    result["group"] = result["peptide"] + "|" + result["hla"]
    result["mutation_class"] = "unspecified"
    result["record_id"] = "tesla:" + result.index.astype(str)
    return result


def parse_braun_alleles(value: object) -> tuple[str, ...]:
    """Normalize the one or more class-I alleles annotated for a Braun epitope."""
    matches = BRAUN_ALLELE.findall(str(value))
    if not matches:
        raise ValueError(f"Could not parse a Braun HLA value: {value!r}")
    return tuple(sorted({normalize_allele_name(match) for match in matches}))


def braun_annotated_constructs(path: Path) -> pd.DataFrame:
    """Return every scoreable Braun construct with its reported best short epitope."""
    frame = pd.read_excel(path, sheet_name="In Vitro")
    required = {
        "Patient_ID",
        "Gene_and_Protein_Change",
        "Short_Epitope",
        "HLA_of_best_short_epitope",
        "Variant_Type",
        "Ttest_pvalue_InVitroStim",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Braun workbook is missing columns: {missing}")
    stimulated = frame.filter(regex=r"^InVitro_PeptideStim_").median(axis=1)
    control = frame.filter(regex=r"^InVitro_NoStim_").median(axis=1)
    result = frame.loc[frame["Variant_Type"].notna()].copy()
    result["peptide"] = result["Short_Epitope"].astype(str).str.strip()
    result = result.loc[
        result["peptide"].map(lambda value: bool(CANONICAL_PEPTIDE.fullmatch(value)))
    ].copy()
    result["outcome"] = (
        (result["Ttest_pvalue_InVitroStim"] < 0.05)
        & (stimulated.loc[result.index] > 3.0 * control.loc[result.index])
    ).astype(int)
    result["mutation_class"] = np.where(
        result["Gene_and_Protein_Change"].astype(str).str.contains("fs", case=False),
        "frameshift",
        np.where(result["Variant_Type"].eq("SNV"), "SNV", "other_indel"),
    )
    result["alleles"] = result["HLA_of_best_short_epitope"].map(parse_braun_alleles)
    result = result.rename(
        columns={
            source: target
            for source, target in {
                "Patient_ID": "patient",
                "Gene_and_Protein_Change": "mutation",
                "Peptide_ID": "construct_id",
                "Vaccine_Peptide": "vaccine_peptide",
            }.items()
            if source in result
        }
    )
    if "construct_id" in result and result["construct_id"].duplicated().any():
        raise ValueError("Braun construct identifiers must be unique")
    return result


def extract_braun(path: Path) -> pd.DataFrame:
    """Collapse Braun in-vitro vaccine measurements to distinct minimal epitopes."""
    result = braun_annotated_constructs(path)
    keys = ["patient", "mutation", "peptide"]
    grouped = (
        result.groupby(keys, as_index=False, sort=True)
        .agg(
            outcome=("outcome", "max"),
            mutation_class=(
                "mutation_class",
                lambda value: "frameshift"
                if (value == "frameshift").any()
                else sorted(set(value))[0],
            ),
            alleles=("alleles", lambda value: tuple(sorted({a for row in value for a in row}))),
        )
    )
    grouped.insert(0, "benchmark", "braun_rcc_2025")
    grouped["group"] = grouped["patient"].astype(str)
    grouped["record_id"] = "braun:" + grouped.index.astype(str)
    return grouped


def extract_braun_constructs(path: Path) -> pd.DataFrame:
    """Retain every scoreable Braun vaccine construct as its own assay record.

    The response remains tied to the authors' reported best short epitope for
    MHC/taxonomy scoring.  It is not evidence that this is the only epitope
    encoded by the longer vaccine peptide.
    """
    result = braun_annotated_constructs(path).copy()
    required = {"construct_id", "vaccine_peptide"}
    missing = sorted(required - set(result.columns))
    if missing:
        raise ValueError(f"Braun construct view requires columns: {missing}")
    result.insert(0, "benchmark", "braun_rcc_2025")
    result["group"] = result["patient"].astype(str)
    result["record_id"] = "braun_construct:" + result["construct_id"].astype(str)
    return result[
        [
            "benchmark",
            "record_id",
            "group",
            "patient",
            "mutation",
            "mutation_class",
            "construct_id",
            "vaccine_peptide",
            "peptide",
            "alleles",
            "outcome",
        ]
    ]


def attach_mhc_scores(frame: pd.DataFrame, process_limit: int) -> tuple[pd.DataFrame, dict]:
    """Score each valid peptide--HLA pair, then retain the best annotated allele."""
    expanded = frame.copy()
    if "alleles" in expanded and expanded["alleles"].notna().any():
        direct = expanded.loc[expanded["alleles"].isna()].drop(columns="alleles")
        multiple = (
            expanded.loc[expanded["alleles"].notna()]
            .drop(columns="hla")
            .explode("alleles", ignore_index=True)
            .rename(columns={"alleles": "hla"})
        )
        expanded = pd.concat((direct, multiple), ignore_index=True, sort=False)
    pairs = (
        expanded[["peptide", "hla"]]
        .drop_duplicates()
        .sort_values(["hla", "peptide"])
        .reset_index(drop=True)
    )
    netmhc, netmhc_provenance = score_netmhcpan(pairs, process_limit)
    mhcflurry, mhcflurry_provenance = score_mhcflurry(pairs, "2.3.0")
    scored = expanded.merge(netmhc, on=["peptide", "hla"], validate="many_to_one")
    scored = scored.merge(mhcflurry, on=["peptide", "hla"], validate="many_to_one")
    score_columns = [
        "netmhcpan_ba_rank",
        "mhcflurry_2_3_0_presentation_score",
    ]
    selected = scored.sort_values(
        ["record_id", "mhcflurry_2_3_0_presentation_score", "netmhcpan_ba_rank"],
        ascending=[True, False, True],
    ).drop_duplicates("record_id")
    if selected[score_columns].isna().any().any():
        raise ValueError("MHC predictors returned missing scores")
    provenance = {
        "unique_peptide_hla_pairs": len(pairs),
        "netmhcpan": netmhc_provenance,
        "mhcflurry": mhcflurry_provenance,
        "mhcflurry_package_version": importlib.metadata.version("mhcflurry"),
        "allele_policy": (
            "For multi-allele Braun annotations, retain the allele with the highest "
            "MHCflurry presentation score; NetMHCpan BA rank breaks ties."
        ),
    }
    return selected, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tesla", type=Path, required=True)
    parser.add_argument("--braun", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--process-limit", type=int, default=4)
    parser.add_argument(
        "--braun-unit",
        choices=("minimal_epitope", "vaccine_construct"),
        default="minimal_epitope",
        help="Use distinct minimal epitopes (default) or every annotated vaccine construct.",
    )
    args = parser.parse_args()
    if args.process_limit < 1:
        parser.error("--process-limit must be at least one")

    tesla = extract_tesla(args.tesla)
    braun = extract_braun(args.braun) if args.braun_unit == "minimal_epitope" else extract_braun_constructs(args.braun)
    combined = pd.concat((tesla, braun), ignore_index=True, sort=False)
    scored, predictor_provenance = attach_mhc_scores(combined, args.process_limit)
    keep = [
        "record_id",
        "benchmark",
        "group",
        "patient",
        "mutation",
        "mutation_class",
        "construct_id",
        "vaccine_peptide",
        "peptide",
        "hla",
        "outcome",
        "netmhcpan_affinity_nm",
        "netmhcpan_ba_rank",
        "mhcflurry_2_3_0_presentation_score",
    ]
    for column in keep:
        if column not in scored:
            scored[column] = "" if column in {"patient", "mutation", "construct_id", "vaccine_peptide"} else np.nan
    result = scored[keep].sort_values("record_id").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False, compression="gzip")
    provenance_path = args.output.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(
            {
                "schema": "external-neoantigen-benchmark-mhc-v1",
                "braun_unit": args.braun_unit,
                "inputs": {
                    str(args.tesla): {"sha256": sha256(args.tesla), "bytes": args.tesla.stat().st_size},
                    str(args.braun): {"sha256": sha256(args.braun), "bytes": args.braun.stat().st_size},
                },
                "benchmarks": result.groupby("benchmark")["outcome"].agg(["size", "sum"]).to_dict("index"),
                "frameshift": result.loc[result["mutation_class"].eq("frameshift"), "outcome"].agg(["size", "sum"]).to_dict(),
                "predictors": predictor_provenance,
                "output_sha256": sha256(args.output),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(result.groupby(["benchmark", "mutation_class"])["outcome"].agg(["size", "sum"]).to_string())
    print(f"Wrote {len(result):,} MHC-scored external rows to {args.output}")


if __name__ == "__main__":
    main()
