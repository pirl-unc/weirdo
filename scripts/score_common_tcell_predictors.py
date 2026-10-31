#!/usr/bin/env python3
"""Score the same peptide-HLA pairs with one frozen predictor stack.

The output preserves every assay row. NetMHCpan and MHCflurry are evaluated
once per unique peptide-HLA pair; Weirdo is evaluated once per unique peptide.
No response label is supplied to any predictor.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from mhctools import NetMHCpan, PRIME
from mhctools.allele_normalization import normalize_allele_name
from mhctools.pred import Kind
from mhcflurry import Class1PresentationPredictor
from mhcflurry.downloads import get_model_release_dir

from weirdo import (
    geometric_mean_scores,
    load_peptide_content_model,
    select_best_peptide_content_model,
    standard_peptide_content_scores,
)


ROOT = Path(__file__).resolve().parents[1]
IMPROVE_INPUT = ROOT / "tmp/viral-host-peptide-content-v1/improve/scored-neoepitopes.csv"
VIRAL_INPUT = ROOT / "tmp/finetune-transfer-report-final/broad-development-transfer/validation-scored.csv"
MODEL_ROOT = ROOT / "data/peptide-content-v1/models"
MODEL_CANDIDATES = (
    tuple(sorted(path for path in MODEL_ROOT.iterdir() if path.is_dir())) if MODEL_ROOT.is_dir() else ()
)
DEFAULT_PRIME = ROOT / "tmp/PRIME-2.1/PRIME"
DEFAULT_MIXMHCPRED = ROOT / "tmp/MixMHCpred-3.0/MixMHCpred"
DEFAULT_MIXMHCPRED_PYTHON = ROOT / "tmp/prime-env/bin/python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_assays() -> pd.DataFrame:
    improve = pd.read_csv(IMPROVE_INPUT)
    improve_rows = pd.DataFrame(
        {
            "source_dataset": "IMPROVE",
            "source_row": np.arange(len(improve)),
            "cohort": improve["cohort"].astype(str),
            "outcome": pd.to_numeric(improve["response"], errors="raise").astype(int),
            "peptide": improve["Mut_peptide"].astype(str),
            "hla": improve["HLA_allele"].map(normalize_allele_name),
            "group": improve["cv_patient_group"].astype(str),
            "patient": improve["Patient"].astype(str),
        }
    )
    viral = pd.read_csv(VIRAL_INPUT)
    if set(viral.get("role", ())) != {"validation"}:
        raise ValueError("Expected the frozen registered viral validation rows only")
    viral_rows = pd.DataFrame(
        {
            "source_dataset": "registered_viral_validation",
            "source_row": np.arange(len(viral)),
            "cohort": viral["cohort"].astype(str),
            "outcome": viral["outcome"].eq("Positive").astype(int),
            "peptide": viral["peptide"].astype(str),
            "hla": viral["hla"].map(normalize_allele_name),
            # Existing overlap-aware peptide components are the response-CV unit.
            "group": viral["peptide_cluster"].astype(str),
            "patient": "",
        }
    )
    rows = pd.concat([improve_rows, viral_rows], ignore_index=True)
    if not rows.peptide.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]{8,15}").all():
        raise ValueError("All assay peptides must be canonical 8-15mers")
    if not rows.outcome.isin([0, 1]).all():
        raise ValueError("Outcomes must be binary")
    return rows


def score_netmhcpan(pairs: pd.DataFrame, process_limit: int) -> tuple[pd.DataFrame, dict]:
    predictor = NetMHCpan(program_name="netMHCpan", process_limit=process_limit)
    results = predictor.predict_pairs(pairs.peptide.tolist(), pairs.hla.tolist())
    records = []
    versions = set()
    for pair, result in zip(pairs.itertuples(index=False), results):
        values = {prediction.kind: prediction for prediction in result.preds}
        affinity = values[Kind.pMHC_affinity]
        presentation = values[Kind.pMHC_presentation]
        versions.update([affinity.predictor_version, presentation.predictor_version])
        records.append(
            {
                "peptide": pair.peptide,
                "hla": pair.hla,
                "netmhcpan_ba_score": affinity.score,
                "netmhcpan_ba_rank": affinity.percentile_rank,
                "netmhcpan_affinity_nm": affinity.value,
                "netmhcpan_el_score": presentation.score,
                "netmhcpan_el_rank": presentation.percentile_rank,
            }
        )
    if len(versions) != 1:
        raise ValueError(f"Mixed NetMHCpan versions: {versions}")
    return pd.DataFrame(records), {"package": "NetMHCpan", "version": versions.pop()}


def score_mhcflurry(pairs: pd.DataFrame, release: str) -> tuple[pd.DataFrame, dict]:
    model_path = Path(get_model_release_dir(release))
    predictor = Class1PresentationPredictor.load(model_path)
    allele_sets = {allele: [allele] for allele in pairs.hla.unique()}
    scored = predictor.predict(
        peptides=pairs.peptide.tolist(),
        alleles=allele_sets,
        sample_names=pairs.hla.tolist(),
        include_affinity_percentile=True,
        verbose=0,
    ).sort_values("peptide_num")
    if not np.array_equal(scored.peptide.to_numpy(), pairs.peptide.to_numpy()):
        raise ValueError("MHCflurry output order does not match input peptide order")
    if not np.array_equal(scored.sample_name.to_numpy(), pairs.hla.to_numpy()):
        raise ValueError("MHCflurry output order does not match input allele order")
    result = pairs.copy()
    for name in (
        "affinity",
        "affinity_percentile",
        "processing_score",
        "presentation_score",
        "presentation_percentile",
    ):
        result[f"mhcflurry_{release.replace('.', '_')}_{name}"] = scored[name].to_numpy(float)
    return result, {
        "package": "mhcflurry",
        "package_version": importlib.metadata.version("mhcflurry"),
        "model_release": release,
        "model_path": str(model_path),
        "provenance": predictor.provenance_string,
    }


def score_prime(
    pairs: pd.DataFrame,
    program_name: Path,
    mixmhcpred_path: Path,
    mixmhcpred_python: Path,
    workers: int,
) -> tuple[pd.DataFrame, dict]:
    """Score the observed peptide-HLA pairs with PRIME 2.1 via mhctools."""
    for path in (program_name, mixmhcpred_path, mixmhcpred_python):
        if not path.is_file():
            raise FileNotFoundError(f"Required PRIME runtime file not found: {path}")

    allele_groups = [
        (allele, part.peptide.tolist())
        for allele, part in pairs.groupby("hla", sort=True)
    ]
    # Probe the nested MixMHCpred runtime once. Repeating its heavy scientific
    # imports in every worker can contend badly enough to hit mhctools' probe
    # timeout even though the runtime is valid.
    runtime_predictor = PRIME(
        alleles=[allele_groups[0][0]],
        program_name=str(program_name),
        mixmhcpred_path=str(mixmhcpred_path),
        mixmhcpred_python=str(mixmhcpred_python),
        timeout=1800,
    )
    runtime_info = runtime_predictor.runtime_info
    validated_mixmhcpred = runtime_predictor._validated_mixmhcpred
    mixmhcpred_runtime = runtime_predictor.mixmhcpred_runtime

    def score_allele(item: tuple[str, list[str]]) -> list[dict]:
        allele, peptides = item
        predictor = PRIME(
            alleles=[allele],
            program_name=str(program_name),
            mixmhcpred_path=str(mixmhcpred_path),
            mixmhcpred_python=str(mixmhcpred_python),
            timeout=1800,
        )
        predictor._validated_mixmhcpred = validated_mixmhcpred
        predictor._mixmhcpred_runtime = mixmhcpred_runtime
        predictor._runtime_resolved = True
        results = predictor.predict(peptides)
        records = []
        for peptide, result in zip(peptides, results):
            predictions = [
                prediction
                for prediction in result.preds
                if prediction.kind == Kind.immunogenicity
            ]
            if len(predictions) != 1:
                raise ValueError(
                    f"PRIME returned {len(predictions)} predictions for "
                    f"{peptide}/{allele}"
                )
            prediction = predictions[0]
            records.append(
                {
                    "peptide": peptide,
                    "hla": allele,
                    "prime_2_1_score": prediction.score,
                    "prime_2_1_rank": prediction.percentile_rank,
                }
            )
        return records

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(score_allele, item): item[0] for item in allele_groups}
        for completed, future in enumerate(as_completed(futures), start=1):
            allele = futures[future]
            allele_records = future.result()
            records.extend(allele_records)
            print(
                f"PRIME complete: {completed}/{len(allele_groups)} alleles ({allele})",
                flush=True,
            )

    result = pd.DataFrame(records)
    if len(result) != len(pairs) or result.duplicated(["peptide", "hla"]).any():
        raise ValueError("PRIME output does not cover each peptide-HLA pair exactly once")
    if result[["prime_2_1_score", "prime_2_1_rank"]].isna().any().any():
        raise ValueError("PRIME output contains missing scores")
    return result, {
        "package": "PRIME",
        "version": "2.1",
        "interface": "mhctools.PRIME",
        "mhctools_version": importlib.metadata.version("mhctools"),
        "program": str(program_name),
        "program_sha256": sha256(program_name),
        "runtime": runtime_info,
        "score_used_in_report": "negative PRIME immunogenicity percentile rank",
        "license_scope": "academic/non-commercial; see the PRIME distribution license",
    }


def score_weirdo(peptides: pd.Series) -> tuple[pd.DataFrame, dict]:
    selected, selection = select_best_peptide_content_model(MODEL_CANDIDATES)
    predictor = load_peptide_content_model(selected)
    probabilities = predictor.predict_proba(peptides.tolist())
    scores = standard_peptide_content_scores(probabilities, predictor.target_categories)
    result = pd.DataFrame({"peptide": peptides})
    for source, target in (
        ("p_viruses", "weirdo_p_viruses"),
        ("p_bacteria", "weirdo_p_bacteria"),
        ("microbial_like", "weirdo_microbial_like"),
        ("p_mammals", "weirdo_p_mammals"),
        ("one_minus_mammals", "weirdo_one_minus_mammals"),
        ("one_minus_human", "weirdo_one_minus_human"),
        ("virus_mammal_like", "weirdo_virus_mammal_like"),
        ("virus_foreign_like", "weirdo_virus_foreign_like"),
        ("microbial_foreign_like", "weirdo_microbial_foreign_like"),
    ):
        result[target] = scores[source]
    config = json.loads((selected / "config.json").read_text())
    evaluation = json.loads((selected / "evaluation.json").read_text())
    return result, {
        "package": "weirdo",
        "version": importlib.metadata.version("weirdo"),
        "model": str(selected.relative_to(ROOT)),
        "architecture": config.get("architecture", "composition_mlp"),
        "base_model": config.get("esm_model"),
        "selection_policy": {
            "timing": "before T-cell response scoring",
            "metric": "microbial_foreign_like_auc",
            "dataset": "shared held-out peptide-content taxonomy split",
            "required_heads": [
                "viruses", "bacteria", "archaea", "fungi", "protists", "mammals", "human"
            ],
            "candidates": selection,
        },
        "selected_metric_value": evaluation["summary"]["microbial_foreign_like_auc"],
        "config_sha256": sha256(selected / "config.json"),
        "evaluation_sha256": sha256(selected / "evaluation.json"),
        "weights_sha256": sha256(selected / "weights.pt"),
    }


def reuse_mhc_scores(rows: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict, dict]:
    """Reuse a verified common-row MHC run while replacing only Weirdo scores."""
    existing = pd.read_csv(path)
    identity = ["source_dataset", "source_row", "cohort", "outcome", "peptide", "hla", "group"]
    if len(existing) != len(rows):
        raise ValueError("Existing score table has a different assay-row count")
    for column in identity:
        old = existing[column].fillna("").astype(str).to_numpy()
        new = rows[column].fillna("").astype(str).to_numpy()
        if not np.array_equal(old, new):
            raise ValueError(f"Existing score table does not match current assay rows: {column}")
    mhc_columns = [
        column for column in existing
        if column.startswith(("netmhcpan_", "mhcflurry_"))
    ]
    pairs = existing[["peptide", "hla", *mhc_columns]].drop_duplicates()
    if pairs.duplicated(["peptide", "hla"]).any():
        raise ValueError("Existing MHC scores are not unique per peptide-HLA pair")
    if pairs[mhc_columns].isna().any().any():
        raise ValueError("Existing MHC scores contain missing values")
    provenance_path = path.with_name("common-tcell-predictions-provenance.json")
    provenance = json.loads(provenance_path.read_text())
    return (
        pairs[["peptide", "hla", *[c for c in mhc_columns if c.startswith("netmhcpan_")]]],
        pairs[["peptide", "hla", *[c for c in mhc_columns if c.startswith("mhcflurry_")]]],
        provenance["predictors"],
    )


def reuse_prime_scores(path: Path) -> tuple[pd.DataFrame, dict] | None:
    """Reuse a complete PRIME result from the existing common-row score file."""
    existing = pd.read_csv(path)
    columns = ["peptide", "hla", "prime_2_1_score", "prime_2_1_rank"]
    if not all(column in existing for column in columns):
        return None
    result = existing[columns].drop_duplicates()
    if result.duplicated(["peptide", "hla"]).any():
        raise ValueError("Existing PRIME scores are not unique per peptide-HLA pair")
    if result[["prime_2_1_score", "prime_2_1_rank"]].isna().any().any():
        raise ValueError("Existing PRIME scores contain missing values")
    provenance_path = path.with_name("common-tcell-predictions-provenance.json")
    provenance = json.loads(provenance_path.read_text())
    return result, provenance["predictors"]["prime"]


def reuse_prime_cache(path: Path, expected_pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict] | None:
    """Reuse the standalone PRIME checkpoint when it exactly matches pair keys."""
    if not path.is_file():
        return None
    result = pd.read_csv(path)
    columns = ["peptide", "hla", "prime_2_1_score", "prime_2_1_rank"]
    if list(result.columns) != columns:
        raise ValueError(f"Unexpected PRIME cache columns in {path}")
    expected = expected_pairs[["peptide", "hla"]].sort_values(["hla", "peptide"])
    observed = result[["peptide", "hla"]].sort_values(["hla", "peptide"])
    if not expected.reset_index(drop=True).equals(observed.reset_index(drop=True)):
        raise ValueError("PRIME cache peptide-HLA keys do not match the current benchmark")
    if result[["prime_2_1_score", "prime_2_1_rank"]].isna().any().any():
        raise ValueError("PRIME cache contains missing scores")
    provenance = json.loads(path.with_suffix(".provenance.json").read_text())
    return result, provenance


def reuse_weirdo_scores(path: Path) -> tuple[pd.DataFrame, dict] | None:
    """Reuse frozen Weirdo scores already verified on the common-row table."""
    if not path.is_file():
        return None
    existing = pd.read_csv(path)
    score_columns = [column for column in existing if column.startswith("weirdo_")]
    if not score_columns:
        return None
    result = existing[["peptide", *score_columns]].drop_duplicates()
    if "weirdo_p_mammals" not in result and "weirdo_one_minus_mammals" in result:
        result["weirdo_p_mammals"] = 1.0 - result["weirdo_one_minus_mammals"]
    # Derived values may have been cached under the former raw-product
    # convention.  Always rebuild them from the primitive frozen heads.
    result["weirdo_virus_mammal_like"] = geometric_mean_scores(
        result["weirdo_p_viruses"], result["weirdo_p_mammals"]
    )
    result["weirdo_virus_foreign_like"] = geometric_mean_scores(
        result["weirdo_p_viruses"], result["weirdo_one_minus_mammals"]
    )
    result["weirdo_microbial_foreign_like"] = geometric_mean_scores(
        result["weirdo_microbial_like"], result["weirdo_one_minus_mammals"]
    )
    score_columns = [column for column in result if column.startswith("weirdo_")]
    if result.duplicated("peptide").any() or result[score_columns].isna().any().any():
        raise ValueError("Existing Weirdo scores are incomplete or inconsistent by peptide")
    provenance_path = path.with_name("common-tcell-predictions-provenance.json")
    provenance = json.loads(provenance_path.read_text())
    return result, provenance["predictors"]["weirdo"]


def ensure_mhcflurry_releases(
    pairs: pd.DataFrame,
    existing: pd.DataFrame,
    old_provenance: dict,
) -> tuple[pd.DataFrame, dict]:
    """Add exact 2.2.0 and 2.3.0 model-release columns to reused scores."""
    result = existing.copy()
    provenance = {"package_version": importlib.metadata.version("mhcflurry"), "model_releases": {}}
    generic = [
        "affinity",
        "affinity_percentile",
        "processing_score",
        "presentation_score",
        "presentation_percentile",
    ]
    if not all(f"mhcflurry_2_3_0_{name}" in result for name in generic):
        if all(f"mhcflurry_{name}" in result for name in generic):
            for name in generic:
                result[f"mhcflurry_2_3_0_{name}"] = result[f"mhcflurry_{name}"]
            provenance["model_releases"]["2.3.0"] = {
                "reused_legacy_columns": True,
                **old_provenance,
            }
        else:
            scored, item = score_mhcflurry(pairs, "2.3.0")
            result = result.merge(scored, on=["peptide", "hla"], validate="one_to_one")
            provenance["model_releases"]["2.3.0"] = item
    else:
        provenance["model_releases"]["2.3.0"] = old_provenance.get(
            "model_releases", {}
        ).get("2.3.0", old_provenance)

    release_220_columns = [f"mhcflurry_2_2_0_{name}" for name in generic]
    if not all(column in result for column in release_220_columns):
        scored, item = score_mhcflurry(pairs, "2.2.0")
        result = result.merge(scored, on=["peptide", "hla"], validate="one_to_one")
        provenance["model_releases"]["2.2.0"] = item
    else:
        provenance["model_releases"]["2.2.0"] = old_provenance.get(
            "model_releases", {}
        ).get("2.2.0", old_provenance)
    return result, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output/cd8-predictor-comparison/common-tcell-predictions.csv.gz",
    )
    parser.add_argument("--process-limit", type=int, default=8)
    parser.add_argument("--prime-workers", type=int, default=4)
    parser.add_argument(
        "--prime-cache",
        type=Path,
        default=ROOT / "output/cd8-predictor-comparison/prime-2.1-pair-scores.csv.gz",
    )
    parser.add_argument("--prime-executable", type=Path, default=DEFAULT_PRIME)
    parser.add_argument("--mixmhcpred-executable", type=Path, default=DEFAULT_MIXMHCPRED)
    parser.add_argument("--mixmhcpred-python", type=Path, default=DEFAULT_MIXMHCPRED_PYTHON)
    parser.add_argument(
        "--rescore-prime",
        action="store_true",
        help="Recompute PRIME even when the existing common-row output contains it.",
    )
    parser.add_argument(
        "--rescore-weirdo",
        action="store_true",
        help="Recompute Weirdo instead of reusing the frozen common-row scores.",
    )
    parser.add_argument(
        "--rescore-mhc",
        action="store_true",
        help="Recompute NetMHCpan and MHCflurry instead of reusing a verified existing output.",
    )
    args = parser.parse_args()
    if args.prime_workers < 1:
        parser.error("--prime-workers must be at least 1")
    rows = load_assays()
    pairs = rows[["peptide", "hla"]].drop_duplicates().sort_values(["hla", "peptide"]).reset_index(drop=True)
    peptides = pairs.peptide.drop_duplicates().sort_values().reset_index(drop=True)
    print(f"Scoring {len(rows):,} assays, {len(pairs):,} peptide-HLA pairs, {len(peptides):,} peptides", flush=True)
    if args.out.is_file() and not args.rescore_mhc:
        netmhc, mhcflurry, old_provenance = reuse_mhc_scores(rows, args.out)
        netmhc_provenance = old_provenance["netmhcpan"]
        mhcflurry, mhcflurry_provenance = ensure_mhcflurry_releases(
            pairs, mhcflurry, old_provenance["mhcflurry"]
        )
        print("Verified existing MHC scores and ensured MHCflurry releases 2.2.0 and 2.3.0", flush=True)
    else:
        netmhc, netmhc_provenance = score_netmhcpan(pairs, args.process_limit)
        print("NetMHCpan complete", flush=True)
        release_frames = []
        release_provenance = {}
        for release in ("2.2.0", "2.3.0"):
            scored, item = score_mhcflurry(pairs, release)
            release_frames.append(scored)
            release_provenance[release] = item
            print(f"MHCflurry model release {release} complete", flush=True)
        mhcflurry = release_frames[0].merge(
            release_frames[1], on=["peptide", "hla"], validate="one_to_one"
        )
        mhcflurry_provenance = {
            "package_version": importlib.metadata.version("mhcflurry"),
            "model_releases": release_provenance,
        }
    reused_prime = (
        reuse_prime_scores(args.out)
        if args.out.is_file() and not args.rescore_prime
        else None
    )
    if reused_prime is None and not args.rescore_prime:
        reused_prime = reuse_prime_cache(args.prime_cache, pairs)
    if reused_prime is None:
        prime, prime_provenance = score_prime(
            pairs,
            program_name=args.prime_executable,
            mixmhcpred_path=args.mixmhcpred_executable,
            mixmhcpred_python=args.mixmhcpred_python,
            workers=args.prime_workers,
        )
        args.prime_cache.parent.mkdir(parents=True, exist_ok=True)
        prime.to_csv(args.prime_cache, index=False)
        args.prime_cache.with_suffix(".provenance.json").write_text(
            json.dumps(prime_provenance, indent=2)
        )
        print(f"Checkpointed PRIME pair scores to {args.prime_cache}", flush=True)
    else:
        prime, prime_provenance = reused_prime
        print("Verified and reused existing PRIME 2.1 scores", flush=True)
    reused_weirdo = (
        reuse_weirdo_scores(args.out)
        if args.out.is_file() and not args.rescore_weirdo
        else None
    )
    if reused_weirdo is None:
        weirdo, weirdo_provenance = score_weirdo(peptides)
        print("Weirdo complete", flush=True)
    else:
        weirdo, weirdo_provenance = reused_weirdo
        print("Verified and reused existing frozen Weirdo scores", flush=True)
    pair_scores = netmhc.merge(mhcflurry, on=["peptide", "hla"], validate="one_to_one")
    pair_scores = pair_scores.merge(prime, on=["peptide", "hla"], validate="one_to_one")
    pair_scores = pair_scores.merge(weirdo, on="peptide", validate="many_to_one")
    result = rows.merge(pair_scores, on=["peptide", "hla"], validate="many_to_one")
    score_columns = [
        column for column in result
        if column.startswith(("netmhcpan_", "mhcflurry_", "prime_", "weirdo_"))
    ]
    if len(result) != len(rows) or result[score_columns].isna().any().any():
        raise ValueError("Every assay row must have every predictor score")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    version_output = subprocess.run(
        ["netMHCpan", "--version", "/dev/null"], capture_output=True, text=True
    )
    provenance = {
        "input_files": {
            str(IMPROVE_INPUT.relative_to(ROOT)): sha256(IMPROVE_INPUT),
            str(VIRAL_INPUT.relative_to(ROOT)): sha256(VIRAL_INPUT),
        },
        "rows": len(result),
        "unique_peptide_hla_pairs": len(pairs),
        "unique_peptides": len(peptides),
        "cohorts": result.groupby("cohort").outcome.agg(["size", "sum"]).to_dict("index"),
        "predictors": {
            "netmhcpan": netmhc_provenance,
            "mhcflurry": mhcflurry_provenance,
            "prime": prime_provenance,
            "weirdo": weirdo_provenance,
        },
        "netmhcpan_version_output": version_output.stdout.strip(),
        "output_sha256": sha256(args.out),
    }
    provenance_path = args.out.with_name("common-tcell-predictions-provenance.json")
    provenance_path.write_text(json.dumps(provenance, indent=2))
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
