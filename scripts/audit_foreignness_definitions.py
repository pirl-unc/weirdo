#!/usr/bin/env python3
"""Compare human/mammal and viral/pathogen-like score definitions.

This is an exploratory transfer audit, not a new lockbox.  The response
coefficient is fit on broad development data and frozen before scoring the
registered validation table.  It exists to prevent changing the taxonomy
summary (for example, human versus mammals) from being confused with a new
T-cell model.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
TRANSFER = ROOT / "scripts" / "evaluate_tcell_transfer.py"

# A deliberately explicit, reviewable set rather than a post-hoc prevalence
# threshold.  Yellow fever is kept out because the validation cohort is a
# vaccine/travel challenge, not ordinary community exposure.
COMMON_EXPOSURE_VIRUSES = frozenset(
    {
        "Influenza A virus",
        "Influenza B virus",
        "Human betaherpesvirus 5",  # CMV
        "human gammaherpesvirus 4",  # EBV
        "Human alphaherpesvirus 1",  # HSV-1
        "Human alphaherpesvirus 2",  # HSV-2
        "Human alphaherpesvirus 3",  # VZV
        "human respiratory syncytial virus",
        "human metapneumovirus",
        "Severe acute respiratory syndrome coronavirus 2",
        "Alphapapillomavirus 7",
        "Measles morbillivirus",
        "Mumps orthorubulavirus",
        "Rotavirus alphagastroenteritidis",
        "Norovirus norwalkense",
    }
)


def _load_transfer():
    spec = importlib.util.spec_from_file_location("tcell_transfer_for_definition_audit", TRANSFER)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not import {TRANSFER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _filter(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[frame["outcome"].isin(("Positive", "Negative"))].copy()
    required = [
        "mhcflurry_presentation_score",
        "mhcflurry_affinity",
        "mhcflurry_processing_score",
    ]
    return result.loc[result[required].notna().all(axis=1)].reset_index(drop=True)


def _joint_filter(development: pd.DataFrame, validation: pd.DataFrame, transfer) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    combined = pd.concat((development, validation), ignore_index=True)
    components = transfer._joint_peptide_components(combined).to_numpy()
    validation_components = components[len(development) :]
    seen = set(validation_components)
    keep = ~np.isin(components[: len(development)], list(seen))
    return development.loc[keep].reset_index(drop=True), validation.reset_index(drop=True), validation_components


def _score_definitions(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    prefix = "esm_finetune__"
    p_virus = frame[f"{prefix}p_viruses"].to_numpy(float)
    p_mammal = frame[f"{prefix}p_mammals"].to_numpy(float)
    p_human = frame[f"{prefix}p_human"].to_numpy(float)
    return {
        "p_viruses": p_virus,
        "one_minus_mammals": 1.0 - p_mammal,
        "one_minus_human": 1.0 - p_human,
        "virus_vs_mammals": p_virus * (1.0 - p_mammal),
        "virus_vs_human": p_virus * (1.0 - p_human),
        "microbial_foreign_like": frame[f"{prefix}microbial_foreign_like"].to_numpy(float),
        "microbial_like": frame[f"{prefix}microbial_like"].to_numpy(float),
    }


def audit(args: argparse.Namespace) -> dict:
    transfer = _load_transfer()
    development = _filter(pd.read_csv(args.development))
    validation = _filter(pd.read_csv(args.validation))
    if "role" in validation:
        validation = validation.loc[validation["role"].eq("validation")].reset_index(drop=True)
    development, validation, validation_groups = _joint_filter(development, validation, transfer)
    train_y = (development["outcome"] == "Positive").to_numpy(int)
    test_y = (validation["outcome"] == "Positive").to_numpy(int)
    train_base, _ = transfer._continuous_mhc_features(development)
    test_base, _ = transfer._continuous_mhc_features(validation)
    baseline, _ = transfer._fit_predict(train_base, train_y, test_base)
    common_mask = validation["source_species"].isin(COMMON_EXPOSURE_VIRUSES).to_numpy()
    subsets = {"all_validation": np.ones(len(validation), dtype=bool), "common_exposure": common_mask}
    definitions = _score_definitions(pd.concat((development, validation), ignore_index=True))
    results = []
    for subset_name, subset_mask in subsets.items():
        if subset_mask.sum() == 0 or len(np.unique(test_y[subset_mask])) < 2:
            continue
        groups = validation_groups[subset_mask]
        for name, all_score in definitions.items():
            train_score = all_score[: len(development)]
            test_score = all_score[len(development) :]
            augmented, coefficient = transfer._fit_predict(
                np.column_stack((train_base, train_score)),
                train_y,
                np.column_stack((test_base[subset_mask], test_score[subset_mask])),
            )
            y = test_y[subset_mask]
            base = baseline[subset_mask]
            raw = test_score[subset_mask]
            bootstrap = transfer._cluster_bootstrap(
                y,
                base,
                augmented,
                groups,
                draws=args.bootstrap_draws,
                seed=args.seed,
            )
            results.append(
                {
                    "subset": subset_name,
                    "score": name,
                    "n_validation": int(len(y)),
                    "positive": int(y.sum()),
                    "raw_score_auc": float(roc_auc_score(y, raw)),
                    "baseline_auc": float(roc_auc_score(y, base)),
                    "augmented_auc": float(roc_auc_score(y, augmented)),
                    "delta_auc": float(roc_auc_score(y, augmented) - roc_auc_score(y, base)),
                    "standardized_coefficient": coefficient,
                    "cluster_bootstrap": bootstrap,
                }
            )
    report = {
        "protocol": "broad-development response calibration; joint peptide-component exclusion; frozen registered validation scoring",
        "common_exposure_virus_set": sorted(COMMON_EXPOSURE_VIRUSES),
        "common_exposure_n": int(common_mask.sum()),
        "development_n": int(len(development)),
        "validation_n": int(len(validation)),
        "validation_positive": int(test_y.sum()),
        "foreignness_scores": "ESM fine-tuned taxonomy outputs already present in input tables",
        "pathogen_like_note": "No independent pathogen head is present in this virus/mammal taxonomy model; microbial_like is effectively virus-like here.",
        "results": results,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--development", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    audit(parser.parse_args())


if __name__ == "__main__":
    main()
