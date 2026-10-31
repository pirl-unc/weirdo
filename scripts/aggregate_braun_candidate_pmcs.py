#!/usr/bin/env python3
"""Aggregate mutation-aware candidate pMHC evidence by vaccine construct.

This is deliberately a *candidate-evidence* aggregation, not a count of
predicted pMHCs.  A candidate receives a fixed, label-free evidence weight

    sqrt(MHCflurry presentation * (1 - NetMHCpan BA rank / 100)).

MHCflurry's presentation predictor already includes its learned antigen-
processing component.  The main construct score is the sum of the three
strongest distinct peptide--HLA candidate weights.  Thus a construct can gain
from several independently strong sequences, while weak windows contribute
almost nothing and a fixed top-three cap prevents a longer construct from
winning merely because it generated more sliding windows.

The stricter secondary score additionally requires MHCflurry presentation
percentile <= 5 and NetMHCpan BA rank <= 2 before a candidate contributes.
Those are biological presentation/binding eligibility rules fixed without
Braun response labels; they are not fitted thresholds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.evaluate_taxonomy_response_experiments import group_bootstrap_delta


TOP_K = 3
PRESENTATION_PERCENTILE_GATE = 5.0
BA_RANK_GATE = 2.0
REQUIRED_COLUMNS = {
    "record_id",
    "patient",
    "mutation_class",
    "outcome",
    "peptide",
    "hla",
    "mhcflurry_2_3_0_presentation_score",
    "mhcflurry_2_3_0_presentation_percentile",
    "netmhcpan_ba_rank",
}
TAXONOMY_WEIGHTS = {
    "one_minus_mammals": "broad9__esm_160k__one_minus_mammals",
    "microbial_like": "broad9__esm_160k__microbial_like",
    "viruses": "broad9__esm_160k__viruses",
}


def candidate_evidence(frame: pd.DataFrame) -> np.ndarray:
    """Return label-free joint pMHC evidence in [0, 1].

    MHCflurry presentation supplies affinity plus antigen-processing evidence;
    NetMHCpan BA rank supplies an independent binding view.  The geometric
    mean requires both to be high without fitting a response model.
    """
    presentation = frame["mhcflurry_2_3_0_presentation_score"].to_numpy(float)
    ba_rank = frame["netmhcpan_ba_rank"].to_numpy(float)
    if not np.isfinite(presentation).all() or ((presentation < 0) | (presentation > 1)).any():
        raise ValueError("MHCflurry presentation scores must be finite in [0, 1]")
    if not np.isfinite(ba_rank).all() or ((ba_rank < 0) | (ba_rank > 100)).any():
        raise ValueError("NetMHCpan BA ranks must be finite in [0, 100]")
    binding = 1.0 - ba_rank / 100.0
    return np.sqrt(presentation * binding)


def _top_k_sum(values: pd.Series, top_k: int = TOP_K) -> float:
    """Add the strongest fixed number of candidate contributions."""
    return float(values.nlargest(top_k).sum())


def _top_k_soft_or(values: pd.Series, top_k: int = TOP_K) -> float:
    """A saturating sensitivity analysis over the same top candidates."""
    selected = values.nlargest(top_k).to_numpy(float)
    return float(1.0 - np.prod(1.0 - selected))


def aggregate_constructs(candidates: pd.DataFrame, top_k: int = TOP_K) -> pd.DataFrame:
    """Produce one response-blind aggregate score per vaccine construct."""
    missing = sorted(REQUIRED_COLUMNS - set(candidates.columns))
    if missing:
        raise ValueError(f"Scored candidates are missing columns: {missing}")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    rows = candidates.copy()
    rows["candidate_evidence"] = candidate_evidence(rows)
    rows["presentation_binding_eligible"] = (
        rows["mhcflurry_2_3_0_presentation_percentile"].to_numpy(float)
        <= PRESENTATION_PERCENTILE_GATE
    ) & (rows["netmhcpan_ba_rank"].to_numpy(float) <= BA_RANK_GATE)
    rows["gated_candidate_evidence"] = rows["candidate_evidence"].where(
        rows["presentation_binding_eligible"], 0.0
    )
    available_taxonomy = {
        name: column for name, column in TAXONOMY_WEIGHTS.items() if column in rows
    }
    for name, column in available_taxonomy.items():
        values = rows[column].to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Frozen taxonomy score {column} must be finite in [0, 1]")
        # Taxonomy changes only the contribution of pMHCs that already pass
        # the identical fixed presentation/binding gate.
        rows[f"gated_evidence_x_{name}"] = rows["gated_candidate_evidence"] * values

    # A repeated peptide/HLA pair can occur at more than one construct
    # position.  It represents one pMHC, not repeated independent evidence.
    rows = (
        rows.sort_values("candidate_evidence", ascending=False)
        .drop_duplicates(["record_id", "peptide", "hla"])
        .reset_index(drop=True)
    )
    identity = ["record_id", "patient", "mutation_class", "outcome"]
    if rows.groupby("record_id")[identity[1:]].nunique().gt(1).any().any():
        raise ValueError("Construct identity fields disagree within record_id")

    aggregates = []
    for record_id, part in rows.groupby("record_id", sort=True):
        first = part.iloc[0]
        evidence = part["candidate_evidence"]
        gated = part["gated_candidate_evidence"]
        result = {
                "record_id": record_id,
                "patient": first.patient,
                "mutation_class": first.mutation_class,
                "outcome": int(first.outcome),
                "n_distinct_candidate_pmhcs": len(part),
                "n_presentation_binding_eligible": int(part["presentation_binding_eligible"].sum()),
                "best_candidate_evidence": float(evidence.max()),
                "top3_sum_candidate_evidence": _top_k_sum(evidence, top_k),
                "top3_soft_or_candidate_evidence": _top_k_soft_or(evidence, top_k),
                "top3_sum_gated_candidate_evidence": _top_k_sum(gated, top_k),
        }
        for name in available_taxonomy:
            result[f"top3_sum_gated_evidence_x_{name}"] = _top_k_sum(
                part[f"gated_evidence_x_{name}"], top_k
            )
        aggregates.append(result)
    return pd.DataFrame(aggregates).sort_values("record_id").reset_index(drop=True)


def evaluate_aggregates(aggregates: pd.DataFrame, bootstrap_draws: int) -> pd.DataFrame:
    """Evaluate fixed aggregate scores; labels are used only in this function."""
    scores = [
        "best_candidate_evidence",
        "top3_sum_candidate_evidence",
        "top3_soft_or_candidate_evidence",
        "top3_sum_gated_candidate_evidence",
    ]
    scores.extend(sorted(column for column in aggregates if column.startswith("top3_sum_gated_evidence_x_")))
    subsets = {
        "all_braun_constructs": aggregates,
        "braun_frameshifts": aggregates.loc[aggregates["mutation_class"].eq("frameshift")],
    }
    rows = []
    for subset_name, part in subsets.items():
        y = part["outcome"].to_numpy(int)
        best_candidate = part["best_candidate_evidence"].to_numpy(float)
        gated_mhc = part["top3_sum_gated_candidate_evidence"].to_numpy(float)
        for index, score_name in enumerate(scores):
            score = part[score_name].to_numpy(float)
            low, high, valid = group_bootstrap_delta(
                y,
                gated_mhc,
                score,
                part["patient"].astype(str).to_numpy(),
                bootstrap_draws,
                20261030 + index,
            )
            rows.append(
                {
                    "endpoint": subset_name,
                    "score": score_name,
                    "n_constructs": len(part),
                    "n_positive": int(y.sum()),
                    "n_patients": int(part["patient"].nunique()),
                    "roc_auc": float(roc_auc_score(y, score)),
                    "average_precision": float(average_precision_score(y, score)),
                    "delta_auc_vs_best_candidate": float(
                        roc_auc_score(y, score) - roc_auc_score(y, best_candidate)
                    ),
                    "delta_auc_vs_gated_mhc": float(
                        roc_auc_score(y, score) - roc_auc_score(y, gated_mhc)
                    ),
                    "delta_auc_vs_gated_mhc_low": low,
                    "delta_auc_vs_gated_mhc_high": high,
                    "gated_mhc_bootstrap_valid": valid,
                }
            )
    return pd.DataFrame(rows)


def write_analysis(output: Path, metrics: pd.DataFrame) -> None:
    """Document fixed rules and descriptive held-out results."""
    result_rows = "\n".join(
        "| "
        + " | ".join(
            [
                row.endpoint.replace("_", " "),
                row.score.replace("_", " "),
                str(row.n_constructs),
                str(row.n_positive),
                f"{row.roc_auc:.4f}",
                f"{row.average_precision:.4f}",
                f"{row.delta_auc_vs_gated_mhc:+.4f} [{row.delta_auc_vs_gated_mhc_low:+.4f}, {row.delta_auc_vs_gated_mhc_high:+.4f}]",
            ]
        )
        + " |"
        for row in metrics.itertuples(index=False)
    )
    text = f"""# Mutation-aware Braun construct aggregation

## Fixed candidate evidence

For each reference-absent 8--11mer candidate paired with the construct's
annotated HLA, the label-free pMHC evidence is

$$e_i = \\sqrt{{p_{{\\mathrm{{MHCflurry\\ presentation}},i}}
(1 - r_{{\\mathrm{{NetMHCpan\\ BA}},i}}/100)}}.$$ 

MHCflurry presentation is already its presentation model, including its learned
processing component.  We therefore do not add the raw MHCflurry processing
score again and double-count that signal.

The prespecified main construct score is the **sum of the three largest distinct
peptide--HLA evidence values**.  It is an additive evidence score rather than a
number of passing pMHCs.  `best candidate` and `top3 soft or` are fixed controls.
The stricter `gated` sensitivity only lets a candidate contribute when
MHCflurry presentation percentile is at most {PRESENTATION_PERCENTILE_GATE:g} and
NetMHCpan BA rank is at most {BA_RANK_GATE:g}; neither threshold was selected on
Braun outcomes.

When present, each `gated evidence x ...` score uses the *identical* gate and
multiplies each eligible candidate's evidence by one frozen broad-nine ESM
taxonomy output before taking its top-three sum: `1 - P(mammals)`, independent
`P(viruses)`, or `microbial like = max[P(virus), P(bacterium), P(archaeon),
P(fungus), P(protist)]`.  These independent sigmoid outputs are scores, not a
probability partition; microbial-like is deliberately a conservative pathogen
proxy.  The taxonomy model and transforms were selected without Braun response
labels.

The candidate enumerator's reference-absence flag is still an operational proxy
for mutation overlap, not proof of it.  These construct-level outcomes are used
only below to evaluate precomputed scores, never to construct them.

## Descriptive evaluation

`ΔAUC` uses a paired patient-bootstrap interval relative to the pMHC-only
gated top-three sum—identical candidate rule and gate, with Weirdo omitted.
This small cohort is not used to select one aggregation rule.

| Endpoint | Score | Constructs | Positive | AUC | AP | ΔAUC vs gated pMHC-only, 95% patient-bootstrap interval |
|---|---|---:|---:|---:|---:|---:|
{result_rows}

## Outputs

- [`raw/construct-aggregates.csv.gz`](raw/construct-aggregates.csv.gz)
- [`tables/aggregate-metrics.csv`](tables/aggregate-metrics.csv)
- [`raw/provenance.json`](raw/provenance.json)
"""
    (output / "analysis.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        parser.error("--bootstrap-draws must be at least 100")

    candidates = pd.read_csv(args.candidates, low_memory=False)
    aggregates = aggregate_constructs(candidates)
    metrics = evaluate_aggregates(aggregates, args.bootstrap_draws)
    raw = args.output / "raw"
    tables = args.output / "tables"
    raw.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    aggregates.to_csv(raw / "construct-aggregates.csv.gz", index=False, compression="gzip")
    metrics.to_csv(tables / "aggregate-metrics.csv", index=False)
    (raw / "provenance.json").write_text(
        json.dumps(
            {
                "schema": "braun-mutation-aware-pmhc-aggregation-v1",
                "candidate_input": str(args.candidates),
                "top_k": TOP_K,
                "presentation_percentile_gate": PRESENTATION_PERCENTILE_GATE,
                "netmhcpan_ba_rank_gate": BA_RANK_GATE,
                "response_labels_used_to_construct_scores": False,
                "response_labels_used_only_for": "descriptive evaluation",
                "candidate_evidence": "sqrt(mhcflurry_presentation * (1 - netmhcpan_ba_rank / 100))",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_analysis(args.output, metrics)
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:+.4f}"))
    print(f"Wrote mutation-aware construct aggregation to {args.output}")


if __name__ == "__main__":
    main()
