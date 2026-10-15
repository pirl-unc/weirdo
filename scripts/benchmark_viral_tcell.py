#!/usr/bin/env python3
"""Exploratory viral T-cell response enrichment against a small WEIRDO-style model.

This is an analysis script, not a released WEIRDO predictor. It deliberately
holds SARS/coronavirus-labelled SwissProt proteins out of training and reports
study-specific, assay-specific comparisons rather than pooling all IEDB rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from datacache import Cache
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier


ROOT = Path(__file__).resolve().parents[1]
PMIDS = ("36254196", "35484232", "33521695")
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
CATEGORIES = (
    "archaea", "bacteria", "fungi", "human", "invertebrates",
    "mammals", "plants", "rodents", "vertebrates", "viruses",
)
HLA_I = re.compile(r"HLA-[ABC]\*\d{2}:\d{2}$")
HELD_OUT_VIRUS = re.compile(r"SARS|COV|CORO", re.IGNORECASE)
MIN_LENGTH, MAX_LENGTH = 8, 14


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_iedb(pmid: str, cache: Cache) -> list[dict]:
    """Fetch one pinned-paper IEDB export in 1,000-record datacache pages."""
    rows: list[dict] = []
    for page in range(100):
        url = "https://query-api.iedb.org/tcell_export?" + urlencode({
            "reference__pmid": "eq." + pmid,
            "order": "assay_id", "limit": 1000, "offset": len(rows),
        })
        path = cache.fetch(
            url, filename=f"iedb-{pmid}-page-{page:03d}.json",
            timeout=45, max_retries=1, max_bytes=60_000_000,
            record_provenance=True,
        )
        batch = json.loads(Path(path).read_text())
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < 1000:
            break
    else:
        raise RuntimeError(f"IEDB pagination exceeded safety limit for {pmid}")
    return rows


def prepare_assays(rows: list[dict], pmid: str) -> list[dict]:
    """Retain exact-HLA class-I peptide assay outcomes and their provenance."""
    result = []
    for row in rows:
        peptide = row.get("epitope__name")
        hla = row.get("mhc_restriction__name")
        outcome = row.get("assay__qualitative_measurement")
        if (
            row.get("host__name") != "Homo sapiens (human)"
            or row.get("mhc_restriction__class") != "I"
            or not isinstance(hla, str) or not HLA_I.fullmatch(hla)
            or not isinstance(peptide, str)
            or not MIN_LENGTH <= len(peptide) <= MAX_LENGTH
            or any(aa not in AA_INDEX for aa in peptide)
            or outcome not in ("Positive", "Negative")
        ):
            continue
        first = row.get("first_in_vivo_process__process_type")
        second = row.get("second_in_vivo_process__process_type")
        if first == "Prophylactic vaccination":
            exposure = "vaccinated"
        elif second == "Prophylactic vaccination":
            exposure = "infected_then_vaccinated"
        elif first == "Occurrence of infectious disease":
            exposure = "infected"
        else:
            exposure = "other"
        result.append({
            "pmid": pmid,
            "assay_id": row.get("assay_id"),
            "peptide": peptide,
            "hla": hla,
            "outcome": outcome,
            "exposure": exposure,
            "assay_method": row.get("assay__method"),
            "culture_condition": row.get("effector_cell__culture_condition"),
            "restriction_evidence": row.get("mhc_restriction__evidence_code"),
            "source_protein": row.get("epitope__source_molecule"),
            "source_organism": row.get("epitope__source_organism"),
        })
    return result


def sample_protein_peptides(path: Path, per_category: int, seed: int):
    """Reservoir-sample one 8–14mer per source protein and organism category."""
    rng = np.random.default_rng(seed)
    reservoirs: dict[str, list[str]] = {cat: [] for cat in CATEGORIES}
    seen = Counter()
    held_out = Counter()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            cat = row["category"]
            protein = row["sequence"]
            if cat not in reservoirs or len(protein) < MIN_LENGTH:
                continue
            if cat == "viruses" and HELD_OUT_VIRUS.search(row["identifier"]):
                held_out["labelled_sars_or_coronavirus_proteins"] += 1
                continue
            length = int(rng.integers(MIN_LENGTH, MAX_LENGTH + 1))
            if len(protein) < length:
                continue
            start = int(rng.integers(0, len(protein) - length + 1))
            peptide = protein[start:start + length]
            if any(aa not in AA_INDEX for aa in peptide):
                continue
            seen[cat] += 1
            slot = reservoirs[cat]
            if len(slot) < per_category:
                slot.append(peptide)
            else:
                replace = int(rng.integers(0, seen[cat]))
                if replace < per_category:
                    slot[replace] = peptide
    peptides = []
    categories = []
    for cat, slot in reservoirs.items():
        peptides.extend(slot)
        categories.extend([cat] * len(slot))
    return peptides, categories, dict(seen), dict(held_out)


def vectorize(peptides: list[str]) -> np.ndarray:
    """Encode whole peptides, with an explicit central-residue TCR-view branch.

    The central-residue branch is a heuristic for class-I exposed positions;
    it does not claim actual TCR contacts for any particular peptide/HLA.
    """
    width = MAX_LENGTH * len(AA) + len(AA) + (MAX_LENGTH - MIN_LENGTH + 1)
    result = np.zeros((len(peptides), width), dtype=np.float32)
    for i, peptide in enumerate(peptides):
        for pos, aa in enumerate(peptide):
            result[i, pos * len(AA) + AA_INDEX[aa]] = 1
        central = peptide[2:-1]
        for aa in central:
            result[i, MAX_LENGTH * len(AA) + AA_INDEX[aa]] += 1 / len(central)
        result[i, MAX_LENGTH * len(AA) + len(AA) + len(peptide) - MIN_LENGTH] = 1
    return result


def fit_exploratory_model(peptides: list[str], categories: list[str], seed: int):
    """Keep the SwissProt category-supervised small-MLP training pattern."""
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16), alpha=0.01,
        batch_size=256, max_iter=35, early_stopping=True,
        n_iter_no_change=6, random_state=seed,
    )
    model.fit(vectorize(peptides), categories)
    return model


def foreignness_scores(model, peptides: list[str]) -> dict[str, float]:
    unique = list(dict.fromkeys(peptides))
    probs = model.predict_proba(vectorize(unique))
    index = {cat: i for i, cat in enumerate(model.classes_)}
    pathogen = probs[:, [index["bacteria"], index["viruses"]]].max(axis=1)
    self_prob = probs[:, [index["human"], index["mammals"], index["rodents"]]].max(axis=1)
    score = pathogen / np.maximum(pathogen + self_prob, 1e-12)
    return dict(zip(unique, map(float, score)))


def within_hla_auc(rows: list[dict]) -> tuple[float | None, int]:
    """Pair positive and negative assay scores only within the same HLA."""
    by_hla = defaultdict(list)
    for row in rows:
        by_hla[row["hla"]].append(row)
    wins = 0.0
    n_pairs = 0
    for members in by_hla.values():
        positive = [r["foreignness"] for r in members if r["outcome"] == "Positive"]
        negative = [r["foreignness"] for r in members if r["outcome"] == "Negative"]
        for pos in positive:
            for neg in negative:
                wins += float(pos > neg) + 0.5 * float(pos == neg)
                n_pairs += 1
    return (wins / n_pairs if n_pairs else None), n_pairs


def summarize(records: list[dict], scores: dict[str, float]):
    """Deduplicate assays by study/exposure/peptide/HLA, excluding conflicts."""
    groups = defaultdict(list)
    for row in records:
        key = (row["pmid"], row["exposure"], row["peptide"], row["hla"])
        groups[key].append(row)
    deduped = []
    conflicts = Counter()
    for key, members in groups.items():
        outcomes = {r["outcome"] for r in members}
        if len(outcomes) != 1:
            conflicts[(key[0], key[1])] += 1
            continue
        row = dict(members[0])
        row["n_assay_rows"] = len(members)
        row["foreignness"] = scores[row["peptide"]]
        deduped.append(row)
    results = []
    for pmid in PMIDS:
        for exposure in sorted({r["exposure"] for r in deduped if r["pmid"] == pmid}):
            subset = [r for r in deduped if r["pmid"] == pmid and r["exposure"] == exposure]
            pos = [r["foreignness"] for r in subset if r["outcome"] == "Positive"]
            neg = [r["foreignness"] for r in subset if r["outcome"] == "Negative"]
            auc = None
            if pos and neg:
                y = [int(r["outcome"] == "Positive") for r in subset]
                auc = float(roc_auc_score(y, [r["foreignness"] for r in subset]))
            matched_auc, n_hla_pairs = within_hla_auc(subset)
            results.append({
                "pmid": pmid, "exposure": exposure, "n_positive": len(pos),
                "n_negative": len(neg), "n_conflicted_groups": conflicts[(pmid, exposure)],
                "mean_positive": float(np.mean(pos)) if pos else None,
                "mean_negative": float(np.mean(neg)) if neg else None,
                "auc": auc,
                "within_hla_auc": matched_auc,
                "n_within_hla_positive_negative_pairs": n_hla_pairs,
            })
    return deduped, results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swissprot-sequences", type=Path,
                        default=ROOT / "data/swissprot-sequences.csv")
    parser.add_argument("--cache-root", type=Path,
                        default=ROOT / "data/viral-tcell-benchmark")
    parser.add_argument("--per-category", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_category < 20:
        parser.error("--per-category must be at least 20")
    cache = Cache("sources", cache_root=str(args.cache_root.resolve()))
    records = []
    raw_counts = {}
    for pmid in PMIDS:
        rows = fetch_iedb(pmid, cache)
        raw_counts[pmid] = len(rows)
        records.extend(prepare_assays(rows, pmid))
    peptides, categories, seen, held_out = sample_protein_peptides(
        args.swissprot_sequences, args.per_category, args.seed,
    )
    model = fit_exploratory_model(peptides, categories, args.seed)
    scores = foreignness_scores(model, [r["peptide"] for r in records])
    deduped, results = summarize(records, scores)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    with (args.cache_root / "scored-assays.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(deduped[0]))
        writer.writeheader()
        writer.writerows(deduped)
    report = {
        "status": "exploratory, unvalidated model; not a released predictor",
        "training_source": str(args.swissprot_sequences),
        "training_source_sha256": file_sha256(args.swissprot_sequences),
        "training_counts": dict(Counter(categories)),
        "eligible_proteins_by_category": seen,
        "held_out_virus_proteins": held_out,
        "model": "whole-peptide one-hot + central composition; MLPClassifier(32,16)",
        "model_validation_score": float(model.best_validation_score_),
        "n_epochs": int(model.n_iter_),
        "seed": args.seed,
        "raw_iedb_rows": raw_counts,
        "n_eligible_assay_rows": len(records),
        "n_deduplicated_assays": len(deduped),
        "results": results,
    }
    (args.cache_root / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
