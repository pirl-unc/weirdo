#!/usr/bin/env python3
"""Compare foreignness definitions on HLA-resolved viral T-cell assays.

The discovery half of this script is hypothesis generation, not a released
WEIRDO predictor. Foreignness models are trained only on SwissProt organism
labels; T-cell assay outcomes are used only for evaluation and for
cross-validated conditional response models. Coronavirus- and
flavivirus-labelled SwissProt proteins are held out of training.

The validation half tests one pre-registered score, direction, comparison and
caliper (see ``PRIMARY_VALIDATION_*`` below) in cohorts that contributed
nothing to the discovery analysis.
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from weirdo.blosum import blosum62_dict


ROOT = Path(__file__).resolve().parents[1]

# Cohorts, not raw PMIDs, are the unit of analysis: the two yellow fever
# papers share one peptide panel and one laboratory, so they are merged and
# deduplicated instead of being counted as independent replications.
COHORTS = (
    {
        "cohort": "36254196",
        "pmids": ("36254196",),
        "role": "discovery",
        "description": (
            "SARS-CoV-2 direct ex-vivo tetramer screen in convalescent donors"
        ),
        "training_holds_out_source_virus_family": True,
    },
    {
        "cohort": "35484232",
        "pmids": ("35484232",),
        "role": "discovery",
        "description": (
            "SARS-CoV-2 IFN-gamma responses after short-term restimulation in "
            "infected and mRNA-vaccinated donors"
        ),
        "training_holds_out_source_virus_family": True,
    },
    {
        "cohort": "33521695",
        "pmids": ("33521695",),
        "role": "discovery",
        "description": (
            "SARS-CoV-2 direct ex-vivo convalescent responses; positives only"
        ),
        "training_holds_out_source_virus_family": True,
    },
    {
        "cohort": "yellow_fever_17d",
        "pmids": ("28386132", "32983097"),
        "role": "validation",
        "description": (
            "Yellow fever 17D direct ex-vivo multimer screens of predicted "
            "binders in vaccinees; one shared panel from one laboratory"
        ),
        "training_holds_out_source_virus_family": True,
    },
    {
        "cohort": "multivirus_concise_set",
        "pmids": ("38608022",),
        "role": "validation",
        "description": (
            "Twelve-virus pMHC binding and reactivity panel in unimmunized "
            "healthy donors after short-term restimulation"
        ),
        # This paper reports two readouts per peptide/HLA pair: whether the
        # pMHC multimer stains (qualitative binding, positive for 928 of 929
        # pairs) and whether T cells activate. Only activation is a T-cell
        # response, so the binding rows are excluded instead of being
        # collapsed against it as contradictory outcomes.
        "assay_methods": ("biological activity",),
        "training_holds_out_source_virus_family": False,
    },
)
COHORT_BY_PMID = {
    pmid: cohort["cohort"] for cohort in COHORTS for pmid in cohort["pmids"]}
ASSAY_METHODS_BY_COHORT = {
    cohort["cohort"]: cohort.get("assay_methods") for cohort in COHORTS}
ROLE_BY_COHORT = {cohort["cohort"]: cohort["role"] for cohort in COHORTS}
DISCOVERY_COHORTS = tuple(
    c["cohort"] for c in COHORTS if c["role"] == "discovery")
VALIDATION_COHORTS = tuple(
    c["cohort"] for c in COHORTS if c["role"] == "validation")
PMIDS = tuple(COHORT_BY_PMID)

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
CATEGORIES = (
    "archaea", "bacteria", "fungi", "human", "invertebrates",
    "mammals", "plants", "rodents", "vertebrates", "viruses",
)
SELF_CATEGORIES = frozenset(("human", "mammals", "rodents"))
PATHOGEN_CATEGORIES = frozenset(("bacteria", "viruses"))
HLA_I = re.compile(r"HLA-[ABC]\*\d{2}:\d{2}$")
POSITIVE_MEASUREMENTS = frozenset((
    "Positive", "Positive-High", "Positive-Intermediate", "Positive-Low"))
# SwissProt coronavirus organism mnemonics include ``_CV...`` (for example,
# ``SPIKE_CVH22``) plus named aliases such as MERS, PEDV, FIPV, and IBV. The
# broad holdout prevents training on obvious coronavirus family members.
HELD_OUT_VIRUS_FAMILIES = {
    "coronavirus": re.compile(
        r"SARS|MERS|COV|CORO|_CV|PEDV|FIPV|IBV", re.IGNORECASE),
    # Flaviviridae mnemonics, so the yellow fever validation cohort has no
    # family-level relatives in the viral training sample. All of these are
    # polyprotein entries; ``_GBV`` and ``_LIV`` are anchored because the bare
    # strings also occur inside unrelated mnemonics.
    "flavivirus": re.compile(
        r"_YEFV|_DEN[1-4]|_DENV|_WNV|_ZIKV|_JAEV|_TBEV|_POWV|_MVEV|_USUV"
        r"|_ILHV|_ROCV|_KOKV|_KUNJ|_KFDV|_OHFV|_ALKV|_BANV|_WSLV|_LANV"
        r"|_LIV|_FLAV|_HCV|_GBV|_BVDV|_CSFV|_PEST", re.IGNORECASE),
}
MIN_LENGTH, MAX_LENGTH = 8, 14
PRESENTATION_CALIPER = 0.1  # MHCflurry percentile-rank points.
PRESENTATION_CALIPER_SENSITIVITY = (0.02, 0.05, 0.1, 0.25)

# Pre-registration. The discovery analysis committed in 3c40e4d nominated the
# anchor-masked nearest-human BLOSUM62 distance as the single least weak lead,
# with higher distance predicting a T-cell response, and fixed the 0.1
# percentile-point caliper before any validation cohort was fetched or scored.
# Nothing below may be re-chosen after the validation numbers are seen.
PRIMARY_VALIDATION_SCORE = "nearest_human_blosum62_tcr"
PRIMARY_VALIDATION_DIRECTION = "higher score predicts Positive"
PRIMARY_VALIDATION_COMPARISONS = ("within_hla_auc", "presentation_matched_auc")
# Reported alongside the primary test, not used for the pass/fail decision.
# Both also use human proteins only, so no viral training data can leak.
SECONDARY_VALIDATION_SCORES = (
    "human_tcr_3mer_surprisal", "human_tcr_4mer_surprisal")
VALIDATION_ALPHA = 0.05
N_PERMUTATIONS = 2000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_iedb(pmid: str, cache) -> list[dict]:
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
    cohort = COHORT_BY_PMID.get(pmid, pmid)
    allowed_methods = ASSAY_METHODS_BY_COHORT.get(cohort)
    for row in rows:
        if (allowed_methods is not None
                and row.get("assay__method") not in allowed_methods):
            continue
        peptide = row.get("epitope__name")
        hla = row.get("mhc_restriction__name")
        measurement = row.get("assay__qualitative_measurement")
        # IEDB grades tetramer and ICS positives; all grades are one Positive
        # class here. The discovery papers report only ungraded outcomes, so
        # this widening affects the validation cohorts alone.
        if measurement in POSITIVE_MEASUREMENTS:
            outcome = "Positive"
        elif measurement == "Negative":
            outcome = "Negative"
        else:
            continue
        if (
            row.get("host__name") != "Homo sapiens (human)"
            or row.get("mhc_restriction__class") != "I"
            or not isinstance(hla, str) or not HLA_I.fullmatch(hla)
            or not isinstance(peptide, str)
            or not MIN_LENGTH <= len(peptide) <= MAX_LENGTH
            or any(aa not in AA_INDEX for aa in peptide)
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
        elif first == "Administration in vivo":
            # Live attenuated vaccine administration in the yellow fever
            # cohort; the same stratum as its prophylactic-vaccination rows.
            exposure = "vaccinated"
        elif first == "No immunization":
            exposure = "natural_exposure"
        else:
            exposure = "other"
        result.append({
            "cohort": cohort,
            "role": ROLE_BY_COHORT.get(cohort, "discovery"),
            "pmid": pmid,
            "assay_id": row.get("assay_id"),
            "peptide": peptide,
            "hla": hla,
            "outcome": outcome,
            "qualitative_measurement": measurement,
            "exposure": exposure,
            "assay_method": row.get("assay__method"),
            "culture_condition": row.get("effector_cell__culture_condition"),
            "restriction_evidence": row.get("mhc_restriction__evidence_code"),
            "source_protein": row.get("epitope__source_molecule"),
            "source_organism": row.get("epitope__source_organism"),
        })
    return result


def deduplicate_assays(records: list[dict]) -> tuple[list[dict], Counter]:
    """Collapse repeat assays within a cohort and exclude conflicting outcomes."""
    groups = defaultdict(list)
    for row in records:
        key = (row["cohort"], row["exposure"], row["peptide"], row["hla"])
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
        row["contributing_pmids"] = "|".join(
            sorted({r["pmid"] for r in members}))
        deduped.append(row)
    return deduped, conflicts


def sample_protein_peptides(path: Path, per_category: int, seed: int):
    """Reservoir-sample one 8-14mer per source protein and category."""
    # Independent streams keep, for example, a broader viral exclusion from
    # changing which human proteins and windows enter the reference sample.
    rngs = {
        cat: np.random.default_rng(np.random.SeedSequence([seed, i]))
        for i, cat in enumerate(CATEGORIES)
    }
    reservoirs: dict[str, list[str]] = {cat: [] for cat in CATEGORIES}
    seen = Counter()
    held_out = Counter()
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            cat = row["category"]
            protein = row["sequence"]
            if cat not in reservoirs or len(protein) < MIN_LENGTH:
                continue
            if cat == "viruses":
                family = next((
                    name for name, pattern in HELD_OUT_VIRUS_FAMILIES.items()
                    if pattern.search(row["identifier"])), None)
                if family is not None:
                    held_out[f"{family}_mnemonic_proteins"] += 1
                    continue
            rng = rngs[cat]
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


def tcr_face(peptide: str) -> str:
    """Return a conservative central-face heuristic (P3 through P-1)."""
    return peptide[2:-1]


def vectorize(peptides: list[str], view: str = "whole") -> np.ndarray:
    """Encode a whole peptide or anchor-masked, length-normalized TCR face."""
    length_width = MAX_LENGTH - MIN_LENGTH + 1
    if view == "whole":
        width = MAX_LENGTH * len(AA) + length_width
        result = np.zeros((len(peptides), width), dtype=np.float32)
        for i, peptide in enumerate(peptides):
            for pos, aa in enumerate(peptide):
                result[i, pos * len(AA) + AA_INDEX[aa]] = 1
            result[i, MAX_LENGTH * len(AA) + len(peptide) - MIN_LENGTH] = 1
        return result
    if view != "tcr_face":
        raise ValueError(f"Unknown peptide view: {view}")

    # Normalize the P3..P-1 face into fixed slots so 8-14mers can share a
    # compact representation without encoding the dominant P2/P-omega anchors.
    slots = MAX_LENGTH - 3
    width = slots * len(AA) + len(AA) + length_width
    result = np.zeros((len(peptides), width), dtype=np.float32)
    for i, peptide in enumerate(peptides):
        face = tcr_face(peptide)
        for pos, aa in enumerate(face):
            slot = round(pos * (slots - 1) / max(len(face) - 1, 1))
            result[i, slot * len(AA) + AA_INDEX[aa]] = 1
            result[i, slots * len(AA) + AA_INDEX[aa]] += 1 / len(face)
        result[i, slots * len(AA) + len(AA) + len(peptide) - MIN_LENGTH] = 1
    return result


def _model(family: str, seed: int):
    if family == "logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, max_iter=500, random_state=seed),
        )
    if family == "mlp":
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(16, 8), alpha=0.05,
                batch_size=256, max_iter=45, early_stopping=True,
                n_iter_no_change=6, random_state=seed,
            ),
        )
    if family == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=128, max_depth=8, min_samples_leaf=20,
            max_features="sqrt", n_jobs=-1, random_state=seed,
        )
    raise ValueError(f"Unknown model family: {family}")


def _positive_probability(model, matrix: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    return model.predict_proba(matrix)[:, classes.index(1)]


def _binary_target(categories: np.ndarray, target: str):
    if target == "pathogen_vs_self":
        keep = np.array([
            cat in PATHOGEN_CATEGORIES or cat in SELF_CATEGORIES
            for cat in categories
        ])
        labels = np.array([int(cat in PATHOGEN_CATEGORIES) for cat in categories])
    elif target == "virus_vs_human":
        keep = np.isin(categories, ["viruses", "human"])
        labels = np.array([int(cat == "viruses") for cat in categories])
    else:
        raise ValueError(f"Unknown binary target: {target}")
    return keep, labels


def deterministic_human_reference(swissprot_path: Path) -> list[str]:
    """Choose one stable window per human protein and supported length."""
    result = []
    with swissprot_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["category"] != "human":
                continue
            protein = row["sequence"]
            for length in range(MIN_LENGTH, MAX_LENGTH + 1):
                if len(protein) < length:
                    continue
                digest = hashlib.sha256(
                    f"{row['identifier']}|{length}".encode()).digest()
                start = int.from_bytes(digest[:8], "big") % (
                    len(protein) - length + 1)
                peptide = protein[start:start + length]
                if all(aa in AA_INDEX for aa in peptide):
                    result.append(peptide)
    return result


def _nearest_human_blosum(
        human_reference_peptides: list[str],
        query_peptides: list[str]) -> np.ndarray:
    """Minimum BLOSUM62 distance from each TCR face to a human face.

    BLOSUM62 scores are small integers, so the float32 position sums below are
    exact and reproduce the equivalent pure-Python comparison value for value.
    """
    substitution = np.asarray(
        [[blosum62_dict[a][b] for b in AA] for a in AA], dtype=np.float32)
    reference_by_length = defaultdict(list)
    for peptide in human_reference_peptides:
        face = tcr_face(peptide)
        reference_by_length[len(face)].append([AA_INDEX[aa] for aa in face])
    encoded = {
        length: np.asarray(faces, dtype=np.int16)
        for length, faces in reference_by_length.items()
    }
    result = []
    for peptide in query_peptides:
        face = tcr_face(peptide)
        reference = encoded.get(len(face))
        if reference is None:
            raise RuntimeError(
                f"No human reference face of length {len(face)} for {peptide}")
        similarity = np.zeros(len(reference), dtype=np.float32)
        for position, aa in enumerate(face):
            similarity += substitution[AA_INDEX[aa]][reference[:, position]]
        self_similarity = sum(blosum62_dict[aa][aa] for aa in face)
        result.append((self_similarity - float(similarity.max())) / len(face))
    return np.asarray(result, dtype=float)


def _residue_log_odds(
        training_peptides: list[str], training_categories: list[str],
        query_peptides: list[str]) -> np.ndarray:
    """Mean virus-versus-human residue log odds over the TCR face."""
    counts = {cat: np.ones(len(AA), dtype=float) for cat in ("human", "viruses")}
    for peptide, category in zip(training_peptides, training_categories):
        if category in counts:
            for aa in tcr_face(peptide):
                counts[category][AA_INDEX[aa]] += 1
    frequencies = {cat: values / values.sum() for cat, values in counts.items()}
    log_ratio = np.log(frequencies["viruses"] / frequencies["human"])
    return np.asarray([
        np.mean([log_ratio[AA_INDEX[aa]] for aa in tcr_face(peptide)])
        for peptide in query_peptides
    ])


def human_tcr_kmer_surprisal(
        swissprot_path: Path, query_peptides: list[str],
        k_values: tuple[int, ...] = (3, 4)) -> dict[str, np.ndarray]:
    """Score TCR faces by smoothed k-mer surprisal in all human proteins."""
    counts = {k: Counter() for k in k_values}
    totals = Counter()
    canonical_amino_acids = frozenset(AA)
    with swissprot_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["category"] != "human":
                continue
            sequence = row["sequence"]
            is_canonical = set(sequence) <= canonical_amino_acids
            for k in k_values:
                n_kmers = max(len(sequence) - k + 1, 0)
                kmers = [
                    sequence[i:i + k]
                    for i in range(n_kmers)
                ]
                canonical = (
                    kmers if is_canonical else
                    [kmer for kmer in kmers
                     if set(kmer) <= canonical_amino_acids]
                )
                counts[k].update(canonical)
                totals[k] += len(canonical)

    result = {}
    pseudocount = 0.5
    for k in k_values:
        denominator = totals[k] + pseudocount * (len(AA) ** k)
        values = []
        for peptide in query_peptides:
            face = tcr_face(peptide)
            kmers = [face[i:i + k] for i in range(len(face) - k + 1)]
            values.append(float(np.mean([
                -np.log((counts[k][kmer] + pseudocount) / denominator)
                for kmer in kmers
            ])))
        result[f"human_tcr_{k}mer_surprisal"] = np.asarray(values)
    return result


def train_foreignness_models(
        peptides: list[str], categories: list[str], query_peptides: list[str],
        seed: int):
    """Train compact discriminative and reference-distance definitions."""
    peptides_array = np.asarray(peptides)
    categories_array = np.asarray(categories)
    all_indices = np.arange(len(peptides))
    train_indices, validation_indices = train_test_split(
        all_indices, test_size=0.2, stratify=categories_array,
        random_state=seed,
    )
    feature_cache = {}

    def features(view: str, which: str):
        key = (view, which)
        if key not in feature_cache:
            source = peptides if which == "training" else query_peptides
            feature_cache[key] = vectorize(source, view)
        return feature_cache[key]

    scores = {}
    metadata = []

    # Preserve the original multiclass training pattern as one comparison.
    multiclass = _model("mlp", seed)
    whole = features("whole", "training")
    multiclass.fit(whole[train_indices], categories_array[train_indices])
    validation_prediction = multiclass.predict(whole[validation_indices])
    metadata.append({
        "model": "multiclass_mlp_whole",
        "family": "MLPClassifier(16,8)",
        "target": "ten SwissProt organism categories",
        "view": "whole peptide including HLA anchors",
        "n_train": len(train_indices),
        "n_validation": len(validation_indices),
        "validation_metric": "accuracy",
        "validation_value": float(accuracy_score(
            categories_array[validation_indices], validation_prediction)),
    })
    probabilities = multiclass.predict_proba(features("whole", "query"))
    class_index = {cat: i for i, cat in enumerate(multiclass.classes_)}
    pathogen = probabilities[:, [
        class_index["bacteria"], class_index["viruses"]]].sum(axis=1)
    selflike = probabilities[:, [
        class_index["human"], class_index["mammals"],
        class_index["rodents"]]].sum(axis=1)
    human = probabilities[:, class_index["human"]]
    virus = probabilities[:, class_index["viruses"]]
    scores["multiclass_mlp_whole__pathogen_vs_self"] = (
        pathogen / np.maximum(pathogen + selflike, 1e-12))
    scores["multiclass_mlp_whole__one_minus_human"] = 1 - human
    scores["multiclass_mlp_whole__virus_vs_human"] = (
        virus / np.maximum(virus + human, 1e-12))

    specifications = (
        ("logistic_whole", "logistic", "whole", "pathogen_vs_self"),
        ("logistic_tcr", "logistic", "tcr_face", "pathogen_vs_self"),
        ("mlp_tcr", "mlp", "tcr_face", "pathogen_vs_self"),
        ("extra_trees_tcr", "extra_trees", "tcr_face", "pathogen_vs_self"),
        ("logistic_tcr_virus", "logistic", "tcr_face", "virus_vs_human"),
    )
    for name, family, view, target in specifications:
        keep, labels = _binary_target(categories_array, target)
        fit_indices = train_indices[keep[train_indices]]
        eval_indices = validation_indices[keep[validation_indices]]
        model = _model(family, seed)
        matrix = features(view, "training")
        model.fit(matrix[fit_indices], labels[fit_indices])
        validation_scores = _positive_probability(model, matrix[eval_indices])
        score_name = f"{name}__{target}"
        scores[score_name] = _positive_probability(
            model, features(view, "query"))
        metadata.append({
            "model": name,
            "family": family,
            "target": target,
            "view": view,
            "n_train": len(fit_indices),
            "n_validation": len(eval_indices),
            "validation_metric": "roc_auc",
            "validation_value": float(roc_auc_score(
                labels[eval_indices], validation_scores)),
        })

    scores["virus_human_residue_log_odds_tcr"] = _residue_log_odds(
        list(peptides_array[train_indices]),
        list(categories_array[train_indices]), query_peptides)
    return scores, metadata


def add_presentation_predictions(rows: list[dict]):
    """Add exact peptide/HLA MHCflurry scores, using no source flanks."""
    import mhcflurry
    from mhcflurry import Class1PresentationPredictor

    predictor = Class1PresentationPredictor.load()
    peptides = [row["peptide"] for row in rows]
    hlas = [row["hla"] for row in rows]
    predictions = predictor.predict(
        peptides=peptides,
        alleles={hla: [hla] for hla in sorted(set(hlas))},
        sample_names=hlas,
        include_affinity_percentile=True,
        verbose=0,
    ).sort_values("peptide_num")
    if len(predictions) != len(rows):
        raise RuntimeError("MHCflurry did not return one prediction per assay")
    columns = (
        "affinity", "affinity_percentile", "processing_score",
        "presentation_score", "presentation_percentile",
    )
    for row, (_, prediction) in zip(rows, predictions.iterrows()):
        for column in columns:
            row[f"mhcflurry_{column}"] = float(prediction[column])
    return {
        "mhcflurry_version": mhcflurry.__version__,
        "model_provenance": predictor.provenance_string,
        "uses_source_flanks": False,
        "explanation": (
            "Presentation combines exact-HLA affinity with peptide-only "
            "processing because reliable source-protein coordinates/flanks "
            "are unavailable in the selected IEDB export."
        ),
    }


def within_hla_auc(
        rows: list[dict], score_column: str = "foreignness",
        presentation_caliper: float | None = None,
        group_fields: tuple[str, ...] = ("hla",)) -> tuple[float | None, int]:
    """Pair outcomes within a stratum and optionally within presentation percentile."""
    by_stratum = defaultdict(list)
    for row in rows:
        by_stratum[tuple(row[field] for field in group_fields)].append(row)
    wins = 0.0
    n_pairs = 0
    for members in by_stratum.values():
        positive = [r for r in members if r["outcome"] == "Positive"]
        negative = [r for r in members if r["outcome"] == "Negative"]
        for pos in positive:
            for neg in negative:
                if presentation_caliper is not None and abs(
                    pos["mhcflurry_presentation_percentile"]
                    - neg["mhcflurry_presentation_percentile"]
                ) > presentation_caliper:
                    continue
                pos_score, neg_score = pos[score_column], neg[score_column]
                wins += float(pos_score > neg_score) + 0.5 * float(
                    pos_score == neg_score)
                n_pairs += 1
    return (wins / n_pairs if n_pairs else None), n_pairs


def _pair_matrices(
        members: list[dict], score_column: str, caliper: float | None):
    """Win and caliper-eligibility matrices over all ordered row pairs."""
    scores = np.asarray([r[score_column] for r in members], dtype=float)
    difference = scores[:, None] - scores[None, :]
    wins = np.where(difference > 0, 1.0, np.where(difference == 0, 0.5, 0.0))
    if caliper is None:
        eligible = np.ones_like(wins)
    else:
        percentile = np.asarray([
            r["mhcflurry_presentation_percentile"] for r in members],
            dtype=float)
        eligible = (
            np.abs(percentile[:, None] - percentile[None, :]) <= caliper
        ).astype(float)
    return wins * eligible, eligible


def within_hla_permutation_p_value(
        rows: list[dict], score_column: str = PRIMARY_VALIDATION_SCORE,
        presentation_caliper: float | None = None,
        n_permutations: int = N_PERMUTATIONS, seed: int = 0,
        chunk: int = 250, group_fields: tuple[str, ...] = ("hla",)):
    """One-sided p-value for the pooled stratified AUC under label permutation.

    Outcomes are shuffled within each stratum, which holds the per-stratum
    positive/negative counts fixed and lets presentation matching re-form under
    every permutation, exactly as it does for the observed labels.
    """
    strata = []
    by_stratum = defaultdict(list)
    for row in rows:
        by_stratum[tuple(row[field] for field in group_fields)].append(row)
    for members in by_stratum.values():
        labels = np.asarray(
            [r["outcome"] == "Positive" for r in members], dtype=float)
        if not labels.any() or labels.all():
            continue
        wins, eligible = _pair_matrices(
            members, score_column, presentation_caliper)
        strata.append((labels, wins, eligible))
    observed_wins = sum(
        float(labels @ wins @ (1 - labels)) for labels, wins, _ in strata)
    observed_pairs = sum(
        float(labels @ eligible @ (1 - labels))
        for labels, _, eligible in strata)
    if not observed_pairs:
        return None, None, 0
    observed = observed_wins / observed_pairs
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, n_permutations, chunk):
        size = min(chunk, n_permutations - start)
        permuted_wins = np.zeros(size)
        permuted_pairs = np.zeros(size)
        for labels, wins, eligible in strata:
            draws = np.stack([rng.permutation(labels) for _ in range(size)])
            complement = 1 - draws
            permuted_wins += np.einsum(
                "pi,pi->p", draws @ wins, complement)
            permuted_pairs += np.einsum(
                "pi,pi->p", draws @ eligible, complement)
        statistic = np.where(
            permuted_pairs > 0,
            permuted_wins / np.maximum(permuted_pairs, 1), -np.inf)
        extreme += int(np.sum(statistic >= observed - 1e-12))
    p_value = (1 + extreme) / (1 + n_permutations)
    return float(p_value), observed, int(observed_pairs)


def _subgroups(rows: list[dict], cohorts: tuple[str, ...]):
    for cohort in cohorts:
        exposures = sorted({
            r["exposure"] for r in rows if r["cohort"] == cohort})
        for exposure in exposures:
            yield cohort, exposure, [
                r for r in rows
                if r["cohort"] == cohort and r["exposure"] == exposure
            ]


def summarize_scores(
        rows: list[dict], score_names: list[str], conflicts: Counter,
        cohorts: tuple[str, ...], p_value_scores: tuple[str, ...] = (),
        n_permutations: int = N_PERMUTATIONS, seed: int = 0):
    """Report unadjusted, within-HLA, and presentation-matched enrichment."""
    results = []
    for cohort, exposure, subset in _subgroups(rows, cohorts):
        y = np.asarray([int(r["outcome"] == "Positive") for r in subset])
        n_positive = int(y.sum())
        n_negative = int(len(y) - y.sum())
        for score_name in score_names:
            values = np.asarray([r[score_name] for r in subset], dtype=float)
            auc = float(roc_auc_score(y, values)) if len(set(y)) == 2 else None
            within_auc, n_hla_pairs = within_hla_auc(subset, score_name)
            matched_auc, n_matched_pairs = within_hla_auc(
                subset, score_name, PRESENTATION_CALIPER)
            sensitivity = {}
            for caliper in PRESENTATION_CALIPER_SENSITIVITY:
                sensitivity_auc, sensitivity_pairs = within_hla_auc(
                    subset, score_name, caliper)
                sensitivity[str(caliper)] = {
                    "auc": sensitivity_auc,
                    "n_pairs": sensitivity_pairs,
                }
            # Pairing within HLA alone still compares peptides from different
            # viruses, so a per-virus composition difference could masquerade
            # as foreignness. This stratum adds the source organism.
            source_auc, n_source_pairs = within_hla_auc(
                subset, score_name, PRESENTATION_CALIPER,
                group_fields=("hla", "source_organism"))
            p_values = {
                "within_hla_p_value": None,
                "presentation_matched_p_value": None,
                "within_hla_and_source_organism_p_value": None,
            }
            if score_name in p_value_scores:
                p_values["within_hla_p_value"] = within_hla_permutation_p_value(
                    subset, score_name, None, n_permutations, seed)[0]
                p_values["presentation_matched_p_value"] = (
                    within_hla_permutation_p_value(
                        subset, score_name, PRESENTATION_CALIPER,
                        n_permutations, seed)[0])
                p_values["within_hla_and_source_organism_p_value"] = (
                    within_hla_permutation_p_value(
                        subset, score_name, PRESENTATION_CALIPER,
                        n_permutations, seed,
                        group_fields=("hla", "source_organism"))[0])
            results.append({
                "cohort": cohort,
                "role": ROLE_BY_COHORT.get(cohort, "discovery"),
                "pmids": sorted({r["pmid"] for r in subset}),
                "exposure": exposure,
                "score": score_name,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "n_conflicted_groups": conflicts[(cohort, exposure)],
                "mean_positive": float(values[y == 1].mean()) if n_positive else None,
                "mean_negative": float(values[y == 0].mean()) if n_negative else None,
                "auc": auc,
                "within_hla_auc": within_auc,
                "n_within_hla_pairs": n_hla_pairs,
                "presentation_matched_auc": matched_auc,
                "n_presentation_matched_pairs": n_matched_pairs,
                "within_hla_and_source_organism_matched_auc": source_auc,
                "n_within_hla_and_source_organism_pairs": n_source_pairs,
                "presentation_caliper_sensitivity": sensitivity,
                **p_values,
            })
    return results


def _oof_logistic_probabilities(
        matrix: np.ndarray, outcomes: np.ndarray, groups: np.ndarray,
        splits: list[tuple[np.ndarray, np.ndarray]], seed: int):
    result = np.full(len(outcomes), np.nan)
    last_feature_coefficients = []
    for train, test in splits:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.25, max_iter=500, random_state=seed),
        )
        model.fit(matrix[train], outcomes[train])
        result[test] = _positive_probability(model, matrix[test])
        last_feature_coefficients.append(float(model[-1].coef_[0, -1]))
    if np.isnan(result).any():
        raise RuntimeError("Cross-validation did not score every assay")
    return result, last_feature_coefficients


def conditional_response_models(
        rows: list[dict], score_names: list[str], seed: int,
        cohorts: tuple[str, ...]):
    """Test incremental signal over presentation + HLA with peptide-grouped CV."""
    results = []
    for cohort, exposure, subset in _subgroups(rows, cohorts):
        outcomes = np.asarray([
            int(r["outcome"] == "Positive") for r in subset], dtype=int)
        if outcomes.sum() < 10 or (1 - outcomes).sum() < 10:
            continue
        groups = np.asarray([r["peptide"] for r in subset])
        hlas = sorted({r["hla"] for r in subset})
        hla_index = {hla: i for i, hla in enumerate(hlas)}
        hla_matrix = np.zeros((len(subset), len(hlas)), dtype=float)
        for i, row in enumerate(subset):
            hla_matrix[i, hla_index[row["hla"]]] = 1
        presentation = np.asarray([
            r["mhcflurry_presentation_score"] for r in subset], dtype=float)
        presentation = np.log(
            np.clip(presentation, 1e-6, 1 - 1e-6)
            / np.clip(1 - presentation, 1e-6, 1 - 1e-6))[:, None]
        baseline_matrix = np.hstack([presentation, hla_matrix])
        splitter = StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=seed)
        splits = list(splitter.split(baseline_matrix, outcomes, groups))
        baseline_probability, _ = _oof_logistic_probabilities(
            baseline_matrix, outcomes, groups, splits, seed)
        baseline_auc = float(roc_auc_score(outcomes, baseline_probability))
        baseline_log_loss = float(log_loss(outcomes, baseline_probability))
        for score_name in score_names:
            foreignness = np.asarray([
                r[score_name] for r in subset], dtype=float)[:, None]
            probability, coefficients = _oof_logistic_probabilities(
                np.hstack([baseline_matrix, foreignness]), outcomes,
                groups, splits, seed)
            auc = float(roc_auc_score(outcomes, probability))
            loss = float(log_loss(outcomes, probability))
            results.append({
                "cohort": cohort,
                "role": ROLE_BY_COHORT.get(cohort, "discovery"),
                "exposure": exposure,
                "score": score_name,
                "n_assays": len(subset),
                "n_unique_peptides": len(set(groups)),
                "n_folds": len(splits),
                "baseline": "MHCflurry presentation score + HLA fixed effects",
                "baseline_auc": baseline_auc,
                "baseline_log_loss": baseline_log_loss,
                "augmented_auc": auc,
                "delta_auc": auc - baseline_auc,
                "augmented_log_loss": loss,
                "delta_log_loss": loss - baseline_log_loss,
                "mean_standardized_foreignness_coefficient": float(
                    np.mean(coefficients)),
                "n_positive_foreignness_coefficient_folds": int(
                    np.sum(np.asarray(coefficients) > 0)),
            })
    return results


def primary_validation_outcome(
        validation_rows: list[dict], n_permutations: int, seed: int) -> dict:
    """Evaluate the pre-registered score against the pre-registered rule."""
    cohorts = []
    for cohort in VALIDATION_COHORTS:
        subset = [r for r in validation_rows if r["cohort"] == cohort]
        if not subset:
            continue
        tests = {}
        for comparison, caliper in (
            ("within_hla_auc", None),
            ("presentation_matched_auc", PRESENTATION_CALIPER),
        ):
            p_value, auc, n_pairs = within_hla_permutation_p_value(
                subset, PRIMARY_VALIDATION_SCORE, caliper,
                n_permutations, seed)
            tests[comparison] = {
                "auc": auc,
                "n_pairs": n_pairs,
                "p_value": p_value,
                "passes": bool(
                    auc is not None and auc > 0.5
                    and p_value is not None and p_value < VALIDATION_ALPHA),
            }
        source_p, source_auc, source_pairs = within_hla_permutation_p_value(
            subset, PRIMARY_VALIDATION_SCORE, PRESENTATION_CALIPER,
            n_permutations, seed, group_fields=("hla", "source_organism"))
        cohorts.append({
            "cohort": cohort,
            # Post-hoc control, deliberately outside the decision rule.
            "sensitivity_within_hla_and_source_organism": {
                "auc": source_auc,
                "n_pairs": source_pairs,
                "p_value": source_p,
                "part_of_decision_rule": False,
            },
            "exposures": sorted({r["exposure"] for r in subset}),
            "n_assays": len(subset),
            "n_positive": sum(r["outcome"] == "Positive" for r in subset),
            "n_negative": sum(r["outcome"] == "Negative" for r in subset),
            "n_hla_alleles": len({r["hla"] for r in subset}),
            "tests": tests,
            "passes": all(test["passes"] for test in tests.values()),
        })
    return {
        "score": PRIMARY_VALIDATION_SCORE,
        "direction": PRIMARY_VALIDATION_DIRECTION,
        "cohorts": cohorts,
        "validated": bool(cohorts) and all(c["passes"] for c in cohorts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swissprot-sequences", type=Path,
                        default=ROOT / "data/swissprot-sequences.csv")
    parser.add_argument("--cache-root", type=Path,
                        default=ROOT / "data/viral-tcell-benchmark")
    parser.add_argument("--per-category", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=N_PERMUTATIONS,
                        help="label permutations behind each reported p-value")
    args = parser.parse_args()
    if args.per_category < 20:
        parser.error("--per-category must be at least 20")
    if args.permutations < 100:
        parser.error("--permutations must be at least 100")

    # datacache and mhcflurry are benchmark-only extras, imported where they
    # are used so the module stays importable with the core dependencies.
    from datacache import Cache

    cache = Cache("sources", cache_root=str(args.cache_root.resolve()))
    records = []
    raw_counts = {}
    for pmid in PMIDS:
        rows = fetch_iedb(pmid, cache)
        raw_counts[pmid] = len(rows)
        records.extend(prepare_assays(rows, pmid))
    deduped, conflicts = deduplicate_assays(records)

    # Validation stays out of sample: a validation peptide that any discovery
    # cohort already tested is dropped rather than scored twice.
    discovery_peptides = {
        row["peptide"] for row in deduped if row["role"] == "discovery"}
    reused = [
        row for row in deduped
        if row["role"] == "validation"
        and row["peptide"] in discovery_peptides
    ]
    deduped = [
        row for row in deduped
        if row["role"] == "discovery"
        or row["peptide"] not in discovery_peptides
    ]
    presentation_metadata = add_presentation_predictions(deduped)

    peptides, categories, seen, held_out = sample_protein_peptides(
        args.swissprot_sequences, args.per_category, args.seed)
    unique_assay_peptides = list(dict.fromkeys(r["peptide"] for r in deduped))
    assay_peptide_set = set(unique_assay_peptides)
    exact_overlap = [
        (peptide, category) for peptide, category in zip(peptides, categories)
        if peptide in assay_peptide_set
    ]
    if exact_overlap:
        retained = [
            (peptide, category)
            for peptide, category in zip(peptides, categories)
            if peptide not in assay_peptide_set
        ]
        peptides = [peptide for peptide, _ in retained]
        categories = [category for _, category in retained]
    score_arrays, model_metadata = train_foreignness_models(
        peptides, categories, unique_assay_peptides, args.seed)
    human_reference = deterministic_human_reference(args.swissprot_sequences)
    score_arrays["nearest_human_blosum62_tcr"] = _nearest_human_blosum(
        human_reference, unique_assay_peptides)
    score_arrays.update(human_tcr_kmer_surprisal(
        args.swissprot_sequences, unique_assay_peptides))
    peptide_index = {peptide: i for i, peptide in enumerate(unique_assay_peptides)}
    score_names = list(score_arrays)
    for row in deduped:
        i = peptide_index[row["peptide"]]
        for score_name, values in score_arrays.items():
            row[score_name] = float(values[i])

    # Presentation is a comparator, not a definition of foreignness.
    comparator_names = ["mhcflurry_presentation_score"]
    reported_names = comparator_names + score_names
    discovery_rows = [r for r in deduped if r["role"] == "discovery"]
    validation_rows = [r for r in deduped if r["role"] == "validation"]
    discovery_results = summarize_scores(
        discovery_rows, reported_names, conflicts, DISCOVERY_COHORTS)
    discovery_conditional = conditional_response_models(
        discovery_rows, score_names, args.seed, DISCOVERY_COHORTS)
    p_value_scores = (
        PRIMARY_VALIDATION_SCORE,) + SECONDARY_VALIDATION_SCORES
    validation_results = summarize_scores(
        validation_rows, reported_names, conflicts, VALIDATION_COHORTS,
        p_value_scores, args.permutations, args.seed)
    validation_conditional = conditional_response_models(
        validation_rows, score_names, args.seed, VALIDATION_COHORTS)
    validation_primary = primary_validation_outcome(
        validation_rows, args.permutations, args.seed)

    args.cache_root.mkdir(parents=True, exist_ok=True)
    deduped.sort(key=lambda row: (row["role"], row["cohort"], row["peptide"]))
    with (args.cache_root / "scored-assays.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(deduped[0]))
        writer.writeheader()
        writer.writerows(deduped)
    report = {
        "status": "exploratory hypothesis generation; not a released predictor",
        "training_source": str(args.swissprot_sequences),
        "training_source_sha256": file_sha256(args.swissprot_sequences),
        "training_counts": dict(Counter(categories)),
        "eligible_proteins_by_category": seen,
        "held_out_virus_proteins": held_out,
        "exact_assay_peptides_removed_from_training": len(exact_overlap),
        "cohorts": [dict(cohort) for cohort in COHORTS],
        "validation_rows_dropped_for_discovery_peptide_reuse": len(reused),
        "deterministic_human_reference_peptides": len(human_reference),
        "training_split": (
            "80/20 stratified peptide split; one sampled peptide per source "
            "protein; homologous proteins are not yet clustered"
        ),
        "tcr_face_definition": (
            "P3 through the penultimate residue, excluding the common class-I "
            "P2 and C-terminal anchors; length-normalized positional one-hot "
            "plus composition and peptide length"
        ),
        "presentation_conditioning": {
            **presentation_metadata,
            "matched_pair_caliper_percentile_points": PRESENTATION_CALIPER,
            "caliper_sensitivity_percentile_points": list(
                PRESENTATION_CALIPER_SENSITIVITY),
            "matched_pair_definition": (
                "positive/negative assay pairs with identical HLA and absolute "
                "MHCflurry presentation-percentile difference <= caliper"
            ),
            "conditional_model": (
                "five-fold peptide-grouped logistic regression; baseline is "
                "MHCflurry presentation score plus HLA fixed effects"
            ),
        },
        "models": model_metadata,
        "score_definitions": {
            "multiclass_mlp_whole__pathogen_vs_self": (
                "sum P(bacteria,viruses) divided by that sum plus "
                "P(human,mammals,rodents)"
            ),
            "multiclass_mlp_whole__one_minus_human": "1 - P(human)",
            "multiclass_mlp_whole__virus_vs_human": (
                "P(viruses) / [P(viruses) + P(human)]"
            ),
            "logistic_whole__pathogen_vs_self": (
                "binary pathogen-versus-selflike probability, whole peptide"
            ),
            "logistic_tcr__pathogen_vs_self": (
                "binary pathogen-versus-selflike probability, TCR face"
            ),
            "mlp_tcr__pathogen_vs_self": (
                "small nonlinear pathogen-versus-selflike probability, TCR face"
            ),
            "extra_trees_tcr__pathogen_vs_self": (
                "depth/leaf-constrained tree ensemble probability, TCR face"
            ),
            "logistic_tcr_virus__virus_vs_human": (
                "binary virus-versus-human probability, TCR face"
            ),
            "nearest_human_blosum62_tcr": (
                "minimum normalized BLOSUM62 distance to a length-matched "
                "TCR face; reference has one deterministic window of every "
                "supported length per human SwissProt protein"
            ),
            "virus_human_residue_log_odds_tcr": (
                "mean smoothed virus-versus-human amino-acid log odds over "
                "the TCR face"
            ),
            "human_tcr_3mer_surprisal": (
                "mean smoothed 3-mer surprisal of the TCR face under all "
                "human SwissProt proteins"
            ),
            "human_tcr_4mer_surprisal": (
                "mean smoothed 4-mer surprisal of the TCR face under all "
                "human SwissProt proteins"
            ),
            "mhcflurry_presentation_score": (
                "positive-control comparator; not foreignness"
            ),
        },
        "validation_prespecification": {
            "fixed_in_commit": "3c40e4d",
            "fixed_before": (
                "any validation cohort was fetched, scored, or inspected"
            ),
            "primary_score": PRIMARY_VALIDATION_SCORE,
            "primary_direction": PRIMARY_VALIDATION_DIRECTION,
            "primary_comparisons": list(PRIMARY_VALIDATION_COMPARISONS),
            "caliper_percentile_points": PRESENTATION_CALIPER,
            "alpha": VALIDATION_ALPHA,
            "n_permutations": args.permutations,
            "permutation_scheme": (
                "outcomes shuffled within HLA, which holds per-allele "
                "positive/negative counts fixed and re-forms presentation "
                "matching under every permutation"
            ),
            "secondary_scores": list(SECONDARY_VALIDATION_SCORES),
            "decision_rule": (
                "the primary score counts as validated only if, in every "
                "validation cohort, the within-HLA AUC and the "
                "presentation-matched AUC both exceed 0.5 with a one-sided "
                "permutation p-value below alpha"
            ),
            "secondary_role": (
                "reported for context; cannot change the pass/fail decision"
            ),
        },
        "validation_primary_outcome": validation_primary,
        "seed": args.seed,
        "raw_iedb_rows": raw_counts,
        "n_eligible_assay_rows": len(records),
        "n_deduplicated_assays": len(deduped),
        "discovery_results": discovery_results,
        "discovery_conditional_cv_results": discovery_conditional,
        "validation_results": validation_results,
        "validation_conditional_cv_results": validation_conditional,
    }
    (args.cache_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
