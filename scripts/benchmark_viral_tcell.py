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
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from urllib.parse import urlencode

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from weirdo.amino_acid_alphabet import canonical_amino_acid_letters
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

# Sorted canonical letters; sorting makes the index order explicit here
# rather than inherited from the package's declaration order.
AA = "".join(sorted(canonical_amino_acid_letters))
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

# Pre-registration, with its real timeline (see ``VALIDATION_PROVENANCE``).
# Commit 3c40e4d (and the matching issue #8 comment) nominated the
# anchor-masked nearest-human BLOSUM62 distance as the single least weak lead
# before the first validation export was fetched, with the 0.1 percentile-point
# caliper and the within-HLA and presentation-matched comparisons. The alpha,
# the permutation test and the every-cohort conjunction were written later,
# before any validation score was computed but after cohort sizes, alleles and
# peptide overlap had been inspected. Nothing below may be re-chosen after the
# validation numbers are seen.
PRIMARY_VALIDATION_SCORE = "nearest_human_blosum62_tcr"
PRIMARY_VALIDATION_DIRECTION = "higher score predicts Positive"
VALIDATION_PROVENANCE = {
    "nominated_before_any_validation_data_were_fetched": {
        "commit": "3c40e4d",
        "committed": "2026-10-15T14:58:42-04:00",
        "first_validation_export_fetched": "2026-10-15T15:00:27-04:00",
        "fixed_there": [
            "the score and its TCR-face definition",
            "the 0.1 percentile-point presentation caliper",
            "within-HLA and presentation-matched AUC as the two comparisons",
            "higher distance predicting a response (the AUC > 0.5 convention)",
        ],
    },
    "written_after_cohort_structure_was_inspected_but_before_any_score": [
        "alpha = 0.05 and the one-sided label-permutation test",
        "the rule that both comparisons must pass in every validation cohort",
        "merging the two yellow fever papers into one cohort",
        "excluding binding-only readouts and mapping 'Administration in vivo' "
        "to the vaccinated stratum",
    ],
    "changed_after_validation_results_were_visible": [
        "permutation draws raised from 2,000 to 50,000 and per-stratum random "
        "streams introduced, because the 2,000-draw p-value near alpha "
        "depended on the permutation seed and on a batching parameter "
        "(Monte-Carlo standard error about 0.005); the estimand is unchanged "
        "but the procedure was not fixed in advance",
    ],
}
# name -> (presentation caliper, pairing strata). The decision loop reads the
# pre-registered names out of this registry, so the published rule and the
# executed rule cannot drift apart. The source-species row is a post-hoc
# control and is deliberately absent from PRIMARY_VALIDATION_COMPARISONS.
PERMUTATION_COMPARISONS = {
    "within_hla_auc": (None, ("hla",)),
    "presentation_matched_auc": (PRESENTATION_CALIPER, ("hla",)),
    "within_hla_and_source_species_matched_auc": (
        PRESENTATION_CALIPER, ("hla", "source_species")),
}
PRIMARY_VALIDATION_COMPARISONS = ("within_hla_auc", "presentation_matched_auc")
# Reported alongside the primary test, not used for the pass/fail decision.
# Both also use human proteins only, so no viral training data can leak.
SECONDARY_VALIDATION_SCORES = (
    "human_tcr_3mer_surprisal", "human_tcr_4mer_surprisal")
# Post-hoc: added after the validation numbers were seen, to ask whether the
# pre-registered score's weakness is the 0.18% self reference. They cannot
# change the decision, and their p-values are Bonferroni-adjusted below.
POSTHOC_VALIDATION_SCORES = (
    "nearest_human_full_blosum62_tcr", "soft_nearest_human_full_blosum62_tcr")
# Self-reference-free comparators that any foreignness score has to beat.
BASELINE_SCORES = (
    "tcr_face_hydrophobicity", "tcr_face_aromatic_fraction",
    "discovery_trained_composition")
VALIDATION_ALPHA = 0.05
# The pre-registered comparison sits near alpha, so the permutation count has
# to make Monte-Carlo error small relative to that distance: 50,000 draws give
# a standard error near 0.001 on a p-value of 0.05, against 0.005 at 2,000.
N_PERMUTATIONS = 50_000
N_BOOTSTRAP = 4000
CLUSTER_KMER = 7  # peptides sharing any 7-mer form one dependence cluster


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


def prepare_assays(
        rows: list[dict], pmid: str) -> tuple[list[dict], Counter]:
    """Retain exact-HLA class-I peptide assay outcomes and their provenance.

    Returns the eligible rows and a count of why the rest were excluded, so
    the funnel from raw export to analysis set stays auditable.
    """
    result = []
    excluded = Counter()
    cohort = COHORT_BY_PMID.get(pmid, pmid)
    allowed_methods = ASSAY_METHODS_BY_COHORT.get(cohort)
    for row in rows:
        if (allowed_methods is not None
                and row.get("assay__method") not in allowed_methods):
            excluded["readout_not_a_t_cell_response"] += 1
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
            excluded["outcome_not_positive_or_negative"] += 1
            continue
        if (
            row.get("host__name") != "Homo sapiens (human)"
            or row.get("mhc_restriction__class") != "I"
            or not isinstance(hla, str) or not HLA_I.fullmatch(hla)
            or not isinstance(peptide, str)
            or not MIN_LENGTH <= len(peptide) <= MAX_LENGTH
            or any(aa not in AA_INDEX for aa in peptide)
        ):
            excluded["not_an_exact_hla_class_i_canonical_peptide"] += 1
            continue
        species = row.get("epitope__species")
        if not isinstance(species, str) or not species.strip():
            # The source-species control stratifies on this field, so a row
            # without it would become a singleton stratum and silently leave
            # that comparison while staying in the other two.
            excluded["missing_source_species"] += 1
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
            "source_species": species.strip(),
        })
    return result, excluded


# Fields that identify or stratify an assay group: members must agree.
AGREEMENT_FIELDS = ("outcome", "source_species")
# Provenance that may legitimately differ between repeat assays of one pair.
# Every distinct value is kept, so nothing depends on which row arrived first.
MULTI_VALUE_FIELDS = (
    "pmid", "assay_id", "qualitative_measurement", "assay_method",
    "culture_condition", "restriction_evidence", "source_protein",
    "source_organism",
)


def deduplicate_assays(records: list[dict]) -> tuple[list[dict], Counter]:
    """Collapse repeat assays within a cohort and exclude conflicting ones."""
    groups = defaultdict(list)
    for row in records:
        key = (row["cohort"], row["exposure"], row["peptide"], row["hla"])
        groups[key].append(row)
    deduped = []
    conflicts = Counter()
    for key in sorted(groups):
        members = sorted(
            groups[key], key=lambda r: (r["pmid"], str(r["assay_id"])))
        disagreement = next((
            field for field in AGREEMENT_FIELDS
            if len({r[field] for r in members}) != 1), None)
        if disagreement is not None:
            conflicts[(key[0], key[1])] += 1
            continue
        row = dict(members[0])
        for field in MULTI_VALUE_FIELDS:
            row[field] = "|".join(sorted({
                str(r[field]) for r in members if r[field] is not None}))
        row["n_assay_rows"] = len(members)
        row["contributing_pmids"] = row["pmid"]
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


def human_proteome_stream(swissprot_path: Path) -> np.ndarray:
    """Encode every human protein as one uint8 stream, 255 between proteins.

    Residues outside the twenty canonical letters also map to 255, so no
    window can span a protein boundary or an ambiguous residue.
    """
    lookup = np.full(256, 255, dtype=np.uint8)
    for aa, index in AA_INDEX.items():
        lookup[ord(aa)] = index
    with swissprot_path.open(newline="") as stream:
        proteins = [
            row["sequence"] for row in csv.DictReader(stream)
            if row["category"] == "human"]
    joined = "*".join(proteins).encode("ascii")
    return lookup[np.frombuffer(joined, dtype=np.uint8)]


def full_human_blosum_scores(
        stream: np.ndarray, query_peptides: list[str], beta: float = 1.0,
        chunk: int = 50_000, query_batch: int = 256) -> dict[str, np.ndarray]:
    """Exact and soft BLOSUM62 distance from each TCR face to ALL human windows.

    The pre-registered nearest-human score searches one window per protein, about
    0.18% of human windows. This searches every window of the face's length, as
    an exact maximum and as a soft maximum ``log(sum(exp(beta*similarity)))/beta``
    that treats near-ties as competing self peptides instead of discarding all
    but one. ``beta`` is fixed a priori rather than tuned.
    """
    substitution = np.asarray(
        [[blosum62_dict[a][b] for b in AA] for a in AA], dtype=np.float32)
    n_aa = len(AA)
    faces = [tcr_face(peptide) for peptide in query_peptides]
    by_length = defaultdict(list)
    for i, face in enumerate(faces):
        by_length[len(face)].append(i)
    nearest = np.empty(len(faces))
    soft = np.empty(len(faces))
    for length, members in by_length.items():
        encoded = np.asarray(
            [[AA_INDEX[aa] for aa in faces[i]] for i in members])
        # Row r of ``scoring`` lays out, for each face position, the BLOSUM
        # row of the query residue; a one-hot window then selects one entry
        # per position, so one matmul sums the whole alignment.
        scoring = np.empty((len(members), length * n_aa), dtype=np.float32)
        for position in range(length):
            scoring[:, position * n_aa:(position + 1) * n_aa] = (
                substitution[encoded[:, position]])
        best = np.full(len(members), -np.inf, dtype=np.float32)
        log_sum = np.full(len(members), -np.inf)
        offsets = np.arange(length) * n_aa
        for start in range(0, len(stream) - length + 1, chunk):
            window = np.lib.stride_tricks.sliding_window_view(
                stream[start:start + chunk + length - 1], length)
            window = window[~(window == 255).any(axis=1)]
            if not len(window):
                continue
            one_hot = np.zeros((len(window), length * n_aa), dtype=np.float32)
            one_hot[np.arange(len(window))[:, None],
                    offsets[None, :] + window] = 1
            for first in range(0, len(members), query_batch):
                batch = slice(first, first + query_batch)
                similarity = scoring[batch] @ one_hot.T
                top = similarity.max(axis=1)
                best[batch] = np.maximum(best[batch], top)
                shifted = np.exp(beta * (similarity - top[:, None]))
                log_sum[batch] = np.logaddexp(
                    log_sum[batch],
                    beta * top.astype(float)
                    + np.log(shifted.sum(axis=1, dtype=np.float64)))
        for slot, i in enumerate(members):
            self_similarity = sum(blosum62_dict[aa][aa] for aa in faces[i])
            nearest[i] = (self_similarity - float(best[slot])) / length
            soft[i] = (self_similarity - log_sum[slot] / beta) / length
    return {
        "nearest_human_full_blosum62_tcr": nearest,
        "soft_nearest_human_full_blosum62_tcr": soft,
    }


# Kyte-Doolittle hydropathy.
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}


def composition_baselines(peptides: list[str]) -> dict[str, np.ndarray]:
    """Self-reference-free TCR-face composition scores.

    A foreignness score is only interesting if it beats properties that need no
    self proteome at all, since bulky hydrophobic and aromatic faces are
    associated with immunogenicity independently of any tolerance argument.
    These are transparent heuristics, not a reimplementation of any published
    immunogenicity model.
    """
    faces = [tcr_face(peptide) for peptide in peptides]
    return {
        "tcr_face_hydrophobicity": np.asarray([
            np.mean([HYDROPATHY[aa] for aa in face]) for face in faces]),
        "tcr_face_aromatic_fraction": np.asarray([
            np.mean([aa in "FWY" for aa in face]) for face in faces]),
    }


def _composition_features(peptides: list[str]) -> np.ndarray:
    result = np.zeros((len(peptides), len(AA) + 1))
    for i, peptide in enumerate(peptides):
        face = tcr_face(peptide)
        for aa in face:
            result[i, AA_INDEX[aa]] += 1 / len(face)
        result[i, -1] = len(peptide)
    return result


def add_discovery_trained_composition(rows: list[dict], seed: int) -> None:
    """Out-of-sample composition baseline trained on discovery T-cell outcomes.

    Discovery rows are scored by a model fitted to the other discovery cohorts
    and validation rows by a model fitted to all of them, so no row is scored by
    a model that saw its own outcome. Only cohorts with both outcomes train it;
    a positives-only cohort would teach it that everything responds.
    """
    trainable = {
        cohort for cohort in DISCOVERY_COHORTS
        if len({r["outcome"] for r in rows if r["cohort"] == cohort}) == 2}

    def fit(cohorts):
        subset = [r for r in rows if r["cohort"] in cohorts]
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=500, random_state=seed))
        model.fit(
            _composition_features([r["peptide"] for r in subset]),
            [int(r["outcome"] == "Positive") for r in subset])
        return model

    models = {None: fit(trainable)}
    for cohort in trainable:
        models[cohort] = fit(trainable - {cohort})
    for cohort in {r["cohort"] for r in rows}:
        members = [r for r in rows if r["cohort"] == cohort]
        model = models.get(cohort if cohort in trainable else None)
        probability = model.predict_proba(_composition_features(
            [r["peptide"] for r in members]))[:, 1]
        for row, value in zip(members, probability):
            row["discovery_trained_composition"] = float(value)


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
        rows: list[dict], score_column: str = PRIMARY_VALIDATION_SCORE,
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
    """Win matrix over ordered row pairs, plus caliper eligibility if any.

    ``None`` eligibility means every positive/negative pair counts, in which
    case the pair total is ``n_positive * n_negative`` and needs no matrix.
    """
    scores = np.asarray([r[score_column] for r in members], dtype=float)
    difference = scores[:, None] - scores[None, :]
    wins = np.where(difference > 0, 1.0, np.where(difference == 0, 0.5, 0.0))
    if caliper is None:
        return wins, None
    percentile = np.asarray([
        r["mhcflurry_presentation_percentile"] for r in members], dtype=float)
    eligible = (
        np.abs(percentile[:, None] - percentile[None, :]) <= caliper
    ).astype(float)
    return wins * eligible, eligible


def _build_strata(
        rows: list[dict], score_column: str, caliper: float | None,
        group_fields: tuple[str, ...]):
    """Sorted strata with both outcomes, each with its pair matrices.

    Shared by the permutation test and the cluster bootstrap so both resample
    exactly the same comparison, in the same canonical order.
    """
    by_stratum = defaultdict(list)
    for row in rows:
        by_stratum[tuple(row[field] for field in group_fields)].append(row)
    strata = []
    for key in sorted(by_stratum):
        members = sorted(
            by_stratum[key],
            key=lambda r: (r["peptide"], r["hla"], str(r["assay_id"])))
        labels = np.asarray(
            [r["outcome"] == "Positive" for r in members], dtype=float)
        if not labels.any() or labels.all():
            continue
        wins, eligible = _pair_matrices(members, score_column, caliper)
        strata.append((key, labels, wins, eligible, members))
    return strata


def _stratum_rng(key: tuple, seed: int) -> np.random.Generator:
    """Give each stratum its own stream, keyed by its identity.

    One shared generator would make the drawn permutations depend on how many
    were batched per call and on the order the strata happened to be built in,
    which would let a memory knob or an upstream row order move a
    pre-registered p-value.
    """
    digest = hashlib.sha256("|".join(map(str, key)).encode()).digest()
    return np.random.default_rng(
        np.random.SeedSequence([seed, int.from_bytes(digest[:8], "big")]))


def stratified_permutation_test(
        rows: list[dict], score_column: str = PRIMARY_VALIDATION_SCORE,
        presentation_caliper: float | None = None,
        n_permutations: int = N_PERMUTATIONS, seed: int = 0,
        chunk: int = 500, group_fields: tuple[str, ...] = ("hla",),
        alpha: float = VALIDATION_ALPHA) -> dict:
    """One-sided p-value for the pooled stratified AUC under label permutation.

    Outcomes are shuffled within each stratum, which holds the per-stratum
    positive/negative counts fixed and lets presentation matching re-form under
    every permutation, exactly as it does for the observed labels.

    Rows are sorted inside each stratum and each stratum draws from its own
    seeded stream, so the result depends only on the data, the score, the
    stratification, ``n_permutations`` and ``seed`` -- not on input row order
    or on ``chunk``, which only bounds memory. The returned Monte-Carlo
    standard error says whether a comparison against ``alpha`` is resolved at
    this number of permutations.
    """
    strata = [
        (key, labels, wins, eligible) for key, labels, wins, eligible, _
        in _build_strata(rows, score_column, presentation_caliper, group_fields)
    ]

    def statistic(draws: np.ndarray, labels, wins, eligible):
        """Pooled win and pair totals for a batch of label vectors."""
        complement = 1 - draws
        batch_wins = np.einsum("pi,pi->p", draws @ wins, complement)
        if eligible is None:
            batch_pairs = draws.sum(axis=1) * complement.sum(axis=1)
        else:
            batch_pairs = np.einsum("pi,pi->p", draws @ eligible, complement)
        return batch_wins, batch_pairs

    observed_wins = 0.0
    observed_pairs = 0.0
    for _, labels, wins, eligible in strata:
        batch_wins, batch_pairs = statistic(
            labels[None, :], labels, wins, eligible)
        observed_wins += float(batch_wins[0])
        observed_pairs += float(batch_pairs[0])
    if not observed_pairs:
        return {
            "auc": None, "n_pairs": 0, "p_value": None,
            "p_value_standard_error": None,
            "n_permutations": 0, "n_undefined_permutations": 0,
            "resolved_at_alpha": False,
        }
    observed = observed_wins / observed_pairs
    rngs = {key: _stratum_rng(key, seed) for key, _, _, _ in strata}
    extreme = 0
    undefined = 0
    for start in range(0, n_permutations, chunk):
        size = min(chunk, n_permutations - start)
        permuted_wins = np.zeros(size)
        permuted_pairs = np.zeros(size)
        for key, labels, wins, eligible in strata:
            rng = rngs[key]
            draws = np.stack([rng.permutation(labels) for _ in range(size)])
            batch_wins, batch_pairs = statistic(draws, labels, wins, eligible)
            permuted_wins += batch_wins
            permuted_pairs += batch_pairs
        # A permutation that leaves no eligible pair has no statistic at all;
        # scoring it as non-extreme while keeping it in the denominator would
        # bias the p-value downwards, so it leaves the reference distribution.
        defined = permuted_pairs > 0
        undefined += int(np.sum(~defined))
        ratio = permuted_wins[defined] / permuted_pairs[defined]
        extreme += int(np.sum(ratio >= observed - 1e-12))
    effective = n_permutations - undefined
    p_value = (1 + extreme) / (1 + effective)
    standard_error = float(
        np.sqrt(max(p_value * (1 - p_value), 0.0) / max(effective, 1)))
    return {
        "auc": observed,
        "n_pairs": int(observed_pairs),
        "p_value": float(p_value),
        "p_value_standard_error": standard_error,
        "n_permutations": int(effective),
        "n_undefined_permutations": undefined,
        "resolved_at_alpha": bool(
            abs(p_value - alpha) > 2 * standard_error),
    }


def peptide_clusters(peptides: list[str], k: int = CLUSTER_KMER) -> dict:
    """Single-linkage clusters of peptides that share any ``k``-mer.

    Panels tile source proteins with nested and overlapping peptides, so rows
    are not independent draws. Sharing a 7-mer links nested peptides (a
    substring of length >= 7 shares its 7-mers) and N/C-terminal extensions.
    """
    unique = sorted(set(peptides))
    parent = list(range(len(unique)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_owner: dict[str, int] = {}
    for i, peptide in enumerate(unique):
        for start in range(len(peptide) - k + 1):
            kmer = peptide[start:start + k]
            owner = first_owner.setdefault(kmer, i)
            if owner != i:
                parent[find(i)] = find(owner)
    # Name each cluster by its alphabetically first member: stable under any
    # input order and meaningful in the scored-assay file.
    members = defaultdict(list)
    for i, peptide in enumerate(unique):
        members[find(i)].append(peptide)
    return {
        peptide: min(group)
        for group in members.values() for peptide in group}


def cluster_bootstrap_auc(
        rows: list[dict], score_column: str = PRIMARY_VALIDATION_SCORE,
        presentation_caliper: float | None = None,
        n_bootstrap: int = N_BOOTSTRAP, seed: int = 0,
        group_fields: tuple[str, ...] = ("hla",),
        confidence: float = 0.95) -> dict:
    """Peptide-cluster bootstrap of the pooled stratified AUC.

    Whole clusters are resampled with replacement, and every row of a cluster
    takes that cluster's multiplicity in every stratum it appears in. That
    preserves the dependence the label permutation ignores. One multinomial
    draw covers all strata, so the result cannot depend on stratum order.
    """
    strata = _build_strata(rows, score_column, presentation_caliper,
                           group_fields)
    if not strata:
        return {"auc": None, "n_pairs": 0}
    cluster_ids = sorted({
        m["peptide_cluster"] for *_, members in strata for m in members})
    index = {cluster: i for i, cluster in enumerate(cluster_ids)}
    n_clusters = len(cluster_ids)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0xB007]))
    counts = rng.multinomial(
        n_clusters, np.full(n_clusters, 1 / n_clusters), size=n_bootstrap
    ).astype(float)

    def pooled(weights_by_stratum):
        wins_total, pairs_total = 0.0, 0.0
        for (_, labels, wins, eligible, _), weights in zip(
                strata, weights_by_stratum):
            a, b = weights * labels, weights * (1 - labels)
            wins_total = wins_total + np.einsum("bi,bi->b", a @ wins, b)
            pairs_total = pairs_total + (
                a.sum(axis=1) * b.sum(axis=1) if eligible is None
                else np.einsum("bi,bi->b", a @ eligible, b))
        return wins_total, pairs_total

    unit = [np.ones((1, len(labels))) for _, labels, *_ in strata]
    observed_wins, observed_pairs = pooled(unit)
    if not observed_pairs[0]:
        return {"auc": None, "n_pairs": 0}
    resampled = [
        counts[:, [index[m["peptide_cluster"]] for m in members]]
        for *_, members in strata]
    wins_total, pairs_total = pooled(resampled)
    defined = pairs_total > 0
    aucs = wins_total[defined] / pairs_total[defined]
    tail = (1 - confidence) / 2
    standard_error = float(aucs.std(ddof=1))
    return {
        "auc": float(observed_wins[0] / observed_pairs[0]),
        "n_pairs": int(observed_pairs[0]),
        "n_clusters": n_clusters,
        "n_bootstrap": int(defined.sum()),
        "standard_error": standard_error,
        "ci_low": float(np.quantile(aucs, tail)),
        "ci_high": float(np.quantile(aucs, 1 - tail)),
        # One-sided percentile-bootstrap p for AUC <= 0.5.
        "p_value": float((1 + np.sum(aucs <= 0.5)) / (1 + len(aucs))),
        "minimum_detectable_auc": minimum_detectable_auc(standard_error),
    }


def minimum_detectable_auc(
        standard_error: float, alpha: float = VALIDATION_ALPHA,
        power: float = 0.8) -> float:
    """Smallest true AUC a one-sided test would flag with the given power.

    Normal approximation around AUC = 0.5 using the cluster-bootstrap standard
    error, so it reflects the effective sample size after peptide dependence.
    """
    normal = NormalDist()
    return 0.5 + (normal.inv_cdf(1 - alpha) + normal.inv_cdf(power)
                  ) * standard_error


def _subgroups(rows: list[dict], cohorts: tuple[str, ...]):
    for cohort in cohorts:
        exposures = sorted({
            r["exposure"] for r in rows if r["cohort"] == cohort})
        for exposure in exposures:
            yield cohort, exposure, [
                r for r in rows
                if r["cohort"] == cohort and r["exposure"] == exposure
            ]


def cached_permutation_test():
    """Memoize permutation runs so one comparison is never computed twice."""
    store: dict = {}

    def run(rows: list[dict], score_column: str, comparison: str,
            stratum_key: tuple, n_permutations: int, seed: int) -> dict:
        key = (stratum_key, score_column, comparison, n_permutations, seed)
        if key not in store:
            caliper, group_fields = PERMUTATION_COMPARISONS[comparison]
            store[key] = stratified_permutation_test(
                rows, score_column, caliper, n_permutations, seed,
                group_fields=group_fields)
        return store[key]

    return run


def cached_cluster_bootstrap():
    """Memoize cluster-bootstrap runs, like :func:`cached_permutation_test`."""
    store: dict = {}

    def run(rows: list[dict], score_column: str, comparison: str,
            stratum_key: tuple, n_bootstrap: int, seed: int) -> dict:
        key = (stratum_key, score_column, comparison, n_bootstrap, seed)
        if key not in store:
            caliper, group_fields = PERMUTATION_COMPARISONS[comparison]
            store[key] = cluster_bootstrap_auc(
                rows, score_column, caliper, n_bootstrap, seed,
                group_fields=group_fields)
        return store[key]

    return run


def summarize_scores(
        rows: list[dict], score_names: list[str], conflicts: Counter,
        cohorts: tuple[str, ...], p_value_scores: tuple[str, ...] = (),
        n_permutations: int = N_PERMUTATIONS, seed: int = 0,
        permutation_test=None, bootstrap_scores: tuple[str, ...] = (),
        n_bootstrap: int = N_BOOTSTRAP, cluster_bootstrap=None):
    """Report unadjusted, within-HLA, and presentation-matched enrichment."""
    permutation_test = permutation_test or cached_permutation_test()
    cluster_bootstrap = cluster_bootstrap or cached_cluster_bootstrap()
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
                if caliper == PRESENTATION_CALIPER:
                    sensitivity_auc, sensitivity_pairs = (
                        matched_auc, n_matched_pairs)
                else:
                    sensitivity_auc, sensitivity_pairs = within_hla_auc(
                        subset, score_name, caliper)
                sensitivity[str(caliper)] = {
                    "auc": sensitivity_auc,
                    "n_pairs": sensitivity_pairs,
                }
            # Pairing within HLA alone still compares peptides from
            # different viruses, so a per-virus composition difference could
            # masquerade as foreignness. The third comparison adds the source
            # species.
            source_auc, n_source_pairs = within_hla_auc(
                subset, score_name, PRESENTATION_CALIPER,
                group_fields=("hla", "source_species"))
            permutation_tests = {}
            if score_name in p_value_scores:
                for comparison in PERMUTATION_COMPARISONS:
                    permutation_tests[comparison] = permutation_test(
                        subset, score_name, comparison,
                        (cohort, exposure), n_permutations, seed)
            bootstrap_tests = {}
            if score_name in bootstrap_scores:
                for comparison in PERMUTATION_COMPARISONS:
                    bootstrap_tests[comparison] = cluster_bootstrap(
                        subset, score_name, comparison,
                        (cohort, exposure), n_bootstrap, seed)
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
                "within_hla_and_source_species_matched_auc": source_auc,
                "n_within_hla_and_source_species_pairs": n_source_pairs,
                "presentation_caliper_sensitivity": sensitivity,
                "permutation_tests": permutation_tests,
                "cluster_bootstrap": bootstrap_tests,
            })
    return results


def _oof_logistic_probabilities(
        matrix: np.ndarray, outcomes: np.ndarray,
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
            baseline_matrix, outcomes, splits, seed)
        baseline_auc = float(roc_auc_score(outcomes, baseline_probability))
        baseline_log_loss = float(log_loss(outcomes, baseline_probability))
        for score_name in score_names:
            foreignness = np.asarray([
                r[score_name] for r in subset], dtype=float)[:, None]
            probability, coefficients = _oof_logistic_probabilities(
                np.hstack([baseline_matrix, foreignness]), outcomes,
                splits, seed)
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


def _pooled_dummies(values: list[str], minimum: int) -> np.ndarray:
    """Dummy-code a category, pooling levels with fewer than ``minimum`` rows."""
    counts = Counter(values)
    levels = sorted(
        {v for v in values if counts[v] >= minimum})
    labels = [v if counts[v] >= minimum else "__pooled__" for v in values]
    if "__pooled__" in labels:
        levels.append("__pooled__")
    # Drop the first level as the reference.
    return np.asarray(
        [[label == level for level in levels[1:]] for label in labels],
        dtype=float).reshape(len(values), max(len(levels) - 1, 0))


def _standardize(values: np.ndarray) -> np.ndarray:
    spread = values.std()
    return (values - values.mean()) / spread if spread > 0 else values * 0


def adjusted_enrichment_models(
        rows: list[dict], score_names: list[str],
        covariate_names: tuple[str, ...], cohorts: tuple[str, ...],
        minimum_level_rows: int = 10) -> list[dict]:
    """Cluster-robust logistic models keeping every row.

    The same-HLA-and-species pairing control discards about 90% of matched
    pairs and so cannot distinguish a confound from lost power. Here allele and
    source species enter as fixed effects alongside the MHCflurry presentation
    logit, optionally with self-reference-free composition baselines, and the
    foreignness score is tested for what it adds. Standard errors are clustered
    on peptide-overlap clusters, so nested and overlapping peptides are not
    counted as independent evidence.
    """
    import statsmodels.api as sm

    def fit_cluster_logit(design, outcome, groups):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.GLM(outcome, design, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": groups})
        if not (np.isfinite(fit.params).all() and np.isfinite(fit.bse).all()):
            raise ValueError("non-finite estimate")
        return fit

    def summarize(fit, position):
        coefficient = float(fit.params[position])
        error = float(fit.bse[position])
        z = coefficient / error
        return {
            "coefficient_per_sd": coefficient,
            "standard_error": error,
            "odds_ratio_per_sd": float(np.exp(coefficient)),
            "z": z,
            "one_sided_p_positive": float(1 - NormalDist().cdf(z)),
            "fit_ok": True,
        }

    results = []
    for cohort in cohorts:
        subset = [r for r in rows if r["cohort"] == cohort]
        if not subset:
            continue
        outcome = np.asarray(
            [r["outcome"] == "Positive" for r in subset], dtype=float)
        if outcome.sum() < 10 or (1 - outcome).sum() < 10:
            continue
        clusters = pd_factorize(
            np.asarray([r["peptide_cluster"] for r in subset]))
        presentation = np.clip(np.asarray(
            [r["mhcflurry_presentation_score"] for r in subset]),
            1e-6, 1 - 1e-6)
        fixed = [
            np.ones((len(subset), 1)),
            _standardize(np.log(presentation / (1 - presentation)))[:, None],
            _pooled_dummies([r["hla"] for r in subset], minimum_level_rows),
            _pooled_dummies(
                [r["source_species"] for r in subset], minimum_level_rows),
        ]
        covariates = [
            _standardize(np.asarray([r[name] for r in subset]))[:, None]
            for name in covariate_names]
        variants = {
            "presentation_hla_species": [],
            "plus_composition_baselines": covariates,
        }
        for variant, extra in variants.items():
            base = np.hstack(fixed + extra)
            context = {
                "cohort": cohort, "variant": variant, "n_assays": len(subset),
                "n_clusters": len(set(clusters)),
            }
            if extra:
                # What the self-reference-free baselines contribute on their
                # own, so a foreignness effect can be read against them.
                try:
                    fit = fit_cluster_logit(base, outcome, clusters)
                    first = base.shape[1] - len(covariate_names)
                    for offset, name in enumerate(covariate_names):
                        results.append({
                            **context, "score": f"__baseline__{name}",
                            "n_parameters": base.shape[1],
                            **summarize(fit, first + offset)})
                except Exception as error:
                    results.append({
                        **context, "score": "__baselines_only__",
                        "fit_ok": False, "failure": type(error).__name__})
            family = []
            for score_name in score_names:
                if score_name in covariate_names:
                    continue
                design = np.hstack([base, _standardize(np.asarray(
                    [r[score_name] for r in subset]))[:, None]])
                record = {
                    **context, "score": score_name,
                    "n_parameters": design.shape[1]}
                try:
                    record.update(summarize(
                        fit_cluster_logit(design, outcome, clusters),
                        design.shape[1] - 1))
                except Exception as error:  # separation, singular design
                    record.update({
                        "fit_ok": False, "failure": type(error).__name__})
                family.append(record)
            # The scores are many looks at one outcome: scale each nominal p
            # by the number of scores tried in this model family.
            fitted = [r for r in family if r["fit_ok"]]
            for record in fitted:
                record["p_bonferroni_across_scores"] = min(
                    1.0, record["one_sided_p_positive"] * len(fitted))
            results.extend(family)
    return results


def pd_factorize(labels: np.ndarray) -> np.ndarray:
    """Integer codes for cluster labels, as statsmodels expects."""
    return np.unique(labels, return_inverse=True)[1]


def annotate_multiplicity(results: list[dict], scores: tuple[str, ...]) -> int:
    """Add a Bonferroni-adjusted p to every non-primary permutation test.

    The pre-registered primary needs no correction. Everything reported
    alongside it is a family of looks at the same two cohorts, so its nominal
    p-values are scaled by the family size instead of being read one by one.
    """
    tests = [
        test for row in results if row["score"] in scores
        for test in row["permutation_tests"].values()
        if test["p_value"] is not None]
    for test in tests:
        test["p_bonferroni"] = min(1.0, test["p_value"] * len(tests))
    return len(tests)


# Where each score's information comes from: what it was fitted to.
SCORE_TRAINING_DATA = {
    "multiclass_mlp_whole__pathogen_vs_self": "SwissProt organism labels",
    "multiclass_mlp_whole__one_minus_human": "SwissProt organism labels",
    "multiclass_mlp_whole__virus_vs_human": "SwissProt organism labels",
    "logistic_whole__pathogen_vs_self": "SwissProt organism labels",
    "logistic_tcr__pathogen_vs_self": "SwissProt organism labels",
    "mlp_tcr__pathogen_vs_self": "SwissProt organism labels",
    "extra_trees_tcr__pathogen_vs_self": "SwissProt organism labels",
    "logistic_tcr_virus__virus_vs_human": "SwissProt organism labels",
    "virus_human_residue_log_odds_tcr": "SwissProt organism labels",
    "nearest_human_blosum62_tcr": "human proteins only (1 window per protein)",
    "human_tcr_3mer_surprisal": "human proteins only",
    "human_tcr_4mer_surprisal": "human proteins only",
    "nearest_human_full_blosum62_tcr": "human proteins only (all windows)",
    "soft_nearest_human_full_blosum62_tcr": "human proteins only (all windows)",
    "tcr_face_hydrophobicity": "none",
    "tcr_face_aromatic_fraction": "none",
    "discovery_trained_composition": (
        "T-cell outcomes of the other discovery cohorts (out of sample)"),
}


def primary_validation_outcome(
        validation_rows: list[dict], results: list[dict],
        n_permutations: int, seed: int, permutation_test,
        n_bootstrap: int = N_BOOTSTRAP, cluster_bootstrap=None) -> dict:
    """Evaluate the pre-registered score against the pre-registered rule."""
    cluster_bootstrap = cluster_bootstrap or cached_cluster_bootstrap()
    exposures_by_cohort = defaultdict(set)
    for row in validation_rows:
        exposures_by_cohort[row["cohort"]].add(row["exposure"])
    # Silently dropping a cohort would let an "every cohort" rule be declared
    # satisfied after testing a subset of it, so check them all before doing
    # any of the expensive work.
    absent = [c for c in VALIDATION_COHORTS if c not in exposures_by_cohort]
    if absent:
        raise RuntimeError(
            f"pre-registered validation cohort(s) {absent} contributed no "
            "eligible assays, so the conjunctive rule cannot be evaluated")
    cohorts = []
    for cohort in VALIDATION_COHORTS:
        subset = [r for r in validation_rows if r["cohort"] == cohort]
        exposures = sorted(exposures_by_cohort[cohort])
        tests = {}
        for comparison in (
            PRIMARY_VALIDATION_COMPARISONS
            + ("within_hla_and_source_species_matched_auc",)
        ):
            # One exposure stratum means the summary row already covers the
            # whole cohort, so the cached run is reused rather than repeated.
            stratum_key = (
                (cohort, exposures[0]) if len(exposures) == 1
                else (cohort, "all_exposures"))
            test = permutation_test(
                subset, PRIMARY_VALIDATION_SCORE, comparison, stratum_key,
                n_permutations, seed)
            decisive = comparison in PRIMARY_VALIDATION_COMPARISONS
            dependence = cluster_bootstrap(
                subset, PRIMARY_VALIDATION_SCORE, comparison, stratum_key,
                n_bootstrap, seed)
            tests[comparison] = {
                **test,
                # Sensitivity analyses outside the decision rule: they treat
                # overlapping peptides as one unit of evidence.
                "cluster_bootstrap": {
                    key: dependence.get(key) for key in (
                        "n_clusters", "standard_error", "ci_low", "ci_high",
                        "p_value", "minimum_detectable_auc")},
                "part_of_decision_rule": decisive,
                "passes": bool(
                    decisive and test["auc"] is not None
                    and test["auc"] > 0.5
                    and test["p_value"] is not None
                    and test["p_value"] < VALIDATION_ALPHA),
            }
        decisive_tests = [
            tests[comparison] for comparison in PRIMARY_VALIDATION_COMPARISONS]
        cohorts.append({
            "cohort": cohort,
            "exposures": exposures,
            "n_assays": len(subset),
            "n_positive": sum(r["outcome"] == "Positive" for r in subset),
            "n_negative": sum(r["outcome"] == "Negative" for r in subset),
            "n_hla_alleles": len({r["hla"] for r in subset}),
            "n_source_species": len({r["source_species"] for r in subset}),
            "tests": tests,
            "passes": all(test["passes"] for test in decisive_tests),
            "all_decisions_resolved_at_alpha": all(
                test["resolved_at_alpha"] for test in decisive_tests),
        })
    unresolved = [
        c["cohort"] for c in cohorts
        if not c["all_decisions_resolved_at_alpha"]]
    return {
        "score": PRIMARY_VALIDATION_SCORE,
        "direction": PRIMARY_VALIDATION_DIRECTION,
        "cohorts": cohorts,
        "validated": bool(cohorts) and all(c["passes"] for c in cohorts),
        "cohorts_with_a_p_value_within_monte_carlo_error_of_alpha": unresolved,
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
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP,
                        help="peptide-cluster bootstrap replicates")
    parser.add_argument("--permutation-seed", type=int, default=1,
                        help="seed for label permutations only, kept separate "
                             "from the training-sample seed so either can be "
                             "varied without disturbing the other")
    args = parser.parse_args()
    if args.per_category < 20:
        parser.error("--per-category must be at least 20")
    if args.permutations < 100:
        parser.error("--permutations must be at least 100")
    if not args.swissprot_sequences.is_file():
        # Checked before the IEDB fetch and the MHCflurry run, which together
        # take minutes, so a mistyped path fails in the first second.
        parser.error(f"--swissprot-sequences not found: {args.swissprot_sequences}")

    # datacache and mhcflurry are benchmark-only extras, imported where they
    # are used so the module stays importable with the core dependencies.
    from datacache import Cache

    cache = Cache("sources", cache_root=str(args.cache_root.resolve()))
    records = []
    raw_counts = {}
    excluded_rows = Counter()
    for pmid in PMIDS:
        rows = fetch_iedb(pmid, cache)
        raw_counts[pmid] = len(rows)
        eligible, excluded = prepare_assays(rows, pmid)
        records.extend(eligible)
        excluded_rows.update(excluded)
    deduped, conflicts = deduplicate_assays(records)

    # Validation stays out of sample: a validation peptide that any discovery
    # cohort already tested is dropped rather than scored twice.
    discovery_peptides = {
        row["peptide"] for row in deduped if row["role"] == "discovery"}
    reused = sum(
        1 for row in deduped
        if row["role"] == "validation"
        and row["peptide"] in discovery_peptides
    )
    deduped = [
        row for row in deduped
        if row["role"] == "discovery"
        or row["peptide"] not in discovery_peptides
    ]
    if not deduped:
        raise RuntimeError(
            "no eligible assays survived selection and deduplication; check "
            "the IEDB export fields before trusting an empty report")
    presentation_metadata = add_presentation_predictions(deduped)

    peptides, categories, seen, held_out = sample_protein_peptides(
        args.swissprot_sequences, args.per_category, args.seed)
    unique_assay_peptides = list(dict.fromkeys(r["peptide"] for r in deduped))
    assay_peptide_set = set(unique_assay_peptides)
    exact_overlap = [
        (peptide, category) for peptide, category in zip(peptides, categories)
        if peptide in assay_peptide_set
    ]
    retained = [
        (peptide, category) for peptide, category in zip(peptides, categories)
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
    score_arrays.update(full_human_blosum_scores(
        human_proteome_stream(args.swissprot_sequences),
        unique_assay_peptides))
    score_arrays.update(composition_baselines(unique_assay_peptides))
    peptide_index = {peptide: i for i, peptide in enumerate(unique_assay_peptides)}
    cluster_of = peptide_clusters(unique_assay_peptides)
    for row in deduped:
        i = peptide_index[row["peptide"]]
        row["peptide_cluster"] = cluster_of[row["peptide"]]
        for score_name, values in score_arrays.items():
            row[score_name] = float(values[i])
    add_discovery_trained_composition(deduped, args.seed)
    score_names = list(score_arrays) + ["discovery_trained_composition"]

    # Presentation is a comparator, not a definition of foreignness.
    comparator_names = ["mhcflurry_presentation_score"]
    reported_names = comparator_names + score_names
    discovery_rows = [r for r in deduped if r["role"] == "discovery"]
    validation_rows = [r for r in deduped if r["role"] == "validation"]
    discovery_results = summarize_scores(
        discovery_rows, reported_names, conflicts, DISCOVERY_COHORTS,
        bootstrap_scores=(PRIMARY_VALIDATION_SCORE,),
        n_bootstrap=args.bootstrap, seed=args.permutation_seed)
    discovery_conditional = conditional_response_models(
        discovery_rows, score_names, args.seed, DISCOVERY_COHORTS)
    p_value_scores = (
        (PRIMARY_VALIDATION_SCORE,) + SECONDARY_VALIDATION_SCORES
        + POSTHOC_VALIDATION_SCORES)
    bootstrap_scores = p_value_scores + BASELINE_SCORES
    permutation_test = cached_permutation_test()
    cluster_bootstrap = cached_cluster_bootstrap()
    validation_results = summarize_scores(
        validation_rows, reported_names, conflicts, VALIDATION_COHORTS,
        p_value_scores, args.permutations, args.permutation_seed,
        permutation_test, bootstrap_scores, args.bootstrap, cluster_bootstrap)
    n_multiplicity_family = annotate_multiplicity(
        validation_results,
        SECONDARY_VALIDATION_SCORES + POSTHOC_VALIDATION_SCORES)
    validation_conditional = conditional_response_models(
        validation_rows, score_names, args.seed, VALIDATION_COHORTS)
    validation_primary = primary_validation_outcome(
        validation_rows, validation_results, args.permutations,
        args.permutation_seed, permutation_test, args.bootstrap,
        cluster_bootstrap)
    adjusted_models = adjusted_enrichment_models(
        validation_rows, score_names, BASELINE_SCORES, VALIDATION_COHORTS)
    cluster_summary = {}
    for cohort in VALIDATION_COHORTS:
        cohort_peptides = {
            r["peptide"] for r in validation_rows if r["cohort"] == cohort}
        sizes = Counter(cluster_of[peptide] for peptide in cohort_peptides)
        cluster_summary[cohort] = {
            "n_peptides": len(cohort_peptides),
            "n_clusters": len(sizes),
            "largest_cluster_peptides": max(sizes.values()),
        }

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
        "validation_rows_dropped_for_discovery_peptide_reuse": reused,
        "excluded_iedb_rows_by_reason": dict(excluded_rows),
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
            "nearest_human_full_blosum62_tcr": (
                "post-hoc: exact minimum normalized BLOSUM62 distance from the "
                "TCR face to EVERY human SwissProt window of that length"
            ),
            "soft_nearest_human_full_blosum62_tcr": (
                "post-hoc: as above with the maximum replaced by "
                "log(sum(exp(similarity)))/1 over all human windows, so "
                "near-ties count as competing self peptides"
            ),
            "tcr_face_hydrophobicity": (
                "baseline: mean Kyte-Doolittle hydropathy of the TCR face; "
                "needs no self reference"
            ),
            "tcr_face_aromatic_fraction": (
                "baseline: fraction of F/W/Y on the TCR face; needs no self "
                "reference"
            ),
            "discovery_trained_composition": (
                "baseline: logistic regression on TCR-face composition and "
                "length fitted to the other discovery cohorts' T-cell "
                "outcomes, so every row is scored out of sample"
            ),
            "mhcflurry_presentation_score": (
                "positive-control comparator; not foreignness"
            ),
        },
        "validation_prespecification": {
            "provenance": VALIDATION_PROVENANCE,
            "primary_score": PRIMARY_VALIDATION_SCORE,
            "primary_direction": PRIMARY_VALIDATION_DIRECTION,
            "primary_comparisons": list(PRIMARY_VALIDATION_COMPARISONS),
            "caliper_percentile_points": PRESENTATION_CALIPER,
            "alpha": VALIDATION_ALPHA,
            "n_permutations": args.permutations,
            "permutation_seed": args.permutation_seed,
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
        "posthoc_analyses": {
            "status": (
                "added after the validation numbers were seen; none of it "
                "enters the pre-registered decision"
            ),
            "scores": list(POSTHOC_VALIDATION_SCORES),
            "baseline_scores": list(BASELINE_SCORES),
            "multiplicity_family_size": n_multiplicity_family,
            "multiplicity_correction": (
                "Bonferroni over the permutation tests of the secondary and "
                "post-hoc scores, stored as p_bonferroni"
            ),
            "peptide_cluster_definition": (
                f"single linkage on any shared {CLUSTER_KMER}-mer, within the "
                "analysed peptides"
            ),
            "validation_peptide_clusters": cluster_summary,
            "adjusted_models": adjusted_models,
        },
        "score_training_data": SCORE_TRAINING_DATA,
        "seed": args.seed,
        "permutation_seed": args.permutation_seed,
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
