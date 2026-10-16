"""Cohort registry and assay preparation for the foreignness-choice benchmark.

Which cohorts exist, which domain each belongs to, and which side of the
development/holdout line it falls on are all decided here, by rule, before any
foreignness score is computed on them. Inclusion is by stated criteria applied
to counts and titles only.

Domains
-------
``viral``       T-cell responses to pathogen-derived peptides.
``neoepitope``  T-cell responses to peptides carrying tumour mutations.

Roles
-----
``dev``      used freely to compare choices and to select a candidate.
``holdout``  never read by the development stage; evaluated once, for the single
             candidate selected per domain, after that choice is committed.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlencode

HLA_I = re.compile(r"HLA-[ABC]\*\d{2}:\d{2}$")
AA = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LENGTH, MAX_LENGTH = 8, 14
POSITIVE = frozenset((
    "Positive", "Positive-High", "Positive-Intermediate", "Positive-Low"))
# Rows from engineered TCR lines and clones test a receptor, not a donor's
# repertoire, so only direct ex-vivo and short-term restimulated rows count.
ALLOWED_CULTURE = ("Direct Ex Vivo", "Short Term Restimulated")
# Assays that measure peptide-MHC binding rather than a T-cell response.
BINDING_ONLY_METHODS = ("surface plasmon resonance (SPR)", "binding assay")
# A cohort enters the pooled model if, after preparation, it still holds both
# outcomes; it is reported on its own only with enough of each to say something.
MIN_POSITIVES = MIN_NEGATIVES = 5
MIN_POSITIVES_REPORTED = MIN_NEGATIVES_REPORTED = 15
HOLDOUT_FRACTION_NUMERATOR = 1   # a quarter of hashed cohorts are held out
HOLDOUT_FRACTION_DENOMINATOR = 4
HOLDOUT_SALT = "foreignness-choices-holdout-v1"

# PMID cohorts, reviewed by title only. ``None`` filters mean every method.
VIRAL_PMIDS = (
    "12393434", "15848276", "18753217", "22508927", "23152369", "23543059",
    "24755960", "26092843", "26107956", "26509579", "26983786", "29698478",
    "29856876", "30737354", "32269122", "32999467", "33128877", "33184509",
    "33427749", "33972535", "34171305", "34211021", "34728796", "34857854",
    "34968416", "35255101", "35636577", "36068240", "36254196", "36656713",
    "37117789", "37193826", "38655616", "38971664", "39864283", "39915487",
    "40342296", "41407706", "42190640", "35484232",
)
NEOEPITOPE_PMIDS = (
    "25837513", "27198675", "28678784", "29545564", "30744692", "31511069",
    "33038342", "33665637", "34244308", "35665491", "35727629", "37586771",
    "38244726", "39753970", "39840067", "40273911",
)
# Second sealed confirmation set (role ``confirm2``): tumour-mutation studies
# found by a title search for neoantigen-related terms after the first holdout
# was spent. Chosen by title and counts alone. Nothing in it has been loaded,
# scored or analysed. The exclusions are listed so that the choice can be audited.
CONFIRM2_NEOEPITOPE_PMIDS = (
    "11325844", "11391614", "15128789", "16247014", "22912393", "22926105",
    "23644516", "24323902", "24979674", "24987109", "26060019", "26389673",
    "26516200", "26681737", "26808500", "26901407", "26945006", "27312342",
    "27827318", "27869121", "27959684", "28031159", "28423700", "28619968",
    "29203539", "29296522", "29563135", "29731959", "29895573", "30256815",
    "30640174", "30714987", "30880120", "30906664", "31011208", "31069153",
    "31164343", "31221207", "31243155", "31266769", "31351799", "31685621",
    "31857430", "31921104", "32226429", "32793190", "32807968", "32831296",
    "33016924", "33257508", "33259803", "33262074", "33262196", "33329543",
    "33479501", "33689574", "33811047", "33879241", "33928024", "34185709",
    "34236583", "34266885", "34321276", "34518289", "34696226", "34771556",
    "34782741", "34813506", "34903219", "35122353", "35379963", "35410325",
    "35413271", "35413272", "35636066", "35648703", "35661819", "35970920",
    "36088370", "36593398", "36792124", "36831375", "36890230", "37076248",
    "37280206", "37359417", "37731488", "37756564", "37939617", "38241630",
    "38349410", "38416845", "38633261", "38740147", "38782542", "38832948",
    "39394480", "39559350", "39720721", "39725148", "39748060", "39832892",
    "39972144", "40205806", "40280722", "40443655", "40554954", "40645179",
    "40681176", "40687378", "41056506", "41136363", "41312368", "41528883",
    "41839756", "42458705",
)
# Title-matched publications deliberately left out, with the reason.
CONFIRM2_EXCLUDED_PMIDS = {
    **{p: "viral or viral-mutant antigen" for p in (
        "11024156", "12667209", "15591521", "24759101", "31736946", "33658348",
        "34222571", "32157447", "40078724", "41280906")},
    **{p: "not a mutated peptide" for p in ("30115740", "38920720")},
    **{p: "post-translational modification, not a sequence change" for p in (
        "28314751", "37353482")},
    **{p: "anchor- or residue-modified peptide, not a natural variant" for p in (
        "35959968", "35720289", "35704596")},
    **{p: "mixed or unclear antigen classes" for p in (
        "14973499", "31624377", "31427818", "30556813", "32582212", "35468205",
        "35484264")},
}

# Cohorts that are not a single publication, or that need a stated rule.
SPECIAL_COHORTS = {
    # Two papers on one yellow fever panel from one laboratory.
    "yellow_fever_17d": {
        "domain": "viral", "filters": [("pmid", "28386132"), ("pmid", "32983097")]},
    # pMHC multimer rows in this paper measure pMHC binding, not T cells.
    "38608022": {
        "domain": "viral", "filters": [("pmid", "38608022")],
        "assay_methods": ("biological activity",)},
    # Oseroff/Pasquetto/Sette direct IEDB submission: an ELISPOT screen of a
    # predicted-binder vaccinia library in Dryvax vaccinees. Many assays are
    # peptide pools, and restriction is "Not determined", so the HLA is the
    # allele that nominated the peptide.
    "vaccinia_dryvax": {
        "domain": "viral",
        "filters": [("submission", "1000279"), ("submission", "1000276")]},
}
FORCED_DEV = frozenset((
    "36254196", "35484232", "yellow_fever_17d", "38608022"))
# Largest binder-matched viral screen, and the standard neoepitope benchmark
# (the TESLA consortium paper, with its tested negatives). Designated before any
# foreignness score was computed on either.
FORCED_HOLDOUT = frozenset(("vaccinia_dryvax", "33038342"))
FORCED_CONFIRM2 = frozenset(CONFIRM2_NEOEPITOPE_PMIDS)


def cohort_specs() -> dict[str, dict]:
    specs = {}
    for pmid in VIRAL_PMIDS:
        specs[pmid] = {"domain": "viral", "filters": [("pmid", pmid)]}
    for pmid in NEOEPITOPE_PMIDS + CONFIRM2_NEOEPITOPE_PMIDS:
        specs[pmid] = {"domain": "neoepitope", "filters": [("pmid", pmid)]}
    for cohort, spec in SPECIAL_COHORTS.items():
        specs[cohort] = dict(spec)
    for cohort, spec in specs.items():
        spec["cohort"] = cohort
        spec["role"] = assign_role(cohort)
    return specs


def assign_role(cohort: str) -> str:
    if cohort in FORCED_CONFIRM2:
        return "confirm2"
    if cohort in FORCED_DEV:
        return "dev"
    if cohort in FORCED_HOLDOUT:
        return "holdout"
    digest = hashlib.sha256(f"{HOLDOUT_SALT}|{cohort}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % HOLDOUT_FRACTION_DENOMINATOR
    return "holdout" if bucket < HOLDOUT_FRACTION_NUMERATOR else "dev"


def fetch_cohort(spec: dict, cache) -> list[dict]:
    """Download every export page for a cohort through datacache."""
    rows: list[dict] = []
    for kind, value in spec["filters"]:
        key = "reference__pmid" if kind == "pmid" else "reference__submission_id"
        tag = f"{kind}-{value}" if kind == "submission" else value
        offset, page = 0, 0
        while True:
            params = {key: "eq." + value, "order": "assay_id",
                      "limit": 1000, "offset": offset}
            if kind == "submission":
                # A direct submission also holds class II and non-human rows.
                params.update({
                    "mhc_restriction__class": "eq.I",
                    "host__name": "eq.Homo sapiens (human)"})
            url = "https://query-api.iedb.org/tcell_export?" + urlencode(params)
            path = cache.fetch(
                url, filename=f"iedb-{tag}-page-{page:03d}.json", timeout=90,
                max_retries=2, max_bytes=80_000_000, record_provenance=True)
            batch = json.loads(Path(path).read_text())
            if not batch:
                break
            rows.extend(batch)
            if len(batch) < 1000:
                break
            offset += len(batch)
            page += 1
            if page > 200:
                raise RuntimeError(f"pagination limit exceeded for {value}")
    return rows


def exposure_of(row: dict) -> str:
    first = row.get("first_in_vivo_process__process_type")
    second = row.get("second_in_vivo_process__process_type")
    disease = (row.get("first_in_vivo_process__disease") or "").lower()
    if first == "Prophylactic vaccination":
        return "vaccinated"
    if second == "Prophylactic vaccination":
        return "infected_then_vaccinated"
    if first == "Administration in vivo":
        return "vaccinated"
    if first == "Occurrence of infectious disease":
        return "infected"
    if first == "No immunization":
        return "natural_exposure"
    if "cancer" in (first or "").lower() or any(
            word in disease for word in (
                "melanoma", "carcinoma", "cancer", "glioblastoma",
                "leukemia", "tumor", "neoplasia", "lymphoma")):
        return "cancer"
    return "other"


def _species(row: dict) -> str | None:
    for field in ("epitope__species", "assay_antigen__species",
                  "epitope__source_organism", "assay_antigen__source_organism"):
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def prepare_cohort(rows: list[dict], spec: dict) -> tuple[list[dict], Counter]:
    """Eligible exact-HLA class-I assays for one cohort, with an exclusion log."""
    excluded: Counter = Counter()
    result = []
    methods = spec.get("assay_methods")
    for row in rows:
        if methods is not None and row.get("assay__method") not in methods:
            excluded["readout_not_a_t_cell_response"] += 1
            continue
        measurement = row.get("assay__qualitative_measurement")
        if measurement in POSITIVE:
            outcome = "Positive"
        elif measurement == "Negative":
            outcome = "Negative"
        else:
            excluded["outcome_not_positive_or_negative"] += 1
            continue
        peptide = row.get("epitope__name")
        hla = row.get("mhc_restriction__name")
        if (row.get("host__name") != "Homo sapiens (human)"
                or row.get("mhc_restriction__class") != "I"
                or not isinstance(hla, str) or not HLA_I.fullmatch(hla)
                or not isinstance(peptide, str)
                or not MIN_LENGTH <= len(peptide) <= MAX_LENGTH
                or set(peptide) - AA):
            excluded["not_an_exact_hla_class_i_canonical_peptide"] += 1
            continue
        if row.get("assay__method") in BINDING_ONLY_METHODS:
            excluded["binding_only_readout"] += 1
            continue
        if row.get("effector_cell__culture_condition") not in ALLOWED_CULTURE:
            excluded["culture_not_direct_or_restimulated"] += 1
            continue
        species = _species(row)
        if spec["domain"] == "neoepitope":
            species = "tumour neoepitope"
        elif species is None:
            excluded["missing_source_species"] += 1
            continue
        tested = row.get("assay__number_of_subjects_tested")
        result.append({
            "cohort": spec["cohort"], "domain": spec["domain"],
            "role": spec["role"],
            "pmid": row.get("reference__pmid") or (
                "submission-" + str(row.get("reference__submission_id"))),
            "assay_id": row.get("assay_id"),
            "peptide": peptide, "hla": hla, "outcome": outcome,
            "qualitative_measurement": measurement,
            "exposure": exposure_of(row),
            "assay_method": row.get("assay__method"),
            "culture_condition": row.get("effector_cell__culture_condition"),
            "restriction_evidence": row.get("mhc_restriction__evidence_code"),
            "source_protein": row.get("epitope__source_molecule"),
            "source_organism": row.get("epitope__source_organism"),
            "source_species": species,
            "n_subjects_tested": int(tested) if isinstance(tested, int)
            and tested > 0 else None,
            "epitope_reference_name": row.get("epitope__reference_name"),
        })
    return result, excluded


def deduplicate(records: list[dict]) -> tuple[list[dict], Counter]:
    """One row per cohort/exposure/peptide/HLA.

    A pair is Positive if any assay of it is positive and Negative only if every
    assay is negative, which is how IEDB itself calls an epitope. Dropping pairs
    with mixed assays instead would discard exactly the peptides that responded
    in some donors and not others: in the neoantigen therapy cohort 39753970 it
    keeps 19 of 67 positive pairs. The number of assays behind a pair is kept,
    because more assays give more chances to score positive.
    """
    groups = defaultdict(list)
    for row in records:
        groups[(row["cohort"], row["exposure"], row["peptide"],
                row["hla"])].append(row)
    result, mixed = [], Counter()
    for key in sorted(groups):
        members = sorted(
            groups[key], key=lambda r: (r["pmid"], str(r["assay_id"])))
        outcomes = [r["outcome"] for r in members]
        n_positive = outcomes.count("Positive")
        row = dict(members[0])
        row["outcome"] = "Positive" if n_positive else "Negative"
        row["n_positive_assays"] = n_positive
        row["n_negative_assays"] = len(outcomes) - n_positive
        if n_positive and n_positive < len(outcomes):
            mixed[key[0]] += 1
        for field in ("pmid", "assay_id", "qualitative_measurement",
                      "assay_method", "culture_condition",
                      "restriction_evidence", "source_protein",
                      "source_organism", "source_species",
                      "epitope_reference_name"):
            row[field] = "|".join(sorted({
                str(r[field]) for r in members if r[field] is not None}))
        tested = [r["n_subjects_tested"] for r in members
                  if r["n_subjects_tested"] is not None]
        row["n_subjects_tested"] = max(tested) if tested else None
        row["n_assay_rows"] = len(members)
        result.append(row)
    return result, mixed


def eligibility_audit(rows: list[dict],
                      minimum: int | None = None) -> list[dict]:
    """Per-cohort counts after preparation, applying the inclusion thresholds.

    ``minimum`` overrides the pooled-inclusion threshold (default 5 of each
    outcome); the stricter threshold for reporting a cohort alone is unchanged.
    """
    minimum = MIN_POSITIVES if minimum is None else minimum
    by = defaultdict(list)
    for row in rows:
        by[row["cohort"]].append(row)
    audit = []
    for cohort, members in sorted(by.items()):
        positives = sum(r["outcome"] == "Positive" for r in members)
        negatives = len(members) - positives
        peptides = {r["peptide"] for r in members}
        included = positives >= minimum and negatives >= minimum
        reported = (positives >= MIN_POSITIVES_REPORTED
                    and negatives >= MIN_NEGATIVES_REPORTED)
        audit.append({
            "cohort": cohort, "domain": members[0]["domain"],
            "role": members[0]["role"], "n_assays": len(members),
            "n_positive": positives, "n_negative": negatives,
            "n_peptides": len(peptides),
            "n_alleles": len({r["hla"] for r in members}),
            "included": included,
            "reported_alone": reported,
            "reason": None if included else (
                f"fewer than {minimum} positives or "
                f"{minimum} negatives after preparation"),
        })
    return audit
