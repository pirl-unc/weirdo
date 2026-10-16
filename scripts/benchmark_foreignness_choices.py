#!/usr/bin/env python3
"""Compare choices of foreignness under docs/foreignness-choices-protocol.md.

Two stages. ``--stage dev`` reads only development cohorts, scores every
registered candidate, and writes a ledger plus a selection per domain.
``--stage holdout`` refuses to run until ``SELECTED_CANDIDATES`` has been filled
in and committed, then evaluates exactly that candidate, once, on the holdout.
Candidates never see T-cell outcomes; only the evaluation does.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import warnings
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_viral_tcell import (  # noqa: E402
    _nearest_human_blosum,
    add_presentation_predictions,
    composition_baselines,
    deterministic_human_reference,
    peptide_clusters,
    vectorize,
)
from scripts.foreignness_cohorts import (  # noqa: E402
    cohort_specs,
    deduplicate,
    eligibility_audit,
    fetch_cohort,
    prepare_cohort,
)
from scripts.foreignness_scores import (  # noqa: E402
    AA,
    KmerIndex,
    PeptideSetIndex,
    ProteomeIndex,
    allele_anchor_pairs,
    nearest_self_distance,
    sample_windows,
)

CACHE_ROOT = ROOT / "data/viral-tcell-benchmark"
OUTPUT_ROOT = ROOT / "data/foreignness-choices"
SWISSPROT = ROOT / "data/swissprot-sequences.csv"
ATLAS_URL = "https://hla-ligand-atlas.org/rel/hla_2020.12.zip"
ATLAS_MEMBER = "hla_2020.12/HLA_aggregated.tsv"

BASELINES = ("tcr_face_hydrophobicity", "tcr_face_aromatic_fraction")
ALPHA_HOLDOUT = 0.025          # 0.05 split across the two domains
MIN_ODDS_RATIO_HOLDOUT = 1.10
DEV_Z_GATE = 1.645
DEV_SIGN_FRACTION = 0.60
MIN_COVERAGE = 0.90
RARE_LEVEL_ROWS = 15
SEED = 7

# Candidates in tie-break order (earlier = simpler). Registered in
# docs/foreignness-choices-protocol.md before any was evaluated.
CANDIDATES: dict[str, str] = {
    "nn_prereg": "incumbent",
    "nn_full": "reference",
    "soft_full": "reference",
    "nn_full_whole": "reference",
    "nn_allele_motif": "presentation-conditioned",
    "soft_allele_motif": "presentation-conditioned",
    "nn_atlas_all": "presented self",
    "nn_atlas_allele": "presented self",
    "soft_atlas_allele": "presented self",
    "nn_full_blosum50": "matrix",
    "nn_full_pmbec": "matrix",
    "nn_full_physchem": "matrix",
    "nn_full_hamming": "matrix",
    "kmer_absent_4": "rarity",
    "kmer_absent_5": "rarity",
    "kmer_absent_6": "rarity",
    "surprisal_3": "rarity",
    "surprisal_4": "rarity",
    "surprisal_5": "rarity",
    "cond_logistic": "learned, presentation-matched",
    "cond_mlp_small": "learned, presentation-matched",
    "cond_mlp_large": "learned, presentation-matched",
    "cond_trees": "learned, presentation-matched",
    "mismatch_whole": "mutation comparator",
    "mismatch_face": "mutation comparator",
    "dai": "mutation comparator",
}
# Filled in and committed between the stages. ``None`` blocks the holdout.
SELECTED_CANDIDATES: dict[str, str | None] = {
    "viral": None, "neoepitope": None}

COND_PANEL = (
    "HLA-A*01:01", "HLA-A*02:01", "HLA-A*03:01", "HLA-A*11:01",
    "HLA-A*24:02", "HLA-B*07:02", "HLA-B*08:01", "HLA-B*15:01",
    "HLA-B*27:05", "HLA-B*35:01", "HLA-B*40:01", "HLA-B*44:02")
NONSELF_CATEGORIES = ("bacteria", "archaea", "fungi", "plants", "invertebrates")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ----------------------------------------------------------------- data --
def assemble_rows(roles: tuple[str, ...], cache) -> tuple[list[dict], dict]:
    """Prepared, deduplicated assays for cohorts with the given roles."""
    records, excluded, raw = [], Counter(), {}
    for cohort, spec in cohort_specs().items():
        if spec["role"] not in roles:
            continue
        rows = fetch_cohort(spec, cache)
        raw[cohort] = len(rows)
        prepared, dropped = prepare_cohort(rows, spec)
        records.extend(prepared)
        excluded.update(dropped)
    deduped, mixed = deduplicate(records)
    audit = eligibility_audit(deduped)
    included = {a["cohort"] for a in audit if a["included"]}
    rows = [r for r in deduped if r["cohort"] in included]
    return rows, {
        "raw_rows": raw, "excluded_rows_by_reason": dict(excluded),
        "pairs_with_mixed_assays": dict(mixed), "audit": audit,
        "n_assays": len(rows)}


def _find(parent: dict, node):
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _gene_token(name: str | None) -> str | None:
    """Gene symbol from a neoepitope reference name such as ``GNL3L R>C``."""
    if name and " " in name.strip():
        return name.split()[0]
    return None


def add_clusters(rows: list[dict]) -> dict:
    """Dependence clusters used for standard errors.

    ``peptide_cluster`` links peptides sharing a 7-mer. ``cluster`` additionally
    merges peptides of the same cohort and source protein (or gene), since
    peptides from one protein share immunodominance.
    """
    overlap = peptide_clusters(sorted({r["peptide"] for r in rows}))
    parent = {c: c for c in set(overlap.values())}
    groups = defaultdict(set)
    for row in rows:
        row["peptide_cluster"] = overlap[row["peptide"]]
        protein = (row["source_protein"] or "").split("|")[0]
        if row["domain"] == "neoepitope":
            protein = _gene_token((row["epitope_reference_name"] or "").split("|")[0]) or ""
        if protein and protein.lower() not in ("unknown", "none"):
            groups[(row["cohort"], protein)].add(row["peptide_cluster"])
    for members in groups.values():
        members = sorted(members)
        for other in members[1:]:
            parent[_find(parent, other)] = _find(parent, members[0])
    for row in rows:
        row["cluster"] = _find(parent, row["peptide_cluster"])
    return {"n_overlap_clusters": len({r["peptide_cluster"] for r in rows}),
            "n_clusters": len({r["cluster"] for r in rows})}


def drop_holdout_reusing_dev(holdout: list[dict], dev: list[dict]) -> list[dict]:
    """Remove holdout rows whose peptide shares a 7-mer with a dev peptide."""
    kept, dropped = [], 0
    by_domain = {}
    for domain in {r["domain"] for r in holdout}:
        peptides = sorted({r["peptide"] for r in dev if r["domain"] == domain})
        by_domain[domain] = {
            p[i:i + 7] for p in peptides for i in range(len(p) - 6)}
    for row in holdout:
        shared = any(
            row["peptide"][i:i + 7] in by_domain.get(row["domain"], ())
            for i in range(len(row["peptide"]) - 6))
        if shared:
            dropped += 1
        else:
            kept.append(row)
    log(f"holdout rows dropped for sharing a 7-mer with dev peptides: {dropped}")
    return kept


# ------------------------------------------------------------ resources --
def load_atlas(cache) -> tuple[PeptideSetIndex, dict]:
    """HLA Ligand Atlas class-I self ligands, pooled and per predicted allele."""
    path = Path(cache.fetch(
        ATLAS_URL, filename="hla_ligand_atlas_2020.12.zip", timeout=300,
        max_retries=2, max_bytes=100_000_000, record_provenance=True))
    csv.field_size_limit(10 ** 9)
    allowed = set(AA)
    pooled, by_allele = set(), defaultdict(set)
    with zipfile.ZipFile(path) as archive, archive.open(ATLAS_MEMBER) as raw:
        lines = (line.decode() for line in raw)
        for row in csv.DictReader(lines, delimiter="\t"):
            if row["hla_class"] not in ("HLA-I", "HLA-I+II"):
                continue
            peptide = row["peptide_sequence"]
            if not 8 <= len(peptide) <= 14 or set(peptide) - allowed:
                continue
            pooled.add(peptide)
            for token in row["donor_alleles"].split(","):
                flag, _, allele = token.partition("/")
                if flag in ("s", "w") and allele[:1] in "ABC" and "-" not in allele:
                    by_allele["HLA-" + allele].add(peptide)
    groups = {None: sorted(pooled)}
    groups.update({a: sorted(p) for a, p in by_allele.items()})
    return PeptideSetIndex(groups), {
        "source": ATLAS_URL, "sha256": file_sha256(path),
        "license": "CC-BY 4.0 (Marcu et al., JITC 2021)",
        "n_class_i_ligands": len(pooled),
        "n_alleles_with_ligands": len(by_allele)}


class Resources:
    """Lazily built shared references and models."""

    def __init__(self, cache, swissprot: Path = SWISSPROT):
        self.cache = cache
        self.swissprot = swissprot
        self._proteome = None
        self._kmers = None
        self._atlas = None
        self._predictor = None
        self.atlas_metadata = None

    @property
    def proteome(self) -> ProteomeIndex:
        if self._proteome is None:
            log("indexing the human proteome")
            self._proteome = ProteomeIndex.from_swissprot(self.swissprot)
        return self._proteome

    @property
    def kmers(self) -> KmerIndex:
        if self._kmers is None:
            log("indexing human 3- to 6-mers")
            self._kmers = KmerIndex(self.proteome, (3, 4, 5, 6))
        return self._kmers

    @property
    def atlas(self) -> PeptideSetIndex:
        if self._atlas is None:
            log("loading the HLA Ligand Atlas")
            self._atlas, self.atlas_metadata = load_atlas(self.cache)
        return self._atlas

    @property
    def affinity(self):
        if self._predictor is None:
            from mhcflurry import Class1PresentationPredictor
            self._predictor = Class1PresentationPredictor.load()
        return self._predictor.affinity_predictor


# ----------------------------------------------------------- candidates --
def _to_rows(rows, keyed, key_of):
    return np.asarray([keyed.get(key_of(r), np.nan) for r in rows], dtype=float)


def train_conditional_classifiers(res: Resources, per_class: int = 600_000,
                                  seed: int = SEED) -> tuple[dict, dict]:
    """Human versus non-viral non-self peptides, matched on predicted presentation.

    Both classes are sampled from whole proteomes, scored against a panel of
    common alleles, and only binders (percentile rank <= 2) are kept. Within
    cells of (rank band, length) the larger class is subsampled to the smaller,
    so binding cannot separate the classes. No virus ever appears in training.
    """
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    with SWISSPROT.open(newline="") as stream:
        non_self = [r["sequence"] for r in csv.DictReader(stream)
                    if r["category"] in NONSELF_CATEGORIES]
    streams = {"self": res.proteome.stream,
               "nonself": ProteomeIndex(non_self).stream}
    lengths = np.array([8, 9, 10, 11])
    weights = np.array([0.10, 0.55, 0.25, 0.10])
    edges = np.array([0, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0])
    pools = {}
    for label, stream in streams.items():
        peptides, alleles = [], []
        for length, weight in zip(lengths, weights):
            n = int(per_class * weight)
            peptides.extend(sample_windows(stream, int(length), n, rng))
        alleles = rng.choice(COND_PANEL, size=len(peptides))
        percentile = np.empty(len(peptides))
        for allele in COND_PANEL:
            index = np.flatnonzero(alleles == allele)
            frame = res.affinity.predict_to_dataframe(
                [peptides[i] for i in index], allele=allele,
                include_percentile_ranks=True)
            percentile[index] = frame["prediction_percentile"].to_numpy()
        keep = percentile <= edges[-1]
        pools[label] = (
            np.asarray(peptides)[keep], percentile[keep])
        log(f"conditional classifier: {label} binders {keep.sum()}")
    chosen_peptides, labels = [], []
    for band in range(len(edges) - 1):
        for length in lengths:
            cells = {}
            for label, (peps, pct) in pools.items():
                in_cell = np.flatnonzero(
                    (pct > edges[band]) & (pct <= edges[band + 1])
                    & (np.char.str_len(peps) == length))
                cells[label] = in_cell
            n = min(len(v) for v in cells.values())
            for label, members in cells.items():
                picked = rng.choice(members, size=n, replace=False)
                chosen_peptides.extend(pools[label][0][picked])
                labels.extend([int(label == "nonself")] * n)
    chosen_peptides = list(chosen_peptides)
    labels = np.asarray(labels)
    matrix = vectorize(chosen_peptides, "tcr_face")
    train, test = train_test_split(
        np.arange(len(labels)), test_size=0.2, stratify=labels,
        random_state=seed)
    specs = {
        "cond_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(
                C=0.25, max_iter=500, random_state=seed)),
        "cond_mlp_small": make_pipeline(
            StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(16, 8), alpha=0.05, early_stopping=True,
                max_iter=200, random_state=seed)),
        "cond_mlp_large": make_pipeline(
            StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(128, 64), alpha=0.01, early_stopping=True,
                max_iter=200, random_state=seed)),
        "cond_trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=10,
            n_jobs=-1, random_state=seed),
    }
    metadata = {"n_matched_peptides": int(len(labels)),
                "validation_roc_auc": {}}
    for name, model in specs.items():
        model.fit(matrix[train], labels[train])
        metadata["validation_roc_auc"][name] = float(roc_auc_score(
            labels[test], model.predict_proba(matrix[test])[:, 1]))
        log(f"  {name}: validation AUC {metadata['validation_roc_auc'][name]:.3f}")
    return specs, metadata


def compute_candidates(rows: list[dict], res: Resources, wanted: list[str],
                       conditional=None) -> dict[str, np.ndarray]:
    """Score every wanted candidate for every row; NaN where undefined."""
    peptides = sorted({r["peptide"] for r in rows})
    pairs = sorted({(r["peptide"], r["hla"]) for r in rows})
    pair_peptides = [p for p, _ in pairs]
    pair_alleles = [a for _, a in pairs]
    by_peptide = {p: i for i, p in enumerate(peptides)}
    by_pair = {k: i for i, k in enumerate(pairs)}
    pep_scores: dict[str, np.ndarray] = {}
    pair_scores: dict[str, np.ndarray] = {}
    want = set(wanted)

    def timed(label, function):
        started = time.time()
        value = function()
        log(f"  {label}: {time.time() - started:.0f}s")
        return value

    if "nn_prereg" in want:
        reference = deterministic_human_reference(res.swissprot)
        pep_scores["nn_prereg"] = timed(
            "nn_prereg", lambda: _nearest_human_blosum(reference, peptides))
    if want & {"nn_full", "soft_full"}:
        out = timed("blosum62 face, all windows", lambda: nearest_self_distance(
            res.proteome, peptides, positions="face", matrix="blosum62",
            beta=1.0))
        pep_scores["nn_full"], pep_scores["soft_full"] = out["nearest"], out["soft"]
    if "nn_full_whole" in want:
        out = timed("blosum62 whole, all windows", lambda: nearest_self_distance(
            res.proteome, peptides, positions="whole", matrix="blosum62",
            beta=None))
        pep_scores["nn_full_whole"] = out["nearest"]
    for name, matrix in (("nn_full_blosum50", "blosum50"),
                         ("nn_full_pmbec", "pmbec"),
                         ("nn_full_physchem", "physchem")):
        if name in want:
            out = timed(name, lambda m=matrix: nearest_self_distance(
                res.proteome, peptides, positions="face", matrix=m, beta=None))
            pep_scores[name] = out["nearest"]
    if want & {"nn_full_hamming", "mismatch_face"}:
        out = timed("hamming face", lambda: nearest_self_distance(
            res.proteome, peptides, positions="face", matrix="hamming",
            beta=None))
        pep_scores["nn_full_hamming"] = out["nearest"]
        pep_scores["mismatch_face"] = out["nearest"] * np.array(
            [len(p) - 3 for p in peptides])
    if want & {"mismatch_whole", "dai"}:
        out = timed("hamming whole + windows", lambda: nearest_self_distance(
            res.proteome, peptides, positions="whole", matrix="hamming",
            beta=None, return_windows=True))
        pep_scores["mismatch_whole"] = out["nearest"] * np.array(
            [len(p) for p in peptides])
        if "dai" in want:
            wild_type = out["window"]
            scores = np.full(len(pairs), np.nan)
            for allele in sorted(set(pair_alleles)):
                index = [i for i, a in enumerate(pair_alleles) if a == allele]
                mutant = [pair_peptides[i] for i in index]
                wt = [wild_type[by_peptide[p]] for p in mutant]
                if any(w is None for w in wt):
                    continue
                aff_mut = np.asarray(res.affinity.predict(mutant, allele=allele))
                aff_wt = np.asarray(res.affinity.predict(wt, allele=allele))
                scores[index] = np.log(aff_wt) - np.log(aff_mut)
            pair_scores["dai"] = scores
    if want & {"kmer_absent_4", "kmer_absent_5", "kmer_absent_6"}:
        for k in (4, 5, 6):
            if f"kmer_absent_{k}" in want:
                pep_scores[f"kmer_absent_{k}"] = res.kmers.absent_fraction(
                    peptides, k)
    for k in (3, 4, 5):
        if f"surprisal_{k}" in want:
            pep_scores[f"surprisal_{k}"] = res.kmers.surprisal(peptides, k)
    if want & {"nn_allele_motif", "soft_allele_motif"}:
        alleles = sorted(set(pair_alleles))
        tables, readable = timed("anchor motifs", lambda: allele_anchor_pairs(
            alleles, res.affinity, res.proteome))
        out = timed("blosum62 face, allele-compatible windows",
                    lambda: nearest_self_distance(
                        res.proteome, pair_peptides, groups=pair_alleles,
                        positions="face", matrix="blosum62", beta=1.0,
                        allowed_pairs=tables))
        pair_scores["nn_allele_motif"] = out["nearest"]
        pair_scores["soft_allele_motif"] = out["soft"]
        res.anchor_summary = readable
    if "nn_atlas_all" in want:
        out = timed("atlas, all alleles", lambda: nearest_self_distance(
            res.atlas, peptides, positions="face", matrix="blosum62",
            beta=None))
        pep_scores["nn_atlas_all"] = out["nearest"]
    if want & {"nn_atlas_allele", "soft_atlas_allele"}:
        out = timed("atlas, allele-specific", lambda: nearest_self_distance(
            res.atlas, pair_peptides, groups=pair_alleles, positions="face",
            matrix="blosum62", beta=1.0))
        pair_scores["nn_atlas_allele"] = out["nearest"]
        pair_scores["soft_atlas_allele"] = out["soft"]
    if want & {n for n in CANDIDATES if n.startswith("cond_")}:
        models = conditional
        matrix = vectorize(peptides, "tcr_face")
        for name in sorted(want & set(models)):
            pep_scores[name] = models[name].predict_proba(matrix)[:, 1]

    result = {}
    for name in wanted:
        if name in pep_scores:
            result[name] = _to_rows(
                rows, dict(zip(peptides, pep_scores[name])),
                lambda r: r["peptide"])
        elif name in pair_scores:
            result[name] = _to_rows(
                rows, dict(zip(pairs, pair_scores[name])),
                lambda r: (r["peptide"], r["hla"]))
        else:
            raise KeyError(f"candidate {name!r} was not computed")
    return result


# ----------------------------------------------------------- statistics --
def rcs_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell restricted cubic spline: the linear term and k-2 curvature terms."""
    t, k = np.asarray(knots, dtype=float), len(knots)

    def cube(u):
        return np.maximum(u, 0.0) ** 3

    columns = [x]
    scale = (t[-1] - t[0]) ** 2
    for j in range(k - 2):
        term = (cube(x - t[j])
                - cube(x - t[k - 2]) * (t[k - 1] - t[j]) / (t[k - 1] - t[k - 2])
                + cube(x - t[k - 1]) * (t[k - 2] - t[j]) / (t[k - 1] - t[k - 2]))
        columns.append(term / scale)
    return np.column_stack(columns)


def pooled_dummies(values, minimum: int) -> np.ndarray:
    """Dummy columns for a category; levels under ``minimum`` rows are pooled."""
    values = [str(v) for v in values]
    counts = Counter(values)
    labels = [v if counts[v] >= minimum else "__other__" for v in values]
    levels = sorted(set(labels))
    if len(levels) < 2:
        return np.empty((len(values), 0))
    return np.asarray(
        [[label == level for level in levels[1:]] for label in labels],
        dtype=float)


def independent_columns(matrix: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """A maximal set of linearly independent columns, in original order.

    Constant columns, dummies duplicated by a single-allele cohort and levels
    emptied by a row subset all make a design rank deficient, and the
    cluster-robust covariance inverts the Hessian directly.
    """
    from scipy.linalg import qr
    if matrix.shape[1] == 0:
        return matrix
    _, r, pivots = qr(matrix, mode="economic", pivoting=True)
    diagonal = np.abs(np.diag(r))
    rank = int((diagonal > tolerance * max(diagonal[0], 1.0)).sum())
    return matrix[:, np.sort(pivots[:rank])]


def standardize_within(values: np.ndarray, strata: list) -> np.ndarray:
    """Z-score within each stratum, ignoring non-finite values."""
    result = np.full(len(values), np.nan)
    index = defaultdict(list)
    for i, stratum in enumerate(strata):
        index[stratum].append(i)
    for members in index.values():
        part = values[members]
        finite = np.isfinite(part)
        if finite.sum() < 2:
            result[members] = np.where(finite, 0.0, np.nan)
            continue
        spread = part[finite].std()
        result[members] = (part - part[finite].mean()) / spread if spread > 0 \
            else np.where(finite, 0.0, np.nan)
    return result


def nuisance_matrix(rows: list[dict], domain: str, stringent: bool) -> np.ndarray:
    """Everything except the score: strata, presentation, allele, species, ..."""
    strata = [f"{r['cohort']}|{r['exposure']}" for r in rows]
    presentation = np.clip(np.asarray(
        [r["mhcflurry_presentation_score"] for r in rows]), 1e-6, 1 - 1e-6)
    logit = np.log(presentation / (1 - presentation))
    knots = np.quantile(logit, [0.05, 0.35, 0.65, 0.95])
    if len(set(knots)) < 4:
        knots = np.linspace(logit.min(), logit.max() + 1e-6, 4)
    tested = np.asarray([r["n_subjects_tested"] or 0 for r in rows], float)
    blocks = [
        np.ones((len(rows), 1)),
        pooled_dummies(strata, 1),
        rcs_basis(logit, knots),
        pooled_dummies([r["hla"] for r in rows], RARE_LEVEL_ROWS),
        pooled_dummies([len(r["peptide"]) for r in rows], 30),
        np.log([r["n_assay_rows"] for r in rows])[:, None],
        np.log1p(tested)[:, None],
        (tested == 0).astype(float)[:, None],
    ]
    if domain == "viral":
        blocks.append(pooled_dummies(
            [r["source_species"] for r in rows], RARE_LEVEL_ROWS))
    if stringent:
        for name in BASELINES:
            blocks.append(standardize_within(
                np.asarray([r[name] for r in rows]), strata)[:, None])
    return np.hstack(blocks)


def fit_logit(design: np.ndarray, outcome: np.ndarray, groups: np.ndarray):
    """Logistic regression with cluster-robust errors; the score is last."""
    import statsmodels.api as sm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sm.GLM(outcome, design, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": groups}, maxiter=100)
    beta, se = float(fit.params[-1]), float(fit.bse[-1])
    if not (np.isfinite(beta) and np.isfinite(se) and se > 0):
        raise ValueError("non-finite estimate")
    return beta, se


def _effect(beta: float, se: float) -> dict:
    z = beta / se
    return {"beta": beta, "se": se, "z": z, "odds_ratio": float(np.exp(beta)),
            "or_ci95": [float(np.exp(beta - 1.96 * se)),
                        float(np.exp(beta + 1.96 * se))],
            "p_one_sided": float(1 - NormalDist().cdf(z))}


def pooled_effect(rows: list[dict], scores: np.ndarray, nuisance: np.ndarray,
                  cluster_field: str = "cluster") -> dict | None:
    """Pooled adjusted effect of one candidate across all strata."""
    strata = [f"{r['cohort']}|{r['exposure']}" for r in rows]
    z = standardize_within(scores, strata)
    keep = np.isfinite(z)
    if keep.sum() < 50:
        return None
    outcome = np.asarray(
        [r["outcome"] == "Positive" for r in rows], dtype=float)[keep]
    if outcome.sum() < 10:
        return None
    groups = np.unique(
        np.asarray([r[cluster_field] for r in rows])[keep],
        return_inverse=True)[1]
    try:
        beta, se = fit_logit(
            np.hstack([independent_columns(nuisance[keep]), z[keep, None]]),
            outcome, groups)
    except Exception as error:   # separation, singular design
        return {"failed": type(error).__name__}
    return {**_effect(beta, se), "n_rows": int(keep.sum()),
            "n_positive": int(outcome.sum()),
            "n_clusters": int(len(set(groups)))}


def cohort_effect(rows: list[dict], scores: np.ndarray) -> dict | None:
    """One cohort alone: presentation, allele, baselines and assay counts."""
    strata = [f"{r['cohort']}|{r['exposure']}" for r in rows]
    z = standardize_within(scores, strata)
    keep = np.isfinite(z)
    sub = [r for r, k in zip(rows, keep) if k]
    outcome = np.asarray([r["outcome"] == "Positive" for r in sub], float)
    if len(sub) < 40 or outcome.sum() < 8 or (1 - outcome).sum() < 8:
        return None
    presentation = np.clip(np.asarray(
        [r["mhcflurry_presentation_score"] for r in sub]), 1e-6, 1 - 1e-6)
    blocks = [
        np.ones((len(sub), 1)),
        np.log(presentation / (1 - presentation))[:, None],
        pooled_dummies([r["hla"] for r in sub], 10),
        np.log([r["n_assay_rows"] for r in sub])[:, None],
    ]
    substrata = [f"{r['cohort']}|{r['exposure']}" for r in sub]
    for name in BASELINES:
        blocks.append(standardize_within(
            np.asarray([r[name] for r in sub]), substrata)[:, None])
    groups = np.unique(
        np.asarray([r["cluster"] for r in sub]), return_inverse=True)[1]
    try:
        beta, se = fit_logit(
            np.hstack([independent_columns(np.hstack(blocks)), z[keep, None]]),
            outcome, groups)
    except Exception:
        return None
    return {"beta": beta, "se": se, "n": len(sub),
            "n_positive": int(outcome.sum())}


def dersimonian_laird(betas, ses) -> dict:
    betas, ses = np.asarray(betas), np.asarray(ses)
    if len(betas) < 2:
        return {}
    w = 1 / ses ** 2
    fixed = (w * betas).sum() / w.sum()
    q = float((w * (betas - fixed) ** 2).sum())
    df = len(betas) - 1
    c = w.sum() - (w ** 2).sum() / w.sum()
    tau2 = max(0.0, (q - df) / c)
    w_star = 1 / (ses ** 2 + tau2)
    beta = float((w_star * betas).sum() / w_star.sum())
    se = float(1 / np.sqrt(w_star.sum()))
    return {"beta": beta, "se": se, "z": beta / se, "tau2": float(tau2),
            "i2": float(max(0.0, (q - df) / q)) if q > 0 else 0.0,
            "n_cohorts": len(betas)}


def evaluate_candidates(rows: list[dict], scores: dict[str, np.ndarray],
                        domain: str) -> list[dict]:
    """Ledger of every candidate in one domain under the registered model."""
    stringent = nuisance_matrix(rows, domain, True)
    plain = nuisance_matrix(rows, domain, False)
    by_cohort = defaultdict(list)
    for i, row in enumerate(rows):
        by_cohort[row["cohort"]].append(i)
    reported = []
    for cohort, index in by_cohort.items():
        n_pos = sum(rows[i]["outcome"] == "Positive" for i in index)
        if n_pos >= 15 and len(index) - n_pos >= 15:
            reported.append((cohort, index))
    ledger = []
    for name, values in scores.items():
        log(f"  {domain}: {name}")
        coverage = float(np.isfinite(values).mean())
        entry = {"candidate": name, "family": CANDIDATES[name],
                 "coverage": coverage}
        entry["stringent"] = pooled_effect(rows, values, stringent)
        entry["structure_only"] = pooled_effect(rows, values, plain)
        entry["protein_free_clusters"] = pooled_effect(
            rows, values, stringent, "peptide_cluster")
        per_cohort = {}
        for cohort, index in reported:
            result = cohort_effect([rows[i] for i in index], values[index])
            if result:
                per_cohort[cohort] = result
        entry["per_cohort"] = per_cohort
        betas = [v["beta"] for v in per_cohort.values()]
        entry["n_cohorts_reported"] = len(betas)
        entry["sign_fraction"] = (
            float(np.mean(np.asarray(betas) > 0)) if betas else None)
        entry["random_effects"] = dersimonian_laird(
            betas, [v["se"] for v in per_cohort.values()])
        ledger.append(entry)
    n_tried = len(ledger)
    for entry in ledger:
        effect = entry["stringent"]
        if effect and "p_one_sided" in effect:
            effect["p_bonferroni"] = min(1.0, effect["p_one_sided"] * n_tried)
    return ledger


def select_candidate(ledger: list[dict]) -> dict:
    """The registered selection rule; ``None`` means nothing advances."""
    eligible = []
    for entry in ledger:
        effect = entry["stringent"]
        if not effect or "beta" not in effect:
            continue
        reasons = []
        if entry["coverage"] < MIN_COVERAGE:
            reasons.append("coverage below 90%")
        if effect["beta"] <= 0:
            reasons.append("pooled coefficient not positive")
        if entry["sign_fraction"] is None or \
                entry["sign_fraction"] < DEV_SIGN_FRACTION:
            reasons.append("positive in fewer than 60% of reported cohorts")
        if effect["z"] < DEV_Z_GATE:
            reasons.append("pooled z below 1.645")
        entry["fails"] = reasons
        if not reasons:
            eligible.append(entry)
    order = list(CANDIDATES)
    eligible.sort(key=lambda e: (-round(e["stringent"]["z"], 9),
                                 order.index(e["candidate"])))
    return {"selected": eligible[0]["candidate"] if eligible else None,
            "n_passing": len(eligible),
            "passing": [e["candidate"] for e in eligible]}


def holdout_verdict(effect: dict) -> dict:
    """Apply the registered confirmation rule to the holdout estimate."""
    se = effect["se"]
    upper = float(np.exp(effect["beta"] + 1.96 * se))
    confirmed = effect["z"] > 1.96 and effect["odds_ratio"] >= MIN_ODDS_RATIO_HOLDOUT
    return {
        "confirmed": bool(confirmed),
        "one_sided_z": effect["z"],
        "odds_ratio_per_sd": effect["odds_ratio"],
        "upper_95_odds_ratio": upper,
        "excludes_effects_at_or_above_minimum": bool(
            upper < MIN_ODDS_RATIO_HOLDOUT),
        "odds_ratio_detectable_with_80pct_power": float(np.exp(2.8 * se)),
        "reading": (
            "confirmed" if confirmed else
            "excludes the minimum effect of interest"
            if upper < MIN_ODDS_RATIO_HOLDOUT else
            "inconclusive: the interval includes both no effect and the "
            "minimum effect of interest"),
    }


# --------------------------------------------------------------- stages --
def attach_baselines(rows: list[dict]) -> None:
    peptides = sorted({r["peptide"] for r in rows})
    baselines = composition_baselines(peptides)
    index = {p: i for i, p in enumerate(peptides)}
    for row in rows:
        for name in BASELINES:
            row[name] = float(baselines[name][index[row["peptide"]]])


def prepare_rows(roles: tuple[str, ...], cache) -> tuple[list[dict], dict]:
    rows, funnel = assemble_rows(roles, cache)
    funnel["clusters"] = add_clusters(rows)
    funnel["presentation"] = add_presentation_predictions(rows)
    attach_baselines(rows)
    return rows, funnel


def ledger_markdown(ledger: list[dict]) -> str:
    lines = ["| Candidate | Family | Coverage | OR/SD (95% CI) | z | p (Bonf.) | "
             "Cohorts + | RE OR |", "| --- | --- | ---: | ---: | ---: | ---: | "
             "---: | ---: |"]
    for entry in sorted(
            ledger, key=lambda e: -(e["stringent"] or {}).get("z", -99)):
        effect = entry["stringent"] or {}
        if "beta" not in effect:
            lines.append(f"| `{entry['candidate']}` | {entry['family']} | "
                         f"{entry['coverage']:.2f} | failed | | | | |")
            continue
        low, high = effect["or_ci95"]
        sign = entry["sign_fraction"]
        sign_text = "-" if sign is None else "{:.0%}".format(sign)
        random = entry["random_effects"]
        random_text = ("-" if not random
                       else "{:.2f}".format(np.exp(random["beta"])))
        lines.append(
            f"| `{entry['candidate']}` | {entry['family']} | "
            f"{entry['coverage']:.2f} | {effect['odds_ratio']:.2f} "
            f"({low:.2f}-{high:.2f}) | {effect['z']:+.2f} | "
            f"{effect['p_one_sided']:.3f} ({effect['p_bonferroni']:.2f}) | "
            f"{sign_text} ({entry['n_cohorts_reported']}) | {random_text} |")
    return "\n".join(lines)


def write_rows(rows: list[dict], path: Path) -> None:
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_dev(args) -> None:
    from datacache import Cache
    import joblib

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cache = Cache("sources", cache_root=str(args.cache_root.resolve()))
    rows, funnel = prepare_rows(("dev",), cache)
    log(f"development rows: {len(rows)} in "
        f"{len({r['cohort'] for r in rows})} cohorts")
    res = Resources(cache)
    wanted = args.candidates or list(CANDIDATES)
    conditional, conditional_metadata = None, None
    if any(name.startswith("cond_") for name in wanted):
        conditional, conditional_metadata = train_conditional_classifiers(res)
        joblib.dump(conditional, OUTPUT_ROOT / "cond-models.joblib")
    scores = compute_candidates(rows, res, wanted, conditional)
    for name, values in scores.items():
        for row, value in zip(rows, values):
            row[name] = float(value)
    report = {
        "stage": "dev", "protocol": "docs/foreignness-choices-protocol.md",
        "data_funnel": funnel, "candidates": CANDIDATES,
        "conditional_classifier": conditional_metadata,
        "atlas": res.atlas_metadata,
        "anchor_pairs_example": {
            a: v for a, v in list(getattr(res, "anchor_summary", {}).items())[:8]},
        "swissprot_sha256": file_sha256(SWISSPROT), "domains": {}}
    for domain in ("viral", "neoepitope"):
        index = [i for i, r in enumerate(rows) if r["domain"] == domain]
        if not index:
            continue
        domain_rows = [rows[i] for i in index]
        domain_scores = {n: v[index] for n, v in scores.items()}
        log(f"evaluating {domain}: {len(domain_rows)} rows, "
            f"{sum(r['outcome'] == 'Positive' for r in domain_rows)} positive")
        ledger = evaluate_candidates(domain_rows, domain_scores, domain)
        selection = select_candidate(ledger)
        report["domains"][domain] = {
            "n_rows": len(domain_rows),
            "n_positive": sum(r["outcome"] == "Positive" for r in domain_rows),
            "n_cohorts": len({r["cohort"] for r in domain_rows}),
            "ledger": ledger, "selection": selection}
        (OUTPUT_ROOT / f"dev-ledger-{domain}.md").write_text(
            ledger_markdown(ledger) + "\n")
        log(f"{domain}: selected {selection['selected']} "
            f"({selection['n_passing']} passing)")
    write_rows(rows, OUTPUT_ROOT / "dev-scored-assays.csv")
    (OUTPUT_ROOT / "dev-report.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n")


def run_holdout(args) -> None:
    from datacache import Cache
    import joblib

    chosen = {d: c for d, c in SELECTED_CANDIDATES.items() if c}
    if not chosen:
        raise SystemExit(
            "SELECTED_CANDIDATES is empty. Commit the development selection "
            "before the holdout is evaluated.")
    if not args.selection_is_committed:
        raise SystemExit(
            "Pass --selection-is-committed to confirm the selection was "
            "committed before this run.")
    cache = Cache("sources", cache_root=str(args.cache_root.resolve()))
    dev_rows, _ = assemble_rows(("dev",), cache)
    rows, funnel = prepare_rows(("holdout",), cache)
    funnel["rows_before_dev_overlap_filter"] = len(rows)
    rows = drop_holdout_reusing_dev(rows, dev_rows)
    funnel["rows_after_dev_overlap_filter"] = len(rows)
    add_clusters(rows)
    res = Resources(cache)
    conditional = None
    if any(c.startswith("cond_") for c in chosen.values()):
        conditional = joblib.load(OUTPUT_ROOT / "cond-models.joblib")
    report = {"stage": "holdout", "selected": chosen, "data_funnel": funnel,
              "registered_confirmation_rule": {
                  "alpha_one_sided": ALPHA_HOLDOUT,
                  "minimum_odds_ratio_per_sd": MIN_ODDS_RATIO_HOLDOUT},
              "domains": {}}
    for domain, candidate in chosen.items():
        domain_rows = [r for r in rows if r["domain"] == domain]
        scores = compute_candidates(domain_rows, res, [candidate], conditional)
        nuisance = nuisance_matrix(domain_rows, domain, True)
        effect = pooled_effect(domain_rows, scores[candidate], nuisance)
        protein_free = pooled_effect(
            domain_rows, scores[candidate], nuisance, "peptide_cluster")
        per_cohort = {}
        for cohort in sorted({r["cohort"] for r in domain_rows}):
            index = [i for i, r in enumerate(domain_rows) if r["cohort"] == cohort]
            result = cohort_effect(
                [domain_rows[i] for i in index], scores[candidate][index])
            if result:
                per_cohort[cohort] = result
        report["domains"][domain] = {
            "candidate": candidate, "n_rows": len(domain_rows),
            "n_positive": sum(r["outcome"] == "Positive" for r in domain_rows),
            "pooled": effect, "peptide_overlap_clusters_only": protein_free,
            "per_cohort": per_cohort,
            "verdict": holdout_verdict(effect)
            if effect and "beta" in effect else None}
        log(f"{domain}: {candidate} -> "
            f"{(report['domains'][domain]['verdict'] or {}).get('reading')}")
    (OUTPUT_ROOT / "holdout-report.json").write_text(
        json.dumps(report, indent=1, default=float) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev", "holdout"), default="dev")
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--candidates", nargs="*", default=None,
                        help="restrict the dev stage to some candidates")
    parser.add_argument("--selection-is-committed", action="store_true")
    args = parser.parse_args()
    unknown = [c for c in args.candidates or [] if c not in CANDIDATES]
    if unknown:
        parser.error(f"unknown candidates: {unknown}")
    (run_dev if args.stage == "dev" else run_holdout)(args)


if __name__ == "__main__":
    main()
