"""Registry of the T-cell evaluation datasets and where they come from.

Unlike :mod:`weirdo.data_manager` (reference proteomes), these tables are not
shipped or downloaded by Weirdo: they are rebuilt by the scripts named in each
entry, into a git-ignored ``data/`` directory (IEDB and CEDAR terms apply, and the
files are large). The registry records what each dataset is, where it comes from,
which script builds it and which report uses it, and ``status`` reports which
local files exist together with their SHA-256, so a result can be tied to the
exact bytes it was computed on.

Fields left ``None`` are not recorded in the repository; the cited report holds
the details.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class EvaluationCohort:
    name: str
    role: str  # development | validation | sealed-confirmation | auxiliary
    description: str
    source: str
    reference: Optional[str]  # DOI or URL of the publication / database
    files: Tuple[str, ...]  # paths relative to the repository root
    builder: Optional[str]  # script that produces the files
    reports: Tuple[str, ...]


EVALUATION_COHORTS: Dict[str, EvaluationCohort] = {c.name: c for c in [
    EvaluationCohort(
        name="improve",
        role="development",
        description="IMPROVE validated neoepitopes (467 positive / 17,053 negative)",
        source="https://github.com/SRHgroup/IMPROVE_paper (data.zip)",
        reference="https://doi.org/10.3389/fimmu.2024.1360281",
        files=("data/improve-validation/02_2_Validated_neoepitopes_calculated_features_TME.txt",
               "data/improve-validation/neoepitopes_predictions.tsv",
               "data/improve-validation/source.json"),
        builder=None,
        reports=("reports/auto-weirdo-v1", "reports/fixed-pmhc-weirdo-combination-v1"),
    ),
    EvaluationCohort(
        name="tesla",
        role="development",
        description="TESLA neoepitope consortium table (no patient identifiers; cluster on group)",
        source="TESLA_neoepitopes.xlsx (supplementary table; URL not recorded)",
        reference=None,
        files=("data/neoantigen-benchmarks/tesla/TESLA_neoepitopes.xlsx",),
        builder=None,
        reports=("reports/auto-weirdo-v1", "reports/fixed-pmhc-weirdo-combination-v1"),
    ),
    EvaluationCohort(
        name="braun-rcc",
        role="development",
        description="Braun RCC mutation-aware construct candidates (128 constructs, 9 patients)",
        source="Braun et al., Nature 2025 supplementary data",
        reference="https://doi.org/10.1038/s41586-024-08507-5",
        files=(),
        builder="scripts/build_braun_mutation_aware_candidates.py",
        reports=("reports/braun-mutation-aware-candidates-v1",
                 "reports/construct-weighted-candidates-v1"),
    ),
    EvaluationCohort(
        name="viral-tcell-benchmark",
        role="development",
        description="Registered viral T-cell studies pulled as IEDB assay pages",
        source="IEDB T-cell export (https://query-api.iedb.org/tcell_export)",
        reference="https://www.iedb.org",
        files=("data/viral-tcell-benchmark",),
        builder=None,
        reports=("reports/auto-weirdo-v1",),
    ),
    EvaluationCohort(
        name="iedb-human-class-i",
        role="auxiliary",
        description="All human-host class I T-cell assay rows; study-grouped training pool",
        source="https://query-api.iedb.org/tcell_export",
        reference="https://www.iedb.org",
        files=("data/iedb-human-class-i/assays.csv", "data/iedb-human-class-i/pool.csv",
               "data/iedb-human-class-i/context.csv"),
        builder="scripts/pull_iedb_human_class_i.py -> scripts/build_iedb_training_pool.py",
        reports=("reports/iedb-grouped-learning-v1",),
    ),
    EvaluationCohort(
        name="fresh-iedb-cohort-v1",
        role="sealed-confirmation",
        description="IEDB studies unused in development, exact-peptide purged (11 studies)",
        source="IEDB (subset of iedb-human-class-i / viral pages)",
        reference="https://www.iedb.org",
        files=("reports/fresh-iedb-cohort-v1/cohort.csv",
               "reports/fresh-iedb-cohort-v1/cohort.funnel.json"),
        builder="scripts/build_fresh_iedb_cohort.py",
        reports=("reports/fresh-iedb-cohort-v1",),
    ),
    EvaluationCohort(
        name="fresh-iedb-cohort-v2",
        role="sealed-confirmation",
        description="Second IEDB confirmation cohort from the full human class I pull (49 studies)",
        source="IEDB full human class I pull",
        reference="https://www.iedb.org",
        files=("reports/fresh-iedb-cohort-v2/cohort.csv",
               "reports/fresh-iedb-cohort-v2/cohort.funnel.json"),
        builder="scripts/build_fresh_iedb_cohort.py --assays",
        reports=("reports/fresh-iedb-cohort-v2",),
    ),
    EvaluationCohort(
        name="fresh-cedar-cohort-v1",
        role="sealed-confirmation",
        description="CEDAR cohort attempt; funnel is empty (no study outside the IEDB pool)",
        source="CEDAR full T-cell export (tcell_full_v3.zip)",
        reference="https://cedar.iedb.org",
        files=("reports/fresh-cedar-cohort-v1/cohort.funnel.json",),
        builder="scripts/convert_cedar_export.py -> scripts/build_fresh_iedb_cohort.py --assays",
        reports=("reports/fresh-cedar-cohort-v1",),
    ),
    EvaluationCohort(
        name="hla-ligand-atlas",
        role="auxiliary",
        description="Presented human self ligands (class I) for the self-ligand analyses",
        source="HLA Ligand Atlas release 2020.12 (hla_ligand_atlas_2020.12.zip)",
        reference="https://hla-ligand-atlas.org",
        files=("data/self-ligands/cohort.csv",),
        builder="scripts/build_self_ligand_cohort.py",
        reports=("reports/epitope-vs-self-ligand-v1", "reports/iedb-grouped-learning-v1"),
    ),
    EvaluationCohort(
        name="uniprot-parent-sequences",
        role="auxiliary",
        description="UniProt sequences for IEDB epitope parent accessions (source context)",
        source="https://rest.uniprot.org/uniprotkb/accessions",
        reference="https://www.uniprot.org/help/license",
        files=("data/iedb-human-class-i/uniprot-sequences.csv",),
        builder="scripts/fetch_uniprot_sequences.py",
        reports=("reports/iedb-grouped-learning-v1",),
    ),
]}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cohort_status(root: Path, with_checksums: bool = False) -> List[dict]:
    """One row per registered cohort: which of its files exist under ``root``."""
    rows = []
    for cohort in EVALUATION_COHORTS.values():
        row = asdict(cohort)
        present = {}
        for rel in cohort.files:
            path = Path(root) / rel
            if path.is_file():
                present[rel] = _sha256(path) if with_checksums else True
            elif path.is_dir():
                present[rel] = True
            else:
                present[rel] = False
        row["present"] = present
        rows.append(row)
    return rows
