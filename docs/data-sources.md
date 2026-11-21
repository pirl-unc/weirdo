# Evaluation data sources

`weirdo data cohorts` lists the T-cell evaluation datasets, where each comes from, which
script builds it, which reports use it and whether the local files exist
(`--checksums` adds SHA-256). The registry lives in `weirdo/evaluation_cohorts.py`.

These tables are not shipped in the wheel and are not downloaded by `weirdo data download`
(which handles the UniProt reference proteomes). IEDB and CEDAR terms apply and the files are
large, so they are rebuilt on demand by the scripts into a git-ignored `data/` directory.

| Cohort | Role | Source | Reference |
|---|---|---|---|
| improve | development | SRHgroup/IMPROVE_paper `data.zip` | [Gartner et al., 2024](https://doi.org/10.3389/fimmu.2024.1360281) |
| tesla | development | TESLA supplementary table (URL not recorded) | not recorded |
| braun-rcc | development | Braun et al., Nature 2025 supplement | [doi](https://doi.org/10.1038/s41586-024-08507-5) |
| viral-tcell-benchmark | development | IEDB T-cell export | [IEDB](https://www.iedb.org) |
| iedb-human-class-i | auxiliary | IEDB query API `tcell_export` | [IEDB](https://www.iedb.org) |
| fresh-iedb-cohort-v1, v2 | sealed confirmation | IEDB studies unused in development, exact-peptide purged | [IEDB](https://www.iedb.org) |
| fresh-cedar-cohort-v1 | sealed confirmation (empty funnel) | CEDAR full T-cell export | [CEDAR](https://cedar.iedb.org) |
| hla-ligand-atlas | auxiliary | HLA Ligand Atlas release 2020.12 | [atlas](https://hla-ligand-atlas.org) |
| uniprot-parent-sequences | auxiliary | UniProt REST | [license](https://www.uniprot.org/help/license) |

Fields marked "not recorded" are not stored in the repository. Pin the retrieval date and
checksum when rebuilding. Rebuild order for the IEDB-derived tables:
`pull_iedb_human_class_i.py` → `build_iedb_training_pool.py` →
`build_fresh_iedb_cohort.py`; the CEDAR route is `convert_cedar_export.py` →
`build_fresh_iedb_cohort.py --assays`. Read untrusted archives with `python -I` from a
separate directory.

## Reports and omitted tables

`reports/*/analysis.md` hold the analyses behind each result, with the small summary tables
they cite. Scored tables larger than 1 MB (for example `fresh-iedb-cohort-v2/scored.csv`
and `auto-weirdo-v1/shipped-weirdo-scores.csv.gz`) are not committed; they are regenerated
by the builders and scorers named in `weirdo data cohorts`.
