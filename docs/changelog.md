# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-01-19

### Added

**Extensible Scorer Architecture**

- `BaseScorer` and `BatchScorer` abstract base classes
- `BaseReference` and `StreamingReference` for reference datasets
- `ScorerRegistry` with plugin-style registration

**FrequencyScorer**

- Scores peptides based on k-mer frequency in reference
- Supports multiple aggregation methods (mean, max, min, sum)
- `get_kmer_scores()` for debugging individual k-mer contributions

**SimilarityScorer**

- Scores peptides based on similarity to reference k-mers
- Supports BLOSUM30, BLOSUM50, BLOSUM62, and PMBEC matrices
- `get_closest_reference()` for finding similar k-mers

**SwissProtReference**

- Loads pre-computed k-mer data from SwissProt database
- Category filtering (human, bacteria, viruses, etc.)
- Memory-efficient modes: `use_set`, `lazy`

**Configuration System**

- `ScorerConfig` dataclass with YAML/JSON support
- Built-in presets: default, human, pathogen, similarity_blosum62, etc.

**High-Level API**

- `score_peptide()` and `score_peptides()` convenience functions
- `create_scorer()` with preset support and caching
- `get_available_presets()` and `get_preset_info()`

**Documentation**

- MkDocs documentation with Material theme
- Getting started guide
- API reference

**CI/CD**

- GitHub Actions for tests (Python 3.9-3.12)
- GitHub Actions for documentation
- Coverage reporting

### Changed

- Updated package version to 1.1.0
- Improved `__init__.py` exports

### Removed

- Deleted stub `weirdo/reference.py` file

## [1.0.1] - Previous

- Initial release with amino acid properties
- BLOSUM and PMBEC matrices
- Peptide vectorization
- Reduced alphabets
