# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [2.1.0] - 2026-02-04

### Added

- Sequence-level statistics (length, repeats, entropy, complexity)
- Reduced alphabet composition features (Murphy/GBMR/SDM, etc.)
- Dipeptide summary statistics (entropy, Gini, maxima, homodimers)

### Changed

- MLP feature vector expanded to 592 non-positional features
- Test warnings suppressed for short-epoch MLP training

## [2.0.3] - 2026-01-27

### Changed

- Removed positional k-mer one-hot features from the MLP feature set
- Updated feature counts and documentation to reflect the non-positional feature set

### Fixed

- Suppressed sklearn convergence warnings in SwissProt MLP tests

## [2.0.2] - 2026-01-27

### Fixed

- Suppressed sklearn convergence warnings in MLP scorer tests

## [2.0.1] - 2026-01-27

### Fixed

- Documentation consistency and reference wording for the MLP-only workflow

## [2.0.0] - 2026-01-27

### Added

- Murphy 8 and Murphy 15 reduced alphabets (`murphy8`, `murphy15`)
- Consistent training metadata keys for model inspection (`n_epochs`, `final_train_loss`, `best_val_loss`)

### Changed

- Default preset now targets the parametric MLP scorer
- MLP scoring now aggregates k-mer probabilities for variable-length peptides
- `create_scorer` returns trainable scorers untrained unless training data is provided
- `score_peptide` / `score_peptides` now require a trained model (or a non-trainable preset)
- CLI scoring now requires a trained model and prints per-category probabilities when available
- Data manager focuses on downloads only (indices removed)

### Removed

- FrequencyScorer from the public API and documentation
- Data index build commands from the CLI

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
