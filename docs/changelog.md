# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Immutable, inspectable score-policy definitions for the stable human
  complement and variable-length mammal, human, mouse, and dog content scores.
  `list_score_policies()` exposes the complete host/category/formula/overlap/
  validation contract, and prediction frames retain it in `DataFrame.attrs`.

## [2.3.0] - 2026-10-25

### Added

- `peptide-content-v1`, `weirdo.predict_peptide_content()`, and the
  `microbial_foreign_like` score for complete 8--15mer ligands. The bundled
  composition MLP has independent multi-label taxonomy heads; separately retained
  CNN, transformer, frozen ESM-2 8M, and fine-tuned ESM-2 8M artifacts support the
  architecture comparison. Protein-family held-out evaluation and grouped
  immunogenicity analyses are documented in `docs/peptide-content.md`.
- Complete, pinned UniProt reference-proteome downloads for exact host-self
  lookup, plus a frozen manifest of 32 mammalian and 512 viral proteomes.
- A reproducible negative-result study of mammalian-host viral peptide labels,
  including Swiss-Prot `OH`/NCBI taxonomy construction, nested experimental
  heads, held-out host discrimination, and patient-grouped IMPROVE evaluation;
  see `docs/viral-host-foreignness-study.md`. The released `p_viruses` score is
  unchanged because the host-aware heads did not improve neoepitope AUC.

### Changed

- `MLPScorer(whole_peptide=True)` scores one complete variable-length peptide
  feature vector rather than aggregating fixed-length windows.

## [2.2.0] - 2026-10-21

### Release contract

- Ship the reviewed species-category model with portable NumPy weights and recorded
  training/evaluation provenance.
- Add checksum-verified, offline `load_pretrained_model()` inference for 8–11-mers,
  bounded batches, explicit human-reference score semantics and complete cache identity.
- Preserve the original training-code provenance and clarify that mnemonic-group splits
  do not establish sequence-homology separation or an immune-response ranking benefit.
- Check the installed wheel in CI, and reject unsafe archive member paths and special files.

### Added

- `swissprot-categories`: a trained species-category predictor that ships with the package
  (0.7 MB of pickle-free `.npz` weights), with `weirdo.predict_categories()`,
  `weirdo.list_bundled_models()` and a `weirdo score` that works without `--model`; see
  `docs/category-predictor.md`. Weights up to 5 MB may ship inside the package (a test enforces
  it); larger models are release assets fetched with `weirdo models download`.
- `MLPScorer(loss=...)`: `'bce'` trains one sigmoid output per label with binary cross-entropy;
  `train()` accepts `cache_features` and `restore_best`
- `foreignness()`, `score()` and `predict_dataframe()` accept `combine='max'|'sum'` or a function
  (default `'max'`, as before), and `predict_dataframe()` adds `pathogen_<combine>` and
  `self_<combine>` columns; `weirdo score --combine max|sum`
- `MLPScorer(positional=True)`: optional position-specific one-hot features of the k-mer
  (`k * 21` more), as used by releases 2.0.0 to 2.0.2; the option is stored with the model
- `weirdo.uniprot` and `weirdo.kmer_table`: download the SwissProt taxonomic-division files from
  UniProt with `datacache` (sizes and MD5s checked against UniProt's release listing, resumable,
  one directory per UniProt release), parse them, and build the sequence table and the k-mer
  reference table with bounded memory
- `scripts/category_data.py`, `scripts/category_eval.py`, `scripts/category_headroom.py` and
  `scripts/train_category_predictor.py`: mnemonic-group splits, evaluation helpers and the
  training/evaluation pipeline

### Changed

- Python 3.10-3.14 are supported (3.9 is dropped). Dependency floors are now numpy >=2.0,
  pandas >=2.2 and scikit-learn >=1.7, and `datacache` >=1.24 is required; on Python 3.11+ pip
  resolves pandas 3 and scikit-learn 1.8+ (those releases need 3.11+). CI runs 3.10-3.14.
- `weirdo data download` builds the reference data from UniProt instead of downloading a prebuilt
  table from a GitHub release (`pirl-unc/weirdo-data`, which no longer exists, so the old command
  failed with a 404). `swissprot-8mers` is built from the new `swissprot-sequences` dataset;
  `weirdo data clear` also removes the raw UniProt downloads.
- Model archives (`weirdo models download`) are downloaded with `datacache`, which retries
  transient failures and checks `expected_sha256`
- Peptide features are computed with vectorised array operations (`extract_features_batch`), about
  35 times faster than the per-peptide functions, which remain as the definition of each feature.
  No worker processes are used, so scripts need no `if __name__ == "__main__":` guard. The
  `parallel_workers` / `parallel_chunksize` options of `train_streaming()` are ignored and emit a
  `DeprecationWarning`.
- Binary multi-label targets are now trained with sigmoid outputs and binary cross-entropy by
  default. Previously the 0/1 labels were fitted by least squares and passed through a sigmoid, so
  `predict_proba` returned values squeezed near 0.5-0.7 that were not probabilities. Models saved
  by earlier versions still load (with a warning); use `loss='squared'` for the old behaviour.
- Early stopping restores the weights of the best validation epoch for sigmoid-head models
- `predict_proba()` scores all windows of all peptides in batches
- Models with sigmoid outputs are saved as a pickle-free `.npz` archive (`model.pt`) with a hash of
  the ordered feature names, which is checked on load. The released-model API additionally
  verifies the complete feature implementation checksum

### Fixed

- A model saved with a different number of features (for example by a development version with
  positional features) loaded without complaint and then failed inside scikit-learn when it was
  used to score; it now fails at load with a message that states both feature counts
- `weirdo data status` and `weirdo data path` returned no exit code; `weirdo score`, `weirdo data
  download` and `weirdo setup` print an error message instead of a traceback when a model cannot be
  loaded or a download fails

## [2.1.2] - 2026-02-06

### Fixed

- Resolved documentation/API mismatches for scorer behavior and utilities
- Implemented FASTA translation path in CLI and clarified clear-data semantics
- Added edge-case validation checks for MLP training and foreignness dataframe categories
- Made PMBEC import silent and aligned amino-acid helper outputs with function names

### Added

- Regression tests for CLI edge cases, data manager cleanup behavior, and logical consistency checks

## [2.1.1] - 2026-02-05

### Fixed

- Packaging metadata now uses SPDX license identifier

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
