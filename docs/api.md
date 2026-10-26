# API Reference

For reproducible downstream inference, use `weirdo.load_pretrained_model()` and its
`score`, `predict_categories`, `manifest`, and `prediction_identity` interface. See the
[stable integration contract](category-predictor.md#stable-integration-api) for input limits,
the explicit human-reference policy, and model provenance. Scalar scores are experimental
category summaries, not immune-response probabilities.

Complete API documentation for WEIRDO.

## High-Level API

### `weirdo.predict_categories`

```python
def predict_categories(
    peptides: Union[str, Sequence[str]],
    model: Optional[Union[str, BaseScorer]] = None,
    model_dir: Optional[str] = None,
    pathogen_categories: Optional[List[str]] = None,
    self_categories: Optional[List[str]] = None,
    aggregate: str = 'mean',
    combine: Union[str, Callable] = 'max',
) -> pandas.DataFrame
```

Probability that each peptide occurs in proteins of each species category, from the bundled
`swissprot-categories` model by default. Returns one row per peptide with a column per
category, `pathogen_<combine>`, `self_<combine>` and `foreignness`; `combine` is `'max'`
(default), `'sum'` or a function that maps an (n_peptides, n_categories_in_group) array to one
value per peptide. See [Species-category predictor](category-predictor.md).

### `weirdo.predict_peptide_content`

```python
def predict_peptide_content(
    peptides: Union[str, Sequence[str]],
    model: Optional[Union[str, BaseScorer]] = None,
    model_dir: Optional[str] = None,
    self_categories: Optional[Sequence[str]] = None,
    microbial_categories: Optional[Sequence[str]] = None,
    exact_self_hits: Optional[Sequence[bool]] = None,
    score_policy: Optional[Union[str, ScorePolicy]] = None,
) -> pandas.DataFrame
```

Scores each complete 8--15mer with the bundled experimental
`peptide-content-v1` model. The category columns are independent sigmoid
probabilities and need not sum to one. The final columns are
`microbial_like`, `self_like`, and
`microbial_foreign_like = microbial_like * (1 - self_like)`. See
[Variable-length peptide content](peptide-content.md). If an exact lookup in a
pinned host reference is available, `exact_self_hits` marks those rows and
forces their learned foreignness summary to zero.

With no overrides, `mammal-microbial-content-v1` preserves the original
numeric default. Pass `human-microbial-content-v1`,
`mouse-microbial-content-v1`, or `dog-microbial-content-v1` for an explicit
species-aware policy. The result's `attrs` contain the complete policy
definition. See [Versioned score policies](score-policies.md).

### Score-policy API

- `get_score_policy(policy_id)` → Immutable `ScorePolicy`
- `list_score_policies(model_id=None)` → JSON-compatible policy definitions
- `apply_score_policy(probabilities, categories, policy, exact_self_hits=None)`
  → source, self, and scalar score arrays

### `weirdo.list_bundled_models`

- `list_bundled_models()` → Models that ship inside the package (name, path, metadata including their evaluation)

### `weirdo.score_peptide`

```python
def score_peptide(
    peptide: str,
    model: Optional[Union[str, BaseScorer]] = None,
    model_dir: Optional[str] = None,
    preset: Optional[str] = None,
    aggregate: str = 'mean',
    **kwargs
) -> float
```

Score a single peptide for foreignness using a trained model.

**Parameters:**

- `peptide`: Peptide sequence to score
- `model`: Model name (from ModelManager) or an instantiated scorer
- `model_dir`: Custom model directory when loading by name
- `preset`: Preset used to construct a scorer (trainable presets need training data)
- `aggregate`: How to aggregate k-mer probabilities for long peptides
- `**kwargs`: Additional arguments passed to `create_scorer()`

**Returns:** Foreignness score (higher = more foreign)

**Example:**

```python
from weirdo import load_model, score_peptide
scorer = load_model('my-mlp')
score = score_peptide('MTMDKSEL', model=scorer)
```

---

### `weirdo.score_peptides`

```python
def score_peptides(
    peptides: Sequence[str],
    model: Optional[Union[str, BaseScorer]] = None,
    model_dir: Optional[str] = None,
    preset: Optional[str] = None,
    aggregate: str = 'mean',
    **kwargs
) -> np.ndarray
```

Score multiple peptides for foreignness using a trained model.

**Parameters:**

- `peptides`: Sequence of peptide strings
- `model`: Model name (from ModelManager) or an instantiated scorer
- `model_dir`: Custom model directory when loading by name
- `preset`: Preset used to construct a scorer (trainable presets need training data)
- `aggregate`: How to aggregate k-mer probabilities for long peptides
- `**kwargs`: Additional arguments passed to `create_scorer()`

**Returns:** Array of foreignness scores

---

### `weirdo.create_scorer`

```python
def create_scorer(
    preset: str = 'default',
    cache: bool = True,
    auto_download: bool = False,
    train_data: Optional[Sequence[str]] = None,
    train_labels: Optional[Any] = None,
    target_categories: Optional[List[str]] = None,
    **overrides
) -> BaseScorer
```

Create a scorer from a preset configuration.

**Parameters:**

- `preset`: Preset name ('default', 'fast')
- `cache`: Whether to cache the scorer instance
- `auto_download`: If True, automatically download data if not present
- `train_data`: Training peptides for trainable scorers
- `train_labels`: Training labels for trainable scorers
- `target_categories`: Category names for multi-label training
- `**overrides`: Override specific config parameters

**Returns:** Configured scorer (trainable scorers are untrained unless training data is provided)

---

## Model Artifacts

### `weirdo.list_models`

- `list_models(model_dir=None)` → List locally installed trained models

### `weirdo.load_model`

- `load_model(name, model_dir=None)` → Load a local model by name

### `weirdo.save_model`

- `save_model(scorer, name, model_dir=None, overwrite=False)` → Save a trained model

### `weirdo.list_pretrained_models`

- `list_pretrained_models(model_dir=None)` → List built-in downloadable model descriptors

### `weirdo.download_pretrained_model`

- `download_pretrained_model(name, model_dir=None, overwrite=False)` → Download/install a built-in pretrained model

### `weirdo.download_model_from_url`

- `download_model_from_url(name, url, model_dir=None, overwrite=False, expected_sha256=None)` → Download/install a model archive from direct URL

---

## Scorer Classes

### `weirdo.scorers.MLPScorer`

```python
class MLPScorer(TrainableScorer):
    def __init__(
        self,
        k: int = 8,
        hidden_layer_sizes: Tuple[int, ...] = (256, 128, 64),
        activation: str = 'relu',
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        early_stopping: bool = True,
        use_dipeptides: bool = True,
        batch_size: int = 256,
        random_state: Optional[int] = None,
        loss: str = 'auto',
        positional: bool = False,
        whole_peptide: bool = False
    )
```

Parametric neural network scorer for category probabilities and foreignness. With binary
multi-label targets (`loss='auto'`) it has one sigmoid output per label trained with binary
cross-entropy.
Set `whole_peptide=True` to extract one feature vector for each complete input
instead of windowing longer sequences into `k`-mers.

**Methods:**

- `train(peptides, labels, target_categories=None, val_peptides=None, val_labels=None, cache_features=False, restore_best=True, ...)` → Train model
- `score(peptides, aggregate='mean', combine='max')` → Foreignness scores in [0, 1]
- `predict_proba(peptides, aggregate='mean')` → Category probabilities
- `foreignness(peptides, aggregate='mean', combine='max')` → Pathogen-vs-self foreignness
- `predict_dataframe(peptides, aggregate='mean', combine='max')` → Full DataFrame output

---

### `weirdo.scorers.SwissProtReference`

```python
class SwissProtReference(StreamingReference):
    def __init__(
        self,
        categories: Optional[List[str]] = None,
        k: int = 8,
        data_path: Optional[str] = None,
        lazy: bool = False,
        use_set: bool = False,
        auto_download: bool = False
    )
```

Reference dataset from SwissProt protein database.

**Methods:**

- `load()` → Load data into memory
- `contains(kmer)` → Check if k-mer exists
- `get_frequency(kmer, default=0.0)` → Get k-mer presence (1.0 if present)
- `get_categories()` → List available categories
- `get_kmer_categories(kmer)` → Get category breakdown for k-mer
- `iter_kmers()` → Iterate over all k-mers

---

## Configuration

### `weirdo.scorers.ScorerConfig`

```python
@dataclass
class ScorerConfig:
    scorer: str = 'mlp'
    reference: str = 'swissprot'
    k: int = 8
    scorer_params: Dict[str, Any] = field(default_factory=dict)
    reference_params: Dict[str, Any] = field(default_factory=dict)
    training_params: Dict[str, Any] = field(default_factory=dict)
```

**Class Methods:**

- `from_preset(name)` → Create from preset
- `from_dict(data)` → Create from dictionary
- `from_yaml(path)` → Load from YAML file
- `from_json(path)` → Load from JSON file

**Methods:**

- `build(auto_load=True, train_data=None, train_labels=None, target_categories=None, **train_overrides)` → Build scorer from config
- `to_dict()` → Convert to dictionary

---

## Registry

### `weirdo.scorers.register_scorer`

```python
@register_scorer('name', description='...')
class MyScorer(BaseScorer):
    ...
```

Decorator to register a custom scorer.

### `weirdo.scorers.register_reference`

```python
@register_reference('name', description='...')
class MyReference(BaseReference):
    ...
```

Decorator to register a custom reference.

### Other Registry Functions

- `list_scorers()` → List registered scorer names
- `list_references()` → List registered reference names
- `create_scorer(name, **params)` → Create scorer by name
- `create_reference(name, **params)` → Create reference by name

---

## Base Classes

### `weirdo.scorers.BaseScorer`

Abstract base class for all scorers.

**Methods to implement:**

- `fit(reference)` → Fit to reference data
- `score(peptides)` → Score peptide(s)

**Provided methods:**

- `fit_score(reference, peptides)` → Convenience method
- `get_params(deep=True)` → Get scorer parameters
- `set_params(**params)` → Set scorer parameters
- `is_fitted` → Property to check if fitted

### `weirdo.scorers.BaseReference`

Abstract base class for reference datasets.

**Methods to implement:**

- `load()` → Load data
- `contains(kmer)` → Check k-mer existence
- `get_frequency(kmer, default)` → Get k-mer presence (1.0 if present)
- `get_categories()` → List categories
- `iter_kmers()` → Iterate over k-mers

---

## Amino Acid Data

### Alphabets

```python
from weirdo import (
    canonical_amino_acids,        # List[AminoAcid]
    canonical_amino_acid_letters, # List[str]
    extended_amino_acids,         # List[AminoAcid]
    extended_amino_acid_letters,  # List[str]
    amino_acid_letter_indices,    # Dict[str, int]
)
```

### Properties

```python
from weirdo.amino_acid_properties import (
    hydropathy, volume, polarity, mass, ...
)
```

### Matrices

```python
from weirdo.blosum import blosum62_dict, blosum62_matrix
from weirdo.pmbec import pmbec_dict, pmbec_matrix
```

### Vectorization

```python
from weirdo import PeptideVectorizer

vectorizer = PeptideVectorizer(max_ngram=2, normalize_row=True)
X = vectorizer.fit_transform(sequences)
```

---

## Data Management

### `weirdo.DataManager`

```python
class DataManager:
    def __init__(
        self,
        data_dir: Optional[Path] = None,
        auto_download: bool = False,
        verbose: bool = True,
        uniprot_url: Optional[str] = None
    )
```

Manages WEIRDO reference data. Datasets are built from UniProtKB/Swiss-Prot: `'swissprot-sequences'`
downloads the taxonomic-division files from UniProt (with `datacache`) and writes the protein
sequence table; `'swissprot-8mers'` builds the k-mer reference table from it. `uniprot_url` (or the
`WEIRDO_UNIPROT_URL` environment variable) selects a mirror.

**Methods:**

- `download(name, force=False)` → Download from UniProt and build a dataset
- `download_all()` → Build all datasets
- `is_downloaded(name)` → Check if dataset exists
- `get_data_path(name)` → Get path to dataset
- `delete_download(name)` → Delete a dataset
- `delete_all_downloads()` → Delete all datasets
- `status()` → Get status dict
- `print_status()` → Print human-readable status
- `clear_all()` → Delete all datasets and the raw UniProt downloads

**Example:**

```python
from weirdo import DataManager

dm = DataManager()
dm.download('swissprot-8mers')
dm.print_status()
dm.clear_all()
```

---

### `weirdo.get_data_manager`

```python
def get_data_manager(
    data_dir: Optional[Path] = None,
    auto_download: bool = False,
    verbose: bool = True
) -> DataManager
```

Get the default DataManager singleton instance.

---

### `weirdo.ensure_data_available`

```python
def ensure_data_available(auto_download: bool = False) -> Path
```

Ensure reference data is available, optionally downloading.

---

## UniProt Data and K-mer Tables

### `weirdo.uniprot`

- `fetch_swissprot_divisions(download_dir, divisions=None, release=None, base_url=None, force=False, show_progress=False, timeout=60)` → `DownloadedRelease(version, base_url, paths, files)`: downloads the SwissProt taxonomic-division files of UniProt's current release with `datacache` into `download_dir/release-<version>/`, checking the sizes and MD5s UniProt lists; reuses cached files; falls back to the newest complete cached release (with a warning) when UniProt cannot be reached
- `fetch_release(base_url=None)` → `UniProtRelease(version, files)`: UniProt's current release listing (`RELEASE.metalink`)
- `iter_swissprot_sequences(path)` → iterator of `(entry name, sequence)` from a `.dat` or `.dat.gz` file
- `build_sequence_table(division_paths, output_path)` → writes `identifier,sequence,category` rows and returns their number
- `read_sequence_table(path_or_dataframe)` → DataFrame with those columns (also reads the older layout with an index column)
- `SWISSPROT_DIVISIONS`, `UNIPROT_BASE_URL`

### `weirdo.kmer_table`

- `build_kmer_table(sequences, output_path, k=8, categories=None, memory_passes=4)` → writes the reference table (`seq`, one `True`/`False` column per category, `label_count`; sorted by k-mer; only k-mers made of the 20 canonical residues) and returns the number of k-mers. Memory use is about `1/memory_passes` of the windows at a time.
- `iter_kmer_masks(sequences, bits, k=8)` → unique packed k-mers with category bitmasks, in sorted order, a block at a time
- `encode_residues`, `window_codes`, `unique_with_masks`, `decode_kmers`, `masks_to_matrix`: the packing primitives (five bits per residue, k up to 12)

### `weirdo.scorers.peptide_features`

- `extract_features(peptide, k=8, use_dipeptides=True, positional=False)` → the definition of each feature for one peptide
- `extract_features_batch(peptides, k=8, use_dipeptides=True, positional=False)` → the same features for many peptides at once (what training and prediction use)
- `get_feature_names(use_dipeptides=True, positional=False, k=8)`, `feature_count(...)`

---

## Command-Line Interface

```bash
# Setup
weirdo setup                  # Download SwissProt from UniProt and build the reference data

# Data management
weirdo data list              # List datasets (alias: ls, status)
weirdo data download          # Build swissprot-8mers from UniProt (or: data download swissprot-sequences)
weirdo data clear             # Clear data
weirdo data path              # Show data directory

# Model training + scoring
weirdo models train --data train.csv --name my-model
weirdo score PEPTIDE...                       # bundled swissprot-categories model
weirdo score --model my-model PEPTIDE...      # a trained model; --combine max|sum, --csv
weirdo models available                       # bundled and downloadable models
```
