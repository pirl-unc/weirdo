# API Reference

Complete API documentation for WEIRDO.

## High-Level API

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
        random_state: Optional[int] = None
    )
```

Parametric neural network scorer for category probabilities and foreignness.

**Methods:**

- `train(peptides, labels, target_categories=None, ...)` → Train model
- `score(peptides, aggregate='mean')` → Foreignness scores in [0, 1]
- `predict_proba(peptides, aggregate='mean')` → Category probabilities
- `foreignness(peptides, aggregate='mean')` → Derived foreignness
- `predict_dataframe(peptides, aggregate='mean')` → Full DataFrame output

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
        verbose: bool = True
    )
```

Manages WEIRDO reference data downloads.

**Methods:**

- `download(name)` → Download a dataset
- `download_all()` → Download all datasets
- `is_downloaded(name)` → Check if dataset exists
- `get_data_path(name)` → Get path to dataset
- `delete_download(name)` → Delete a dataset
- `delete_all_downloads()` → Delete all datasets
- `status()` → Get status dict
- `print_status()` → Print human-readable status
- `clear_all()` → Delete all downloads

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

## Command-Line Interface

```bash
# Setup
weirdo setup                  # Download reference data

# Data management
weirdo data list              # List datasets (alias: ls, status)
weirdo data download          # Download reference data
weirdo data clear             # Clear data
weirdo data path              # Show data directory

# Model training + scoring
weirdo models train --data train.csv --name my-model
weirdo score --model my-model PEPTIDE...
```
