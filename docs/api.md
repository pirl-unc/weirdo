# API Reference

Complete API documentation for WEIRDO.

## High-Level API

### `weirdo.score_peptide`

```python
def score_peptide(
    peptide: str,
    preset: str = 'default',
    auto_download: bool = False,
    **kwargs
) -> float
```

Score a single peptide for foreignness.

**Parameters:**

- `peptide`: Peptide sequence to score
- `preset`: Scoring preset to use (default: 'default')
- `auto_download`: If True, automatically download data if not present
- `**kwargs`: Additional arguments passed to `create_scorer()`

**Returns:** Foreignness score (higher = more foreign)

**Example:**

```python
from weirdo import score_peptide
score = score_peptide('MTMDKSEL')

# Auto-download data on first use
score = score_peptide('MTMDKSEL', auto_download=True)
```

---

### `weirdo.score_peptides`

```python
def score_peptides(
    peptides: Sequence[str],
    preset: str = 'default',
    auto_download: bool = False,
    **kwargs
) -> np.ndarray
```

Score multiple peptides for foreignness.

**Parameters:**

- `peptides`: Sequence of peptide strings
- `preset`: Scoring preset to use
- `auto_download`: If True, automatically download data if not present
- `**kwargs`: Additional arguments passed to `create_scorer()`

**Returns:** Array of foreignness scores

---

### `weirdo.create_scorer`

```python
def create_scorer(
    preset: str = 'default',
    cache: bool = True,
    auto_download: bool = False,
    **overrides
) -> BaseScorer
```

Create a scorer from a preset configuration.

**Parameters:**

- `preset`: Preset name ('default', 'human', 'pathogen', etc.)
- `cache`: Whether to cache the scorer instance
- `auto_download`: If True, automatically download data if not present
- `**overrides`: Override specific config parameters

**Returns:** Configured and fitted scorer

---

## Scorer Classes

### `weirdo.scorers.FrequencyScorer`

```python
class FrequencyScorer(BatchScorer):
    def __init__(
        self,
        k: int = 8,
        pseudocount: float = 1e-10,
        aggregate: str = 'mean',
        category: Optional[str] = None,
        batch_size: int = 10000
    )
```

Frequency-based foreignness scorer.

**Methods:**

- `fit(reference)` → Fit to reference dataset
- `score(peptides)` → Score peptide(s)
- `fit_score(reference, peptides)` → Fit and score in one call
- `get_kmer_scores(peptide)` → Get individual k-mer scores
- `score_batch(peptides, show_progress=False)` → Batch scoring

---

### `weirdo.scorers.SimilarityScorer`

```python
class SimilarityScorer(BatchScorer):
    def __init__(
        self,
        k: int = 8,
        matrix: str = 'blosum62',
        distance_metric: str = 'min_distance',
        max_candidates: int = 1000,
        aggregate: str = 'mean',
        batch_size: int = 1000
    )
```

Similarity-based foreignness scorer using substitution matrices.

**Methods:**

- `fit(reference)` → Fit to reference dataset
- `score(peptides)` → Score peptide(s)
- `get_closest_reference(kmer, n=5)` → Find closest reference k-mers

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
        use_set: bool = False
    )
```

Reference dataset from SwissProt protein database.

**Methods:**

- `load()` → Load data into memory
- `contains(kmer)` → Check if k-mer exists
- `get_frequency(kmer, default=0.0)` → Get k-mer frequency
- `get_categories()` → List available categories
- `get_kmer_categories(kmer)` → Get category breakdown for k-mer
- `iter_kmers()` → Iterate over all k-mers

---

## Configuration

### `weirdo.scorers.ScorerConfig`

```python
@dataclass
class ScorerConfig:
    scorer: str = 'frequency'
    reference: str = 'swissprot'
    k: int = 8
    scorer_params: Dict[str, Any] = field(default_factory=dict)
    reference_params: Dict[str, Any] = field(default_factory=dict)
```

**Class Methods:**

- `from_preset(name)` → Create from preset
- `from_dict(data)` → Create from dictionary
- `from_yaml(path)` → Load from YAML file
- `from_json(path)` → Load from JSON file

**Methods:**

- `build(auto_load=True)` → Build scorer from config
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
- `get_frequency(kmer, default)` → Get k-mer frequency
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

Manages WEIRDO reference data and indices.

**Methods:**

- `download(name)` → Download a dataset
- `download_all()` → Download all datasets
- `is_downloaded(name)` → Check if dataset exists
- `get_data_path(name)` → Get path to dataset
- `delete_download(name)` → Delete a dataset
- `delete_all_downloads()` → Delete all datasets
- `build_index(name)` → Build an index
- `rebuild_indices()` → Rebuild all indices
- `delete_index(name)` → Delete an index
- `delete_all_indices()` → Delete all indices
- `status()` → Get status dict
- `print_status()` → Print human-readable status
- `clear_all()` → Delete everything

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
weirdo setup                  # Download data and build indices

# Data management
weirdo data list              # List datasets and indices (alias: ls, status)
weirdo data download          # Download reference data
weirdo data clear             # Clear data
weirdo data index             # Build indices
weirdo data path              # Show data directory

# Scoring
weirdo score PEPTIDE...       # Score peptides
weirdo score -p human PEP     # Use specific preset
weirdo score --auto-download  # Auto-download if needed
```
