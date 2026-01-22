# Foreignness Scorers

WEIRDO provides an extensible architecture for foreignness scoring with multiple
scorer implementations and reference datasets.

## Architecture Overview

The scoring system uses three main components:

1. **References**: Load and query k-mer data from protein databases
2. **Scorers**: Compute foreignness scores using reference data
3. **Registry**: Plugin system for adding new implementations

```
Reference (SwissProt) → Scorer (Frequency/Similarity) → Scores
```

## FrequencyScorer

Scores peptides based on k-mer frequency in the reference dataset.

### How it works

1. Decompose peptide into overlapping k-mers
2. Look up each k-mer's frequency in the reference
3. Compute `-log10(frequency + pseudocount)` for each k-mer
4. Aggregate k-mer scores using specified method

```python
from weirdo.scorers import FrequencyScorer, SwissProtReference

ref = SwissProtReference(categories=['human']).load()
scorer = FrequencyScorer(
    k=8,
    pseudocount=1e-10,
    aggregate='mean'
).fit(ref)

scores = scorer.score(['MTMDKSEL', 'XXXXXXXX'])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 8 | K-mer size |
| `pseudocount` | float | 1e-10 | Added to frequencies to avoid log(0) |
| `aggregate` | str | 'mean' | Aggregation method ('mean', 'max', 'min', 'sum') |

### Inspecting K-mer Scores

```python
# Get individual k-mer contributions
kmer_scores = scorer.get_kmer_scores('MTMDKSELVQKAKLAE')
for kmer, score in kmer_scores:
    print(f"  {kmer}: {score:.3f}")
```

## MLPScorer

Trainable MLP (multi-layer perceptron) for learning foreignness from labeled data.
Uses one-hot encoded amino acid features and scikit-learn's MLPRegressor.

### How it works

1. Convert peptides to one-hot encoded features (k positions × 21 amino acids)
2. Train an MLP regressor on labeled peptide data
3. Predict foreignness scores for new peptides

```python
from weirdo.scorers import MLPScorer

# Create and train
scorer = MLPScorer(
    k=8,
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
)
scorer.train(peptides, labels, epochs=200)

# Score new peptides
scores = scorer.score(['MTMDKSEL', 'XXXXXXXX'])

# Save and load
scorer.save('my_model')
loaded = MLPScorer.load('my_model')
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 8 | K-mer size |
| `hidden_layer_sizes` | tuple | (256, 128, 64) | Sizes of hidden layers |
| `activation` | str | 'relu' | Activation function ('relu', 'tanh', 'logistic') |
| `alpha` | float | 0.0001 | L2 regularization strength |
| `early_stopping` | bool | True | Use early stopping with validation split |

### Training Parameters

```python
scorer.train(
    peptides=peptides,       # Training sequences
    labels=labels,           # Target foreignness scores
    epochs=200,              # Maximum iterations
    learning_rate=0.001,     # Initial learning rate
    verbose=True,            # Print progress
)
```

### Model Management

```python
from weirdo import list_models, load_model, save_model

# Save a trained model
save_model(scorer, 'my-foreignness-model')

# List available models
for model in list_models():
    print(f"{model.name}: {model.scorer_type}")

# Load a model
scorer = load_model('my-foreignness-model')
```

### CLI Training

```bash
# Train from CSV (columns: peptide, label)
weirdo models train --data train.csv --name my-model --hidden-layers 256,128

# List trained models
weirdo models list

# Show model info
weirdo models info my-model
```

## SimilarityScorer

Scores peptides based on minimum distance to reference k-mers using
substitution matrices (BLOSUM, PMBEC).

### How it works

1. Decompose peptide into overlapping k-mers
2. For each k-mer, find the most similar reference k-mer
3. Compute distance based on substitution matrix
4. Aggregate distances across k-mers

```python
from weirdo.scorers import SimilarityScorer, SwissProtReference

ref = SwissProtReference(categories=['human']).load()
scorer = SimilarityScorer(
    k=8,
    matrix='blosum62',
    max_candidates=1000,
    distance_metric='min_distance'
).fit(ref)

scores = scorer.score(['MTMDKSEL', 'XXXXXXXX'])
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 8 | K-mer size |
| `matrix` | str | 'blosum62' | Substitution matrix ('blosum30', 'blosum50', 'blosum62', 'pmbec') |
| `max_candidates` | int | 1000 | Maximum reference k-mers to compare |
| `distance_metric` | str | 'min_distance' | Distance computation method |
| `aggregate` | str | 'mean' | Aggregation method |

### Finding Closest Matches

```python
# Find closest reference k-mers
matches = scorer.get_closest_reference('MTMDKSEL', n=5)
for ref_kmer, distance in matches:
    print(f"  {ref_kmer}: distance={distance:.3f}")
```

## SwissProtReference

Reference dataset from the SwissProt protein database, containing pre-computed
k-mer presence across organism categories.

### Categories

- archaea, bacteria, fungi, human, invertebrates
- mammals, plants, rodents, vertebrates, viruses

```python
from weirdo.scorers import SwissProtReference

# Load all categories
ref = SwissProtReference().load()

# Load specific categories
ref = SwissProtReference(categories=['human', 'mammals']).load()

# Memory-efficient mode (presence only, no category info)
ref = SwissProtReference(use_set=True).load()

# Lazy mode (stream from disk, low memory)
ref = SwissProtReference(lazy=True).load()
```

### Querying the Reference

```python
# Check if k-mer exists
if 'MTMDKSEL' in ref:
    print("K-mer found!")

# Get category breakdown
cats = ref.get_kmer_categories('MTMDKSEL')
print(f"Human: {cats['human']}, Bacteria: {cats['bacteria']}")

# Iterate over all k-mers
for kmer in ref.iter_kmers():
    process(kmer)
```

## Configuration System

Use `ScorerConfig` for reproducible configurations:

```python
from weirdo.scorers import ScorerConfig

# From preset
config = ScorerConfig.from_preset('default')

# From dictionary
config = ScorerConfig.from_dict({
    'scorer': 'frequency',
    'reference': 'swissprot',
    'k': 8,
    'scorer_params': {'aggregate': 'max'},
    'reference_params': {'categories': ['human']},
})

# From YAML file
config = ScorerConfig.from_yaml('my_config.yaml')

# Build scorer
scorer = config.build()
```

## Adding Custom Scorers

Register new scorers using the decorator:

```python
from weirdo.scorers import register_scorer, BaseScorer
import numpy as np

@register_scorer('my_scorer', description='My custom scorer')
class MyScorer(BaseScorer):
    def __init__(self, k=8, my_param=1.0, **kwargs):
        super().__init__(**kwargs)
        self._params.update({'k': k, 'my_param': my_param})

    def fit(self, reference):
        self._reference = reference
        self._is_fitted = True
        return self

    def score(self, peptides):
        self._check_is_fitted()
        peptides = self._ensure_list(peptides)
        # Custom scoring logic
        return np.array([self._score_one(p) for p in peptides])

    def _score_one(self, peptide):
        # Your implementation here
        pass

# Now available via registry
from weirdo.scorers import create_scorer
scorer = create_scorer('my_scorer', k=8, my_param=2.0)
```

## Performance Tips

### 1. Use category filters to reduce reference size

```python
ref = SwissProtReference(categories=['human']).load()  # Faster
```

### 2. Use set mode when you don't need frequency information

```python
ref = SwissProtReference(use_set=True).load()  # Less memory
```

### 3. Use batch scoring for large datasets

```python
scorer = FrequencyScorer(batch_size=10000).fit(ref)
scores = scorer.score_batch(large_peptide_list, show_progress=True)
```

### 4. Cache scorers for repeated use

```python
from weirdo import create_scorer
scorer = create_scorer('human', cache=True)  # Cached by default
```
