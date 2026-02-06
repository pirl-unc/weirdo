# Foreignness Scorers

WEIRDO provides multiple methods for scoring peptide foreignness, from simple k-mer lookup to trained neural networks.

## MLPScorer (Recommended)

A multi-layer perceptron trained on SwissProt k-mer data to predict organism category membership.

### Training Data

The training data comes from SwissProt, containing ~100M unique 8-mers with binary labels for 10 organism categories:

| Category | Description |
|----------|-------------|
| human | Homo sapiens |
| rodents | Mouse, rat |
| mammals | Other mammals (dog, cow, primates, etc.) |
| vertebrates | Fish, birds, reptiles, amphibians |
| invertebrates | Insects, worms, mollusks |
| bacteria | Bacterial proteins |
| viruses | Viral proteins |
| archaea | Archaeal proteins |
| fungi | Fungal proteins |
| plants | Plant proteins |

```python
from weirdo.scorers import SwissProtReference

ref = SwissProtReference().load()

# Get multi-label training data
peptides, labels = ref.get_training_data(
    target_categories=['human', 'viruses', 'bacteria', 'mammals'],
    multi_label=True,
    max_samples=100000  # Optional memory limit
)
# labels.shape = (100000, 4)
```

### Feature Extraction

The MLP extracts **592 features** from each peptide:

#### Amino Acid Properties (48 features)

12 physicochemical properties, each summarized with 4 statistics (mean, std, min, max):

| Property | Description |
|----------|-------------|
| hydropathy | Kyte-Doolittle scale |
| hydrophilicity | Hopp-Woods scale |
| mass | Molecular weight |
| volume | Residue volume |
| polarity | Grantham polarity |
| pK_side_chain | Side chain pKa |
| accessible_surface_area | Unfolded ASA |
| accessible_surface_area_folded | Folded ASA |
| local_flexibility | B-factor correlation |
| refractivity | Molar refractivity |
| solvent_exposed_area | Fraction exposed |
| prct_exposed_residues | % typically exposed |

#### Structural Features (27 features)

| Feature Group | Count | Features |
|---------------|-------|----------|
| Secondary structure | 12 | helix/sheet/turn propensity × 4 stats |
| Category fractions | 9 | positive_charged, negative_charged, hydrophobic, aromatic, aliphatic, polar_uncharged, tiny, small, cysteine |
| Charge features | 4 | net_charge, charge_transitions, max_cluster, arginine_ratio |
| Disorder features | 2 | disorder_promoting, order_promoting fractions |

**Viral-discriminating features:**
- `frac_cysteine`: Viruses often have more cysteines (disulfide bonds)
- `arginine_ratio` (R/(R+K)): Viruses often have depleted arginine
- `frac_disorder_promoting`: Viral proteins often more disordered

#### Composition Features (420 features)

| Feature | Count | Description |
|---------|-------|-------------|
| AA frequencies | 20 | Fraction of each amino acid |
| Dipeptide frequencies | 400 | Fraction of each AA pair (20×20) |

#### Sequence Statistics (12 features)

| Feature | Description |
|---------|-------------|
| seq_length | Peptide length |
| seq_log_length | log(1 + length) |
| seq_sqrt_length | sqrt(length) |
| frac_unknown | Fraction of non-canonical residues |
| unique_frac | Unique AA fraction (unique/20) |
| max_run_frac | Longest homopolymer run / length |
| repeat_frac | Adjacent repeats / (length-1) |
| entropy_aa | Normalized AA entropy |
| effective_aa | Effective alphabet size (normalized) |
| max_aa_freq | Most frequent AA fraction |
| top2_aa_freq | Sum of top-2 AA fractions |
| gini_aa | Gini impurity of AA distribution |

#### Reduced Alphabet Frequencies (80 features)

Composition across common reduced alphabets (Murphy, GBMR, SDM, etc.).
Each alphabet contributes one feature per reduced group.

#### Dipeptide Summary (5 features)

| Feature | Description |
|---------|-------------|
| dipep_entropy | Normalized dipeptide entropy |
| dipep_gini | Dipeptide Gini impurity |
| dipep_max_freq | Max dipeptide frequency |
| dipep_top2_freq | Sum of top-2 dipeptide frequencies |
| dipep_homodimer_frac | Sum of homodipeptide frequencies |

### Training

```python
from weirdo.scorers import SwissProtReference, MLPScorer

ref = SwissProtReference().load()

categories = [
    'archaea', 'bacteria', 'fungi', 'human', 'invertebrates',
    'mammals', 'plants', 'rodents', 'vertebrates', 'viruses'
]

peptides, labels = ref.get_training_data(
    target_categories=categories,
    multi_label=True
)

scorer = MLPScorer(
    k=8,
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    alpha=0.0001,
    early_stopping=True,
)

scorer.train(
    peptides, labels,
    target_categories=categories,
    epochs=200,
    learning_rate=0.001,
    verbose=True
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | int | 8 | K-mer window size for aggregating long peptides |
| `hidden_layer_sizes` | tuple | (256, 128, 64) | Hidden layer dimensions |
| `activation` | str | 'relu' | Activation: 'relu', 'tanh', 'logistic' |
| `alpha` | float | 0.0001 | L2 regularization strength |
| `early_stopping` | bool | True | Stop when validation loss plateaus |
| `use_dipeptides` | bool | True | Include dipeptide frequencies + summary stats |

### Prediction Methods

#### `predict_proba(peptides, aggregate='mean')` - Category Probabilities

Returns sigmoid-activated probabilities for each category:

```python
probs = scorer.predict_proba(['MTMDKSEL', 'SIINFEKL'], aggregate='mean')
# Shape: (2, n_categories)
# Values in [0, 1]
```

#### `foreignness(peptides)` - Foreignness Score

Computes: `max(pathogens) / (max(pathogens) + max(self))`

```python
scores = scorer.foreignness(
    ['MTMDKSEL', 'SIINFEKL'],
    pathogen_categories=['bacteria', 'viruses'],
    self_categories=['human', 'mammals', 'rodents']
)
# Values in [0, 1], higher = more foreign
```

#### `predict_dataframe(peptides)` - Full Results

Returns DataFrame with all category probabilities and foreignness:

```python
df = scorer.predict_dataframe([
    'MTMDKSEL',           # 8-mer
    'MTMDKSELVQKAKLAE',   # Variable length (aggregates k-mers)
    'SIINFEKL',
])
```

Output:
```
         peptide  human  viruses  bacteria  mammals  ...  foreignness
        MTMDKSEL   0.82     0.12      0.08     0.79  ...        0.127
MTMDKSELVQKAKLAE   0.78     0.15      0.10     0.75  ...        0.161
        SIINFEKL   0.15     0.73      0.21     0.18  ...        0.802
```

Options:
- `aggregate='mean'|'max'|'min'`: How to combine k-mer scores for long peptides
- `pathogen_categories`: Categories for foreignness numerator
- `self_categories`: Categories for foreignness denominator

#### `features_dataframe(peptides)` - Feature Extraction

Returns DataFrame with all 592 features:

```python
df = scorer.features_dataframe(['MTMDKSEL', 'SIINFEKL'])
# Shape: (2, 593) - 592 features + peptide column

# Get feature names
names = scorer.get_feature_names()
```

### Model Persistence

```python
from weirdo import save_model, load_model, list_models

# Save
save_model(scorer, 'my-model')

# List
for m in list_models():
    print(f"{m.name}: {m.scorer_type}")

# Load
scorer = load_model('my-model')

# Models stored in ~/.weirdo/models/
```

---

## SwissProtReference

Reference dataset providing k-mer presence across organism categories.

```python
from weirdo.scorers import SwissProtReference

# Load all categories
ref = SwissProtReference().load()

# Load specific categories
ref = SwissProtReference(categories=['human', 'viruses']).load()

# Memory-efficient modes
ref = SwissProtReference(use_set=True).load()   # Presence only
ref = SwissProtReference(lazy=True).load()      # Stream from disk
```

### Querying

```python
# Check if k-mer exists
if 'MTMDKSEL' in ref:
    print("Found!")

# Get category breakdown
cats = ref.get_kmer_categories('MTMDKSEL')
# {'human': True, 'bacteria': False, 'viruses': False, ...}

# Iterate over k-mers
for kmer, categories in ref.iter_kmers_with_categories():
    process(kmer, categories)
```

### Training Data Generation

```python
# Single-label (foreignness to human)
peptides, labels = ref.get_training_data(target_categories=['human'])
# labels: 0 if in human, 1 if not

# Multi-label (all categories)
peptides, labels = ref.get_training_data(
    target_categories=['human', 'viruses', 'bacteria', 'mammals'],
    multi_label=True
)
# labels.shape: (n_kmers, 4)
```

---

## SimilarityScorer

Reference-based scorer using substitution matrices (BLOSUM/PMBEC) to measure
distance from reference k-mers.

```python
from weirdo.scorers import SimilarityScorer, SwissProtReference

ref = SwissProtReference(categories=['human']).load()
scorer = SimilarityScorer(matrix='blosum62', distance_metric='min_distance')
scorer.fit(ref)
scores = scorer.score(['MTMDKSEL', 'SIINFEKL'])
```

Key parameters:

- `matrix`: `blosum30`, `blosum50`, `blosum62`, `pmbec`
- `distance_metric`: `min_distance`, `mean_distance`, `max_similarity`
- `aggregate`: `mean`, `max`, `min`, `sum`

---

## CLI Commands

```bash
# Data management
weirdo data download    # Download SwissProt reference
weirdo data list        # Show data status
weirdo data path        # Show data directory

# Model management
weirdo models list      # List trained models
weirdo models train --data train.csv --name my-model
weirdo models info my-model
weirdo models delete my-model
```
