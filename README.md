[![Tests](https://github.com/pirl-unc/weirdo/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/weirdo/actions/workflows/tests.yml)
[![Documentation](https://github.com/pirl-unc/weirdo/actions/workflows/docs.yml/badge.svg)](https://github.com/pirl-unc/weirdo/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/weirdo.svg)](https://pypi.python.org/pypi/weirdo/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# WEIRDO

**W**idely **E**stimated **I**mmunological **R**ecognition and **D**etection of **O**utliers

A Python library for computing peptide foreignness scores—predicting whether a peptide sequence is likely from a pathogen (bacteria, virus) or from self (human, mammalian).

## Overview

WEIRDO trains a multi-layer perceptron (MLP) on k-mer presence data from SwissProt to predict organism category membership. Given any peptide, it outputs:

- **Category probabilities**: likelihood of appearing in human, bacteria, viruses, mammals, etc.
- **Foreignness score**: `max(pathogens) / (max(pathogens) + max(self))`

## Quick Start

```python
from weirdo.scorers import SwissProtReference, MLPScorer

# Load reference data (SwissProt 8-mers with organism labels)
ref = SwissProtReference().load()

# Define organism categories
categories = [
    'archaea', 'bacteria', 'fungi', 'human', 'invertebrates',
    'mammals', 'plants', 'rodents', 'vertebrates', 'viruses'
]

# Get training data: each 8-mer labeled with organism presence
peptides, labels = ref.get_training_data(
    target_categories=categories,
    multi_label=True,
    max_samples=200000  # Optional: sample for faster training
)

# Train the MLP
scorer = MLPScorer(k=8, hidden_layer_sizes=(256, 128, 64))
scorer.train(peptides, labels, target_categories=categories, epochs=200)

# Score new peptides (any length)
df = scorer.predict_dataframe(['MTMDKSEL', 'SIINFEKL', 'NLVPMVATV'])
print(df)
```

**Output:**
```
    peptide  human  viruses  bacteria  mammals  ...  foreignness
   MTMDKSEL   0.82     0.12      0.08     0.79  ...        0.127
   SIINFEKL   0.15     0.73      0.21     0.18  ...        0.802
  NLVPMVATV   0.31     0.68      0.15     0.35  ...        0.660
```

## Installation

```bash
pip install weirdo
```

Download reference data (~2.5 GB compressed / ~7.5 GB uncompressed) for training:
```bash
weirdo data download
```

## Training Data

WEIRDO uses pre-computed 8-mer data from SwissProt (~100M unique k-mers):

| Category | Description |
|----------|-------------|
| human | Homo sapiens proteins |
| rodents | Mouse, rat proteins |
| mammals | Other mammals (dog, cow, primates, etc.) |
| vertebrates | Fish, birds, reptiles, amphibians |
| invertebrates | Insects, worms, mollusks |
| bacteria | Bacterial proteins |
| viruses | Viral proteins |
| archaea | Archaeal proteins |
| fungi | Fungal proteins |
| plants | Plant proteins |

Each 8-mer has True/False labels for each category, indicating whether it appears in proteins from that organism group.

## Feature Extraction

The MLP uses **495 features** extracted from each peptide:

### Amino Acid Properties (48 features)
12 physicochemical properties × 4 statistics (mean, std, min, max):
- Hydropathy, hydrophilicity
- Mass, volume
- Polarity, pK side chain
- Accessible surface area (folded/unfolded)
- Local flexibility, refractivity
- Solvent exposed area, % exposed residues

### Structural Features (27 features)
- **Secondary structure propensities** (12): helix, sheet, turn × 4 stats
- **Category fractions** (9): positive/negative charged, hydrophobic, aromatic, aliphatic, polar, tiny, small, cysteine
- **Charge features** (4): net charge, charge transitions, max cluster, R/(R+K) ratio
- **Disorder features** (2): disorder/order promoting fractions

### Composition Features (420 features)
- **Amino acid frequencies** (20): fraction of each amino acid
- **Dipeptide frequencies** (400): fraction of each amino acid pair

## API Reference

### Training

```python
from weirdo.scorers import SwissProtReference, MLPScorer

# Load reference
ref = SwissProtReference().load()

# Get training data
peptides, labels = ref.get_training_data(
    target_categories=['human', 'viruses', 'bacteria', 'mammals'],
    multi_label=True,
    max_samples=100000  # Optional: limit for memory
)

# Train
scorer = MLPScorer(
    k=8,
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    alpha=0.0001,  # L2 regularization
)
scorer.train(
    peptides, labels,
    target_categories=['human', 'viruses', 'bacteria', 'mammals'],
    epochs=200,
    learning_rate=0.001
)
```

### Prediction

```python
# Category probabilities (sigmoid-activated)
probs = scorer.predict_proba(['MTMDKSEL'])
# Shape: (1, n_categories)

# Foreignness score
foreign = scorer.foreignness(
    ['MTMDKSEL'],
    pathogen_categories=['bacteria', 'viruses'],
    self_categories=['human', 'mammals', 'rodents']
)
# Returns: max(pathogens) / (max(pathogens) + max(self))

# Full DataFrame output (handles variable-length peptides)
df = scorer.predict_dataframe(['MTMDKSEL', 'SIINFEKL', 'NLVPMVATV'])
```

### Feature Extraction

```python
# Extract features as DataFrame
df = scorer.features_dataframe(['MTMDKSEL', 'SIINFEKL'])
# Shape: (2, 496) - 495 features + peptide column

# Feature names
names = scorer.get_feature_names()
# ['hydropathy_mean', 'hydropathy_std', ..., 'dipep_YY']
```

### Model Persistence

```python
from weirdo import save_model, load_model, list_models

# Save trained model
save_model(scorer, 'my-foreignness-model')

# List saved models
for model in list_models():
    print(f"{model.name}: {model.scorer_type}")

# Load model
scorer = load_model('my-foreignness-model')
```

### CLI

```bash
# Data management
weirdo data download        # Download SwissProt reference
weirdo data list            # Show data status

# Model management
weirdo models list          # List trained models
weirdo models train --data train.csv --name my-model
weirdo models info my-model # Show model details

# Scoring
weirdo score --model my-model MTMDKSEL SIINFEKL
```

## Architecture

```
weirdo/
├── scorers/
│   ├── mlp.py          # MLPScorer with feature extraction
│   ├── swissprot.py    # SwissProtReference (training data)
│   ├── config.py       # Presets and configuration
│   ├── registry.py     # Scorer registry
│   └── trainable.py    # TrainableScorer base class
├── model_manager.py    # Save/load trained models
├── amino_acid_properties.py  # 12 AA property dictionaries
└── api.py              # High-level functions
```

## Citation

```bibtex
@software{weirdo,
  title = {WEIRDO: Widely Estimated Immunological Recognition and Detection of Outliers},
  author = {PIRL-UNC},
  url = {https://github.com/pirl-unc/weirdo}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
