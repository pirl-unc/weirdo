# WEIRDO

**W**idely **E**stimated **I**mmunological **R**ecognition and **D**etection of **O**utliers

A Python library that predicts, for any peptide, the probability that it occurs in proteins of each of ten species categories, and summarizes them as a pathogen-versus-mammalian foreignness score.

## How It Works

WEIRDO trains a neural network on k-mer presence data from SwissProt to predict organism category membership, and ships a trained model (`swissprot-categories`):

```
Peptide → Feature Extraction (592 features) → MLP → 10 sigmoid probabilities → Pathogen vs self
             ↓                                              ↓                         ↓
    - AA properties (48)                  P(bacteria), P(viruses), P(human),   pathogen_max = max(P(bacteria), P(viruses))
    - Structural (27)                     P(rodents), P(mammals), P(fungi),    self_max     = max(P(human), P(rodents), P(mammals))
    - Composition (420)                   P(plants), P(archaea), ...           foreignness  = pathogen_max / (pathogen_max + self_max)
    - Sequence stats (12)
    - Reduced alphabets (80)
    - Dipeptide summary (5)
```

The outputs are close to calibrated probabilities but only weakly informative (mean AUC about 0.6
on protein families not seen in training); see [Species-category predictor](category-predictor.md).

```python
from weirdo import predict_categories

predict_categories(['SIINFEKL', 'NLVPMVATV', 'GILGFVFTL'])
```

## Quick Start

```python
from weirdo.scorers import SwissProtReference, MLPScorer

# Load reference data
ref = SwissProtReference().load()

# Get training data (8-mers with organism labels)
categories = ['human', 'viruses', 'bacteria', 'mammals', 'rodents',
              'vertebrates', 'invertebrates', 'archaea', 'fungi', 'plants']
peptides, labels = ref.get_training_data(
    target_categories=categories,
    multi_label=True,
    max_samples=200000  # Optional: sample for faster training
)

# Train MLP
scorer = MLPScorer(k=8, hidden_layer_sizes=(256, 128, 64))
scorer.train(peptides, labels, target_categories=categories, epochs=200)

# Score peptides
df = scorer.predict_dataframe(['MTMDKSEL', 'SIINFEKL', 'NLVPMVATV'])
print(df[['peptide', 'human', 'viruses', 'bacteria', 'pathogen_max', 'self_max', 'foreignness']])
```

## Installation

```bash
pip install weirdo

# Build reference data from UniProt (~670 MB download, then a ~8 GB 8-mer table)
weirdo data download
```

## Key Features

### Multi-Category Prediction

Predict the probability of a peptide appearing in each of 10 organism categories
(archaea, bacteria, fungi, human, invertebrates, mammals, plants, rodents, vertebrates, viruses).
The default foreignness groups are **pathogens** = bacteria, viruses and **self** = human,
rodents, mammals; both are configurable, and so is how the probabilities in a group are
combined (`combine='max'` by default, or `'sum'`). Archaea, fungi, invertebrates, plants and
vertebrates are in neither default group.

### Rich Feature Extraction

592 features capturing peptide properties:

| Feature Group | Count | Description |
|---------------|-------|-------------|
| AA Properties | 48 | Hydropathy, mass, volume, etc. (12 props × 4 stats) |
| Structural | 27 | Secondary structure, charge patterns, disorder |
| Composition | 420 | AA frequencies (20) + dipeptides (400) |
| Sequence Stats | 12 | Length, repeats, entropy, complexity |
| Reduced Alphabets | 80 | Grouped composition across reduced alphabets |
| Dipeptide Summary | 5 | Entropy, Gini, maxima, homodimers |

### Variable-Length Peptides

Score peptides of any length—long peptides are decomposed into overlapping k-mers and aggregated:

```python
df = scorer.predict_dataframe([
    'MTMDKSEL',           # 8-mer (single k-mer)
    'MTMDKSELVQKAKLAE',   # 16-mer (9 overlapping k-mers, averaged)
])
```

### Model Persistence

Save and load trained models:

```python
from weirdo import save_model, load_model

save_model(scorer, 'my-model')
scorer = load_model('my-model')
```

## Documentation

- [Getting Started](getting-started.md) - Installation and setup
- [Scorers](scorers.md) - MLPScorer and SwissProtReference
- [Amino Acids](amino-acids.md) - Property data and alphabets
- [API Reference](api.md) - Full API documentation

## Links

- [GitHub Repository](https://github.com/pirl-unc/weirdo)
- [PyPI Package](https://pypi.org/project/weirdo/)

## License

Apache License 2.0
