# WEIRDO

**W**idely **E**stimated **I**mmunological **R**ecognition and **D**etection of **O**utliers

A Python library for computing peptide foreignness scores—predicting whether a peptide sequence is likely from a pathogen (bacteria, virus) or from self (human, mammalian).

## How It Works

WEIRDO trains a neural network on k-mer presence data from SwissProt to predict organism category membership:

```
Peptide → Feature Extraction (495 features) → MLP → Category Probabilities → Foreignness Score
             ↓                                              ↓
    - AA properties (48)                        human: 0.82
    - Structural (27)                          viruses: 0.12
    - Composition (420)                       bacteria: 0.08
                                                    ↓
                                    foreignness = max(pathogens) / (max(pathogens) + max(self))
                                                = 0.12 / (0.12 + 0.82) = 0.13
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
print(df[['peptide', 'human', 'viruses', 'bacteria', 'foreignness']])
```

## Installation

```bash
pip install weirdo

# Download reference data (~2.5 GB compressed / ~7.5 GB uncompressed)
weirdo data download
```

## Key Features

### Multi-Category Prediction

Predict probability of peptide appearing in 10 organism categories:

- **Self**: human, rodents, mammals, vertebrates, invertebrates
- **Pathogens**: bacteria, viruses, archaea, fungi
- **Other**: plants

### Rich Feature Extraction

495 features capturing peptide properties:

| Feature Group | Count | Description |
|---------------|-------|-------------|
| AA Properties | 48 | Hydropathy, mass, volume, etc. (12 props × 4 stats) |
| Structural | 27 | Secondary structure, charge patterns, disorder |
| Composition | 420 | AA frequencies (20) + dipeptides (400) |

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
