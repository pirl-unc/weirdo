# Getting Started

This guide will help you get started with WEIRDO for foreignness scoring of peptides.

## Installation

Install from PyPI:

```bash
pip install weirdo
```

Or install from source for development:

```bash
git clone https://github.com/pirl-unc/weirdo.git
cd weirdo
./develop.sh
```

## Downloading Reference Data

WEIRDO requires reference data (~2.5 GB compressed / ~7.5 GB uncompressed) for training. You can download it using the CLI or Python API.

### Option 1: CLI Setup (Recommended)

```bash
# Download data
weirdo setup

# Or just download data
weirdo data download
```

### Option 2: Python API

```python
from weirdo import get_data_manager

dm = get_data_manager()
dm.download('swissprot-8mers')  # Download reference data
dm.print_status()               # Show what's downloaded
```

### Managing Data

```bash
weirdo data list         # List datasets (or: ls, status)
weirdo data clear --all  # Clear all data
weirdo data path         # Show data directory
```

## Basic Usage

Train a model and score peptides:

```python
from weirdo.scorers import SwissProtReference, MLPScorer

categories = [
    'archaea', 'bacteria', 'fungi', 'human', 'invertebrates',
    'mammals', 'plants', 'rodents', 'vertebrates', 'viruses'
]

ref = SwissProtReference().load()
peptides, labels = ref.get_training_data(
    target_categories=categories,
    multi_label=True,
    max_samples=100000,  # Sample for faster training
    shuffle=True,
    seed=42,
)

scorer = MLPScorer(k=8, hidden_layer_sizes=(256, 128, 64))
scorer.train(peptides, labels, target_categories=categories, epochs=100)

# Foreignness scores in [0, 1]
scores = scorer.score(['MTMDKSEL', 'SIINFEKL'])

# Per-category probabilities + foreignness
df = scorer.predict_dataframe(['MTMDKSEL', 'SIINFEKL'])
```

Save and reload models:

```python
from weirdo import save_model, load_model, score_peptide

save_model(scorer, 'my-model')
loaded = load_model('my-model')
score = score_peptide('MTMDKSEL', model=loaded)
```

CLI scoring:

```bash
weirdo score --model my-model MTMDKSEL SIINFEKL
```

Download pretrained weights:

```bash
# Built-in named models (if configured in your installed version)
weirdo models available
weirdo models download MODEL_NAME

# Or a custom archive URL (GitHub release asset, etc.)
weirdo models download --url https://github.com/ORG/REPO/releases/download/vX.Y.Z/MODEL_NAME.tar.gz --save-as MODEL_NAME
```

## Understanding Scores

Foreignness scores are in [0, 1]:

| Score Range | Interpretation |
|-------------|----------------|
| 0.0 | Strongly self-like |
| 0.5 | Ambiguous / mixed |
| 1.0 | Strongly pathogen-like |

## Using Presets

Presets define model architecture defaults:

```python
from weirdo import get_available_presets, create_scorer

print(get_available_presets())
# ['default', 'fast']

scorer = create_scorer('fast')
scorer.train(peptides, labels, target_categories=categories, epochs=50)
```

## Custom Configuration

For more control, configure the scorer directly:

```python
from weirdo.scorers import MLPScorer

scorer = MLPScorer(
    k=8,
    hidden_layer_sizes=(128, 64),
    use_dipeptides=False,
)
scorer.train(peptides, labels, target_categories=categories, epochs=50)
```

## Aggregation Methods

When scoring long peptides, k-mer probabilities are aggregated:

| Method | Description |
|--------|-------------|
| `mean` | Average across k-mers (default) |
| `max` | Maximum (most pathogen-like) |
| `min` | Minimum (most self-like) |

```python
for agg in ['mean', 'max', 'min']:
    score = scorer.score(['MTMDKSELVQKAKLAE'], aggregate=agg)[0]
    print(f"{agg}: {score:.3f}")
```

## Next Steps

- Learn about [Foreignness Scorers](scorers.md) for detailed scorer documentation
- Explore [Amino Acid Data](amino-acids.md) for amino acid property data
- See the [API Reference](api.md) for complete API documentation
