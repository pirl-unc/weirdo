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

The bundled model needs no data. Training your own model and the SwissProt reference scorers need
reference data, which WEIRDO builds from UniProtKB/Swiss-Prot itself: it downloads the taxonomic-division
files from UniProt's FTP site (about 670 MB, with [datacache](https://github.com/openvax/datacache);
transfers are retried and resumed, and the sizes and MD5s UniProt lists are checked), builds a
table of the ~570,000 protein sequences, and from it the 8-mer reference table (~100M k-mers,
about 8 GB on disk, a few minutes). The files stay in `~/.weirdo/downloads/uniprot/release-<version>/`,
so each UniProt release is downloaded once.

### Option 1: CLI Setup (Recommended)

```bash
# Sequences only (~670 MB download)
weirdo data download swissprot-sequences

# Sequences and the 8-mer table (the default dataset)
weirdo data download
weirdo setup            # the same, with a status summary
```

Set `WEIRDO_UNIPROT_URL` to use a mirror, for example
`https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/taxonomic_divisions`.
UniProt serves the division files of its current release only, so the data follow UniProt's
releases; the release a table was built from is recorded in `~/.weirdo/metadata.json`.

### Option 2: Python API

```python
from weirdo import get_data_manager

dm = get_data_manager()
dm.download('swissprot-8mers')  # Download SwissProt from UniProt and build the table
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

# Foreignness scores in [0, 1]: max(pathogen) / (max(pathogen) + max(self))
# (combine='sum' compares summed probabilities instead)
scores = scorer.score(['MTMDKSEL', 'SIINFEKL'])

# Per-category probabilities + pathogen/self values + foreignness
df = scorer.predict_dataframe(['MTMDKSEL', 'SIINFEKL'])
```

Or skip training and use the bundled model, which needs no download:

```python
from weirdo import predict_categories

predict_categories(['MTMDKSEL', 'SIINFEKL'])
```

See [Species-category predictor](category-predictor.md) for what the outputs mean and how the
bundled model was evaluated.

Save and reload models:

```python
from weirdo import save_model, load_model, score_peptide

save_model(scorer, 'my-model')
loaded = load_model('my-model')
score = score_peptide('MTMDKSEL', model=loaded)
```

CLI scoring:

```bash
weirdo score MTMDKSEL SIINFEKL                # bundled swissprot-categories model
weirdo score --combine sum MTMDKSEL SIINFEKL  # compare summed probabilities instead of maxima
weirdo score --model my-model MTMDKSEL SIINFEKL
weirdo score MTMDKSEL --csv > scores.csv      # CSV only, no extra text
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

Foreignness scores are in [0, 1] and compare how strongly the pathogen categories (bacteria,
viruses) are predicted with how strongly the mammalian categories (human, rodents, mammals) are
predicted: `max(pathogen) / (max(pathogen) + max(self))` by default, or with `combine='sum'` the
expected number of categories in each group that contain the peptide:

| Score Range | Interpretation |
|-------------|----------------|
| 0.0 | Only self categories are predicted |
| 0.5 | Equal strength (also returned when both are zero) |
| 1.0 | Only pathogen categories are predicted |

With the bundled model these are weak signals: the underlying category probabilities have a mean
AUC of about 0.6 on unseen protein families, and a typical peptide scores about 0.81 (peptides found
only in mammalian proteins: median 0.76; only in bacteria/viruses: 0.84; with `combine='sum'`: 0.67,
0.60 and 0.71). Compare a score with those reference distributions in
[Species-category predictor](category-predictor.md), not with 0.5, and do not read the order of single
peptides: the score separates groups of peptides, not individual ones.

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
