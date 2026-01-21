# Getting Started

This guide will help you get started with WEIRDO for foreignness scoring of peptides.

## Installation

Install from PyPI:

```bash
pip install weirdo
```

Or install from source for development:

```bash
git clone https://github.com/openvax/weirdo.git
cd weirdo
./develop.sh
```

## Downloading Reference Data

WEIRDO requires reference data (~7.5 GB) for scoring. You can download it using the CLI or Python API.

### Option 1: CLI Setup (Recommended)

```bash
# Download data and build indices
weirdo setup

# Or just download data
weirdo data download
```

### Option 2: Auto-download on First Use

```python
from weirdo import score_peptide

# Automatically download data if not present
score = score_peptide('MTMDKSEL', auto_download=True)
```

### Option 3: Python API

```python
from weirdo import get_data_manager

dm = get_data_manager()
dm.download('swissprot-8mers')  # Download reference data
dm.print_status()               # Show what's downloaded
```

### Managing Data

```bash
weirdo data list         # List datasets and indices (or: ls, status)
weirdo data clear --all  # Clear all data
weirdo data path         # Show data directory
```

## Basic Usage

The simplest way to score peptides is using the high-level API:

```python
from weirdo import score_peptide, score_peptides

# Score a single peptide
score = score_peptide('MTMDKSEL')
print(f"Foreignness score: {score:.3f}")

# Score multiple peptides
peptides = ['MTMDKSEL', 'ACDEFGHI', 'XXXXXXXX']
scores = score_peptides(peptides)
for pep, score in zip(peptides, scores):
    print(f"{pep}: {score:.3f}")
```

## Understanding Scores

Higher scores indicate more "foreign" peptides:

| Score Range | Interpretation |
|-------------|----------------|
| ~0 | K-mers commonly found in reference proteome |
| ~5 | Mix of known and unknown k-mers |
| ~10 | K-mers rare or absent from reference |

## Using Presets

WEIRDO provides several preset configurations:

```python
from weirdo import get_available_presets, create_scorer

# List available presets
print(get_available_presets())
# ['default', 'fast', 'human', 'pathogen', 'similarity_blosum62', 'similarity_pmbec']

# Create scorer with specific preset
scorer = create_scorer('human')  # Uses human-only reference
scores = scorer.score(['MTMDKSEL'])
```

### Available Presets

| Preset | Description |
|--------|-------------|
| `default` | Frequency-based scoring against all SwissProt categories |
| `human` | Frequency-based scoring against human proteins only |
| `pathogen` | Frequency-based scoring against bacteria and viruses |
| `similarity_blosum62` | Similarity-based scoring using BLOSUM62 matrix |
| `similarity_pmbec` | Similarity-based scoring using PMBEC matrix |
| `fast` | Memory-efficient mode (presence only, no frequencies) |

## Custom Configuration

For more control, use the scorer classes directly:

```python
from weirdo.scorers import FrequencyScorer, SwissProtReference

# Load reference with specific categories
ref = SwissProtReference(
    categories=['human', 'mammals'],
    use_set=False  # Keep frequency information
).load()

# Configure scorer
scorer = FrequencyScorer(
    k=8,              # K-mer size
    pseudocount=1e-10,  # For log computation
    aggregate='mean'    # How to combine k-mer scores
).fit(ref)

# Score peptides
scores = scorer.score(['MTMDKSEL', 'XXXXXXXX'])
```

## Aggregation Methods

The `aggregate` parameter controls how k-mer scores are combined:

| Method | Description |
|--------|-------------|
| `mean` | Average score across all k-mers (default) |
| `max` | Maximum (most foreign) k-mer score |
| `min` | Minimum (least foreign) k-mer score |
| `sum` | Sum of all k-mer scores |

```python
# Compare aggregation methods
for agg in ['mean', 'max', 'min']:
    scorer = FrequencyScorer(aggregate=agg).fit(ref)
    score = scorer.score(['MTMDKSELVQKAKLAE'])[0]
    print(f"{agg}: {score:.3f}")
```

## Next Steps

- Learn about [Foreignness Scorers](scorers.md) for detailed scorer documentation
- Explore [Amino Acid Data](amino-acids.md) for amino acid property data
- See the [API Reference](api.md) for complete API documentation
