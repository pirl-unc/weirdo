[![Tests](https://github.com/pirl-unc/weirdo/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/weirdo/actions/workflows/tests.yml)
[![Documentation](https://github.com/pirl-unc/weirdo/actions/workflows/docs.yml/badge.svg)](https://github.com/pirl-unc/weirdo/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/weirdo.svg)](https://pypi.python.org/pypi/weirdo/)
[![Python Version](https://img.shields.io/pypi/pyversions/weirdo.svg)](https://pypi.python.org/pypi/weirdo/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# WEIRDO

**W**idely **E**stimated **I**mmunological **R**ecognition and **D**etection of **O**utliers

A Python library for computing metrics of immunological foreignness for candidate T-cell epitopes.

## Quick Start

```python
# Simple API
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

**Higher scores = more foreign** (peptide k-mers are rare or absent in the reference proteome)

## Installation

```bash
pip install weirdo
```

For development:

```bash
git clone https://github.com/pirl-unc/weirdo.git
cd weirdo
./develop.sh
```

## Setup: Downloading Reference Data

WEIRDO requires reference data (~2.5 GB compressed, ~7.5 GB uncompressed) for scoring.

### Option 1: CLI Setup (Recommended)

```bash
# Complete setup: download data and build indices
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

### Data Management CLI

```bash
weirdo data list       # List datasets and indices (alias: ls, status)
weirdo data download   # Download reference data
weirdo data clear      # Delete downloaded data
weirdo data index      # Build/rebuild indices
weirdo data path       # Show data directory location
```

## Features

### Foreignness Scoring

Score peptides for immunological foreignness using multiple methods:

```python
from weirdo.scorers import FrequencyScorer, SimilarityScorer, SwissProtReference

# Load reference proteome (human proteins)
ref = SwissProtReference(categories=['human']).load()

# Frequency-based scoring
freq_scorer = FrequencyScorer(k=8, aggregate='mean').fit(ref)
scores = freq_scorer.score(['MTMDKSEL', 'XXXXXXXX'])

# Similarity-based scoring (using BLOSUM62)
sim_scorer = SimilarityScorer(k=8, matrix='blosum62').fit(ref)
scores = sim_scorer.score(['MTMDKSEL', 'XXXXXXXX'])
```

### Presets

Use built-in presets for common configurations:

```python
from weirdo import create_scorer, get_available_presets

# List available presets
print(get_available_presets())
# ['default', 'fast', 'human', 'pathogen', 'similarity_blosum62', 'similarity_pmbec']

# Create scorer with preset
scorer = create_scorer('human')
scores = scorer.score(['MTMDKSEL'])
```

### Amino Acid Properties

Access comprehensive amino acid property data:

```python
from weirdo.amino_acid_properties import hydropathy, volume, polarity
from weirdo.blosum import blosum62_dict
from weirdo.pmbec import pmbec_dict

# Single residue properties
print(f"Alanine hydropathy: {hydropathy['A']}")

# Substitution matrices
print(f"A→V substitution score: {blosum62_dict['A']['V']}")
```

### Peptide Vectorization

Convert peptides to numerical feature vectors:

```python
from weirdo import PeptideVectorizer

vectorizer = PeptideVectorizer(max_ngram=2, normalize_row=True)
X = vectorizer.fit_transform(['ACDEFGHIK', 'KLMNPQRST'])
```

## Architecture

WEIRDO uses an extensible plugin architecture:

```
weirdo/
├── scorers/           # Foreignness scoring system
│   ├── base.py        # BaseScorer, BatchScorer ABCs
│   ├── reference.py   # BaseReference, StreamingReference ABCs
│   ├── registry.py    # Plugin registration
│   ├── config.py      # Configuration and presets
│   ├── frequency.py   # FrequencyScorer
│   ├── similarity.py  # SimilarityScorer
│   └── swissprot.py   # SwissProtReference
├── api.py             # High-level convenience functions
├── amino_acid_*.py    # Amino acid data
├── blosum.py          # BLOSUM matrices
├── pmbec.py           # PMBEC matrix
└── peptide_vectorizer.py
```

### Adding Custom Scorers

```python
from weirdo.scorers import register_scorer, BaseScorer

@register_scorer('my_scorer', description='My custom scorer')
class MyScorer(BaseScorer):
    def fit(self, reference):
        self._reference = reference
        self._is_fitted = True
        return self

    def score(self, peptides):
        self._check_is_fitted()
        # Custom scoring logic
        pass
```

## Reference Data

WEIRDO uses pre-computed k-mer data from SwissProt:

- **10 organism categories**: human, mammals, bacteria, viruses, archaea, fungi, invertebrates, plants, rodents, vertebrates
- **~100M unique 8-mers** across all categories
- **Category filtering** for targeted analysis

## Development

```bash
# Install development dependencies
./develop.sh

# Run linting
./lint.sh

# Run tests
./test.sh

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
```

## Citation

If you use WEIRDO in your research, please cite:

```
@software{weirdo,
  title = {WEIRDO: Widely Estimated Immunological Recognition and Detection of Outliers},
  author = {OpenVax},
  url = {https://github.com/pirl-unc/weirdo}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## Related Projects

- [pepdata](https://github.com/openvax/pepdata) - Amino acid property data
- [mhctools](https://github.com/openvax/mhctools) - MHC binding prediction
- [vaxrank](https://github.com/openvax/vaxrank) - Personalized cancer vaccine design
