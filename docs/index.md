# WEIRDO

**W**idely **E**stimated **I**mmunological **R**ecognition and **D**etection of **O**utliers

A Python library for computing metrics of immunological foreignness for candidate T-cell epitopes.

## Features

- **Foreignness Scoring**: Quantify how "foreign" a peptide is relative to a reference proteome
- **Amino Acid Properties**: Access physical/chemical properties of amino acids
- **Substitution Matrices**: Work with BLOSUM and PMBEC similarity matrices
- **Peptide Vectorization**: Convert peptides to numerical feature vectors

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

## How It Works

WEIRDO decomposes peptides into overlapping k-mers (default k=8) and computes a foreignness score based on:

1. **Frequency-based scoring**: How often each k-mer appears in the reference proteome
2. **Similarity-based scoring**: Minimum distance to reference k-mers using substitution matrices

```
Peptide: MTMDKSELVQKA
         ├────────┤     → MTMDKSEL (k-mer 1)
          ├────────┤    → TMDKSELV (k-mer 2)
           ├────────┤   → MDKSELVQ (k-mer 3)
            ├────────┤  → DKSELVQK (k-mer 4)
             ├────────┤ → KSELVQKA (k-mer 5)

Each k-mer is scored → Scores are aggregated (mean/max/min)
```

## License

Apache License 2.0
