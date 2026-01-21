# Amino Acid Data

WEIRDO provides extensive amino acid property data and substitution matrices
for peptide analysis.

## Amino Acid Alphabets

### Canonical Amino Acids (20)

```python
from weirdo import canonical_amino_acids, canonical_amino_acid_letters

# List of AminoAcid objects
for aa in canonical_amino_acids:
    print(f"{aa.letter}: {aa.full_name}")

# Single-letter codes
print(canonical_amino_acid_letters)
# ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
```

### Extended Amino Acids (24)

Includes rare amino acids (selenocysteine, pyrrolysine) and wildcards:

```python
from weirdo import extended_amino_acids, extended_amino_acid_letters

# Includes U (selenocysteine), O (pyrrolysine), X (unknown), B, Z, J
```

### Amino Acid Indexing

```python
from weirdo import amino_acid_letter_indices

# Get index for a letter
idx = amino_acid_letter_indices['A']  # 0
idx = amino_acid_letter_indices['V']  # 19
```

## Physical/Chemical Properties

Single residue properties are available as dictionaries:

```python
from weirdo.amino_acid_properties import (
    hydropathy,
    volume,
    polarity,
    pK_side_chain,
    prct_exposed_residues,
    hydrophilicity,
    accessible_surface_area,
    refractivity,
    local_flexibility,
    accessible_surface_area_folded,
    mass,
)

# Access properties by single-letter code
print(f"Alanine hydropathy: {hydropathy['A']}")
print(f"Tryptophan volume: {volume['W']}")
print(f"Lysine pK: {pK_side_chain['K']}")
```

### Available Properties

| Property | Description |
|----------|-------------|
| `hydropathy` | Kyte-Doolittle hydropathy index |
| `volume` | Residue volume (Å³) |
| `polarity` | Grantham polarity |
| `pK_side_chain` | Side chain pK values |
| `prct_exposed_residues` | Percent exposed residues |
| `hydrophilicity` | Hopp-Woods hydrophilicity |
| `accessible_surface_area` | Accessible surface area |
| `refractivity` | Residue refractivity |
| `local_flexibility` | Local flexibility parameter |
| `accessible_surface_area_folded` | ASA in folded state |
| `mass` | Residue mass (Da) |

### Chou-Fasman Secondary Structure

```python
from weirdo.chou_fasman import (
    alpha_helix_score,
    beta_sheet_score,
    turn_score,
)

# Propensity scores for secondary structure
print(f"Alanine helix propensity: {alpha_helix_score['A']}")
```

## Substitution Matrices

### BLOSUM Matrices

Block Substitution Matrices for sequence alignment scoring:

```python
from weirdo.blosum import (
    blosum30_dict,
    blosum50_dict,
    blosum62_dict,
    blosum30_matrix,
    blosum50_matrix,
    blosum62_matrix,
)

# Dictionary access
score = blosum62_dict['A']['V']  # Score for A→V substitution

# NumPy matrix access (faster for batch operations)
from weirdo import amino_acid_letter_indices
i = amino_acid_letter_indices['A']
j = amino_acid_letter_indices['V']
score = blosum62_matrix[i, j]
```

### PMBEC Matrix

Potential of Mean Force Energy Contact scores:

```python
from weirdo.pmbec import pmbec_dict, pmbec_matrix

# Dictionary access
score = pmbec_dict['A']['V']

# NumPy matrix access
score = pmbec_matrix[i, j]
```

### Contact Energy Matrices

Residue contact energies for different secondary structures:

```python
from weirdo.residue_contact_energies import (
    helix_vs_coil,
    helix_vs_strand,
    strand_vs_coil,
    coil_vs_helix,  # Transpose
    strand_vs_helix,  # Transpose
    coil_vs_strand,  # Transpose
)

# Interaction energy between residues in different structures
energy = helix_vs_coil['A']['G']
```

## Reduced Alphabets

Map amino acids to reduced alphabets for dimensionality reduction:

```python
from weirdo.reduced_alphabet import alphabets

# Available reduced alphabets
print(alphabets.keys())
# ['gbmr4', 'sdm12', 'hp2', 'murphy10', 'alex6', 'murphy15', 'murphy8', 'hp_vs_aromatic']

# Map amino acid to reduced alphabet
hp2 = alphabets['hp2']
print(hp2['A'])  # Maps to hydrophobic/hydrophilic group

# Convert sequence
sequence = 'ACDEFG'
reduced = ''.join(hp2[aa] for aa in sequence)
```

### Available Reduced Alphabets

| Alphabet | Groups | Description |
|----------|--------|-------------|
| `hp2` | 2 | Hydrophobic vs hydrophilic |
| `gbmr4` | 4 | GBMR 4-group clustering |
| `alex6` | 6 | Custom 6-group clustering |
| `murphy8` | 8 | Murphy 8-group clustering |
| `murphy10` | 10 | Murphy 10-group clustering |
| `sdm12` | 12 | SDM 12-group clustering |
| `murphy15` | 15 | Murphy 15-group clustering |

## Peptide Vectorizer

Convert peptides to numerical feature vectors:

```python
from weirdo import PeptideVectorizer

# Create vectorizer
vectorizer = PeptideVectorizer(
    max_ngram=2,          # Up to 2-mers (dimers)
    normalize_row=True,   # L1 normalization
    reduced_alphabet=None # Optional alphabet reduction
)

# Fit and transform sequences
sequences = ['ACDEFGHIK', 'KLMNPQRST', 'VWXYZZZZZ']
X = vectorizer.fit_transform(sequences)
print(X.shape)  # (3, num_features)

# Transform new sequences
X_new = vectorizer.transform(['NEWSEQUENCE'])
```

### With Reduced Alphabet

```python
from weirdo.reduced_alphabet import alphabets

vectorizer = PeptideVectorizer(
    max_ngram=3,
    reduced_alphabet=alphabets['murphy10']
)
X = vectorizer.fit_transform(sequences)
```
