[![Tests](https://github.com/pirl-unc/weirdo/actions/workflows/tests.yml/badge.svg)](https://github.com/pirl-unc/weirdo/actions/workflows/tests.yml)
[![Documentation](https://github.com/pirl-unc/weirdo/actions/workflows/docs.yml/badge.svg)](https://github.com/pirl-unc/weirdo/actions/workflows/docs.yml)
[![PyPI](https://img.shields.io/pypi/v/weirdo.svg)](https://pypi.python.org/pypi/weirdo/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# WEIRDO

**W**idely **E**stimated **I**mmunological **R**ecognition and **D**etection of **O**utliers

A Python library that predicts, for any peptide, the probability that it occurs in proteins of each of ten species categories, and summarizes them as a pathogen-versus-mammalian foreignness score.

## Overview

WEIRDO trains a multi-layer perceptron (MLP) on k-mer presence data from SwissProt to predict organism category membership, and ships a trained model, `swissprot-categories`. Given any peptide of 8 residues or more, it outputs:

- **Category probabilities**: one independent sigmoid output for each of ten categories (archaea, bacteria, fungi, human, invertebrates, mammals, plants, rodents, vertebrates, viruses), the probability that the peptide occurs in SwissProt proteins of that category. The outputs are close to calibrated but only weakly informative: on protein families not seen in training the mean AUC is about 0.6.
- **Pathogen and self values**: the largest probability among the pathogen categories (bacteria, viruses) and among the mammalian categories (human, rodents, mammals).
- **Foreignness score**: `max(pathogen) / (max(pathogen) + max(self))`. The combination is a parameter: `combine="sum"` compares the summed probabilities instead (the expected number of categories in each group that contain the peptide), and `combine` also accepts your own function. Five categories (archaea, fungi, invertebrates, plants, vertebrates) are in neither default group; pass `pathogen_categories` / `self_categories` to change the groups.

[docs/category-predictor.md](docs/category-predictor.md) explains what the numbers mean, how the model was evaluated and what its limits are. A typical peptide scores about 0.81 on the foreignness scale (peptides found only in mammalian proteins: median 0.76; only in bacteria/viruses: 0.84), so compare a score with those reference distributions rather than with 0.5. Nothing here has been shown to predict T-cell immunogenicity. The bundled weights derive from UniProtKB/Swiss-Prot sequences (CC BY 4.0).

## Quick Start

```python
from weirdo import predict_categories

predict_categories(['SIINFEKL', 'NLVPMVATV', 'GILGFVFTL'])
```

```
  peptide  archaea  bacteria  fungi  human  invertebrates  mammals  plants  rodents  vertebrates  viruses  pathogen_max  self_max  foreignness
 SIINFEKL    0.077     0.389  0.141  0.044          0.158    0.025   0.102    0.048        0.026    0.057         0.389     0.048        0.891
NLVPMVATV    0.026     0.461  0.100  0.092          0.069    0.047   0.116    0.106        0.040    0.027         0.461     0.106        0.813
GILGFVFTL    0.045     0.435  0.118  0.083          0.076    0.041   0.112    0.093        0.038    0.031         0.435     0.093        0.823
```

`predict_categories(peptides, combine="sum")` gives `pathogen_sum` and `self_sum` columns instead.

Individual outputs stay close to the base rates (the two viral epitopes get a virus probability of about 0.03); see [docs/category-predictor.md](docs/category-predictor.md).

```bash
weirdo score SIINFEKL NLVPMVATV GILGFVFTL
```

### Training your own model

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

# Train the MLP: binary multi-label targets use one sigmoid per category and
# binary cross-entropy, so predict_proba returns probabilities.
scorer = MLPScorer(k=8, hidden_layer_sizes=(256, 128, 64))
scorer.train(peptides, labels, target_categories=categories, epochs=200)

# Score new peptides (any length)
df = scorer.predict_dataframe(['MTMDKSEL', 'SIINFEKL', 'NLVPMVATV'])
```

Early stopping needs validation peptides that are not near-duplicates of the training ones (overlapping 8-mers of one protein share 7 residues); pass `val_peptides` / `val_labels` built from different protein families. `scripts/train_category_predictor.py` does this and was used to train the bundled model.

## Installation

```bash
pip install weirdo
```

The bundled model needs no extra data. To train your own model or use the SwissProt reference
scorers, build the reference data from UniProt:
```bash
weirdo data download swissprot-sequences   # ~670 MB from UniProt, builds a ~230 MB table
weirdo data download                       # also builds the 8-mer table (several GB, some minutes)
```
The files are fetched with [datacache](https://github.com/openvax/datacache) from UniProt's FTP site
(`WEIRDO_UNIPROT_URL` selects a mirror), checked against the sizes and MD5s UniProt lists and kept
in `~/.weirdo/downloads/uniprot/release-<version>/`, so they are downloaded once per UniProt release.

## Training Data

WEIRDO builds its 8-mer data from UniProtKB/Swiss-Prot (~100M unique k-mers) with `weirdo data download`:

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

The MLP uses **592 features** extracted from each peptide:

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

### Sequence Statistics (12 features)
- Length, log-length, sqrt-length
- Unknown fraction, unique AA fraction
- Max run length, repeat fraction
- Entropy/complexity (entropy, effective AAs, Gini, top-2/maximum frequency)

### Reduced Alphabet Frequencies (80 features)
- Composition across common reduced alphabets (Murphy, GBMR, SDM, etc.)

### Dipeptide Summary (5 features)
- Entropy, Gini, max/top2 frequency, homodipeptide fraction

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
# Category probabilities, one independent sigmoid per category
probs = scorer.predict_proba(['MTMDKSEL'])
# Shape: (1, n_categories)

# Foreignness score
foreign = scorer.foreignness(
    ['MTMDKSEL'],
    pathogen_categories=['bacteria', 'viruses'],
    self_categories=['human', 'mammals', 'rodents'],
    combine='max',  # default: max(pathogens) / (max(pathogens) + max(self)); 'sum' adds the probabilities
)

# Full DataFrame output (handles variable-length peptides)
df = scorer.predict_dataframe(['MTMDKSEL', 'SIINFEKL', 'NLVPMVATV'])
```

### Feature Extraction

```python
# Extract features as DataFrame
df = scorer.features_dataframe(['MTMDKSEL', 'SIINFEKL'])
# Shape: (2, 593) - 592 features + peptide column

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
weirdo data download        # Download SwissProt from UniProt and build the reference data
weirdo data list            # Show data status

# Model management
weirdo models list          # List trained models
weirdo models train --data train.csv --name my-model
weirdo models info my-model # Show model details
weirdo models available     # List built-in downloadable pretrained models
weirdo models download NAME # Download pretrained weights by name
weirdo models download --url https://.../model.tar.gz --save-as my-model

# Scoring
weirdo score MTMDKSEL SIINFEKL                 # bundled swissprot-categories model
weirdo score --combine sum MTMDKSEL           # compare summed instead of maximal probabilities
weirdo score --model my-model MTMDKSEL       # a model you trained
```

## Long-Run Training on Modal

Use `scripts/train_modal_long_run.py` to run long remote training and export
weights as a `.tar.gz` archive:

```bash
# One-time setup: seed the SwissProt CSV (built by `weirdo data download`) into the Modal data volume
modal volume put weirdo-data-cache ~/.weirdo/downloads/swissprot-8mers.csv downloads/swissprot-8mers.csv --force
```

```bash
modal run scripts/train_modal_long_run.py \
  --model-name swissprot-mlp-modal \
  --epochs 1000 \
  --max-samples 2000000 \
  --output-archive ./swissprot-mlp-modal.tar.gz
```

The script:
- trains `MLPScorer` remotely
- saves model files to a Modal volume
- packages weights into `MODEL_NAME.tar.gz`
- can return archive bytes to your local machine (`--output-archive`)
- reads SwissProt from `--swissprot-path` (default: `/root/.weirdo/downloads/swissprot-8mers.csv`)
- treats `--max-samples 0` (default) as "use all available rows"

## Distributing Model Weights

Recommended flow:

1. Train and export `MODEL_NAME.tar.gz` (Modal script above).
2. Upload archive to a GitHub Release asset (or other HTTPS hosting).
3. Download after install via CLI:

```bash
# Direct URL (no code change needed)
weirdo models download --url https://github.com/ORG/REPO/releases/download/vX.Y.Z/MODEL_NAME.tar.gz --save-as MODEL_NAME
```

If you want named built-in downloads (`weirdo models download MODEL_NAME`),
add an entry to `PRETRAINED_MODELS` in `weirdo/model_manager.py` and release a
new package version.

## Architecture

```
weirdo/
├── scorers/
│   ├── mlp.py              # MLPScorer (sigmoid heads, foreignness)
│   ├── peptide_features.py # The 592 peptide features (vectorised)
│   ├── swissprot.py        # SwissProtReference (training data)
│   ├── config.py           # Presets and configuration
│   ├── registry.py         # Scorer registry
│   └── trainable.py        # TrainableScorer base class
├── pretrained/             # Models that ship in the package (weights up to 5 MB)
├── uniprot.py              # Download SwissProt from UniProt (datacache), build sequences
├── kmer_table.py           # Build the SwissProt k-mer reference table
├── data_manager.py         # ~/.weirdo reference data
├── model_manager.py        # Save/load trained models
├── amino_acid_properties.py  # 12 AA property dictionaries
└── api.py                  # High-level functions
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
