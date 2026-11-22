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

For complete MHC-length peptides, the experimental `peptide-content-v1` model is
the better entry point. It was trained directly on variable-length 8--15mers and
uses independent heads for viruses, bacteria, archaea, fungi, plants, protists,
invertebrates, fish, non-mammalian tetrapods, mammals, human, mouse, and dog:

```python
from weirdo import predict_peptide_content

result = predict_peptide_content(
    ["SIINFEKL", "GILGFVFTL"],
    score_policy="human-microbial-content-v1",
)
```

Its primary summary is
`microbial_foreign_like = max(microbial heads) * (1 - max(selected self heads))`.
The score is useful for ranking peptide content, but it did not add a validated
T-cell immunogenicity signal beyond MHC presentation or affinity. See
[docs/peptide-content.md](docs/peptide-content.md) for the split, model comparison,
host-specific guidance, and evaluation.
[docs/score-policies.md](docs/score-policies.md) defines the versioned human,
mouse, dog, generic-mammal, and stable human-complement scoring contracts.

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
weirdo data reference-proteome dog          # complete dog proteome for exact self lookup
weirdo data reference-proteome cattle       # Bos taurus / UP000009136
weirdo data reference-proteome --proteome UP000009136  # any UniProt reference proteome
weirdo data reference-proteome --manifest --dry-run  # inspect the 544-proteome frozen corpus
weirdo data reference-proteome --manifest --group mammals  # resumably fetch its 32 mammals
```
The files are fetched with [datacache](https://github.com/openvax/datacache) from UniProt's FTP site
(`WEIRDO_UNIPROT_URL` selects a mirror), checked against the sizes and MD5s UniProt lists and kept
in `~/.weirdo/downloads/uniprot/release-<version>/`, so they are downloaded once per UniProt release.
Complete host references (`human`, `dog`, `cat`, `mouse`, `rat`, and `cattle`) are separately fetched as pinned
UniProt reference-proteome FASTAs, with a local SHA-256/content summary.
The same command also handles frozen multi-proteome selections. `--manifest`
downloads every selected FASTA through datacache, skips verified local files on
re-run, and supports `--group mammals`, `--group viruses`, or a specific leaf.

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

## CLI

```bash
# Scoring
weirdo score MTMDKSEL SIINFEKL                 # bundled swissprot-categories model
weirdo score --combine sum MTMDKSEL            # compare summed instead of maximal probabilities
weirdo score --model my-model MTMDKSEL         # a model you trained

# Data & model management
weirdo data download                           # download SwissProt from UniProt and build reference data
weirdo data reference-proteome dog             # complete host proteome for exact self lookup
weirdo data list                               # show data status
weirdo models list                             # list trained models
weirdo models download NAME                    # download pretrained weights
```

## Documentation

Full guides and API reference live in `docs/`:

- [Getting started](docs/getting-started.md)
- [Category predictor](docs/category-predictor.md) — evaluation, calibration, and limits of the bundled `swissprot-categories` model
- [Peptide-content models](docs/peptide-content.md) and [score policies](docs/score-policies.md) — variable-length 8–15mer models and host-specific scoring contracts
- [Scorers & feature extraction](docs/scorers.md) and [amino-acid properties](docs/amino-acids.md) — the 592 physicochemical/composition features and `MLPScorer` training API
- [Data sources](docs/data-sources.md) and [API reference](docs/api.md)
- [Contributing](docs/contributing.md) and [changelog](docs/changelog.md)

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
