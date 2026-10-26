# Species-category predictor

## Stable integration API

```python
from weirdo import load_pretrained_model

model = load_pretrained_model("swissprot-categories-v1")
categories = model.predict_categories(["ACDEFGHI", "MTMDKSELV"])
scores = model.score(["ACDEFGHI", "MTMDKSELV"])
print(model.prediction_identity)
```

This interface verifies the wheel's immutable manifest and every artifact checksum before
loading portable NumPy weights. It never downloads files or searches a user model cache.
An explicit `model_dir` may point to an identical local copy. Missing or modified files fail
with an error. The manifest records the training-table and split-file hashes, feature
implementation, evaluation, attribution and a reproducible inference value. The original
`67e6224-dirty` training-code provenance is retained; the release commit is not presented as
the code used for training.

The integration accepts canonical ASCII amino-acid sequences of lengths **8–11**, the lengths
covered by the category evaluation below. It normalizes lowercase, preserves order and
duplicates, returns an empty result for an empty batch, and rejects other lengths, ambiguous
residues and whitespace. Distinct peptides are scored in bounded batches. Longer supported
peptides average the category outputs of their overlapping 8-mer windows.

Its explicit policy, `human-self-complement-v1`, returns **one minus the human-category
output**. It supports human reference taxon 9606 only; a rodent category is not a mouse-specific
self model. This is a descriptive score relative to the training population, not a probability
of processing, presentation or immune recognition. It has no independently established
immune-response ranking benefit and is intended only for opt-in experimental ranking.
`prediction_identity` includes the artifact, input and feature contract, policy and host;
downstream caches must retain the complete identity.
`model.score_policy_definition` and the returned frame's
`attrs["score_policy_definition"]` expose the complete host, category, formula,
overlap, direction, and validation contract. See
[Versioned score policies](score-policies.md).

The older `predict_categories(...)` and `MLPScorer.foreignness(...)` APIs below expose the
configurable pathogen/mammalian ratio. That ratio and the human-category complement answer
different questions; the integration policy is named explicitly so they cannot share a cache.

WEIRDO ships a trained model, `swissprot-categories`, that returns for any peptide **one
probability per species category**. The categories are the ten UniProt taxonomic divisions of
SwissProt: archaea, bacteria, fungi, human, invertebrates, mammals, plants, rodents, vertebrates
and viruses.

```python
from weirdo import predict_categories

predict_categories(["SIINFEKL", "NLVPMVATV", "GILGFVFTL"]).round(3)
```

```
  peptide  archaea  bacteria  fungi  human  invertebrates  mammals  plants  rodents  vertebrates  viruses  pathogen_max  self_max  foreignness
 SIINFEKL    0.077     0.389  0.141  0.044          0.158    0.025   0.102    0.048        0.026    0.057         0.389     0.048        0.891
NLVPMVATV    0.026     0.461  0.100  0.092          0.069    0.047   0.116    0.106        0.040    0.027         0.461     0.106        0.813
GILGFVFTL    0.045     0.435  0.118  0.083          0.076    0.041   0.112    0.093        0.038    0.031         0.435     0.093        0.823
```

```bash
weirdo score SIINFEKL NLVPMVATV GILGFVFTL            # table
weirdo score SIINFEKL NLVPMVATV --csv > scores.csv   # machine readable
```

!!! warning "What this is, and is not"
    The outputs are close to calibrated probabilities, but they are only weakly informative about
    where a peptide comes from: on protein families the model has never seen, the mean AUC over
    categories is about 0.6 (0.5 is chance). The two viral epitopes above (NLVPMVATV, GILGFVFTL)
    get a virus probability of about 0.03, barely different from the 0.04 base rate. Do not read
    the order of single peptides: SIINFEKL (chicken ovalbumin, found in SwissProt only in
    `OVAL_CHICK`, a vertebrate protein) scores 0.89, higher than GILGFVFTL (influenza M1, found
    only in 114 viral proteins) at 0.82, because the model sees eight residues and not where they
    come from, and the score of one peptide varies far more (standard deviation 0.17 among
    peptides found only in mammalian proteins) than the gap between these two. For a peptide that
    occurs in SwissProt, [looking it up](#looking-a-peptide-up-instead-of-predicting-it) is exact.
    Nothing here has been shown to predict T-cell immunogenicity; the related benchmark of
    foreignness scores against T-cell data is discussed in
    [issue #8](https://github.com/pirl-unc/weirdo/issues/8).

## What an output means

Output `c` is the probability that the peptide **occurs in SwissProt proteins of category `c`**.

* The ten outputs are independent sigmoids. They are **not** mutually exclusive and need not sum
  to one: a conserved peptide can belong to human, mammals and rodents at once.
* The probability is relative to the training population, the unique 8-mers of SwissProt. In
  that population about 40% of 8-mers occur in bacteria and about 4% in viruses, so for an
  unremarkable peptide the output stays near these base rates. A peptide that is not from SwissProt at all
  (a new pathogen, a neoantigen) is still scored against SwissProt categories.
* Because discrimination is weak, outputs rarely stray far from the base rates. On 100,000
  held-out 8-mers, 90% of the virus outputs fall between 0.020 and 0.061 and the largest is 0.148
  (base rate 0.037), whereas 90% of the bacteria outputs, the most informative ones, span 0.16 to 0.65.
* Peptides longer than 8 residues are scored as the **mean over their 8-mer windows**. Shorter
  peptides are padded and unreliable.
* If your population has different base rates, shift the logit of each output by
  `log(π_new/(1−π_new)) − log(π_old/(1−π_old))`; the training prevalences are stored in the
  model metadata.

## Pathogen versus self

The outputs can be summarized as one value for each group and their ratio:

```
pathogen_max = max(P(bacteria), P(viruses))
self_max     = max(P(human), P(rodents), P(mammals))
foreignness  = pathogen_max / (pathogen_max + self_max)
```

How the probabilities inside a group are combined is a parameter. `combine="sum"` adds them
instead (`pathogen_sum`, `self_sum`): because the outputs are probabilities, each sum is the
**expected number of categories in the group that contain the peptide**, so `foreignness` then
compares the expected number of pathogen categories with the expected number of mammalian ones
(the groups have two and three members, which shifts the neutral point). `combine` also accepts
a function that maps an array of shape (n_peptides, n_categories_in_group) to one value per
peptide. On the test fold the two built-in choices separate k-mers found only in pathogens from
those found only in mammals equally well (AUC 0.692 and 0.693), so the choice is one of
interpretation. The groups are the defaults and can be changed too:

```python
from weirdo import load_model

model = load_model("swissprot-categories")
model.predict_dataframe(
    ["SIINFEKL"],
    pathogen_categories=["bacteria", "viruses", "fungi"],
    self_categories=["human"],
    combine="sum",      # default "max": max(pathogen) / (max(pathogen) + max(self))
)
```

The other five categories are in neither default group: **archaea, fungi, invertebrates, plants
and vertebrates** (UniProt's "vertebrates" division is the non-mammalian vertebrates: fish,
birds, reptiles and amphibians; "invertebrates" includes the worms and insects but also
protozoan parasites such as *Plasmodium* and *Toxoplasma*; chicken ovalbumin is a vertebrate
protein). They are foreign to a human as well, but not pathogens in general, so they are left
out; move them into `pathogen_categories` if that is the question you want to ask.

Read `foreignness` against the reference distributions in [How well it works](#how-well-it-works),
not against 0.5: bacteria is in about 40% of 8-mers against much lower rates for the mammalian
categories, so a typical peptide scores about 0.81 with `combine="max"` (0.67 with `"sum"`) and
even peptides found only in mammalian proteins have a median of 0.76 (0.60).

## Looking a peptide up instead of predicting it

The model estimates how often an 8-mer of this composition occurs in each category. For a peptide
that is in SwissProt, the exact answer is in the sequence table (`weirdo data download
swissprot-sequences`), and it can differ completely from the prediction:

| Peptide | Found in SwissProt | Predicted `foreignness` | From the sequences (`max`) |
| --- | --- | --- | --- |
| GILGFVFTL (influenza A M1) | 114 viral proteins, nothing else | 0.82 | 1.0 (viruses only) |
| NLVPMVATV (CMV pp65) | 3 viral proteins, nothing else | 0.81 | 1.0 (viruses only) |
| SIINFEKL (chicken ovalbumin) | `OVAL_CHICK`, a vertebrate protein | 0.89 | 0.5 (in neither group) |

```python
import pandas as pd

sequences = pd.read_csv("~/.weirdo/downloads/swissprot-sequences.csv")
hits = sequences[sequences.sequence.str.contains("GILGFVFTL", regex=False)]
hits.category.value_counts()      # viruses    114
```

Both SwissProt releases checked (2023_02 and 2026_03) give the same counts. The sequences only
know the SwissProt release they were downloaded from, so they say nothing about a neoantigen or
a pathogen that is not in SwissProt; that is what the predictor is for. A combined score that
looks a peptide up first and predicts only when it is absent is not part of WEIRDO.

## How well it works

On protein families the model never saw (the test fold, 600,000 8-mers, scored once after training), the mean AUC over the ten categories is **0.615** (95% interval 0.610–0.619); 0.5 is chance. Per category it runs from plants 0.576 to archaea 0.664. The probabilities are close to calibrated: mean expected calibration error 0.0079 and calibration slopes between 0.83 and 0.97, i.e. slightly over-confident on new families. The log loss improves on always predicting each category's base rate by 2.4% on average. That is a small gain: the outputs order peptides only a little better than chance.

For the pathogen-versus-self summary, the default `max(pathogen) / (max(pathogen) + max(self))` separates k-mers found only in bacteria/viruses from those found only in human/rodents/mammals with AUC **0.692** (0.686–0.698); the `sum` version gives 0.693 and the difference of sums 0.690. These are results for groups of k-mers: the distributions of the two groups overlap widely (see the table below), so the score of a single peptide says little.

A naive random split of 8-mers would have reported a mean AUC of 0.620 rather than 0.615. The inflation is small because a model built on composition features cannot memorize individual 8-mers; the tables below nevertheless use protein families that do not cross folds.

### Held-out families: test fold (600,000 8-mers)

Intervals are 95% percentile intervals from 100 bootstrap resamples of 20 family buckets.

| Category | Prevalence | AUC (95% interval) | Average precision | Log-loss skill (95% interval) | ECE | Calibration slope |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| archaea | 0.036 | 0.664 (0.655–0.672) | 0.068 | 0.038 (0.034–0.042) | 0.0033 | 0.96 |
| bacteria | 0.386 | 0.655 (0.648–0.661) | 0.528 | 0.054 (0.048–0.060) | 0.0269 | 0.90 |
| fungi | 0.145 | 0.586 (0.580–0.592) | 0.191 | 0.013 (0.010–0.015) | 0.0173 | 0.89 |
| human | 0.100 | 0.628 (0.623–0.634) | 0.161 | 0.030 (0.027–0.032) | 0.0071 | 0.90 |
| invertebrates | 0.090 | 0.611 (0.603–0.621) | 0.144 | 0.024 (0.019–0.029) | 0.0063 | 0.89 |
| mammals | 0.042 | 0.602 (0.597–0.608) | 0.061 | 0.015 (0.013–0.017) | 0.0023 | 0.86 |
| plants | 0.095 | 0.576 (0.571–0.582) | 0.123 | 0.009 (0.007–0.011) | 0.0094 | 0.84 |
| rodents | 0.100 | 0.628 (0.621–0.635) | 0.158 | 0.029 (0.026–0.032) | 0.0045 | 0.89 |
| vertebrates | 0.041 | 0.603 (0.591–0.611) | 0.059 | 0.015 (0.010–0.019) | 0.0013 | 0.83 |
| viruses | 0.038 | 0.592 (0.577–0.606) | 0.056 | 0.016 (0.002–0.032) | 0.0003 | 0.97 |
| **mean** | 0.107 | **0.615** (0.610–0.619) | 0.155 | **0.024** (0.022–0.026) | 0.0079 | 0.89 |

Log-loss skill is `1 − log loss / log loss of always predicting the training prevalence`; 0 means no better than the base rate. ECE is the expected calibration error over 10 bins; a calibration slope of 1 is perfect.

### Pathogen versus self

AUC for telling 253,867 k-mers found only in bacteria/viruses from 106,390 found only in human/rodents/mammals.

| Score | AUC (95% interval) |
| --- | ---: |
| max(pathogen) / (max(pathogen) + max(self))  *(default, `combine='max'`)* | 0.692 (0.686–0.698) |
| sum(pathogen) / (sum(pathogen) + sum(self))  *(`combine='sum'`)* | 0.693 (0.688–0.699) |
| sum(pathogen) − sum(self) | 0.690 (0.684–0.697) |
| previous model (squared loss), max ratio | 0.678 (0.671–0.686) |

A typical peptide scores above 0.5, because the bacteria output has a base rate near 0.4 against much lower base rates for the mammalian outputs. Read a `foreignness` value against these reference distributions on the test fold, not against 0.5:

| Score | k-mers found | Number | 5th percentile | Median | 95th percentile |
| --- | --- | ---: | ---: | ---: | ---: |
| `combine='max'` (default) | only in bacteria/viruses | 253,867 | 0.571 | 0.842 | 0.941 |
| `combine='max'` (default) | only in human/rodents/mammals | 106,390 | 0.364 | 0.761 | 0.912 |
| `combine='max'` (default) | all test k-mers | 600,000 | 0.459 | 0.814 | 0.933 |
| `combine='sum'` | only in bacteria/viruses | 253,867 | 0.410 | 0.711 | 0.875 |
| `combine='sum'` | only in human/rodents/mammals | 106,390 | 0.249 | 0.604 | 0.822 |
| `combine='sum'` | all test k-mers | 600,000 | 0.321 | 0.672 | 0.859 |

### Previous squared-loss head versus the sigmoid head (same data, same test fold)

| Model | Mean AUC | Mean log loss | Log loss of the base rate | Mean ECE |
| --- | ---: | ---: | ---: | ---: |
| previous (squared loss + sigmoid) | 0.607 | 0.726 | 0.302 | 0.419 |
| sigmoid + cross-entropy | 0.615 | 0.293 | 0.302 | 0.0079 |

### How much the split protocol matters

| Evaluation set | 8-mers | Mean AUC | Mean log-loss skill | Mean ECE |
| --- | ---: | ---: | ---: | ---: |
| training sample (seen during training) | 100,000 | 0.627 | 0.030 | 0.0048 |
| random k-mers from the training families (not seen) | 300,000 | 0.620 | 0.026 | 0.0046 |
| validation fold (new families; used for early stopping) | 300,000 | 0.616 | 0.025 | 0.0056 |
| **test fold (new families)** | 600,000 | 0.615 | 0.024 | 0.0079 |

### Longer peptides (8-mer windows averaged)

The model is trained on 8-mers; a longer peptide is scored as the mean of its 8-mer windows. Labels are presence of the whole peptide in a category, and the base rate used for the skill is that set's own prevalence.

| Length | Peptides | Mean AUC | Mean log-loss skill | Mean calibration slope | Pathogen-vs-self AUC (max ratio) | Pathogen-vs-self AUC (sum ratio) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 400,000 | 0.620 | 0.026 | 0.97 | 0.701 | 0.702 |
| 10 | 400,000 | 0.625 | 0.028 | 1.03 | 0.707 | 0.708 |
| 11 | 400,000 | 0.627 | 0.030 | 1.08 | 0.710 | 0.711 |


## Why the ceiling is low

The label of an 8-mer is which taxonomic divisions have a protein containing it. Eight residues
carry little information about that, and the divisions overlap: a human peptide is often also a
mammal and a rodent peptide. Three unrelated learners trained from scratch were tried on the same splits
and land in the same place:

| Learner | Training 8-mers | Mean validation AUC | Mean validation log loss |
| --- | ---: | ---: | ---: |
| MLP, squared loss + sigmoid (previous head) | 1,000,000 | 0.607 | 0.7258 |
| MLP, cross-entropy, 592 composition features | 1,000,000 | 0.612 | 0.2863 |
| MLP, cross-entropy, composition + per-position residues | 1,000,000 | 0.611 | 0.2865 |
| MLP (512, 256, 128), composition + per-position residues | 1,000,000 | 0.612 | 0.2865 |
| **shipped model**: MLP, cross-entropy, 592 composition features | 3,000,000 | 0.616 | 0.2857 |
| CNN over the 8 residues (PyTorch) | 3,000,000 | 0.617 | 0.2854 |
| *always predict the base rate* | – | 0.500 | 0.2945 |

Gradient boosting on the same 592 features (500,000 training 8-mers), for the categories tried, against the shipped MLP's validation AUC:

| Category | Gradient boosting | Shipped MLP |
| --- | ---: | ---: |
| viruses | 0.599 | 0.591 |
| bacteria | 0.649 | 0.652 |
| human | 0.624 | 0.633 |
| plants | 0.572 | 0.577 |

The first four rows are scored on 200,000 validation 8-mers and the others on 300,000 (same fold), so
differences of a few thousandths in AUC are within sampling noise; the 95% interval of the test-fold mean
AUC is ±0.005. The learners tried agree to within about 0.005 AUC, which points to the information in an
8-mer, rather than the model class, as the limit. Pretrained protein language models were not tried.
`scripts/category_headroom.py` reproduces the table.

## Differences from earlier versions

Earlier versions trained the network with squared error on the 0/1 labels and then applied a
sigmoid to the regression output. Those "probabilities" were confined to a narrow band around
0.5–0.7, and `foreignness` computed from them was close to 0.5 for almost every peptide. Models
saved by earlier versions still load and still produce those values, with a warning. The sigmoid
head trained with binary cross-entropy is now the default whenever the targets are binary and
multi-label (`MLPScorer(loss="auto")`); `loss="squared"` restores the old behaviour.

`foreignness()` keeps its earlier default, `max(pathogen) / (max(pathogen) + max(self))`, and
gained the `combine` parameter (`"max"`, `"sum"` or a function). `predict_dataframe()` now also
returns the two group values (`pathogen_max`, `self_max`).

## Reproducing the model

The model was trained from the SwissProt sequences with `scripts/train_category_predictor.py`,
which uses the family-aware splits from `scripts/category_data.py`:

```bash
python scripts/category_data.py build --sequences data/swissprot-sequences.csv \
    --out data/category-splits --n-all 0 --n-train-heldout 300000
for K in 9 10 11; do
  python scripts/category_data.py build --sequences data/swissprot-sequences.csv \
      --out data/category-splits-k$K --k $K --n-train 0 --n-val 0 --n-all 0 --n-test 400000
done
python scripts/train_category_predictor.py --splits data/category-splits \
    --length-splits 9=data/category-splits-k9,10=data/category-splits-k10,11=data/category-splits-k11 \
    --legacy-baseline --out weirdo/pretrained/swissprot-categories \
    --report report.json --markdown report.md
```

**Splits.** Proteins are assigned to train, validation and test folds (70/15/15) by a hash of
their UniProt entry-name mnemonic (`RS19` in `RS19_ECOLI`). This keeps identical mnemonic
prefixes together, but is a heuristic rather than a sequence-homology clustering method:
homologs with different names can cross folds. "Family" in the tables denotes these mnemonic
groups. Only 8-mers found in exactly one fold are kept, so no identical 8-mer crosses folds and
each kept 8-mer's labels come entirely from its own fold. The test fold is scored once, after
training.

**Data.** `swissprot-sequences.csv` was built from the UniProt taxonomic-division files dated
2023-05-03 (`scripts/download-swissprot-data.sh`); its per-category entry counts equal those of
the division files. Labels rebuilt from it match the released 8-mer table on the first 300,000
rows of that table (not a random sample): 99.88% of those 8-mers are found, and 99.84% of the found
ones have the same category vector (`--verify-table` repeats the check).

## Data and attribution

The model was trained on protein sequences from UniProtKB/Swiss-Prot
([uniprot.org](https://www.uniprot.org)), which is distributed under the
[Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/). The
bundled weights are a derived work; please cite UniProt (The UniProt Consortium, *Nucleic Acids
Research*) when you use them.

## Limitations

* The target is presence in SwissProt, not the organism a peptide was observed in, and SwissProt
  covers model organisms and well-studied pathogens much better than the rest of life.
* Discrimination is modest (see the tables); do not use the probabilities to assign a peptide to
  an organism.
* Mnemonic groups do not establish homology-disjoint evaluation. Sequence-cluster and held-out
  taxon validation remain tracked in [issue #6](https://github.com/pirl-unc/weirdo/issues/6).
* The model has only seen 8-mers. Longer peptides are scored by averaging windows, which is
  checked above for 9–11-mers but not beyond.
* The summed pathogen/self score is a descriptive summary. Whether it adds anything to MHC
  presentation for predicting T-cell recognition has not been tested on held-out data.
