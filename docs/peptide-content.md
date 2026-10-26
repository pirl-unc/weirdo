# Variable-length peptide-content model

This experimental, opt-in model scores complete 8--15mer peptides. It is a
different model from the older `swissprot-categories` 8mer-window model.

## Outputs

The ten coarse taxonomy heads are `viruses`, `bacteria`, `archaea`, `fungi`,
`plants`, `protists`, `invertebrates`, `fish`,
`non_mammalian_tetrapods`, and `mammals`. `human`, `mouse`, and `dog` are
additional diagnostic heads nested under `mammals`.

Every head is an independent sigmoid trained with binary cross-entropy. The
outputs are not a softmax and do not sum to one. If the same peptide occurs in
both a mammal and a bacterium, both labels are positive. Exact peptide matches
that cross protein-family folds are removed before training and evaluation.

The default summaries are:

```text
microbial_like = max(viruses, bacteria, archaea, fungi, protists)
self_like = max(the caller's selected self categories)
microbial_foreign_like = microbial_like * (1 - self_like)
```

The maxima prevent a peptide that matches several overlapping categories from
getting an artificially large score. `microbial_foreign_like` is a bounded
ranking score, not a calibrated posterior probability of biological
foreignness: training examples were deliberately balanced across taxonomic
strata and peptide lengths.

For a human, mouse, or dog view, reasonable learned self bundles are
`("mammals", "human")`, `("mammals", "mouse")`, and
`("mammals", "dog")`. Cats and rats currently use `("mammals",)` as the
learned fallback. An exact selected-host proteome lookup should take precedence
over the learned score: an exact self hit is self even when its short sequence
also looks microbial.

```python
from weirdo import predict_peptide_content

scores = predict_peptide_content(
    ["SIINFEKL", "GILGFVFTL"],
    score_policy="human-microbial-content-v1",
)
```

For mouse analyses, use `score_policy="mouse-microbial-content-v1"`.
Each result records the complete host, category, formula, overlap, direction,
and validation contract in `scores.attrs["score_policy_definition"]`; see
[Versioned score policies](score-policies.md). Explicit category tuples remain
available for custom exploratory definitions and are labeled
`custom-unversioned`.

After an exact lookup against a pinned host reference, pass the resulting
boolean flags to `exact_self_hits=[...]`; exact hits are returned as
`exact_self_hit=True` and their foreignness summary is forced to zero.

## Corpus and split

The source SwissProt corpus contains 569,516 proteins and 205,866,895 amino
acids. Protein mnemonic families are assigned to 70/15/15 train/validation/test
folds. Candidate 8--15mers are sampled approximately evenly by source category
and length, relabeled by exact lookup against every protein in their fold, and
dropped if the same peptide occurs in another fold.

| Category | Unique NCBI taxa | Proteins | Amino acids |
| --- | ---: | ---: | ---: |
| viruses | 2,703 | 17,320 | 7,860,752 |
| bacteria | 3,247 | 335,891 | 105,368,988 |
| archaea | 233 | 19,708 | 5,710,974 |
| fungi | 807 | 36,360 | 18,390,340 |
| plants | 2,344 | 44,402 | 16,733,771 |
| protists | 195 | 6,560 | 3,461,461 |
| invertebrates | 1,597 | 22,708 | 8,119,214 |
| fish | 460 | 6,014 | 2,302,214 |
| non-mammalian tetrapods | 943 | 12,865 | 4,288,901 |
| mammals | 1,910 | 67,688 | 33,630,280 |

“Taxa” means distinct NCBI taxon IDs present in SwissProt; strain- or
subspecies-level records can therefore count separately and this is a coverage
measure rather than a strict biological species count.

The nested species heads contain 20,422 human proteins (11,392,335 amino
acids), 17,155 mouse proteins (9,733,237 amino acids), and 847 dog proteins
(386,387 amino acids). These are **SwissProt reviewed-entry counts**, not
complete species proteomes: this corpus is strongly biased toward human and
mouse curation. For example, UniProt's dog reference proteome `UP000805418`
has 20,987 annotated genes and 43,615 protein sequences. The v1 dog head is
therefore diagnostic; `mammals` remains the safer learned fallback for dog and
cat work.

### Proposed balanced taxonomy (v2)

The model above is the trained v1 model. The following is the proposed v2
label schema, computed from the same 2023 SwissProt records and their NCBI
lineages. It is not yet a bundled model. The leaves are mutually exclusive;
parent scores such as `bacteria_like` and `mammals_like` are derived with a
maximum rather than trained as competing softmax classes.

Splitting the 105.4-million-residue bacterial class at evolutionary boundaries
gives seven leaves below the target ceiling of approximately 20 million amino
acids:

| Proposed bacterial leaf | Unique NCBI taxa | Proteins | Amino acids |
| --- | ---: | ---: | ---: |
| `enterobacterales` | 306 | 63,270 | 19,541,142 |
| `other_gammaproteobacteria` | 482 | 59,950 | 18,839,311 |
| `alphaproteobacteria` | 352 | 35,475 | 11,037,821 |
| `other_pseudomonadota` | 357 | 40,419 | 12,562,068 |
| `bacilli` | 522 | 55,794 | 17,561,382 |
| `actinomycetota` | 520 | 24,691 | 8,228,218 |
| `other_bacteria` | 708 | 56,292 | 17,599,046 |

Classification order matters: Enterobacterales is removed before the rest of
Gammaproteobacteria, and those groups plus Alphaproteobacteria are removed
before the rest of Pseudomonadota. `other_bacteria` contains all remaining
bacterial phyla and the remaining Bacillota. The derived score is:

```text
bacteria_like = max(the seven bacterial leaves)
```

The analogous four-way mammalian partition, measured in the reviewed-only v1
corpus, is:

| Candidate mammalian leaf | Unique NCBI taxa | Proteins | Amino acids |
| --- | ---: | ---: | ---: |
| `primates` | 238 | 27,194 | 14,068,551 |
| `glires` | 626 | 28,054 | 14,984,497 |
| `carnivora` | 175 | 1,696 | 649,057 |
| `other_mammals` | 871 | 10,744 | 3,928,175 |

The small Carnivora count is a **source-data limitation**, not a reason to
merge its biology away. Its parent Laurasiatheria has 858 taxa, 11,674 proteins,
and 4,347,314 amino acids in SwissProt, dominated by 8,973 artiodactyl proteins
(3,377,321 amino acids); it mixes carnivorans with ungulates/cetaceans, bats,
horses, shrews, and other lineages. That is a reasonable broad evolutionary
roll-up, but not a substitute for a carnivoran self label.

V2 should instead draw one complete UniProt reference proteome per selected
species (reviewed and unreviewed entries) and retain full proteomes for exact
self lookup. The classifier should deduplicate sequences, select a primary
protein per gene where available, and cap/sampling-balance each species and
leaf during training. Thus approximately 20 million residues is a *training
sample budget per leaf*, not a reason to define leaves around sparse SwissProt
coverage. On that corpus, `carnivora` can remain an exclusive leaf; `laurasiatheria`
is an optional parent roll-up. Exact dog or cat proteome lookup still takes
priority.

The current complete-reference starting set illustrates the scale difference:

| Host | UniProt reference proteome | Annotated genes | Protein sequences | Amino acids |
| --- | --- | ---: | ---: | ---: |
| dog | [UP000805418](https://www.uniprot.org/proteomes/UP000805418) | 20,987 | 43,615 | 26,920,080 |
| cat | [UP000823872](https://www.uniprot.org/proteomes/UP000823872) | 19,324 | 58,973 | 38,470,872 |
| mouse | [UP000000589](https://www.uniprot.org/proteomes/UP000000589) | 21,860 | 54,855 | 23,132,845 |
| rat | [UP000002494](https://www.uniprot.org/proteomes/UP000002494) | 21,471 | 52,824 | 32,555,332 |

The protein count exceeds the gene count because these proteomes include
multiple translated products and annotations. Include them all for conservative
exact-self exclusion; avoid letting them multiply a host's weight when training
the learned taxonomy classifier.

Complete host references are managed separately from the SwissProt taxonomy
corpus:

```bash
weirdo data reference-proteome --list
weirdo data reference-proteome dog
weirdo data reference-proteome cattle  # Bos taurus; cow and bovine are aliases
weirdo data reference-proteome --proteome UP000009136  # any explicit accession
weirdo data reference-proteome --manifest --dry-run
weirdo data reference-proteome --manifest --group mammals
weirdo data reference-proteome --manifest  # all 544 FASTAs; resumable
```

The download is a compressed FASTA from a pinned UniProt reference-proteome ID,
fetched with datacache and accompanied by a local SHA-256/content-summary
record. The six short host names are a convenience manifest in code, not a
copy of UniProt's catalog: `--proteome` resolves any UniProt *reference*
proteome accession at download time. The initial frozen v2 input selection is
`weirdo/reference_proteome_manifests/reference-proteomes-v1.yaml`: 32 diverse
mammalian proteomes and 512 viral proteomes (128 each from Riboviria,
Duplodnaviria, Varidnaviria, and Monodnaviria). It is a compact, versioned
selection plus policy—not a vendored copy of UniProt's changing whole catalog.
All entries are handled by the same datacache/FASTA/checksum path as a named
host or an explicit `--proteome` accession. To build a variable-length corpus
from the complete frozen selection, use
`scripts/reference_proteome_data.py build --reference-dir ~/.weirdo/downloads/reference-proteomes --out <directory>`.
It creates independent, nested heads for viruses and the four viral realms,
mammals and the four mammalian leaves, plus human/mouse/dog/cat/cattle. The
Modal runner exposes that whole workflow as
`modal run scripts/train_peptide_content_modal.py --stage reference-all`.

This is deliberately the primary sequence layer: UniProt gives us stable
protein identifiers, a broad organism catalog, and an uncomplicated FASTA
source. A reference proteome is already a selected UniProtKB collection, so
do not blindly union it with SwissProt for the same species. `pyensembl` is
useful as an optional auxiliary layer when a caller needs Ensembl coordinates,
transcript-to-gene choice, or variants, but is not required to download or
search host proteomes.

Together with the unsplit categories, v2 would have 19 mutually exclusive
coarse leaves: `viruses`, the seven bacterial leaves, `archaea`, `fungi`,
`plants`, `protists`, `invertebrates`, `fish`,
`non_mammalian_tetrapods`, and the four mammalian leaves. `human`, `mouse`,
and `dog` remain overlapping diagnostic heads; `laurasiatheria` is a derived
parent roll-up. This preserves useful roll-ups without forcing related or
multiply occurring peptides into a single softmax class.

After removing 101,160 cross-fold peptide candidates, the dataset contains
799,357 training, 153,159 validation, and 150,807 held-out test peptides.

## Architecture comparison

All architectures see the complete peptide rather than averaged 8mer windows.
The test AUC for `microbial_foreign_like` distinguishes peptides found only in
the five microbial categories from peptides found only in mammals.

| Architecture | Mean category AUC | `microbial_foreign_like` AUC |
| --- | ---: | ---: |
| composition MLP | 0.613 | 0.659 |
| residue CNN | 0.616 | 0.666 |
| small transformer | 0.617 | 0.654 |
| frozen ESM-2 8M | 0.625 | 0.665 |
| fine-tuned ESM-2 8M | **0.625** | **0.673** |

Fine-tuning ESM-2 is the best tested taxonomy model, but the improvement over
the small CNN is modest. The composition MLP is the dependency-free bundled
model; larger Torch/ESM artifacts are retained separately.

For the fine-tuned ESM-2 model, `microbial_foreign_like` AUC increases from
0.643 for 8mers to 0.697 for 15mers. The full category AUCs remain modest
(for example, mammals 0.648, human 0.618, mouse 0.612, and dog 0.585), so the
learned output should complement exact host-proteome lookup rather than replace
it.

## Incremental immunogenicity evaluation

The taxonomy models were never fitted to T-cell outcomes. Their scores were
added to MHCflurry baselines in repeated 5x5-fold grouped cross-validation;
peptide-overlap clusters stayed within folds. Architecture selection used only
taxonomy metrics. Confidence intervals use 1,000 cluster bootstrap samples.

| Cohort and MHC baseline | Baseline AUC | With fine-tuned ESM | Delta AUC (95% CI) |
| --- | ---: | ---: | ---: |
| broad neoepitope, presentation | 0.8801 | 0.8796 | -0.0005 (-0.0008, -0.0003) |
| broad infection, presentation | 0.7422 | 0.7420 | -0.0003 (-0.0005, -0.0001) |
| registered viral validation, presentation | 0.6619 | 0.6647 | +0.0028 (-0.0052, +0.0111) |
| registered viral validation, affinity | 0.6367 | 0.6454 | +0.0087 (-0.0037, +0.0219) |

The registered viral validation results are suggestive, especially against the
affinity-only baseline, but both confidence intervals cross zero and the
discovery split had a negative delta. The result therefore does **not** validate
an incremental T-cell immunogenicity signal beyond MHC. Use the model as an
experimental taxonomy/content ranking feature, not as an immunogenicity
predictor.

### Current complete-reference run

The follow-up run uses the frozen `reference-proteomes-v1` corpus rather than
the older SwissProt-only split: 1,460,196 proteins and 865,882,189 amino acids
from 32 mammalian and 512 selected viral reference proteomes (511 contributed
canonical protein sequences). The taxonomy split
contains 351,301 train, 52,802 validation, and 52,268 test peptides. The 1,076
unique registered-validation peptides were excluded from candidate sampling
before training.

| Model | Mean taxonomy AUC | Foreignness AUC |
| --- | ---: | ---: |
| composition MLP | 0.6840 | 0.7555 |
| residue CNN | 0.6899 | 0.7557 |
| frozen ESM-2 8M | 0.6929 | 0.7592 |
| fine-tuned ESM-2 8M | **0.6956** | **0.7662** |

For the prospective T-cell test, the response coefficient was fitted on the
768 discovery assays (638 positive) and then frozen before scoring the 1,274
validation assays (385 positive; 861 peptide clusters). This is stricter than
the grouped-CV rows above and is the result to use for “does it help on Tc
epitopes?”

| Content model added to MHCflurry presentation + HLA + cohort | Validation AUC | Delta AUC (200-draw cluster 95% CI) |
| --- | ---: | ---: |
| none (baseline) | 0.6129 | — |
| composition MLP foreignness | 0.6090 | -0.0040 (-0.0091, +0.0010) |
| residue CNN foreignness | 0.6084 | -0.0046 (-0.0105, +0.0017) |
| frozen ESM-2 8M foreignness | 0.6007 | -0.0123 (-0.0206, -0.0045) |

The table above is a useful stress test, but its negative deltas are not a
clean estimate of the taxonomy effect.  The discovery response model is fit
mostly on SARS-CoV-2 cohorts, and the direction of the MHCflurry presentation
score changes between the domains: discovery-only presentation AUC is about
0.40, whereas it is about 0.67 on the untouched validation set.  A coefficient
learned from discovery therefore transfers with the wrong sign.  We keep that
naive result in the report as a calibration-failure diagnostic rather than
calling it evidence that better foreignness intrinsically hurts
immunogenicity.

### Domain-robust reanalysis

To remove that artifact, the response calibrator was fit on the broad
development table (after removing every validation peptide cluster and exact
peptide) and then applied once to the registered validation set.  The model
uses continuous presentation, affinity, and processing features; the taxonomy
score is the only added feature.  This is still exploratory because the broad
development table helped choose the calibration protocol.

| Content signal added | Validation AUC | Delta AUC | Cluster-bootstrap 95% CI |
| --- | ---: | ---: | ---: |
| none | 0.6690 | — | — |
| frozen ESM `virus_foreign_like` | 0.6717 | +0.0027 | -0.0022, +0.0069 |
| frozen ESM `p_viruses` | 0.6715 | +0.0025 | -0.0035, +0.0080 |
| frozen ESM `1 - p_mammals` | 0.6716 | +0.0026 | -0.0029, +0.0077 |
| fine-tuned ESM `virus_foreign_like` | 0.6709 | +0.0019 | -0.0048, +0.0082 |

The fitted taxonomy coefficient is positive (standardized coefficient +0.042
for frozen ESM `virus_foreign_like`, +0.054 for fine-tuned ESM), and the log
loss improves slightly.  Thus the
current evidence is compatible with a small positive content prior, not with a
systematic foreignness penalty.  The interval includes zero, so this is not a
confirmed immunogenicity improvement.  The three foreignness definitions are
effectively tied; the product form is not uniquely validated.

Adding the two available TCR-face composition features gives a stronger
baseline.  With the fine-tuned ESM model, the residual `virus_foreign_like`
increment is +0.0013 (95% CI -0.0015, +0.0042); without TCR chemistry it is
+0.0019 (95% CI -0.0048, +0.0082).  In matched-HLA, matched-presentation
pairs, the fine-tuned ESM content score alone has AUC 0.526 and the MHC+TCR
baseline has AUC 0.584; the augmented model is 0.584.  Fine-tuning improves
taxonomy discrimination but does not turn the taxonomy score into a strong
T-cell-response predictor.

The practical conclusion is to keep Weirdo's taxonomy output as a weak,
calibrated content prior and to combine it with MHC presentation, processing,
and TCR-facing chemistry.  A viral-only benchmark cannot test self-versus-
nonself discrimination: every source peptide is already nonself.  The next
decisive benchmark must contain matched human/self peptides (or mutation-matched
neoepitopes) and hold out entire source families or cohorts.

### Where the assay data came from

Both evaluation tables are custom, reproducible selections from
[IEDB T-cell assay exports](https://query-api.iedb.org/docs/swagger/), not
pre-existing benchmark downloads and not model-training labels.

The broad development collection has 8,059 deduplicated peptide/HLA/cohort
observations. Its 13 neoepitope cohorts contribute 2,583 observations (486
positive), while 31 infection cohorts contribute 5,476 observations (1,622
positive). The latter collection is mostly viral but also includes
*Trypanosoma cruzi* and *Mycobacterium tuberculosis*, so “infection” is more
accurate than “viral.” Rows are canonical 8--14mer, exact-HLA class-I T-cell
assays in humans. Binding-only assays are excluded; repeated assay rows are
collapsed so a peptide/HLA/cohort item is positive if any contributing assay
was positive and negative only if all were negative. This collection was used
for exploratory development and is not independent confirmation.

The fixed viral benchmark has 2,042 observations: 768 discovery observations
(638 positive) and 1,274 validation observations (385 positive). Its component
cohorts are:

| Role | Source | Observations | Positive | Negative | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| discovery | [PMID 36254196](https://pubmed.ncbi.nlm.nih.gov/36254196/) | 93 | 49 | 44 | SARS-CoV-2, direct ex-vivo tetramers |
| discovery | [PMID 35484232](https://pubmed.ncbi.nlm.nih.gov/35484232/) | 152 | 66 | 86 | SARS-CoV-2, short-term-restimulated IFN-gamma |
| discovery | [PMID 33521695](https://pubmed.ncbi.nlm.nih.gov/33521695/) | 523 | 523 | 0 | SARS-CoV-2 positives only; cannot support an AUC alone |
| validation | [PMIDs 28386132](https://pubmed.ncbi.nlm.nih.gov/28386132/) and [32983097](https://pubmed.ncbi.nlm.nih.gov/32983097/) | 363 | 131 | 232 | yellow-fever 17D, one laboratory/panel |
| validation | [PMID 38608022](https://pubmed.ncbi.nlm.nih.gov/38608022/) | 911 | 254 | 657 | multi-virus T-cell reactivity; binding-only rows removed |

Thirteen validation observations whose peptide appeared in discovery were
removed. Here “pre-registered” means that the cohort roles, score direction,
matching caliper, and comparisons were committed in git before fetching the
validation exports; it does not mean registration in an external registry.
The protocol history also discloses that the permutation count was increased
after initial results to reduce Monte Carlo error. “Precommitted viral
benchmark” is therefore the less ambiguous name.

### Does the model identify more “virus-y” viral peptides?

There are two different questions:

1. `p(viruses)` asks whether sequence content resembles SwissProt viral
   proteins rather than the other taxonomy bundles. Its held-out taxonomy AUC
   is 0.610 for the fine-tuned ESM-2 model: real signal, but weak.
2. `virus_foreign_like = p(viruses) * (1 - p(mammals))` asks whether that
   content adds T-cell-response ranking within viral assay cohorts after an MHC
   baseline.

For the second question, the fine-tuned ESM result is mixed:

| Assay split | Baseline AUC | With `virus_foreign_like` | Delta AUC (95% CI) |
| --- | ---: | ---: | ---: |
| discovery | 0.9255 | 0.9229 | -0.0027 (-0.0085, +0.0014) |
| validation | 0.6618 | 0.6639 | +0.0021 (-0.0069, +0.0106) |
| validation, source-species adjusted | 0.7137 | 0.7177 | +0.0040 (-0.0050, +0.0132) |
| broad infection development set | 0.7422 | 0.7424 | +0.0002 (-0.0007, +0.0012) |

The validation direction is interesting, but the intervals cross zero, the
discovery direction reverses, and the broad result is effectively zero. A
clean next test should hold out entire viral families represented in each
assay cohort, rank positive against negative peptides within virus/source
species and HLA, and keep MHC presentation in the baseline. This tests
generalizable viral sequence content instead of memorization of related viral
proteins.

### Count-aware matched-strata lockboxes

The next analysis is implemented in
`scripts/tcell_stratified_eval.py`. It
normalizes both assay tables into strata keyed by

```text
(HLA, peptide length, presentation bin, processing bin,
 assay/cohort, exposure, source family)
```

and compares positive and tested-negative assay counts only within those
strata. `n_positive_assays`, `n_negative_assays`, `n_assay_rows`, and
`n_subjects_tested` are preserved. A table that lacks counts (the current
registered validation export is one) is marked
`count_source=binary_outcome_fallback` and receives one explicit tested assay;
it is not silently interpreted as donor-level evidence.

The script makes two independent, label-blind development lockboxes: one in
which complete source families are held out, and one in which complete
TCR-face similarity components are held out. Face components require two
shared four-mers, rather than one common motif, to avoid a biologically
unhelpful giant connected component. The external validation table is never
deleted to force an artificial overlap-free result; group overlap is reported
as an audit. The baseline is a regularized binomial GLM with spline terms for
presentation, processing, affinity, and length plus HLA/cohort/exposure/source
family effects. Reports include within-stratum PR-AUC, top-10%/top-20% yield
and lift, weighted calibration, and bootstrap intervals over leakage groups.

Run it after the feature choices are frozen:

```bash
python scripts/tcell_stratified_eval.py \
  --development tmp/finetune-transfer-report-final/broad-development-transfer/development-scored.csv \
  --validation tmp/finetune-transfer-report-final/broad-development-transfer/validation-scored.csv \
  --out tmp/tcell-stratified-v1 \
  --iedb-raw-dir data/viral-tcell-benchmark
```

When a curation table is available, add
`--neoepitope-pairs pairs.csv`, where `pairs.csv` has
`wild_type_peptide`, `mutant_peptide`, `hla` (and optionally `cohort` and
`exposure`). The explicit table is copied to
`neoepitope-wt-mutant-pairs.csv`; it supplies sequence pairing only and does
not invent response counts. Alternatively, `--iedb-raw-dir` extracts
`related_object__name` (WT) and `assay_antigen__name` (mutant) from cached IEDB
JSON and joins assay context; the current raw exports supply WT sequences but
only a small subset of matching WT assay rows, so response availability is
reported per pair rather than assumed.

The current run writes `stratified-development.csv`, `stratified-validation.csv`,
`lockbox.csv`, `matched-pairs.csv`, and `report.json` under
`tmp/tcell-stratified-v1`. The present assay exports contain no explicit
`wt_peptide`/`wild_type_peptide` column, so the default scored-table-only run
records zero pairs rather than substituting a nearest-human peptide. The
provided raw-IEDB run records the explicit WT sequences and marks their WT
response availability per pair; a true paired response analysis still needs a
curated WT panel for the untested parents.

### Foreignness score families to test

Taxonomy probability is only one framing of non-self. The next model matrix
should keep these scores separate and add them only after the count-aware MHC
baseline has been frozen:

| Family | Construction | Main leakage/control |
| --- | --- | --- |
| exact self | exhaustive exact match in a pinned host reference proteome | selected-host lookup; exact hit overrides learned score |
| nearest self | minimum BLOSUM/PMBEC/physicochemical distance over every host peptide of the same length | exhaustive windows; HLA/length-matched decoys |
| TCR-face rarity | host-vs-pathogen log-odds, k-mer surprisal, or masked-LM likelihood on P3..PΩ-1 | hold out face components and source families |
| mutation contrast | mutant minus WT score, including substitutions at each TCR-facing position | paired HLA/length/presentation/processing assays |
| HLA-conditioned self distance | distance to the presented self ligand distribution for that HLA, not to the whole proteome | HLA Ligand Atlas or matched immunopeptidomics; avoid fixed P3..PΩ-1 alone |
| positive-selection coverage | similarity to the set of peptides likely to have driven thymic selection | report non-monotonic effects; do not assume “farther is always better” |
| protein-LM contrast | pseudo-log-likelihood or embedding distance under host versus pathogen/reference-family models | family-held-out calibration; keep taxonomy prevalence out of the target |

These are complementary hypotheses, not interchangeable definitions. TCR-facing
chemistry should be exposed as position-aware features (hydrophobicity,
aromaticity, charge, polarity, and substitution deltas), with peptide length
and HLA anchor positions modeled separately. HLA Ligand Atlas provides a route
to presentation-conditioned self distributions; PRIME-style allele-aware
motifs provide a stronger chemistry baseline than fixed anchor positions.
Positive-selection analyses also motivate testing a non-monotonic self-distance
term rather than hard-coding “more foreign is more immunogenic.”

The reproducible comparison is in `scripts/audit_foreignness_definitions.py`.
On the frozen registered validation transfer, after joint peptide-component
exclusion, the result was:

| Added score | All validation ΔAUC | Common-exposure viruses ΔAUC |
| --- | ---: | ---: |
| `p_viruses` | +0.0017 | +0.0027 |
| `1 - p_mammals` | +0.0023 | +0.0031 |
| `1 - p_human` | +0.0016 | +0.0020 |
| `p_viruses × (1 - p_mammals)` | +0.0021 | +0.0029 |
| `p_viruses × (1 - p_human)` | +0.0017 | +0.0025 |

The common-exposure set contains 705 validation rows and was specified before
scoring; yellow-fever vaccine/challenge rows were kept separate. Every interval
still crosses zero. The current taxonomy model has no independent pathogen
head, and its `microbial_like` output is effectively the virus head here, so
“pathogen-like” cannot yet be tested as a distinct definition. The full audit
is saved as `tmp/tcell-stratified-v1/foreignness-definition-audit.json`.

## Reproduction

`scripts/peptide_content_data.py` builds the multi-label splits,
`scripts/train_peptide_content_models.py` trains and evaluates the architectures,
and `scripts/evaluate_peptide_content_immunogenicity.py` performs grouped
out-of-fold incremental T-cell-response evaluation. The remote orchestration is
in `scripts/train_peptide_content_modal.py`. `scripts/analyze_taxonomy_buckets.py`
recomputes the proposed bacterial and mammalian bucket counts directly from
SwissProt DAT files.
