# Viral T-cell response benchmark

`scripts/benchmark_viral_tcell.py` compares several definitions of peptide
foreignness on exact-HLA viral T-cell assays. It has two halves that must not
be read as one analysis:

- a **discovery** half over three SARS-CoV-2 studies, which is hypothesis
  generation and was free to look at every definition; and
- a **validation** half over two cohorts that contributed nothing to
  discovery, which tests one score, one direction, one comparison and one
  caliper, all fixed in commit `3c40e4d` before the validation data were
  fetched.

Neither half produced a usable foreignness predictor. The pre-registered score
did not validate; the details are below.

The foreignness scores themselves are allele-independent. Presentation and
antigen identity enter the evaluation only through the comparisons:

1. each peptide/HLA pair receives an MHCflurry presentation score;
2. positives and negatives are compared within the same HLA, within an
   MHCflurry presentation-percentile caliper, and in a sensitivity analysis
   within the same HLA *and* the same source organism;
3. a five-fold, peptide-grouped response model tests whether each foreignness
   score adds information beyond MHCflurry presentation and HLA fixed effects;
   and
4. a one-sided label-permutation test (outcomes shuffled within stratum, so
   per-allele counts stay fixed and presentation matching re-forms under every
   permutation) gives a p-value for the pre-registered comparisons.

Keeping these concepts separate is intentional. A WEIRDO score should describe
a peptide's relationship to self. A downstream immunogenicity model can combine
that score with presentation.

## Reproduce the analysis

```bash
python -m pip install -e '.[benchmark]'
mhcflurry-downloads fetch models_class1_presentation
python scripts/benchmark_viral_tcell.py \
  --swissprot-sequences data/swissprot-sequences.csv \
  --per-category 5000 --permutations 2000 --seed 42
```

IEDB exports are downloaded through
[datacache](https://github.com/openvax/datacache). Query URLs, checksums, and
fetch dates remain in datacache provenance receipts. The script writes the
ignored files `data/viral-tcell-benchmark/report.json` and
`data/viral-tcell-benchmark/scored-assays.csv`.

The report records the SwissProt input SHA-256, random seed, model validation
metrics, MHCflurry version/provenance, the pre-registration block, all
assay-level scores, and every enrichment result. The run described below used
MHCflurry 2.3.14 with the local 2.3.0 presentation weights, and covers 2,043
deduplicated assays, 1,598 unique peptides and 51 alleles.

## Cohorts

Cohorts, not PMIDs, are the unit of analysis: the two yellow fever papers share
one peptide panel and one laboratory, so they are merged and deduplicated
rather than counted as independent replications.

| Cohort | Role | Design | Assays (pos/neg) |
| --- | --- | --- | ---: |
| [36254196](https://pmc.ncbi.nlm.nih.gov/articles/PMC9568370/) | discovery | SARS-CoV-2 direct ex-vivo tetramers in convalescent donors | 93 (49/44) |
| [35484232](https://pmc.ncbi.nlm.nih.gov/articles/PMC9064790/) | discovery | SARS-CoV-2 IFN-γ after short-term restimulation, infected and mRNA-vaccinated | 152 (66/86) |
| [33521695](https://pmc.ncbi.nlm.nih.gov/articles/PMC7837622/) | discovery | SARS-CoV-2 direct ex-vivo convalescent responses; positives only | 523 (523/0) |
| yellow fever 17D ([28386132](https://pubmed.ncbi.nlm.nih.gov/28386132/), [32983097](https://pubmed.ncbi.nlm.nih.gov/32983097/)) | validation | Direct ex-vivo multimer screen of predicted binders in YF-17D vaccinees, 43 alleles | 363 (131/232) |
| multi-virus concise set ([38608022](https://pubmed.ncbi.nlm.nih.gov/38608022/)) | validation | T-cell activation against a 12-virus epitope panel in unimmunized healthy donors, 25 alleles | 912 (255/657) |

PMID 33521695 cannot support an AUC and only contributes positive coverage.
In PMID 38608022 each peptide/HLA pair carries two readouts, pMHC multimer
staining (positive for 928 of 929 pairs) and T-cell activation; only activation
is a T-cell response, so the binding rows are excluded instead of being
collapsed against it as contradictory outcomes. IEDB grades some positives
(`Positive-High/Intermediate/Low`); all grades are one positive class, which
affects the validation cohorts only because the discovery papers report
ungraded outcomes. Repeat assays of the same peptide/HLA pair within a cohort
are collapsed, and pairs with contradictory outcomes are dropped: one in the
yellow fever cohort, where the two papers disagree, and one in the infected
stratum of PMID 35484232.

## Foreignness definitions

The benchmark trains on one randomly sampled 8–14mer per SwissProt source
protein. It uses an 80/20 organism-stratified split and never trains on T-cell
outcomes. The TCR-face view uses P3 through the penultimate residue, excluding
the common class-I P2 and C-terminal anchors, aligned into fixed positional
bins plus composition and peptide length.

| Definition | Representation | Training objective |
| --- | --- | --- |
| Multiclass MLP posterior | Whole peptide | Ten SwissProt organism categories, converted to `1 - P(human)`, pathogen-vs-self, or virus-vs-human scores |
| Logistic regression | Whole peptide or TCR face | Bacteria/virus versus human/mammal/rodent |
| Small MLP | TCR face | Bacteria/virus versus human/mammal/rodent |
| Constrained Extra Trees | TCR face | Bacteria/virus versus human/mammal/rodent |
| Logistic regression | TCR face | Virus versus human |
| Nearest-human distance | TCR face | Minimum normalized BLOSUM62 distance to a deterministic, length-matched human reference |
| Residue log odds | TCR face | Smoothed virus-versus-human amino-acid log odds |
| Human 3-mer/4-mer surprisal | TCR face | Rarity under every human SwissProt protein; no viral training data |

The binary SwissProt validation AUCs were 0.591–0.619; ten-category MLP
accuracy was 0.133. These deliberately small models are weak organism
classifiers. The nearest-human and human-surprisal scores use no viral training
examples at all, so no virus family can leak into them; the deterministic
human reference holds 142,897 peptides, one window of every supported length
per eligible human protein, and does not depend on the training seed.

## Discovery result

No definition showed repeatable enrichment after conditioning on presentation.
The nearest-human BLOSUM62 TCR-face distance was the only weak
infection-specific lead: within-HLA AUC 0.591 and 0.620 in the two infected
cohorts, near null after presentation matching in the direct ex-vivo study,
large in the smaller restimulation study, and reversed in mRNA-vaccinated
donors.

| Cohort | MHCflurry presentation AUC | Nearest-human AUC | Within-HLA AUC | Matched AUC (0.1) | Matched AUC (0.05) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 36254196, infected | 0.699 | 0.537 | 0.591 | 0.533 (167 pairs) | 0.518 (110 pairs) |
| 35484232, infected | 0.637 | 0.619 | 0.620 | 0.821 (56 pairs) | 0.903 (31 pairs) |
| 35484232, vaccinated | 0.432 | 0.475 | 0.357 | 0.348 (56 pairs) | 0.520 (25 pairs) |

The 0.903 value comes from 31 non-independent pairs in a short-term
restimulation study, while the stronger direct ex-vivo study is near null after
matching, so it was never confirmatory. Adding the nearest-human score to
presentation plus HLA changed cross-validated AUC by -0.009, +0.006 and -0.005
in those three rows. The full-proteome surprisal scores, which cannot memorize
viruses, were also null, and the learned taxonomy classifiers were mostly
*below* 0.5 in the infected cohorts, which is a reverse association rather than
enrichment.

On that evidence the nearest-human TCR-face distance was nominated as the
single least weak candidate, and its formula, direction, comparison and caliper
were frozen in commit `3c40e4d`.

## Pre-registered validation

The rule recorded in `report.json` under `validation_prespecification`:

- **score** `nearest_human_blosum62_tcr`, **direction** higher distance
  predicts a positive response;
- **comparisons** within-HLA AUC and presentation-matched AUC at the 0.1
  percentile-point caliper;
- **inference** one-sided within-stratum label permutation, 2,000 draws,
  α = 0.05;
- **decision** validated only if *both* comparisons exceed 0.5 with p < α in
  *every* validation cohort. The two human-surprisal scores are reported as
  secondary context and cannot change the decision.

### Outcome: not validated

| Cohort | Score | Within-HLA AUC (p) | Matched AUC (p) | Same HLA **and** virus (p) |
| --- | --- | ---: | ---: | ---: |
| yellow fever 17D | MHCflurry presentation | 0.809 | 0.677 | 0.677 |
| yellow fever 17D | **nearest-human (primary)** | 0.478 (0.69) | 0.527 (0.29) | 0.527 (0.29) |
| yellow fever 17D | human 3-mer surprisal | 0.498 (0.52) | 0.444 (0.85) | 0.444 (0.85) |
| yellow fever 17D | human 4-mer surprisal | 0.483 (0.65) | 0.449 (0.83) | 0.449 (0.83) |
| multi-virus set | MHCflurry presentation | 0.605 | 0.507 | 0.478 |
| multi-virus set | **nearest-human (primary)** | 0.563 (0.013) | 0.560 (0.051) | 0.531 (0.30) |
| multi-virus set | human 3-mer surprisal | 0.556 (0.023) | 0.574 (0.016) | 0.544 (0.21) |
| multi-virus set | human 4-mer surprisal | 0.544 (0.054) | 0.562 (0.038) | 0.551 (0.17) |

Pair counts: 1,420 within-HLA and 405 matched in yellow fever; 38,569
within-HLA, 7,616 matched and 813 matched within HLA and source organism in the
multi-virus set. The yellow fever cohort has one source organism, so its
source-stratified column is identical by construction, which is a useful check
that the extra stratum behaves as intended.

The primary score is null in the yellow fever cohort in both pre-registered
comparisons. It clears the unmatched comparison in the multi-virus cohort
(AUC 0.563, p = 0.013) and misses the matched one at the boundary
(AUC 0.560, p = 0.051), so that cohort fails the conjunctive rule too.
`validation_primary_outcome.validated` is therefore `false`.

The multi-virus signal is stable across calipers (matched AUC 0.556–0.560 from
0.02 to 0.25 percentile points), so it is not an artifact of the matching
threshold, but it falls to 0.531 (p = 0.30) once pairs must share both the
allele and the source virus. Part of that drop is lost power — the control
keeps 813 of 7,616 pairs — yet the effect size shrinks along with the p-value,
so the signal is at least as consistent with a between-virus composition
difference as with foreignness. The human 3-mer surprisal is the only score to
clear both arms in that cohort, and it behaves the same way under the control
(0.574 → 0.544, p = 0.21). With three leakage-free scores, two comparisons and
two cohorts, a single nominal pass is weak evidence; nothing survives a
Bonferroni correction over those twelve tests.

In the conditional response models, adding the primary score to presentation
plus HLA moved cross-validated AUC by -0.007 in yellow fever and +0.006 in the
multi-virus cohort. Every definition had a positive standardized coefficient in
all five multi-virus folds, with deltas from +0.001 to +0.016 — a coherent but
very small increment in the one cohort where peptides were pre-selected for
stable pMHC binding, and absent where they were not.

The learned taxonomy classifiers reached matched AUC 0.607 (Extra Trees) and
0.630 (virus-versus-human logistic) in yellow fever, with flaviviruses held out
of training. Those definitions were *below* 0.5 in both infected discovery
cohorts, so this is a direction flip between cohorts, not a replication, and
they were not pre-registered. The multi-virus cohort cannot test them cleanly:
its influenza, herpesvirus, papillomavirus, poxvirus and retrovirus source
organisms are not excluded from the viral training corpus, so those rows carry
family-level leakage that the primary and secondary scores do not.

Presentation remains the only strong, repeatable predictor in this benchmark:
within-HLA AUC 0.809 in yellow fever and 0.605 in the multi-virus cohort.

## Leakage and independence controls

- 418 coronavirus-mnemonic and 126 flavivirus-mnemonic SwissProt proteins are
  excluded from the viral training sample, covering the discovery virus and the
  yellow fever validation virus at family level. This is still not a
  homology-clustered split of the full corpus; that work is tracked in
  [weirdo #6](https://github.com/pirl-unc/weirdo/issues/6).
- One of the 1,598 unique assay peptides also appeared in the sampled training
  peptides and was removed before training.
- 13 validation rows whose peptide had already been tested in a discovery
  cohort were dropped, so validation is genuinely out of sample.
- Foreignness models never see assay outcomes; assay outcomes are only used for
  evaluation and for the cross-validated conditional models.
- Random streams are seeded per organism category, so changing a viral
  exclusion cannot perturb the human reference sample.

## Interpretation limits

MHCflurry presentation combines exact-HLA affinity with peptide-only
processing, because reliable source-protein flanks are not available for all
selected IEDB rows; this is weaker than a flank-aware natural-presentation
analysis.

These studies establish T-cell recognition of supplied peptide/HLA complexes,
or responses after peptide restimulation. They do not prove that infected cells
naturally process and present every peptide. Tetramer and multimer candidates
were selected partly for predicted or measured HLA binding, which creates
ascertainment bias, and the two validation cohorts differ in what their
negatives mean: yellow fever negatives are predicted binders that were never
recognized, whereas multi-virus negatives are peptides with confirmed stable
pMHC that were still not recognized. Restriction evidence, assay method,
readout, exposure and source annotations stay in the scored assay file.

The remaining implementation and validation gaps are tracked in
[weirdo #4](https://github.com/pirl-unc/weirdo/issues/4),
[#5](https://github.com/pirl-unc/weirdo/issues/5),
[#7](https://github.com/pirl-unc/weirdo/issues/7), and
[#8](https://github.com/pirl-unc/weirdo/issues/8).
