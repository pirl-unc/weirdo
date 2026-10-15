# Exploratory viral T-cell response benchmark

`scripts/benchmark_viral_tcell.py` compares several definitions of peptide
foreignness on exact-HLA viral T-cell assays. This is a discovery analysis,
not a released WEIRDO predictor.

The foreignness scores themselves are allele-independent. Presentation enters
the evaluation in three explicit ways:

1. each peptide/HLA pair receives an MHCflurry presentation score;
2. positives and negatives are compared within the same HLA and within a
   specified MHCflurry presentation-percentile caliper; and
3. a five-fold, peptide-grouped response model tests whether each foreignness
   score adds information beyond MHCflurry presentation and HLA fixed effects.

Keeping these concepts separate is intentional. A future WEIRDO score should
describe the peptide's relationship to self. A downstream immunogenicity or
ranking model can combine that score with presentation.

## Reproduce the analysis

```bash
python -m pip install -e '.[benchmark]'
mhcflurry-downloads fetch models_class1_presentation
python scripts/benchmark_viral_tcell.py \
  --swissprot-sequences data/swissprot-sequences.csv \
  --per-category 5000
```

IEDB exports are downloaded through
[datacache](https://github.com/openvax/datacache). Query URLs, checksums, and
fetch dates remain in datacache provenance receipts. The script writes the
ignored files `data/viral-tcell-benchmark/report.json` and
`data/viral-tcell-benchmark/scored-assays.csv`.

The report records the SwissProt input SHA-256, random seed, model validation
metrics, MHCflurry version/provenance, all assay-level scores, and every
enrichment result. The run described below used MHCflurry 2.3.14 and the local
2.3.0 presentation weights.

## Foreignness definitions

The benchmark trains on one randomly sampled 8–14mer per SwissProt source
protein. It uses an 80/20 organism-stratified split and never trains on the
T-cell outcomes. The TCR-face view uses P3 through the penultimate residue,
excluding the common class-I P2 and C-terminal anchors. It aligns this face
into fixed positional bins and adds composition and peptide length.

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

The binary SwissProt validation AUCs were 0.568–0.625; ten-category MLP
accuracy was 0.134. These deliberately small models are weak organism
classifiers. The nearest-human and human-surprisal scores use no viral training
examples at all.

The run excluded 418 proteins with SARS or coronavirus SwissProt mnemonics,
including `_CV...`, MERS, PEDV, FIPV, and IBV conventions. None of the 521
unique assay peptides occurred exactly in the sampled training peptides.
This blocks exact and obvious coronavirus-family leakage, but it is not yet a
homology-clustered split across the full SwissProt corpus. That remaining work
is tracked in [weirdo #6](https://github.com/pirl-unc/weirdo/issues/6).

The nearest-human reference contains 142,897 peptides: one deterministic
window of every supported length from each eligible human protein. Its value
does not depend on the training seed.

## Discovery result

No definition showed repeatable enrichment after conditioning on presentation.
The nearest-human BLOSUM62 TCR-face distance is the only weak infection-specific
lead: its within-HLA AUC was 0.591 and 0.620 in the infected cohorts. The
presentation-matched result was nearly null in the direct ex-vivo study and
large in the smaller restimulation study. The vaccinated cohort reversed the
association.

| Cohort | MHCflurry presentation AUC | Nearest-human AUC | Within-HLA AUC | Presentation-matched AUC (0.1) | Matched AUC (0.05) |
| --- | ---: | ---: | ---: | ---: | ---: |
| PMID 36254196, infected | 0.699 | 0.537 | 0.591 | 0.533 (167 pairs) | 0.518 (110 pairs) |
| PMID 35484232, infected | 0.637 | 0.619 | 0.620 | 0.821 (56 pairs) | 0.903 (31 pairs) |
| PMID 35484232, vaccinated | 0.432 | 0.475 | 0.357 | 0.348 (56 pairs) | 0.520 (25 pairs) |

The 0.903 value comes from only 31 non-independent positive/negative pairs in
a short-term-restimulation study. The stronger direct ex-vivo study is near
null after presentation matching, so the large value is not confirmatory.

Adding the nearest-human score to presentation plus HLA changed
cross-validated AUC by -0.009, +0.006, and -0.005 in the three rows above. It
slightly improved log loss only in the second infected cohort. Its standardized
coefficient was positive in all five folds of that cohort, but only two of five
and one of five folds in the other cohorts.

The full-human-proteome surprisal scores avoid viral memorization and sampling
of the self reference. They were also null. The 3-mer score's unadjusted AUCs
were 0.518, 0.542, and 0.513, while its presentation-matched AUCs were 0.533,
0.536, and 0.357. The 4-mer score's matched AUCs were 0.473, 0.571, and 0.429.
Neither improved the conditional response model consistently.

The learned taxonomy classifiers did not yield repeatable positive
enrichment. Most had matched AUC below 0.5 in the infected cohorts. Some raised
conditional-model AUC by learning a negative coefficient, which is evidence
of a reverse association rather than enrichment for foreignness. In the
vaccinated cohort, the constrained TCR-face tree model added 0.013 AUC and
slightly improved log loss, but this did not repeat in the infected cohorts
with a positive coefficient. No learned definition improved both
discrimination and log loss across cohorts.

If one score is carried into a new, untouched validation dataset, the
nearest-human TCR-face distance is the least weak candidate because its
within-HLA direction repeated in both infected cohorts and it cannot memorize
viral families. Its formula and direction should be fixed before that dataset
is opened. The caliper sensitivity results at 0.02, 0.05, 0.1, and 0.25
percentile points are all stored in `report.json`; this avoids choosing the
matching threshold after seeing one favorable value.

## Assay data and interpretation

| PMID | Evidence | Eligible exact-HLA class-I records |
| --- | --- | ---: |
| [36254196](https://pmc.ncbi.nlm.nih.gov/articles/PMC9568370/) | Direct ex-vivo tetramers in convalescent donors; tested positive and negative peptide/HLA pairs | 93 |
| [35484232](https://pmc.ncbi.nlm.nih.gov/articles/PMC9064790/) | IFN-γ responses after short-term restimulation in infected and mRNA-vaccinated donors; repeated peptide/HLA observations | 167 |
| [33521695](https://pmc.ncbi.nlm.nih.gov/articles/PMC7837622/) | Direct ex-vivo convalescent responses; positives only in this export | 523 |

After outcome deduplication and removal of conflicts, the two studies with
tested negatives contribute 49/44 positive/negative assays for PMID 36254196,
27/37 for the infected stratum of PMID 35484232, and 37/30 for its vaccinated
stratum. PMID 33521695 is useful for positive coverage but cannot support an
AUC.

MHCflurry presentation combines exact-HLA affinity with peptide-only
processing here. Reliable source-protein coordinates and flanks are not
available for all selected IEDB rows, so this is weaker than a flank-aware
natural-presentation analysis.

These studies establish T-cell recognition of supplied peptide/HLA complexes
or responses after peptide restimulation. They do not prove that infected
cells naturally process and present every peptide. Tetramer candidates were
also selected partly for HLA binding, which creates ascertainment bias.
Restriction evidence, assay method, exposure, and source annotations remain
in the scored assay file.

The implementation and validation gaps are tracked in
[weirdo #4](https://github.com/pirl-unc/weirdo/issues/4),
[#5](https://github.com/pirl-unc/weirdo/issues/5),
[#7](https://github.com/pirl-unc/weirdo/issues/7), and
[#8](https://github.com/pirl-unc/weirdo/issues/8).
