# Mammalian viral host range is weakly identifiable from MHC-length peptides and does not improve neoepitope ranking

## Abstract

Sequence similarity to microbial antigens has been proposed as a component of
T-cell recognition and neoantigen quality. We asked whether the broad viral
probability produced by Weirdo could be improved by separating viruses known to
infect mammals from viruses with other hosts. Reviewed Swiss-Prot viral proteins
were labeled from their `OH` host-taxonomy annotations and current NCBI
lineages. Of 7,860,752 viral amino acids, 5,168,799 (65.75%) belonged to viruses
with at least one mammalian host, 2,670,445 (33.97%) to viruses with annotated
non-mammalian hosts, and 21,508 (0.27%) remained unresolved. An ESM-2 8M model
was fine-tuned on leakage-controlled 8--15mer membership labels. Although the
two host-aware heads achieved one-versus-all test AUCs near 0.61, discrimination
between mammal-host and other-host viral peptides was weak: AUC 0.537 for the
direct mammal-host probability and 0.593 for the difference between the two
host-head probabilities. On 17,464 IMPROVE neoepitopes, the general viral score
added more ROC-AUC to an MHC/sequence baseline than the direct mammal-host score
(+0.00306 versus +0.00158), and neither estimate excluded zero. Adding the
mammal-host score after the general viral score changed ROC-AUC by -0.00013
(95% patient-bootstrap interval -0.00120 to +0.00088). These results do not
support replacing or augmenting the general viral score with a mammalian-host
head for MHC-length peptide ranking. They suggest that host-range determinants
require longer protein or genomic context than an isolated epitope provides.

## Introduction

Neoepitope immunogenicity depends on antigen processing and presentation as
well as recognition of the peptide--MHC complex by T cells. Prior work has used
similarity to known pathogen antigens as a component of tumour neoantigen
fitness, motivating sequence-derived measures of microbial resemblance
([Łuksza et al., 2017](https://doi.org/10.1038/nature24473)). Other models find
that physicochemical properties of TCR-facing residues provide complementary
information ([Schmidt et al., 2021](https://doi.org/10.1016/j.xcrm.2021.100194)).
The IMPROVE study assembled 17,500 candidate neoepitopes from 70 patients and
showed that presentation, peptide chemistry, and patient context all contribute
to prediction ([Gartner et al., 2024](https://doi.org/10.3389/fimmu.2024.1360281)).

The existing Weirdo peptide-content model assigns an independent probability
that an 8--15mer occurs in viral proteins. This broad class combines viruses of
mammals with bacteriophages and viruses of plants, fungi, invertebrates, and
other hosts. Because mammal-infecting viruses are more plausible sources of
prior human T-cell exposure, we hypothesized that a host-aware viral label might
provide a more relevant foreignness signal.

Viral host assignment is tractable at the database level. UniProtKB viral
records encode susceptible hosts as NCBI TaxIDs in `OH` lines, and resources
such as the Virus--Host Database have demonstrated the value of linking viral
sequences to curated host taxonomy
([Mihara et al., 2016](https://doi.org/10.3390/v8030066)). Host range can also
be predicted from longer viral sequences. Genome-fragment models and recent
protein-language-model approaches report substantial host-prediction accuracy
([Mock et al., 2021](https://doi.org/10.1093/bioinformatics/btaa740);
[Rasool et al., 2026](https://doi.org/10.1038/s41598-026-37765-8)). Whether the
same information remains detectable in a single MHC-length peptide is unknown.

## Methods

### Viral host labels

The source corpus was the local May 2023 Swiss-Prot viral division: 17,320
reviewed proteins, 2,703 viral TaxIDs, and 7,860,752 amino acids. Host TaxIDs
were parsed from `OH` lines, aggregated across proteins sharing a viral TaxID,
resolved through `merged.dmp`, and traversed through the NCBI taxonomy tree.

A viral TaxID was labeled `mammal_host_viruses` if any explicit host was
Mammalia or a descendant. Viruses with mammalian and non-mammalian hosts,
including vector-borne viruses, remained positive because the label denotes the
ability to infect a mammal rather than mammal exclusivity. Viruses with known
hosts but no mammalian host were labeled `other_host_viruses`. Thirteen viral
TaxIDs without host annotations were left unlabeled. Two `Vertebrata`-only
TaxIDs were resolved as fowlpox and assigned to the non-mammalian group. No
missing host was silently treated as negative.

| Viral host group | TaxIDs | Proteins | Amino acids | Corpus fraction |
| --- | ---: | ---: | ---: | ---: |
| Any mammalian host | 1,676 | 10,198 | 5,168,799 | 65.75% |
| Other annotated hosts | 1,014 | 7,096 | 2,670,445 | 33.97% |
| Unresolved | 13 | 26 | 21,508 | 0.27% |

The mammalian-host group contained 3,973,206 amino acids from viruses annotated
only with mammalian hosts and 1,195,593 amino acids from mixed-host viruses.

### Peptide data and model

Candidate 8--15mers were sampled by source stratum and length. UniProt mnemonic
protein families were assigned to 70/15/15 train/validation/test folds. Exact
peptide membership was then computed against every protein in each fold, and
peptides occurring across folds were removed. The resulting dataset contained
892,585 training, 170,691 validation, and 168,687 test peptides.

The model used the 8-million-parameter ESM-2 encoder
([Lin et al., 2023](https://doi.org/10.1126/science.ade2574)) with a pooled
multi-label sigmoid head over 15 categories. The original `viruses` output was
retained, and `mammal_host_viruses` and `other_host_viruses` were added as
nested outputs. Fine-tuning used binary cross-entropy, AdamW, learning rate
3e-5, batch size 512, and early stopping with patience three. Validation loss
was lowest after epoch 3; training stopped after epoch 6 and restored the
epoch-3 checkpoint.

### Neoepitope evaluation

The IMPROVE table contained 17,520 candidate neoepitopes and 467 recognized
positives. Fifty-six rows, including one positive, overlapped existing project
benchmarks and were excluded before analysis, leaving 17,464 rows, 466
positives, and 70 patients.

Logistic response models were evaluated with five repeats of five-fold
stratified patient-group cross-validation. Training rows sharing an exact mutant
peptide with a held-out fold were purged. The primary baseline contained MHC
binding ranks, peptide length, HLA allele, cohort, hydrophobicity, and aromatic
residue proportion. Frozen peptide-content scores were added individually.
Uncertainty was estimated with 500 patient-cluster bootstrap samples. All score
forms were specified before inspecting their results.

## Results

### Short peptides weakly encode viral host range

Across the full held-out peptide set, the general viral, mammal-host viral, and
other-host viral heads had similar one-versus-all AUCs: 0.6102, 0.6103, and
0.6076, respectively. This apparent performance was largely shared detection
of viral rather than host-specific sequence. Among 32,634 peptides occurring
in exactly one viral host group, the direct mammal-host probability achieved
only 0.5371 AUC. Comparing the two heads as
`p_mammal_host_viruses - p_other_host_viruses` increased within-virus AUC to
0.5933, indicating modest relative host information but poor absolute
separation.

### Host-aware scores did not improve IMPROVE ranking

The primary baseline achieved 0.62077 ROC-AUC. The best tested viral score was
the unchanged general `p_viruses` output. The direct mammal-host probability,
its contrast with mammalian self, and the mammal-versus-other host margin did
not improve upon it.

| Added score | Augmented AUC | Delta AUC | Patient-bootstrap 95% interval |
| --- | ---: | ---: | ---: |
| `p_viruses` | 0.62383 | +0.00306 | -0.00069 to +0.00642 |
| `p_other_host_viruses` | 0.62322 | +0.00245 | -0.00086 to +0.00530 |
| `p_mammal_host_viruses * (1 - p_mammals)` | 0.62310 | +0.00233 | -0.00110 to +0.00528 |
| `p_mammal_host_viruses` | 0.62235 | +0.00158 | -0.00160 to +0.00442 |
| `max(p_mammal_host_viruses, p_other_host_viruses)` | 0.62230 | +0.00153 | -0.00145 to +0.00422 |
| `p_mammal_host_viruses - p_other_host_viruses` | 0.62045 | -0.00031 | -0.00086 to +0.00017 |

The host label also failed an incremental test. Adding
`p_mammal_host_viruses` to a model that already contained `p_viruses` changed
ROC-AUC from 0.62383 to 0.62370: delta -0.00013, with a 95% interval of
-0.00120 to +0.00088. Adding both host heads produced delta -0.00093
(95% interval -0.00201 to +0.00001).

The previously bundled model's `p_viruses` result was stronger (+0.00624,
95% interval +0.00094 to +0.01116), although that checkpoint was trained before
the two new strata were added and is therefore contextual rather than a strict
paired architecture comparison.

## Discussion

The host labels were technically successful: 99.73% of viral amino acids could
be assigned without treating missing annotations as negative, and both sides
contained millions of residues. The negative downstream result therefore does
not arise from an unusably small or grossly imbalanced source corpus.

Instead, the learned outputs indicate that most signal available to an
MHC-length peptide is generic viral composition. Host range often depends on
interactions involving complete receptor-binding, entry, replication, and
immune-evasion proteins. Whole proteins and 100--400 nucleotide fragments used
by successful host-prediction systems retain such context; an isolated 8--15mer
usually does not. The modest 0.593 within-virus AUC from the difference of the
two heads is consistent with weak host-associated composition that is
insufficient to improve neoepitope ranking.

The result also argues against interpreting a viral probability as evidence of
likely prior exposure. In this experiment, `p_other_host_viruses` performed at
least as well as several mammal-specific forms after peptide chemistry was in
the baseline. The useful component of `p_viruses` may therefore be a broad
sequence-composition correlate rather than a calibrated probability of
cross-reactive viral memory.

### Limitations

1. Swiss-Prot is manually reviewed but taxonomically and historically biased;
   host annotations may be incomplete even when present.
2. The fold key is a conservative protein-mnemonic family proxy, not a strict
   ICTV-family holdout. Taxonomy metrics may therefore retain related-virus
   signal, making the weak within-virus result, if anything, more cautionary.
3. The ESM-2 8M encoder is small. Larger protein models can encode more host
   information, but an epitope still lacks whole-protein context.
4. IMPROVE is highly imbalanced and represents one assembled cancer benchmark.
   Confidence intervals account for patients but not for selecting among many
   explored score forms.
5. The study evaluates ranking association, not causal T-cell cross-reactivity
   or patient exposure histories.

## Conclusion

Mammalian viral host range is a high-coverage annotation for reviewed viral
proteins, but it is only weakly identifiable from individual 8--15mer peptides
and does not improve neoepitope-response AUC beyond a general viral score. The
host-aware heads should remain experimental and should not replace the released
`p_viruses` feature. Future work should test whole-protein embeddings,
virus-family-held-out evaluation, and explicit exposure or serology data before
revisiting host-specific viral foreignness.

## Reproducibility

- Host-label and split construction: `scripts/viral_host_peptide_content_data.py`
- Shared exact-membership builder: `scripts/peptide_content_data.py`
- Model training and held-out evaluation: `scripts/train_peptide_content_models.py`
- Patient-grouped IMPROVE evaluation: `scripts/evaluate_improve_foreignness.py`
- Cloud orchestration: `scripts/train_peptide_content_modal.py`
- Split SHA-256: `a1f0b0fc17efb956ff2dbfe5529da100648a119b222e909b5426258aa56ef2c8`

The experimental model is not installed as the package default because the
pre-specified comparison did not support deployment of the new score.
