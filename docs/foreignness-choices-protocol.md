# Foreignness-choice benchmark: protocol

This page fixes how candidate definitions of foreignness are compared, selected
and tested, **before any candidate is evaluated against T-cell outcomes** on the
development cohorts. Its commit hash and timestamp are the registration. Changes
made afterwards are listed under [Deviations](#deviations) with the date and the
reason; nothing is edited in place.

The question is the one in [weirdo #8](https://github.com/pirl-unc/weirdo/issues/8):
does any definition of how foreign a peptide is add information about T-cell
recognition beyond MHC presentation, and does it do so on data that did not
choose it? The previous benchmark
([viral T-cell benchmark](viral-tcell-benchmark.md)) found no validated score,
with weak power and one cheap composition-like association. This protocol widens
the evidence and tests a larger set of choices under one fixed procedure.

## Domains

| Domain | Peptides | Why |
| --- | --- | --- |
| `viral` | Pathogen-derived peptides | Where most T-cell data with tested negatives exist |
| `neoepitope` | Peptides carrying tumour mutations | Where nearest-self distance has real dynamic range, and what the issue requires before WEIRDO can affect vaccine ranking |

Domains are analysed and decided separately. A definition is never credited in
one domain for a result in the other.

## Data

All data come from IEDB T-cell assay exports (class I, human host, exact HLA
allele, 8-14mers, positive or negative outcome), plus the HLA Ligand Atlas
(CC-BY 4.0; Marcu et al., JITC 2021) as a self-immunopeptidome reference.
Cohorts are listed in `scripts/foreignness_cohorts.py`; they were chosen from a
census of publications by counts and titles only, with no score computed.

- Only direct ex-vivo and short-term restimulated rows are used. Rows from
  engineered TCR lines and clones test a receptor, not a donor repertoire.
- Assays that measure pMHC binding rather than a T-cell response are removed.
- A peptide-HLA pair within a cohort and exposure stratum is **Positive if any
  assay is positive and Negative only if every assay is negative**, which is how
  IEDB calls an epitope. An earlier draft of this rule dropped pairs with mixed
  assays; that was changed before any scoring because it kept 19 of 67 positive
  pairs in one neoantigen cohort and 1 of 31 in another.
- A cohort enters the pooled model with at least 5 positives and 5 negatives
  after preparation, and is reported on its own with at least 15 of each.

### Development and holdout

Each cohort is assigned a role by the rule in `assign_role`: four prior viral
cohorts stay in development; the vaccinia Dryvax screen and the TESLA
consortium cohort (PMID 33038342) are held out by design as the largest
binder-matched viral screen and the standard neoepitope benchmark; every other
cohort is hashed (SHA-256 of a fixed salt and the cohort id), and a quarter are
held out.

The development stage never reads holdout rows. Holdout rows whose peptide
shares a 7-mer with any development peptide of the same domain are dropped from
the holdout before it is scored, because a peptide tested in both would let the
development data speak for the holdout.

Known properties of the holdout, stated in advance:

- **Vaccinia Dryvax** (IEDB direct submission 1000279/1000276): an ELISPOT screen
  of a predicted-binder library in vaccinees. Many assays are peptide pools, so a
  peptide's negative is inferred from a negative pool, and restriction evidence
  is "Not determined", so the HLA is the allele that nominated the peptide. It is
  large (about 21,400 pairs, 137 positive) and one virus.
- **Neoepitope holdout**: TESLA plus one hashed cohort, 64 positive pairs among
  1,688. It is **underpowered** for modest effects.

Approximate power of the holdout test (one-sided alpha = 0.025, a unit-variance
score, with 0-25% of the information lost to covariate adjustment):

| True odds ratio per SD | Neoepitope holdout | Vaccinia holdout |
| ---: | ---: | ---: |
| 1.10 | 9-11% | 16-20% |
| 1.20 | 24-30% | 45-57% |
| 1.30 | 43-54% | 76-86% |
| 1.50 | 79-89% | 98-100% |

So an effect the size of the best viral association in the previous benchmark
(about 1.2 per SD) would be confirmed in the neoepitope holdout roughly a quarter
of the time even if it were real. **A failed holdout test will therefore be
inconclusive about small effects**, and the report will say which effect sizes it
can and cannot exclude. Only a large effect can be confirmed here.

## Candidates

Each candidate is a score per peptide (per peptide-allele pair where the
reference depends on the allele). The incumbent is the previously nominated
score. Registered candidates:

| Id | Family | Definition |
| --- | --- | --- |
| `nn_prereg` | incumbent | Nearest-human BLOSUM62 distance on the TCR face, one sampled window per protein |
| `nn_full` | reference | Exact nearest distance to every human window (TCR face, BLOSUM62) |
| `soft_full` | reference | Soft maximum, beta = 1, over every human window |
| `nn_full_whole` | reference | As `nn_full` over the whole peptide, anchors included |
| `nn_allele_motif` | presentation-conditioned | Exact nearest distance among human windows whose P2/P-omega pair is compatible with the query allele |
| `soft_allele_motif` | presentation-conditioned | Soft maximum over the same allele-compatible windows |
| `nn_atlas_all` | presented self | Nearest distance to HLA Ligand Atlas class-I ligands of the same length |
| `nn_atlas_allele` | presented self | Nearest distance to Atlas ligands predicted to bind the query allele (pooled fallback below 200) |
| `soft_atlas_allele` | presented self | Soft maximum over the same set |
| `nn_full_blosum50` | matrix | `nn_full` with BLOSUM50 |
| `nn_full_pmbec` | matrix | `nn_full` with the PMBEC matrix |
| `nn_full_physchem` | matrix | `nn_full` in a space of hydropathy, volume and polarity from the package tables |
| `nn_full_hamming` | matrix | Fraction of mismatched face positions to the nearest window |
| `kmer_absent_4`, `kmer_absent_5`, `kmer_absent_6` | rarity | Fraction of face k-mers absent from the human proteome |
| `surprisal_3`, `surprisal_4`, `surprisal_5` | rarity | Mean smoothed surprisal of face k-mers under the human proteome |
| `cond_logistic`, `cond_mlp_small`, `cond_mlp_large`, `cond_trees` | learned, presentation-matched | Probability that a binder is non-self, trained on human versus non-viral, non-human peptides **matched on predicted presentation**; no virus is ever in training |
| `mismatch_whole`, `mismatch_face` | mutation comparators | Mismatches to the nearest human window, whole peptide or face |
| `dai` | mutation comparator | Differential agretopicity: predicted affinity of the nearest human window over the peptide's |

Baselines enter the models as covariates, not as candidates: TCR-face
hydropathy and TCR-face aromatic fraction.

A candidate may be added later, but it is then recorded under Deviations and its
development numbers are exploratory. The holdout is not used to choose.

## Estimand

For candidate `s` in a domain, the pooled effect is the coefficient of the
within-stratum standardized score in a logistic regression with: stratum (cohort
and exposure) fixed effects; a restricted cubic spline of the logit MHCflurry
presentation score; allele and, for viral peptides, source-species fixed effects
(levels under 15 rows pooled); peptide-length effects; log assay count and log
subjects tested; and the two composition baselines. Standard errors are
clustered on peptide-overlap clusters (peptides sharing a 7-mer), merged with
same-cohort source-protein groups where a protein is annotated. A per-cohort
version without stratum effects gives the sign-consistency check and a random
effects (DerSimonian-Laird) summary.

## Selection

For each domain, from candidates defined for at least 90% of development rows:

1. keep candidates whose pooled coefficient is positive and positive in at least
   60% of cohorts reported alone;
2. keep those with pooled z of at least 1.645;
3. select the largest z; ties go to the simpler family in the order of the table.

If none survive, **no candidate advances and the holdout is not evaluated**. That
is a reported outcome.

The development ledger lists every candidate tried and its Bonferroni-adjusted
p. It is optimistic by construction, because the maximum of many candidates is
chosen on it; the holdout is the unbiased estimate.

## Holdout test

The selected candidate per domain is committed to this page's Deviations log and
to `scripts/benchmark_foreignness_choices.py` **before** the holdout stage runs.
It is then evaluated once, in the same model, on that domain's holdout.

It counts as **confirmed** in a domain only if both hold: a one-sided
cluster-robust z above 1.96 (alpha = 0.025, splitting 0.05 across two domains)
and an odds ratio of at least 1.10 per SD. The size criterion exists because a
very large holdout can make a trivial effect significant.

A confirmed neoepitope result is the minimum before WEIRDO may influence
default neoepitope ranking, and even then it is evidence about the score, not
an implementation. A viral-only result does not license that.

## Development results

Run on 2026-10-15 after the protocol commit `b58da11`, over 44 development
cohorts: 31 viral (5,476 assays, 1,622 positive) and 13 neoepitope (2,583
assays, 486 positive). Peptide-overlap clusters number 3,773, merging to 1,751
once same-cohort source proteins are joined. The learned classifiers trained on
56,142 presentation-matched peptides and reached validation AUC 0.625-0.638
with binding held equal between the classes. The Atlas reference holds 90,425
class-I ligands across 51 alleles. The anchor-motif estimator recovers the
expected motifs (HLA-A*02:01: L/M at P2 with V/L/I at P-omega).

In every table the odds ratio is per within-cohort SD of the score, adjusted as
in [Estimand](#estimand); `Cohorts +` is the share of cohorts reported alone
(15 or more of each outcome) with a positive coefficient, and `RE OR` is the
DerSimonian-Laird summary of those cohort estimates. p is one-sided; the
Bonferroni column multiplies by the 26 candidates tried. These numbers are
optimistic, because the best of 26 was chosen on them.

### Viral: no candidate advances

| Candidate | Family | Coverage | OR/SD (95% CI) | z | p (Bonf.) | Cohorts + | RE OR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cond_trees` | learned, presentation-matched | 1.00 | 1.06 (0.99-1.13) | +1.68 | 0.047 (1.00) | 50% (18) | 0.94 |
| `nn_prereg` | incumbent | 1.00 | 1.07 (0.97-1.18) | +1.40 | 0.081 (1.00) | 61% (18) | 1.07 |
| `cond_logistic` | learned, presentation-matched | 1.00 | 1.05 (0.98-1.12) | +1.39 | 0.083 (1.00) | 39% (18) | 0.92 |
| `cond_mlp_small` | learned, presentation-matched | 1.00 | 1.05 (0.98-1.12) | +1.31 | 0.095 (1.00) | 39% (18) | 0.95 |
| `soft_allele_motif` | presentation-conditioned | 1.00 | 1.08 (0.96-1.21) | +1.31 | 0.096 (1.00) | 67% (18) | 1.14 |
| `nn_allele_motif` | presentation-conditioned | 1.00 | 1.06 (0.96-1.17) | +1.19 | 0.118 (1.00) | 67% (18) | 1.13 |
| `kmer_absent_5` | rarity | 1.00 | 1.04 (0.96-1.11) | +0.96 | 0.169 (1.00) | 44% (18) | 1.27 |
| `nn_full_whole` | reference | 1.00 | 1.04 (0.95-1.14) | +0.89 | 0.187 (1.00) | 78% (18) | 1.17 |
| `kmer_absent_4` | rarity | 1.00 | 1.01 (0.98-1.05) | +0.75 | 0.225 (1.00) | 100% (6) | 1.94 |
| `surprisal_4` | rarity | 1.00 | 1.03 (0.95-1.12) | +0.74 | 0.231 (1.00) | 50% (18) | 0.99 |
| `cond_mlp_large` | learned, presentation-matched | 1.00 | 1.03 (0.96-1.10) | +0.73 | 0.233 (1.00) | 56% (18) | 0.96 |
| `surprisal_3` | rarity | 1.00 | 1.03 (0.94-1.13) | +0.70 | 0.243 (1.00) | 50% (18) | 0.99 |
| `mismatch_whole` | mutation comparator | 1.00 | 1.03 (0.90-1.19) | +0.45 | 0.325 (1.00) | 67% (18) | 1.07 |
| `dai` | mutation comparator | 1.00 | 1.01 (0.94-1.09) | +0.31 | 0.377 (1.00) | 44% (18) | 0.96 |
| `surprisal_5` | rarity | 1.00 | 1.01 (0.92-1.12) | +0.29 | 0.385 (1.00) | 44% (18) | 1.02 |
| `soft_full` | reference | 1.00 | 1.01 (0.85-1.20) | +0.09 | 0.463 (1.00) | 56% (18) | 1.08 |
| `nn_atlas_all` | presented self | 0.90 | 1.00 (0.93-1.07) | -0.08 | 0.533 (1.00) | 44% (16) | 1.00 |
| `nn_full_pmbec` | matrix | 1.00 | 0.99 (0.90-1.08) | -0.23 | 0.592 (1.00) | 56% (18) | 1.14 |
| `nn_full` | reference | 1.00 | 0.98 (0.85-1.14) | -0.24 | 0.594 (1.00) | 56% (18) | 1.10 |
| `nn_full_physchem` | matrix | 1.00 | 0.98 (0.84-1.14) | -0.26 | 0.602 (1.00) | 72% (18) | 1.12 |
| `nn_full_blosum50` | matrix | 1.00 | 0.98 (0.84-1.14) | -0.28 | 0.612 (1.00) | 61% (18) | 1.12 |
| `soft_atlas_allele` | presented self | 0.90 | 0.97 (0.89-1.07) | -0.59 | 0.723 (1.00) | 50% (16) | 1.03 |
| `nn_atlas_allele` | presented self | 0.90 | 0.97 (0.88-1.07) | -0.60 | 0.727 (1.00) | 44% (16) | 1.04 |
| `mismatch_face` | mutation comparator | 1.00 | 0.98 (0.90-1.05) | -0.64 | 0.739 (1.00) | 61% (18) | 1.10 |
| `nn_full_hamming` | matrix | 1.00 | 0.97 (0.91-1.04) | -0.85 | 0.802 (1.00) | 67% (18) | 1.08 |
| `kmer_absent_6` | rarity | 0.97 | 0.95 (0.88-1.02) | -1.50 | 0.933 (1.00) | 56% (18) | 1.06 |

No candidate meets the gate. The best, a presentation-matched tree ensemble,
has z = +1.68 and is positive in only half the cohorts; the incumbent reaches
+1.40. Between-cohort heterogeneity is very high (I-squared about 0.85). Exact
and soft nearest-self distance, the presented-self references from the HLA
Ligand Atlas, and the allele-motif references do no better. Under the
registered rule the **viral holdout is not evaluated** and stays sealed for a
future candidate.

### Neoepitope: one selection

| Candidate | Family | Coverage | OR/SD (95% CI) | z | p (Bonf.) | Cohorts + | RE OR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `nn_full_whole` | reference | 1.00 | 1.21 (1.05-1.40) | +2.70 | 0.003 (0.09) | 100% (7) | 1.29 |
| `mismatch_face` | mutation comparator | 1.00 | 1.21 (1.04-1.40) | +2.54 | 0.005 (0.14) | 57% (7) | 1.15 |
| `nn_full_hamming` | matrix | 1.00 | 1.20 (1.04-1.38) | +2.49 | 0.006 (0.17) | 57% (7) | 1.15 |
| `nn_full_physchem` | matrix | 1.00 | 1.20 (1.03-1.40) | +2.35 | 0.009 (0.24) | 57% (7) | 1.16 |
| `nn_full_pmbec` | matrix | 1.00 | 1.20 (1.03-1.39) | +2.30 | 0.011 (0.28) | 86% (7) | 1.30 |
| `kmer_absent_6` | rarity | 0.97 | 1.20 (1.02-1.41) | +2.25 | 0.012 (0.32) | 67% (6) | 1.36 |
| `kmer_absent_5` | rarity | 1.00 | 1.13 (1.00-1.27) | +1.91 | 0.028 (0.73) | 20% (5) | 0.94 |
| `nn_full` | reference | 1.00 | 1.15 (0.99-1.34) | +1.88 | 0.030 (0.77) | 57% (7) | 1.17 |
| `soft_full` | reference | 1.00 | 1.18 (0.99-1.40) | +1.84 | 0.033 (0.85) | 86% (7) | 1.29 |
| `nn_full_blosum50` | matrix | 1.00 | 1.15 (0.99-1.33) | +1.81 | 0.035 (0.91) | 71% (7) | 1.16 |
| `surprisal_3` | rarity | 1.00 | 1.15 (0.99-1.33) | +1.80 | 0.036 (0.93) | 57% (7) | 1.13 |
| `mismatch_whole` | mutation comparator | 1.00 | 1.11 (0.98-1.27) | +1.65 | 0.050 (1.00) | 83% (6) | 1.07 |
| `surprisal_4` | rarity | 1.00 | 1.12 (0.97-1.30) | +1.52 | 0.064 (1.00) | 57% (7) | 1.12 |
| `nn_allele_motif` | presentation-conditioned | 1.00 | 1.12 (0.94-1.34) | +1.28 | 0.100 (1.00) | 86% (7) | 1.24 |
| `soft_allele_motif` | presentation-conditioned | 1.00 | 1.12 (0.93-1.35) | +1.21 | 0.112 (1.00) | 86% (7) | 1.24 |
| `surprisal_5` | rarity | 1.00 | 1.08 (0.94-1.25) | +1.06 | 0.145 (1.00) | 57% (7) | 1.05 |
| `nn_prereg` | incumbent | 1.00 | 1.05 (0.88-1.24) | +0.53 | 0.297 (1.00) | 71% (7) | 1.10 |
| `cond_mlp_large` | learned, presentation-matched | 1.00 | 1.03 (0.91-1.18) | +0.50 | 0.307 (1.00) | 43% (7) | 1.05 |
| `cond_logistic` | learned, presentation-matched | 1.00 | 1.03 (0.90-1.19) | +0.42 | 0.338 (1.00) | 29% (7) | 0.86 |
| `cond_trees` | learned, presentation-matched | 1.00 | 1.04 (0.88-1.22) | +0.42 | 0.339 (1.00) | 29% (7) | 0.88 |
| `nn_atlas_all` | presented self | 1.00 | 1.02 (0.87-1.20) | +0.28 | 0.390 (1.00) | 86% (7) | 1.28 |
| `cond_mlp_small` | learned, presentation-matched | 1.00 | 1.01 (0.89-1.16) | +0.19 | 0.426 (1.00) | 43% (7) | 0.88 |
| `soft_atlas_allele` | presented self | 1.00 | 1.00 (0.84-1.21) | +0.05 | 0.479 (1.00) | 71% (7) | 1.31 |
| `nn_atlas_allele` | presented self | 1.00 | 1.00 (0.84-1.20) | +0.03 | 0.490 (1.00) | 71% (7) | 1.29 |
| `dai` | mutation comparator | 1.00 | 0.96 (0.81-1.12) | -0.55 | 0.710 (1.00) | 57% (7) | 0.95 |
| `kmer_absent_4` | rarity | 1.00 | 0.94 (0.81-1.09) | -0.84 | 0.799 (1.00) | 100% (1) | - |

`nn_full_whole`, the exact BLOSUM62 distance from the whole peptide to its
nearest human window, is selected: OR 1.21 per SD (1.05-1.40), z = +2.70, a
positive coefficient in all seven cohorts reported alone and a homogeneous
random-effects summary (OR 1.29, I-squared = 0). With 26 candidates tried the
Bonferroni-adjusted p is 0.09. Several correlated scores, all variants of how
far a peptide sits from any human window, form a cluster just behind it, so the
effective number of independent looks is well below 26. Two further candidates
(`mismatch_face`, `nn_full_hamming`) have z of 2.5 but fail the sign-consistency
gate at 57%. The pre-registered viral score, the Atlas references and the
learned classifiers show nothing, and differential agretopicity does not help.

### What the selected score is separating (exploratory, after selection)

These descriptive checks were run after the selection and did not change it.
Across the 13 neoepitope cohorts, the positive rate by whole-peptide mismatches
to the nearest human window is:

| Mismatches | 0 (self-identical) | 1 | 2 | 3 | 4 or more |
| --- | ---: | ---: | ---: | ---: | ---: |
| Peptides | 119 | 2,112 | 163 | 147 | 42 |
| Positive rate | 14% | 16% | 37% | 35% | 38% |

The rate is flat from zero to one mismatch and then more than doubles. So the
selected score mostly separates point mutants (and self-identical peptides) from
peptides that differ more substantially, such as frameshift, fusion or
multi-residue changes, instead of grading foreignness among point mutants.
Whether the mutated residue sits at an anchor or on the TCR face makes no
visible difference (positive rate 20% for anchor-only changes, 18% for
TCR-facing ones), which is not what the usual TCR-facing-mutation argument
predicts. Both observations are in development data only and are hypotheses for
the holdout and for future cohorts.

## Selection committed

The viral domain has no selection. The neoepitope domain's selection,
`nn_full_whole`, is recorded in `SELECTED_CANDIDATES` in
`scripts/benchmark_foreignness_choices.py` in the same commit that adds this
section, before the holdout stage is run. The holdout stage then opens only the
neoepitope holdout (TESLA, PMID 33038342, and cohort 31511069), after dropping
rows whose peptide shares a 7-mer with a development peptide.

The holdout code path was verified before use by running it on development rows
(`--stage holdout --dry-run`), where it reproduces the development estimate to
six decimals. That check reads no holdout row.

## Deviations

Logged in the order they happened; none changes a registered rule.

1. Before the development run, a dry-run mode and a restriction to domains with
   a selection were added to the holdout stage so that a bug cannot waste the
   holdout and the viral holdout stays unopened. Implementation only.
2. After the development run, the mismatch-count tables above were computed
   from development data. They are exploratory and did not change the selection.
