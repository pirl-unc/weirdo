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

**Correction.** An earlier revision of this section read the raw table below as
evidence that the selected score mostly separates point mutants from substantially
novel sequence. That reading was wrong, and the second-generation development stage
below shows why. Descriptive checks run after the selection, over the 13 neoepitope
cohorts, give these raw positive rates by whole-peptide mismatches to the nearest
human window:

| Mismatches | 0 (self-identical) | 1 | 2 | 3 | 4 or more |
| --- | ---: | ---: | ---: | ---: | ---: |
| Peptides | 119 | 2,112 | 163 | 147 | 42 |
| Positive rate | 14% | 16% | 37% | 35% | 38% |

The crude odds ratio for two or more mismatches against fewer is 3.04, but
stratified by cohort alone it is **1.16**. The cohorts with the most multi-mismatch
peptides (33665637 at 93%, 38244726 at 98%, 40273911 at 86%) simply have high
base rates (47%, 65%, 33%), so the raw difference is mostly cohort composition.
Whether the mutated residue is at an anchor or on the TCR face also makes no clear
difference in the raw rates (20% against 18%), but that comparison carries the same
confounding and is not evidence either way.

### Second-generation development ranking

Run after the registration in [Second generation](#second-generation-a-sealed-set-for-the-novelty-hypothesis),
on the 13 neoepitope development cohorts only, under the identical model:

| Candidate | OR/SD (95% CI) | z | p (Bonf., 4 tried) | Cohorts + | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `nn_full_whole` | 1.21 (1.05-1.40) | +2.70 | 0.003 (0.01) | 100% (7) | passes |
| `mismatch_whole` | 1.11 (0.98-1.27) | +1.65 | 0.050 (0.20) | 83% (6) | passes |
| `novel_2plus` | 1.11 (0.98-1.26) | +1.65 | 0.049 (0.20) | 50% (6) | fails the 60% sign gate |
| `novel_log` | 1.04 (0.91-1.19) | +0.60 | 0.276 (1.00) | 67% (6) | fails z |

Adjusted within cohorts, the two-or-more-mismatch indicator is weak and unstable
(its per-cohort coefficients run from -0.70 to +5.45), so the novelty hypothesis
that motivated the second sealed set is **not supported by the development data**.
The graded whole-peptide distance is the only candidate that is positive in every
cohort reported alone (coefficients +0.13 to +0.53, homogeneous across cohorts).
Under the registered rule the selection is therefore `nn_full_whole` again: the
same score that was inconclusive on the first holdout is the one tested on the
second sealed set. That is a replication attempt, not a test of the novelty idea.

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

## Holdout result: not confirmed

Run once, at 23:39 on 2026-10-15 after the selection commit `d801c9d`, with
`--stage holdout --selection-is-committed`. Only the neoepitope domain was
opened. 84 of 1,688 holdout rows were dropped for sharing a 7-mer with a
development peptide, leaving 1,604 pairs with 61 positives (TESLA, PMID
33038342, and cohort 31511069).

| Candidate | Odds ratio per SD (95% CI) | z (one-sided) | Required | Reading |
| --- | ---: | ---: | --- | --- |
| `nn_full_whole` | 0.90 (0.67-1.19) | -0.75 | z > 1.96 and OR >= 1.10 | **not confirmed; inconclusive** |

Both holdout cohorts point the wrong way (coefficients -0.05 for TESLA, with 39
positives, and -0.20 for 31511069, with 22). The interval excludes the
development estimate's centre but does not exclude the minimum effect of
interest (upper bound 1.19), so the registered reading is *inconclusive*, not
*refuted*. The holdout could only have confirmed a large effect: the odds ratio
detectable with 80% power was 1.51. The development estimate (log odds ratio
0.19, standard error 0.07) and the holdout estimate (-0.11, standard error 0.15)
differ by z of about 1.85, which is what a winner's-curse estimate looks like
after the best of 26 candidates was chosen on the development data.

**A tempting explanation that does not hold (post hoc, inputs only).** Looking only
at the holdout's peptide sequences, never at its outcomes, 94.0% are single point
mutants and 4.6% (74 peptides) have two or more mismatches to the nearest human
window, against 13.6% in development, and the score's spread is smaller (SD 0.34
against 0.45). It is tempting to say the holdout lacked the variation that drove the
development effect. The second-generation development ranking below shows that
the multi-mismatch contrast does not carry the development effect (within-cohort
odds ratio 1.16), so this does not explain the miss. The simpler readings are that
the development estimate was inflated by choosing the best of 26 candidates and
that the effect differs across studies.

**What this means.** No definition of foreignness has been validated in either
domain. The neoepitope holdout is spent for this candidate: reusing it to test a
refined candidate would not be an independent confirmation. The viral holdout,
including the 21,419-pair vaccinia screen, is sealed and available to any future
viral candidate that clears development.

## Second generation: a sealed set for the novelty hypothesis

Registered on 2026-10-15, after the first neoepitope holdout was spent and
before any outcome in the set below was analysed.

**Why.** The first holdout was inconclusive and 94% point mutants, so it could not
test what the development data suggested: that peptides with two or more
mismatches to the nearest human window are more often recognized than point
mutants, at equal predicted presentation. Testing that needs fresh tumour-mutation
data. None of it is independent of the *idea* (the idea came from the development
data and from learning that the first holdout lacked such peptides), but all of it
is independent *data*.

**Data.** A title search of IEDB for neoantigen, neoepitope, frameshift, mutanome,
mutated, mutant, somatic and fusion terms found 269 publications outside the
registry. 140 of them have at least 12 positive or negative class-I rows; the other
129 were not considered because they cannot reach the inclusion threshold. Of the
140, reading titles alone, 116 are included as tumour-mutation studies
(`CONFIRM2_NEOEPITOPE_PMIDS`) and 24 are excluded with a stated reason in
`CONFIRM2_EXCLUDED_PMIDS` (viral or viral-mutant antigens, unmutated peptides,
post-translational or residue-modified peptides, mixed antigen classes). Titles
that do not contain those terms are not in the search at all, so the set is
title-matched, not exhaustive. The set
has role `confirm2` and was never scored or analysed before this registration;
only marginal counts and the distribution of mismatches to the nearest human window
(never cross-tabulated with an outcome) were inspected, in order to state the power
below.

Under the usual preparation rules most of these studies drop out: 33 report only
positives, 19 have both outcomes but fewer than five of one, and a large share of
rows are class II or lack an exact class-I allele. The primary analysis keeps
cohorts with at least five of each outcome, as in development: **15
cohorts, 984 pairs, 197 positive**, after dropping 14 rows whose peptide
shares a 7-mer with a development peptide. Of these, 19% match a human window
exactly, 73% differ at one position and 8.4%
(83 pairs) differ at two or more. A registered *sensitivity* analysis keeps cohorts with
at least one of each outcome (34 cohorts, about 1,500 pairs, about 340 positive). It
is reported but cannot decide the verdict.

**Power of the primary analysis** (one-sided alpha 0.025, standardized score, 0-25%
of the information lost to adjustment):

| True odds ratio per SD | 1.1 | 1.2 | 1.3 | 1.5 |
| --- | ---: | ---: | ---: | ---: |
| Power | 12%-14% | 32%-40% | 57%-69% | 91%-97% |

The development data imply roughly 1.4-1.5 per SD for a two-or-more-mismatch
indicator before any shrinkage, which this set could confirm; a more modest true
effect would often go unconfirmed, and the report will state which effect sizes it
can exclude.

**Candidates** (`G2_EVALUATED`), scored on the development neoepitope cohorts only:
`nn_full_whole` (the first-generation winner, as a reference), `mismatch_whole`
(count of mismatches to the nearest human window), `novel_2plus` (one if two or
more mismatches) and `novel_log` (log of one plus the count). The last two are
post hoc, so their development numbers are exploratory.

**Selection.** The identical rule: positive pooled coefficient, positive in at
least 60% of cohorts reported alone, pooled z of at least 1.645, 90% coverage; the
largest z wins and ties go to the earlier candidate (`nn_full_whole`, then
`mismatch_whole`, then the two new ones). If none qualifies, the sealed set is not
opened.

**Confirmation.** The selected candidate is committed before the stage runs, then
evaluated once with `--stage confirm2`. Confirmed only if the one-sided
cluster-robust z exceeds 1.96 and the odds ratio is at least 1.10 per SD, exactly as
for the first holdout. A confirmed result would justify considering a score
feature for neoepitope ranking and nothing more.

## Second sealed set: not confirmed

Run once, at 23:53 on 2026-10-15 after the selection commit `74c249c`, with
`--stage confirm2 --selection-is-committed`. Only the neoepitope domain was opened;
22 of 1,511 pairs were dropped for sharing a 7-mer with a development peptide.

| Analysis | Cohorts | Pairs (positive) | Odds ratio per SD (95% CI) | z (one-sided) | Reading |
| --- | ---: | ---: | ---: | ---: | --- |
| **Primary**: at least 5 of each outcome | 15 | 984 (197) | 1.09 (0.85-1.40) | +0.67 | inconclusive |
| Inclusive sensitivity: at least 1 of each | 34 | 1,489 (335) | 1.14 (0.92-1.41) | +1.18 | inconclusive |

`nn_full_whole` is **not confirmed**: z is far below 1.96. The point estimates are
positive but small and the intervals include both no effect and the minimum effect
of interest. The odds ratio detectable with 80% power was about 1.43, so only a
large effect could have been confirmed. Only 3 cohorts had enough rows to be
reported alone, which is why per-cohort sign consistency is not informative here.

### Synthesis of the confirmatory evidence (post hoc)

An inverse-variance pool of the two independent confirmatory results, which
neither chose the candidate nor were seen before the selection:

| Evidence | Odds ratio per SD (95% CI) | z |
| --- | ---: | ---: |
| Development, 13 cohorts (best of 26 chosen here, so optimistic) | 1.21 (1.05-1.40) | +2.70 |
| First holdout, TESLA + 31511069 | 0.90 (0.67-1.19) | -0.75 |
| Second sealed set, primary | 1.09 (0.85-1.40) | +0.67 |
| **Pool of the two confirmatory results, primary** | **1.00 (0.83-1.21)** | +0.01 |
| Pool of the two confirmatory results, inclusive | 1.05 (0.88-1.24) | +0.50 |

The development estimate differs from the confirmatory pool by z = 1.6. So the
confirmatory data are consistent with no effect and exclude, at 95%, anything above
about 1.2 per SD, and the development estimate sits at the very edge of that range,
as expected when the best of many candidates is reported.

**What this means.** For neoepitopes, an exact distance from the whole peptide to
its nearest human window does not add detectable information beyond predicted
presentation and TCR-face composition in these data. A small effect (up to about
1.2 per SD) is not excluded. No candidate is validated in either domain. Both
neoepitope sealed sets are now spent for this candidate. The viral holdout, including
the 21,419-pair vaccinia screen, is still sealed.

## Third generation: a composition feature, not foreignness

Registered on 2026-10-15 after both neoepitope sets were analysed and before any
sealed set was touched for this hypothesis.

**What and why.** The first benchmark carried two TCR-face composition measures as
covariates: the mean hydropathy of residues P3 to P-omega-minus-one, and the share of
F, W and Y among them (`tcr_face_aromatic_fraction`). Tested on their own, under the
same adjusted model but each adjusted for the *other* instead of itself, the
aromatic share was stronger and more consistent on development data than any of
the 26 foreignness definitions, in both domains. It is not a measure of foreignness
and nothing here is evidence about foreignness. It is evidence about what a ranking
model should contain. Large aromatic residues at TCR-contact positions have been
reported to enrich for immunogenicity (for example Calis et al., PLoS Comput Biol
2013), so this is a replication on new data rather than a new idea.

**Development ranking** (the identical registered rule, 2 candidates; the
development numbers are optimistic because this feature was chosen after they were
seen as covariates):

| Domain | Candidate | OR/SD (95% CI) | z | Cohorts + |
| --- | --- | ---: | ---: | ---: |
| viral | aromatic share | 1.26 (1.16-1.38) | +5.48 | 83% (18) |
| viral | hydropathy | 1.07 (1.00-1.15) | +2.03 | 61% (18) |
| neoepitope | aromatic share | 1.24 (1.09-1.40) | +3.34 | 100% (6) |
| neoepitope | hydropathy | 1.09 (0.95-1.24) | +1.19 | 43% (7) |

The rule selects the aromatic share in both domains.

**Test.** One pre-specified hypothesis per domain: the aromatic share has an odds
ratio above 1 per SD beyond predicted presentation, stratum, allele, species,
length, assay counts and hydropathy, with peptide- and protein-cluster-robust
errors. It is evaluated once on sealed data that did not choose it:

- *viral*: the viral holdout (the vaccinia Dryvax screen and eight hashed cohorts),
  never opened. Rows sharing a 7-mer with a development peptide are dropped.
  One-sided alpha 0.025.
- *neoepitope*: the first holdout and the second sealed set together. Both were
  analysed once already, for the foreignness hypothesis, and not for this one, so
  their use is paid for by halving the threshold to alpha 0.0125. The primary
  analysis keeps cohorts with at least five of each outcome; an inclusive
  sensitivity analysis (at least one of each) is reported and cannot decide.

Confirmed in a domain only if the one-sided z exceeds the critical value for its
alpha **and** the odds ratio is at least 1.10 per SD. The two domains are judged
separately.

**Power** (approximate, from marginal counts only; 0-25% of information lost to
adjustment; odds ratio per SD):

| | 1.10 | 1.15 | 1.25 | 1.35 |
| --- | ---: | ---: | ---: | ---: |
| Viral, alpha 0.025 | 30%-38% | 56%-68% | 92%-97% | 99%-100% |
| Neoepitope, alpha 0.0125 | 11%-14% | 22%-30% | 55%-69% | 83%-92% |

At the development estimates (about 1.25 per SD) the viral test is well powered and
the neoepitope test is only moderately so; a neoepitope miss would again be
inconclusive about smaller effects.

**Meaning.** A confirmed viral result would support adding TCR-face aromatic content
to a presentation-based ranking for pathogen epitopes, and a confirmed neoepitope
result would support it for neoepitopes. Neither would say anything about
foreignness, and neither would be an implementation.

## Deviations

Logged in the order they happened; none changes a registered rule.

1. Before the development run, a dry-run mode and a restriction to domains with
   a selection were added to the holdout stage so that a bug cannot waste the
   holdout and the viral holdout stays unopened. Implementation only.
2. After the development run, the mismatch-count tables above were computed
   from development data. They are exploratory and did not change the selection.
3. After the holdout result, the holdout's peptide mismatch distribution and
   score spread were computed from sequences alone, with no outcome read, to
   explain why the test was inconclusive. It is a post-hoc description, not part
   of the registered test, and it did not alter the result or the selection.
4. After the first neoepitope holdout was spent, a second sealed confirmation set,
   the novelty candidates and the inclusive sensitivity analysis were registered
   (section above) before the new candidates were ranked and before any
   outcome in the set was analysed.
5. After the second-generation development ranking, the mismatch contrast was
   re-examined stratified by cohort. It showed the raw novelty table was a
   cohort-composition artefact, and the exploratory statements that relied on it
   were withdrawn in the same commit that records the second selection.
6. The second confirmation set was analysed once and the confirmatory evidence
   pooled (section above). The pooling is a post-hoc description, not a registered
   test, and the registered verdict for each set stands on its own.
7. After the neoepitope sealed sets were analysed for the foreignness hypothesis,
   a third hypothesis (the aromatic share) was registered above and the same sets
   were reserved for it at a halved alpha. The viral holdout is used for the first
   time.
