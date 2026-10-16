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

## Deviations

*None yet.*
