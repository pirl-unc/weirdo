# Using Weirdo with MHCflurry presentation and NetMHCpan BA

All scores below use the **shipped** `peptide-content-v1-ensemble` heads (23 per
peptide) and are label-free except where a fitted model is named. Datasets:
IMPROVE bladder/melanoma/Basket (one family), two viral CD8 panels (one family),
TESLA, and Braun RCC constructs (128 constructs, 9 patients). Family means keep
IMPROVE's three cohorts from dominating. Braun frameshift is a 19-row subgroup
of Braun and is not an endpoint.

## Result

`weirdo.pmhc_weirdo_score` is a fixed combination:

    core  = mean(rank(MHCflurry 2.3 presentation), rank(-NetMHCpan BA rank))
    score = core + 0.1 * (rank(virus_foreign_like) - 0.5)
                 + 0.1 * (rank(face_surprise)      - 0.5)

Ranks use a frozen, label-free reference (`pmhc-rank-reference-v1.json`).

| Family AUC | Presentation | BA | Core | Core + Weirdo |
|---|---:|---:|---:|---:|
| IMPROVE tumour | 0.539 | 0.589 | 0.572 | 0.583 |
| Viral | 0.699 | 0.635 | 0.689 | 0.694 |
| TESLA | 0.841 | 0.816 | 0.858 | 0.866 |
| Braun (128 constructs) | 0.582 | 0.615 | 0.606 | 0.608 |
| **Macro** | 0.665 | 0.664 | 0.681 | **0.688** |

The core beats either predictor alone on macro AUC (neither dominates because
both are rank-normalized). The two Weirdo terms add **+0.0066 macro AUC** and
improve all seven datasets (+0.001 to +0.021). That is a small effect, not a
major contribution: see the null below and the structural limit.

## Why Weirdo cannot be a major contributor on these benchmarks

In every benchmark here, positives and negatives come from the same source
(viral against viral, mutant self-like against mutant self-like), so foreignness
barely varies across the candidates being ranked. The shipped taxonomy score
reflects this: its mean `virus_foreign_like` is 0.277 for the multivirus panel
and 0.25-0.26 for tumour cohorts, and only 1.7% of its variance lies between
datasets. An isolated 8-11mer carries little species information (MHC-filtered
Swiss-Prot AUC about 0.60), so retraining the same kind of classifier cannot
raise this ceiling. Foreignness would matter on mixed-source tasks, such as
presented foreign peptides against presented self ligands; none of the response
benchmarks tests that.

## Everything that was tried

| Hypothesis | Fitting | Outcome |
|---|---|---|
| Fixed additive head, 23 heads x 4 weights | none | best cell +0.0044 macro (microbial-like, 0.1); max-over-92-cells permutation p = 0.025 for that cell, 0.085 for the chosen `virus_foreign_like` at 0.1 |
| Logistic on core + all 23 heads | leave-one-family-out | macro -0.046 to -0.058 vs core |
| Logistic on core + 4 summary heads | leave-one-family-out | macro -0.008 to -0.010 |
| Gated: core x heads interactions | leave-one-family-out | macro -0.021 to 0.000; TESLA up, Braun down |
| Weirdo heads only | leave-one-family-out | macro 0.48-0.54; below 0.5 on TESLA (sign flips between datasets) |
| Mutant-minus-wild-type Weirdo contrast (IMPROVE only, 17,303 rows) | none / leave-one-cohort-out | best +0.0079 macro, inside the null (p = 0.077); fitted -0.005 to -0.010 |
| Rank-weighted construct sums over candidate windows (Braun) | none | see `reports/construct-weighted-candidates-v1`; no reliable gain, candidate count alone gives 0.619 |
| Self-relative TCR-face surprise (zero fit, sign fixed in advance) | none | standalone AUC above 0.5 on all 7 datasets (0.518-0.610); +0.0048 macro at 0.1, max-null p < 0.002, positive in all four families |
| Hydrophobicity / aromaticity of the face | none | hydrophobicity flips sign (TESLA 0.369) and hurts; aromaticity mixed |
| Fitted logistic on core + face features | leave-one-family-out | macro -0.003 to -0.034; TESLA loses 0.03-0.13 |
| virus_foreign_like + face surprise, both fixed at 0.1 | none | +0.0066 macro, all four families positive (7 weight pairs were looked at) |
| Earlier IMPROVE-fitted response adapters | IMPROVE only | failed external confirmation (`reports/tcr-visible-nonself-v1`) |

## Interpretation

- Taxonomy heads do not transfer as a fitted response signal: their relation to
  response changes sign across datasets, so any learned weight overfits the
  training families. A small fixed weight is the only form that did not hurt.
- The advantage of the core comes from putting the two pMHC predictors on a
  common scale. Presentation wins on TESLA and viral sets, BA wins on the tumour
  cohorts, and their rank average is near the best of both everywhere.
- The Weirdo weight was fixed a priori from earlier exploratory screens on these
  same datasets, so the benchmark is not an independent confirmation. A new
  cohort is needed before claiming the +0.003.

## Caveats

- The rank reference is the pooled evaluation peptides; it is label-free but not
  an independent reference set of random self peptides.
- Braun's 9 patients give wide intervals; per-dataset paired bootstrap intervals
  for the best cells include zero in nearly every dataset.
- MHCflurry 2.3 and NetMHCpan BA ranks are required inputs; the score does not
  run them.
