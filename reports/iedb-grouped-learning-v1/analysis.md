# Fresh-cohort confirmation and study-grouped learning

## Question

Does Weirdo give a non-trivial boost over NetMHCpan BA + MHCflurry 2.3 presentation
for T-cell epitope prediction? "Non-trivial" was set in advance as a within-study
AUC gain of about +0.02 or more with an interval excluding zero.

All metrics are **within-study** AUC (studies with >= 20 pairs and >= 5 of each
class), because pooled AUC mixes study prevalences. Intervals are study-clustered
bootstrap.

## Sealed fresh cohorts (frozen shipped score, weights 0.1/0.1, nothing tuned)

Both cohorts contain only IEDB studies never used before, share no peptide with any
earlier dataset, and the second also excludes everything the first consumed.

| Cohort | Eligible studies (rows) | Presentation | Core | Full | Full - core [95% CI] | Studies improved |
|---|---:|---:|---:|---:|---|---:|
| fresh-1 | 11 (1,103) | 0.650 | 0.651 | 0.667 | +0.0156 [-0.007, +0.039] | 7 / 11 |
| fresh-2 | 49 (4,978) | 0.572 | 0.578 | 0.582 | +0.0045 [-0.003, +0.012] | 24 / 49 |

The direction replicates but the size is trivial and not distinguishable from zero.
Split by source type (secondary, post hoc): the face-surprise term gains +0.011 in
the 19 human-sourced fresh-2 studies and nothing (-0.002) in the 21 pathogen studies.

### Short-list view (secondary, post hoc, added after the AUC result)

R-precision is the fraction of positives in each study's top-k, where k is that
study's number of positives; a random ranking scores the study prevalence.

| Cohort | Random | Core | Full | Full - core [95% CI] |
|---|---:|---:|---:|---|
| fresh-1 (11 studies) | 0.544 | 0.631 | 0.651 | +0.020 [-0.011, +0.064] |
| fresh-2 (49 studies) | 0.468 | 0.529 | 0.524 | -0.006 [-0.022, +0.010] |

The short-list view does not rescue the primary result; the larger cohort goes
slightly the wrong way. It also shows how weak the baseline is: the core is only
about six points above random in the top-k.

## Study-grouped learning on all of IEDB (training regime, data composition, output space)

34,946 labelled human class I peptide-HLA pairs from 2,376 studies; 137 held-out
studies are eligible. XGBoost `rank:pairwise` with the study as the query group
learns to order pairs within a study. Folds hold out whole studies and drop any
training pair whose peptide occurs in the held-out fold. Feature sets: `pmhc`
(rank features, raw log-odds, length), `pmhc_face` (+ TCR-face composition),
`pmhc_weirdo` (+ the 23 shipped taxonomy heads and face surprise). Mean gap is
AUC(model) - AUC(core).

| Mode / config | pmhc | pmhc_face | pmhc_weirdo | Weirdo heads effect [CI] | Face effect [CI] |
|---|---:|---:|---:|---|---|
| residual, shallow | -0.0026 | +0.0023 | +0.0016 | +0.0042 [-0.001, +0.009] | +0.0048 [-0.000, +0.010] |
| residual, default | -0.0156 | -0.0073 | -0.0104 | +0.0052 [-0.005, +0.015] | +0.0083 [+0.000, +0.017] |
| residual, deep | -0.0438 | -0.0379 | -0.0421 | +0.0017 [-0.015, +0.017] | +0.0059 [-0.009, +0.020] |
| plain (from scratch), default | -0.0376 | -0.0281 | -0.0320 | +0.0056 [-0.010, +0.021] | +0.0094 [-0.006, +0.024] |

Residual mode boosts on top of the core (`base_margin = 4 * (core - 0.5)`); plain
learns from scratch. The best learned model is +0.0023 over the core (interval
includes zero), and presentation alone is -0.0139 against the core.

## Same-HLA self-ligand neighbourhood (tested on the same 137 studies)

Distance from each peptide to human self ligands presented by the same HLA (HLA
Ligand Atlas strong binders, same length; TCR-face mismatches, BLOSUM, counts within
1-2 face mismatches, exact-hit flag) was added as a feature set. Effect over the
`pmhc`-only learner, three declared configs: shallow -0.0015 [-0.004, +0.001],
default +0.0027 [-0.003, +0.008], deep -0.0014 [-0.014, +0.011]. On top of all other
features: +0.0014, -0.0011, +0.0001. Null.

## Source-protein context (new information, same 137 studies)

Each peptide was mapped into its curated UniProt parent protein (63.1% of pairs; the
IEDB start position only breaks ties). Features: protein length, relative position,
distance to termini, MHCflurry 2.3 presentation with 10-residue flanks (a better pMHC
baseline, not Weirdo), and Weirdo's taxonomy profile of the source protein (mean of
four heads over all 9-mer windows plus the peptide's deviation from its own
protein). Effects over the `pmhc`-only learner (shallow / default / deep configs):

| Feature set | shallow | default | deep |
|---|---:|---:|---:|
| flank-aware presentation | -0.0009 | -0.0009 | -0.0061 |
| position / protein length | +0.0006 | +0.0009 | -0.0050 |
| Weirdo protein profile | +0.0045 [-0.001, +0.010] | +0.0049 [-0.005, +0.015] | +0.0006 [-0.018, +0.018] |
| Weirdo given all context | +0.0029 [-0.002, +0.008] | +0.0047 [-0.006, +0.016] | +0.0066 [-0.011, +0.023] |

Flank-aware presentation does not improve the baseline here, so processing context
is not the missing signal, and the Weirdo protein profile lands on the same
+0.005 ceiling as every other Weirdo route.

## Conclusion

**Weirdo does not make a major contribution to T-cell epitope prediction on any
test run here.** Every estimate of the Weirdo-attributable gain lies between about
+0.002 and +0.016 within-study AUC, mostly +0.005, and nearly every interval
includes zero. The result is stable across five development benchmarks, two sealed
fresh cohorts, and 137 held-out studies of learned ranking. Re-learning the pMHC
weighting from IEDB makes things worse, not better, so the shipped fixed rank core
remains the right baseline.

## Public data is exhausted

The CEDAR (cancer IEDB) full T-cell export (157,263 rows, 4,066 studies) was converted
and filtered the same way. Every human class I study with both response classes is
already in the IEDB pool; the only 5 studies outside it (121 rows) are positive-only
and cannot support an AUC. No further public sealed cohort exists in IEDB or CEDAR.

## What was deliberately not done

- Pooled-across-study evaluation, where foreignness predicts the source (human vs
  pathogen) and therefore study prevalence, would show a large Weirdo "boost" that
  is a prevalence artifact rather than epitope prediction. It was not used.
- The fresh cohorts are now spent; tuning weights on them would void the
  confirmation.
