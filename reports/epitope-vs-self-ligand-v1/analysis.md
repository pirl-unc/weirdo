# T-cell epitopes versus presented self ligands (pMHC-matched)

This is a different task from the within-study ranking used elsewhere. Every
candidate is presented, but one side is a tolerated human self ligand (HLA Ligand
Atlas: class I, 8-11mers, exactly one atlas-predicted strong allele, 11,755
sampled, no peptide shared with the IEDB pool) and the other an IEDB peptide
tested for T-cell reactivity. Each IEDB peptide is matched without replacement to
a self ligand of the same HLA and length, nearest in (log presentation, log BA
rank) within a caliper, so the pMHC core is near chance by construction.

The decisive control is the same matching for IEDB peptides that were tested and
found **non-reactive**. A score that separates both from self ligands equally
detects foreignness or source, not epitope-ness; only a positive-specific excess
is epitope signal.

## Result (caliper 0.05 SD, 1,774 positive pairs and 1,650 assay-negative pairs)

| Score | AUC, positives vs self | AUC, assay-negatives vs self | Positive-specific excess |
|---|---:|---:|---:|
| pMHC core | 0.483 | 0.479 | +0.004 |
| BA rank | 0.500 | 0.500 | 0.000 |
| virus_foreign_like | 0.560 | 0.537 | +0.023 |
| p_viruses | 0.558 | 0.546 | +0.012 |
| one_minus_mammals | 0.528 | 0.494 | +0.035 |
| face surprise (self-relative TCR face) | 0.586 | 0.572 | +0.014 |
| packaged full score | 0.519 | 0.506 | +0.013 |

Looser calipers (0.15, 0.5) leave every Weirdo score unchanged while the pMHC core
drifts below 0.5 (0.46, 0.44) because matched self ligands remain better presented
than matched epitopes; that residual imbalance is why the calipers were tightened.

## Interpretation

- Even in the task most favourable to foreignness, Weirdo separates presented
  epitopes from presented self ligands only weakly (AUC 0.52-0.59).
- Most of that separation is shared with peptides that were tested and did **not**
  elicit a response, so it is largely foreignness or MS-versus-assay provenance, not
  T-cell epitope-ness; the positive-specific excess is +0.00 to +0.04.
- The best single feature is the zero-parameter self-relative face surprise, which
  is residue composition and is also the most exposed to MS detection bias (for
  example, under-detection of cysteine-containing ligands).
- No model or combination here is a major contribution, and no response data
  was used to build any score.
