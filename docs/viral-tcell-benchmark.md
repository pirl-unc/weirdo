# Exploratory viral T-cell response benchmark

`scripts/benchmark_viral_tcell.py` downloads assay-level IEDB T-cell exports
through [datacache](https://github.com/openvax/datacache), then scores exact-HLA
class-I peptides with an exploratory, whole-peptide classifier. It uses the
existing local `data/swissprot-sequences.csv` as its training input and records
that file's SHA-256 in the report. The 2023 local SwissProt file is not a pinned
public training release; rebuilding a versioned protein-level source is tracked
in [weirdo #6](https://github.com/pirl-unc/weirdo/issues/6).

```bash
python -m pip install -e '.[benchmark]'
python scripts/benchmark_viral_tcell.py \
  --swissprot-sequences data/swissprot-sequences.csv \
  --per-category 5000
```

The script caches source responses and writes `report.json` and
`scored-assays.csv` under ignored `data/viral-tcell-benchmark/`. Cache hits are
reused offline. IEDB query URLs, checksums, and fetch dates are retained in
datacache provenance receipts. No assay data or model weights are committed.

The three source papers are:

| PMID | Evidence | Exact-HLA class-I records with 8–14mer canonical peptides |
| --- | --- | ---: |
| [36254196](https://pmc.ncbi.nlm.nih.gov/articles/PMC9568370/) | Direct ex-vivo tetramers in convalescent donors; tested positive and negative peptide/HLA pairs | 93 |
| [35484232](https://pmc.ncbi.nlm.nih.gov/articles/PMC9064790/) | IFN-γ responses after short-term restimulation in infected and mRNA-vaccinated donors; repeated peptide/HLA observations | 167 |
| [33521695](https://pmc.ncbi.nlm.nih.gov/articles/PMC7837622/) | Direct ex-vivo convalescent responses; positives only in this export | 523 |

The model samples 5,000 complete 8–14mer windows per SwissProt category, uses
position-specific amino-acid identity and a central-residue composition branch,
and trains a 32/16-unit MLP classifier. Proteins whose identifiers contain
`SARS`, `COV`, or `CORO` are excluded from its viral training sample. This is a
limited exclusion, not a homology-cluster or viral-family split. The classifier
is an analysis baseline, not a released WEIRDO model; its category validation
accuracy was 0.1532 across ten balanced categories in the run below.

With seed 42 and 5,000 sampled peptides per category, the direct ex-vivo
tetramer panel yielded AUC 0.493 (49 positives, 44 tested negatives), and its
within-HLA positive/negative comparison yielded 0.404 across 314 pairs. The
infected IFN-γ subset yielded AUC 0.426 (27 positives, 37 negatives); the
vaccinated subset yielded AUC 0.573 (37 positives, 30 negatives), but its
within-HLA comparison was 0.445. The infected-then-vaccinated stratum has only
two positives. The positive-only study cannot support an AUC. These are
descriptive results: **this exploratory model did not show consistent positive
enrichment**. The scored rows and exact counts are in `report.json` and
`scored-assays.csv` after running the script.

These studies measure T-cell recognition of supplied peptide/HLA complexes or
responses to peptide restimulation. They support a set of plausible viral
epitopes. They do not, by themselves, establish natural processing and
presentation by infected cells. IEDB restriction evidence codes and assay
methods remain in the scored rows. A tested negative is a negative for its
reported assay context; untested viral subsequences are not negatives.

The next analysis needs protein-cluster and held-out-virus-family splits,
reliable class probabilities, an HLA-aware TCR-facing representation, and
uncertainty estimates grouped by peptide/protein/donor. These are tracked in
[weirdo #4](https://github.com/pirl-unc/weirdo/issues/4),
[#5](https://github.com/pirl-unc/weirdo/issues/5),
[#7](https://github.com/pirl-unc/weirdo/issues/7), and
[#8](https://github.com/pirl-unc/weirdo/issues/8).
