# Mutation-aware Braun construct aggregation

## Fixed candidate evidence

For each reference-absent 8--11mer candidate paired with the construct's
annotated HLA, the label-free pMHC evidence is

$$e_i = \sqrt{p_{\mathrm{MHCflurry\ presentation},i}
(1 - r_{\mathrm{NetMHCpan\ BA},i}/100)}.$$ 

MHCflurry presentation is already its presentation model, including its learned
processing component.  We therefore do not add the raw MHCflurry processing
score again and double-count that signal.

The prespecified main construct score is the **sum of the three largest distinct
peptide--HLA evidence values**.  It is an additive evidence score rather than a
number of passing pMHCs.  `best candidate` and `top3 soft or` are fixed controls.
The stricter `gated` sensitivity only lets a candidate contribute when
MHCflurry presentation percentile is at most 5 and
NetMHCpan BA rank is at most 2; neither threshold was selected on
Braun outcomes.

When present, each `gated evidence x ...` score uses the *identical* gate and
multiplies each eligible candidate's evidence by one frozen broad-nine ESM
taxonomy output before taking its top-three sum: `1 - P(mammals)`, independent
`P(viruses)`, or `microbial like = max[P(virus), P(bacterium), P(archaeon),
P(fungus), P(protist)]`.  These independent sigmoid outputs are scores, not a
probability partition; microbial-like is deliberately a conservative pathogen
proxy.  The taxonomy model and transforms were selected without Braun response
labels.

The candidate enumerator's reference-absence flag is still an operational proxy
for mutation overlap, not proof of it.  These construct-level outcomes are used
only below to evaluate precomputed scores, never to construct them.

## Descriptive evaluation

`ΔAUC` uses a paired patient-bootstrap interval relative to the pMHC-only
gated top-three sum—identical candidate rule and gate, with Weirdo omitted.
This small cohort is not used to select one aggregation rule.

| Endpoint | Score | Constructs | Positive | AUC | AP | ΔAUC vs gated pMHC-only, 95% patient-bootstrap interval |
|---|---|---:|---:|---:|---:|---:|
| all braun constructs | best candidate evidence | 128 | 60 | 0.5797 | 0.5295 | -0.0159 [-0.0771, +0.0377] |
| all braun constructs | top3 sum candidate evidence | 128 | 60 | 0.5556 | 0.5320 | -0.0400 [-0.0754, +0.0009] |
| all braun constructs | top3 soft or candidate evidence | 128 | 60 | 0.5691 | 0.5407 | -0.0265 [-0.0604, +0.0194] |
| all braun constructs | top3 sum gated candidate evidence | 128 | 60 | 0.5956 | 0.5769 | +0.0000 [+0.0000, +0.0000] |
| all braun constructs | top3 sum gated evidence x microbial like | 128 | 60 | 0.6103 | 0.5555 | +0.0147 [-0.0079, +0.0341] |
| all braun constructs | top3 sum gated evidence x one minus mammals | 128 | 60 | 0.5958 | 0.5536 | +0.0002 [-0.0100, +0.0057] |
| all braun constructs | top3 sum gated evidence x viruses | 128 | 60 | 0.5909 | 0.5586 | -0.0047 [-0.0342, +0.0251] |
| braun frameshifts | best candidate evidence | 25 | 10 | 0.6000 | 0.5450 | -0.0400 [-0.2585, +0.1259] |
| braun frameshifts | top3 sum candidate evidence | 25 | 10 | 0.6100 | 0.5949 | -0.0300 [-0.1082, +0.0058] |
| braun frameshifts | top3 soft or candidate evidence | 25 | 10 | 0.6167 | 0.5932 | -0.0233 [-0.0945, +0.0408] |
| braun frameshifts | top3 sum gated candidate evidence | 25 | 10 | 0.6400 | 0.6260 | +0.0000 [+0.0000, +0.0000] |
| braun frameshifts | top3 sum gated evidence x microbial like | 25 | 10 | 0.6000 | 0.5010 | -0.0400 [-0.1857, +0.0428] |
| braun frameshifts | top3 sum gated evidence x one minus mammals | 25 | 10 | 0.6133 | 0.5325 | -0.0267 [-0.1111, +0.0588] |
| braun frameshifts | top3 sum gated evidence x viruses | 25 | 10 | 0.6133 | 0.4955 | -0.0267 [-0.1181, +0.1222] |

## Outputs

- [`raw/construct-aggregates.csv.gz`](raw/construct-aggregates.csv.gz)
- [`tables/aggregate-metrics.csv`](tables/aggregate-metrics.csv)
- [`raw/provenance.json`](raw/provenance.json)
