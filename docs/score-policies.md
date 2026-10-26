# Versioned score policies

WEIRDO category heads are independent probabilities of sequence occurrence in
training-data categories. A scalar summary needs additional choices that do not
belong to the model weights: the intended host, source and self categories,
overlap handling, aggregation formula, exact-match behavior, direction, and
interpretation. A score policy records all of those choices under a stable ID.

```python
from weirdo import list_score_policies

for policy in list_score_policies():
    print(policy["policy_id"], policy["host_name"], policy["formula"])
```

## Built-in policies

| Policy | Model | Host | Self heads | Formula |
| --- | --- | --- | --- | --- |
| `human-self-complement-v1` | `swissprot-categories-v1` | human, TaxID 9606 | `human` | `1 - self_like` |
| `mammal-microbial-content-v1` | `peptide-content-v1` | generic Mammalia clade, TaxID 40674 | `mammals` | `microbial_like * (1 - self_like)` |
| `human-microbial-content-v1` | `peptide-content-v1` | human, TaxID 9606 | `mammals`, `human` | same |
| `mouse-microbial-content-v1` | `peptide-content-v1` | mouse, TaxID 10090 | `mammals`, `mouse` | same |
| `dog-microbial-content-v1` | `peptide-content-v1` | dog, TaxID 9615 | `mammals`, `dog` | same |

The content policies define `microbial_like` as the maximum of the independent
`viruses`, `bacteria`, `archaea`, `fungi`, and `protists` heads. They use a
maximum for self as well, so overlapping labels are not counted repeatedly.
A peptide may still be high in both groups; multiplication by `1 - self_like`
downweights, rather than deletes, that overlap. If the caller supplies an exact
match from the selected host reference proteome, content policies set
`self_like=1` and the aggregate score to zero.

## Human and mouse examples

```python
from weirdo import predict_peptide_content

human = predict_peptide_content(
    ["SIINFEKL", "GILGFVFTL"],
    score_policy="human-microbial-content-v1",
)
mouse = predict_peptide_content(
    ["SIINFEKL", "GILGFVFTL"],
    score_policy="mouse-microbial-content-v1",
)

print(human.attrs["score_policy_definition"])
print(mouse.attrs["host_taxon_id"])
```

`predict_peptide_content()` retains the original numeric default through
`mammal-microbial-content-v1`. It is deliberately labeled as a generic clade,
not a species policy. Existing calls that pass `self_categories` or
`microbial_categories` still work, but their result is marked
`custom-unversioned` in `DataFrame.attrs`. A named policy cannot be combined
with custom category arguments.

The stable released category model exposes the same contract:

```python
from weirdo import load_pretrained_model

model = load_pretrained_model(score_policy="human-self-complement-v1")
print(model.score_policy_definition)
scores = model.predict_dataframe(["ACDEFGHI"])
print(scores.attrs["score_policy_definition"])
```

Downstream caches should retain the policy ID and full model prediction
identity. A host is never inferred from a peptide or ambient configuration.

## Interpretation and validation

Every built-in definition states that higher values mean more foreign-like
*under that model and training distribution*. The values are not probabilities
of antigen processing, presentation, T-cell recognition, infection, or prior
exposure.

The registered viral transfer audit compared the relevant content forms after
an MHC baseline. Delta AUC was +0.0017 for `p_viruses`, +0.0023 for
`1 - p_mammals`, +0.0016 for `1 - p_human`, +0.0021 for
`p_viruses * (1 - p_mammals)`, and +0.0017 for
`p_viruses * (1 - p_human)`; every interval crossed zero. The mammalian-host
viral follow-up likewise found no incremental benefit over the general viral
head. The policies therefore encode reproducible content summaries, not an
independently validated immune-response default.

Use exact host-proteome lookup when available, and treat the learned score as
an experimental ranking feature alongside MHC presentation, processing, and
TCR-facing chemistry.
