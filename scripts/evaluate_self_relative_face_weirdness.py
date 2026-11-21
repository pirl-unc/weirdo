#!/usr/bin/env python3
"""Zero-fit, self-proteome-relative weirdness of the TCR-facing residues.

For each peptide the exposed positions P3..P(L-1) (the same face used in the
tcr-visible report) are scored against the human Swiss-Prot residue distribution:

* ``surprise``     mean -log f_human(aa)          (rare-in-self residues score high)
* ``hydrophobic``  mean Kyte-Doolittle value       (published scale)
* ``aromatic``     fraction of F/W/Y/H

Declared protocol: each feature has a fixed positive sign (more unusual or more
hydrophobic/aromatic exposed residues => more likely a T-cell epitope), enters as
``core + lam * (rank(feature) - 0.5)`` with lam in {0.1, 0.2, 0.33}, and is judged
against a within-dataset permutation null that takes the maximum over the same grid.
No sign, position set or weight was chosen after looking at outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_lodo_weirdo_models import FAMILIES, ecdf, load  # noqa: E402
from scripts.evaluate_weirdo_term_null import fast_auc  # noqa: E402

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
AROMATIC = set("FWYH")
LAMBDAS = (0.1, 0.2, 0.33)
FEATURES = ("surprise", "hydrophobic", "aromatic")


def human_surprise(sequences: pd.DataFrame) -> dict[str, float]:
    """-log residue frequency over reviewed human Swiss-Prot proteins."""
    counts = pd.Series(list("".join(sequences.loc[sequences["category"].eq("human"), "sequence"]))).value_counts()
    counts = counts.reindex(list(KYTE_DOOLITTLE)).fillna(0) + 1
    return (-np.log(counts / counts.sum())).to_dict()


def face(peptide: str) -> str:
    """P3..P(L-1) exposed residues (1-based), i.e. peptide[2:-1]."""
    return peptide[2:-1]


def features(peptides: pd.Series, surprise: dict[str, float]) -> pd.DataFrame:
    """The three declared face features."""
    rows = []
    for peptide in peptides:
        residues = face(peptide)
        rows.append(
            {
                "surprise": np.mean([surprise[a] for a in residues]),
                "hydrophobic": np.mean([KYTE_DOOLITTLE[a] for a in residues]),
                "aromatic": np.mean([a in AROMATIC for a in residues]),
            }
        )
    return pd.DataFrame(rows, index=peptides.index)


def gain(outcome, core, indices, base, rank, lam):
    """Macro (family-mean) AUC gain over the core for one feature and weight."""
    score = core + lam * (rank - 0.5)
    per_dataset = {d: fast_auc(outcome[i], score[i]) - base[d] for d, i in indices.items()}
    per_family = [np.mean([per_dataset[d] for d in members]) for members in FAMILIES.values()]
    return float(np.mean(per_family)), per_family


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--weirdo", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=500)
    args = parser.parse_args()
    frame, _ = load(args.internal, args.external, args.weirdo)
    surprise = human_surprise(pd.read_csv(args.sequences, usecols=["sequence", "category"]))
    feats = features(frame["peptide"], surprise)
    outcome = frame["outcome"].to_numpy(int)
    core = frame["core"].to_numpy(float)
    indices = frame.groupby("dataset").indices
    base = {d: fast_auc(outcome[i], core[i]) for d, i in indices.items()}
    ranks = {name: ecdf(feats[name].to_numpy(float)) for name in FEATURES}
    rows = []
    for name in FEATURES:
        standalone = {d: fast_auc(outcome[i], feats[name].to_numpy()[i]) for d, i in indices.items()}
        for lam in LAMBDAS:
            macro, per_family = gain(outcome, core, indices, base, ranks[name], lam)
            rows.append({"feature": name, "lam": lam, "d_macro": macro, **dict(zip(FAMILIES, per_family)),
                         "standalone_auc_min": min(standalone.values()), "standalone_auc_max": max(standalone.values())})
    table = pd.DataFrame(rows).sort_values("d_macro", ascending=False)
    rng = np.random.default_rng(84_342)
    matrix = np.column_stack([ranks[n] for n in FEATURES])
    null_max = []
    for _ in range(args.permutations):
        shuffled = matrix.copy()
        for i in indices.values():
            shuffled[i] = matrix[rng.permutation(i)]
        null_max.append(max(gain(outcome, core, indices, base, shuffled[:, j], lam)[0]
                            for j in range(len(FEATURES)) for lam in LAMBDAS))
    null_max = np.array(null_max)
    table["max_null_p"] = [(null_max >= g).mean() for g in table["d_macro"]]
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "self_relative_face_weirdness.csv", index=False)
    pd.set_option("display.width", 220)
    print(table.round(4).to_string(index=False))
    print("null max: median %.4f, 95th %.4f" % (np.median(null_max), np.percentile(null_max, 95)))
    per = pd.DataFrame({n: {d: fast_auc(outcome[i], feats[n].to_numpy()[i]) for d, i in indices.items()} for n in FEATURES})
    print(per.round(3))


if __name__ == "__main__":
    main()
