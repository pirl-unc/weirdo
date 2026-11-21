#!/usr/bin/env python3
"""One-shot confirmation of the frozen pMHC + Weirdo score on the fresh IEDB cohort.

Primary (declared before scoring): mean over studies with >= 20 pairs and >= 5 of
each class of within-study AUC(full) - AUC(core); interval by study-clustered
bootstrap. Secondary: pooled AUC, which mixes study prevalences. Weights are the
shipped defaults; nothing is tuned here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import weirdo  # noqa: E402

MIN_PAIRS, MIN_CLASS = 20, 5


def score_variants(frame: pd.DataFrame) -> pd.DataFrame:
    """Frozen variants of the packaged score plus the two raw predictors."""
    args = (
        frame["peptide"],
        frame["mhcflurry_2_3_0_presentation_score"],
        frame["netmhcpan_ba_rank"],
    )
    heads = frame["virus_foreign_like"]
    out = pd.DataFrame(index=frame.index)
    out["presentation"] = frame["mhcflurry_2_3_0_presentation_score"].to_numpy()
    out["ba"] = -frame["netmhcpan_ba_rank"].to_numpy()
    out["core"] = weirdo.pmhc_weirdo_score(*args, weirdo_weight=0.0, face_weight=0.0)["score"].to_numpy()
    out["core_vfl"] = weirdo.pmhc_weirdo_score(*args, weirdo_scores=heads, face_weight=0.0)["score"].to_numpy()
    out["core_face"] = weirdo.pmhc_weirdo_score(*args, weirdo_weight=0.0)["score"].to_numpy()
    out["full"] = weirdo.pmhc_weirdo_score(*args, weirdo_scores=heads)["score"].to_numpy()
    return out


def eligible(frame: pd.DataFrame) -> list[str]:
    counts = frame.groupby("pmid")["outcome"].agg(["size", "sum"])
    ok = (counts["size"] >= MIN_PAIRS) & (counts["sum"] >= MIN_CLASS) & (counts["size"] - counts["sum"] >= MIN_CLASS)
    return counts.index[ok].astype(str).tolist()


def within_study(frame: pd.DataFrame, scores: pd.DataFrame, studies: list[str], a: str, b: str) -> float:
    gaps = []
    for study in studies:
        rows = frame["pmid"].astype(str).eq(study).to_numpy()
        y = frame["outcome"].to_numpy(int)[rows]
        gaps.append(roc_auc_score(y, scores[a].to_numpy()[rows]) - roc_auc_score(y, scores[b].to_numpy()[rows]))
    return float(np.mean(gaps))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    args = parser.parse_args()
    frame = pd.read_csv(args.scored, low_memory=False)
    frame["pmid"] = frame["pmid"].astype(str)
    scores = score_variants(frame)
    studies = eligible(frame)
    y = frame["outcome"].to_numpy(int)
    rows = [
        {
            "score": name,
            "pooled_auc": roc_auc_score(y, scores[name]),
            "mean_within_study_auc": np.mean([
                roc_auc_score(y[m], scores[name].to_numpy()[m])
                for m in (frame["pmid"].eq(s).to_numpy() for s in studies)
            ]),
        }
        for name in scores.columns
    ]
    table = pd.DataFrame(rows)
    study_rows = {s: frame["pmid"].eq(s).to_numpy() for s in studies}
    gaps = pd.DataFrame(
        {
            name: [
                roc_auc_score(y[m], scores[name].to_numpy()[m]) - roc_auc_score(y[m], scores["core"].to_numpy()[m])
                for m in study_rows.values()
            ]
            for name in ("core_vfl", "core_face", "full", "presentation", "ba")
        },
        index=studies,
    )
    rng = np.random.default_rng(84_342)
    picks = rng.integers(0, len(studies), (args.bootstrap_draws, len(studies)))
    out = []
    for name in gaps.columns:
        boot = gaps[name].to_numpy()[picks].mean(axis=1)
        out.append({"score": name, "delta_within_study_auc_vs_core": gaps[name].mean(),
                    "low": np.percentile(boot, 2.5), "high": np.percentile(boot, 97.5),
                    "studies_improved": int((gaps[name] > 0).sum()), "studies": len(studies)})
    deltas = pd.DataFrame(out)
    organism = frame.groupby("pmid")["source_organism"].first().fillna("")
    kind = organism.map(lambda o: "unknown" if not o else ("human" if o == "Homo sapiens" else "pathogen"))
    strata = []
    for label in ("pathogen", "human", "unknown"):
        members = [s for s in studies if kind[s] == label]
        if members:
            strata.append({"source": label, "studies": len(members),
                           **{f"d_{n}": gaps.loc[members, n].mean() for n in ("core_vfl", "core_face", "full")}})
    strata = pd.DataFrame(strata)
    per_study = pd.DataFrame(
        {s: {n: roc_auc_score(y[m], scores[n].to_numpy()[m]) for n in scores.columns}
         for s in studies for m in [frame["pmid"].eq(s).to_numpy()]}
    ).T
    per_study["n"] = frame.groupby("pmid").size().reindex(studies).to_numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output / "fresh_cohort_scores.csv", index=False)
    deltas.to_csv(args.output / "fresh_cohort_deltas.csv", index=False)
    strata.to_csv(args.output / "fresh_cohort_by_source.csv", index=False)
    per_study.to_csv(args.output / "fresh_cohort_per_study.csv")
    pd.set_option("display.width", 200)
    print("eligible studies:", len(studies), "rows", int(frame["pmid"].isin(studies).sum()))
    for shown in (table, deltas, strata):
        print(shown.round(4).to_string(index=False))
    print(per_study.round(3).head(20))


if __name__ == "__main__":
    main()
