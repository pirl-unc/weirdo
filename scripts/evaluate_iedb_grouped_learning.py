#!/usr/bin/env python3
"""Study-grouped pairwise ranking on the pooled IEDB pairs, with Weirdo ablations.

Objective matches the metric: XGBoost ``rank:pairwise`` with the study as the query
group learns to order peptide-HLA pairs *within* a study, so prevalence differences
between studies cannot be exploited. Folds hold out whole studies; any training pair
whose peptide also occurs in the held-out fold is dropped (exact-peptide purge).

Feature sets (declared up front):

* ``pmhc``          rank features of MHCflurry presentation / NetMHCpan BA, their core,
                    raw log-odds, peptide length
* ``pmhc_face``     pmhc + TCR-face composition (self-relative surprise, aromatic
                    fraction, Kyte-Doolittle hydrophobicity)
* ``pmhc_weirdo``   pmhc + the shipped ensemble's 23 taxonomy heads + face surprise
* ``all``           everything
* ``pmhc_selfnbr``  pmhc + same-HLA self-ligand neighbourhood (``--extra-features``,
                    from ``build_hla_self_ligand_neighborhood.py``)
* ``all_selfnbr``   everything + the neighbourhood
* ``pmhc_flank``    pmhc + flank-aware MHCflurry presentation (``ctx_flank_presentation``,
                    ``ctx_mapped``): a better pMHC baseline, not Weirdo
* ``pmhc_position`` pmhc + protein length / relative position / distance to termini
* ``pmhc_prot``     pmhc + Weirdo taxonomy profile of the source protein (``prot_*``)
* ``pmhc_ctx``      pmhc + all non-Weirdo source context
* ``pmhc_ctx_weirdo`` pmhc_ctx + protein profile + shipped heads + face surprise; its
                    gain over ``pmhc_ctx`` is Weirdo's contribution given context

Modes (declared before the pooled data were scored): ``residual`` (primary) boosts on
top of the core with ``base_margin = 4 * (core - 0.5)``, so the learner only fits
structure the core misses; ``plain`` learns from scratch and is reported as a control.

Primary metric: mean over eligible held-out studies (>= 20 pairs, >= 5 per class) of
within-study AUC(model) - AUC(core), study-clustered bootstrap interval. The Weirdo
contribution is ``pmhc_weirdo`` (or ``all``) against ``pmhc`` -- not against the core.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import weirdo  # noqa: E402
from weirdo.pmhc_score import face_surprise  # noqa: E402

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5, "G": -0.4,
    "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8,
    "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}
CONFIGS = {
    "default": dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0),
    "shallow": dict(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, min_child_weight=10, reg_lambda=10.0),
    "deep": dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
                 colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0),
}
MIN_PAIRS, MIN_CLASS = 20, 5
NON_HEADS = {
    "pmid", "peptide", "hla", "outcome", "source_organism", "netmhcpan_ba_score", "netmhcpan_ba_rank",
    "netmhcpan_affinity_nm", "netmhcpan_el_score", "netmhcpan_el_rank",
}


def heads_of(frame: pd.DataFrame) -> list[str]:
    """The ensemble's taxonomy head columns (those added by Weirdo scoring)."""
    return [c for c in frame.columns if c not in NON_HEADS and not c.startswith("mhcflurry_")]


def build_features(frame: pd.DataFrame, extra: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Feature matrices for each declared feature set (``extra`` adds the neighbourhood sets)."""
    ranked = weirdo.pmhc_weirdo_score(
        frame["peptide"], frame["mhcflurry_2_3_0_presentation_score"], frame["netmhcpan_ba_rank"],
        weirdo_weight=0.0, face_weight=0.0,
    )
    pmhc = pd.DataFrame(
        {
            "presentation_rank": ranked["presentation_rank"].to_numpy(),
            "binding_rank": ranked["binding_rank"].to_numpy(),
            "core": ranked["pmhc_core"].to_numpy(),
            "log_presentation": np.log10(frame["mhcflurry_2_3_0_presentation_score"].clip(1e-6, 1)).to_numpy(),
            "log_ba_rank": np.log10(frame["netmhcpan_ba_rank"].clip(1e-3, None)).to_numpy(),
            "length": frame["peptide"].str.len().to_numpy(),
        }
    )
    face = pd.DataFrame(
        {
            "face_surprise": [face_surprise(p) for p in frame["peptide"]],
            "face_aromatic": [np.mean([a in "FWYH" for a in p[2:-1]]) for p in frame["peptide"]],
            "face_hydrophobic": [np.mean([KYTE_DOOLITTLE[a] for a in p[2:-1]]) for p in frame["peptide"]],
        }
    )
    heads = frame[heads_of(frame)].reset_index(drop=True)
    sets = {
        "pmhc": pmhc,
        "pmhc_face": pd.concat([pmhc, face], axis=1),
        "pmhc_weirdo": pd.concat([pmhc, heads, face[["face_surprise"]]], axis=1),
        "all": pd.concat([pmhc, face, heads], axis=1),
    }
    if extra is not None:
        groups = {p: [c for c in extra.columns if c.startswith(p)] for p in ("selfnbr_", "ctx_", "prot_")}
        if groups["selfnbr_"]:
            sets["pmhc_selfnbr"] = pd.concat([pmhc, extra[groups["selfnbr_"]]], axis=1)
            sets["all_selfnbr"] = pd.concat([sets["all"], extra[groups["selfnbr_"]]], axis=1)
        if groups["ctx_"]:
            flank = [c for c in ("ctx_flank_presentation", "ctx_mapped") if c in extra.columns]
            position = [c for c in groups["ctx_"] if c not in ("ctx_flank_presentation", "ctx_mapped")]
            sets["pmhc_flank"] = pd.concat([pmhc, extra[flank]], axis=1)
            sets["pmhc_position"] = pd.concat([pmhc, extra[position + ["ctx_mapped"]]], axis=1)
            sets["pmhc_ctx"] = pd.concat([pmhc, extra[groups["ctx_"]]], axis=1)
        if groups["prot_"]:
            sets["pmhc_prot"] = pd.concat([pmhc, extra[groups["prot_"]]], axis=1)
        if groups["ctx_"] and groups["prot_"]:
            sets["pmhc_ctx_weirdo"] = pd.concat(
                [sets["pmhc_ctx"], extra[groups["prot_"]], heads, face[["face_surprise"]]], axis=1
            )
    return sets


MARGIN_SCALE = 4.0


def margin(matrix: pd.DataFrame, scale: float) -> np.ndarray | None:
    """Base margin from the pMHC core (None for the plain control)."""
    return scale * (matrix["core"].to_numpy(float) - 0.5) if scale else None


def fit_predict(
    train: pd.DataFrame, x_train: pd.DataFrame, x_test: pd.DataFrame, params: dict, seed: int, scale: float
) -> np.ndarray:
    """Rank model trained with the study as the query group."""
    from xgboost import XGBRanker  # lazy: its OpenMP runtime must not load next to torch's

    qid = pd.factorize(train["pmid"])[0]
    order = np.argsort(qid, kind="stable")
    model = XGBRanker(objective="rank:pairwise", random_state=seed, n_jobs=4, tree_method="hist", **params)
    base = margin(x_train, scale)
    model.fit(
        x_train.iloc[order], train["outcome"].to_numpy(int)[order], qid=qid[order],
        base_margin=None if base is None else base[order],
    )
    return model.predict(x_test, base_margin=margin(x_test, scale))


def eligible_studies(frame: pd.DataFrame) -> list[str]:
    counts = frame.groupby("pmid")["outcome"].agg(["size", "sum"])
    ok = (counts["size"] >= MIN_PAIRS) & (counts["sum"] >= MIN_CLASS) & (counts["size"] - counts["sum"] >= MIN_CLASS)
    return counts.index[ok].tolist()


def cross_validate(
    frame: pd.DataFrame, features: dict[str, pd.DataFrame], params: dict, folds: int, seed: int, scale: float
) -> pd.DataFrame:
    """Out-of-fold scores for every feature set (held-out studies, peptide-purged training)."""
    both = frame.groupby("pmid")["outcome"].transform(lambda s: 0 < s.sum() < len(s)).to_numpy()
    studies = frame["pmid"].to_numpy()
    out = pd.DataFrame(np.nan, index=frame.index, columns=list(features))
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in splitter.split(frame, groups=studies):
        test_peptides = set(frame["peptide"].to_numpy()[test_idx])
        train_idx = train_idx[both[train_idx] & ~frame["peptide"].iloc[train_idx].isin(test_peptides).to_numpy()]
        for name, matrix in features.items():
            out.iloc[test_idx, out.columns.get_loc(name)] = fit_predict(
                frame.iloc[train_idx], matrix.iloc[train_idx], matrix.iloc[test_idx], params, seed, scale
            )
    return out


def within_study_gaps(frame: pd.DataFrame, scores: pd.DataFrame, reference: np.ndarray, studies: list[str]) -> pd.DataFrame:
    """Per-study AUC(model) - AUC(reference) for every column of ``scores``."""
    rows = {}
    for study in studies:
        m = frame["pmid"].eq(study).to_numpy()
        y = frame["outcome"].to_numpy(int)[m]
        base = roc_auc_score(y, reference[m])
        rows[study] = {name: roc_auc_score(y, scores[name].to_numpy()[m]) - base for name in scores.columns}
    return pd.DataFrame(rows).T


def summarize(gaps: pd.DataFrame, draws: int, seed: int = 84_342) -> pd.DataFrame:
    picks = np.random.default_rng(seed).integers(0, len(gaps), (draws, len(gaps)))
    rows = []
    for name in gaps.columns:
        boot = gaps[name].to_numpy()[picks].mean(axis=1)
        rows.append({"model": name, "mean_gap": gaps[name].mean(), "low": np.percentile(boot, 2.5),
                     "high": np.percentile(boot, 97.5), "studies_improved": int((gaps[name] > 0).sum()), "studies": len(gaps)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=84_342)
    parser.add_argument("--bootstrap-draws", type=int, default=2_000)
    parser.add_argument("--extra-features", type=Path, nargs="+", help="feature CSVs keyed by (pmid,) peptide, hla")
    parser.add_argument("--margin-scale", type=float, default=MARGIN_SCALE, help="0 = plain control")
    parser.add_argument("--only", choices=sorted(CONFIGS), help="run a single config")
    args = parser.parse_args()
    frame = pd.read_csv(args.pool, low_memory=False)
    frame["pmid"] = frame["pmid"].astype(str)
    extra = None
    if args.extra_features:
        parts = []
        for path in args.extra_features:
            table = pd.read_csv(path, dtype={"pmid": str})
            keys = [k for k in ("pmid", "peptide", "hla") if k in table.columns]
            merged = frame[keys].merge(table.drop_duplicates(keys), on=keys, how="left")
            parts.append(merged.drop(columns=keys).reset_index(drop=True))
        extra = pd.concat(parts, axis=1)
        extra = extra.loc[:, ~extra.columns.duplicated()]
    features = build_features(frame, extra)
    core = features["pmhc"]["core"].to_numpy()
    studies = eligible_studies(frame)
    print("eligible held-out studies:", len(studies))
    args.output.mkdir(parents=True, exist_ok=True)
    selected = {args.only: CONFIGS[args.only]} if args.only else CONFIGS
    for label, params in selected.items():
        scores = cross_validate(frame, features, params, args.folds, args.seed, args.margin_scale)
        scores["presentation"] = frame["mhcflurry_2_3_0_presentation_score"].to_numpy()
        gaps = within_study_gaps(frame, scores, core, studies)
        table = summarize(gaps, args.bootstrap_draws)
        boot = np.random.default_rng(1).integers(0, len(gaps), (args.bootstrap_draws, len(gaps)))
        effects = []
        for label_, with_, base_ in (
            ("weirdo_heads_effect (pmhc_weirdo - pmhc)", "pmhc_weirdo", "pmhc"),
            ("face_effect (pmhc_face - pmhc)", "pmhc_face", "pmhc"),
            ("selfnbr_effect (pmhc_selfnbr - pmhc)", "pmhc_selfnbr", "pmhc"),
            ("selfnbr_on_top (all_selfnbr - all)", "all_selfnbr", "all"),
            ("flank_effect (pmhc_flank - pmhc)", "pmhc_flank", "pmhc"),
            ("position_effect (pmhc_position - pmhc)", "pmhc_position", "pmhc"),
            ("protein_weirdo_effect (pmhc_prot - pmhc)", "pmhc_prot", "pmhc"),
            ("context_effect (pmhc_ctx - pmhc)", "pmhc_ctx", "pmhc"),
            ("weirdo_given_context (pmhc_ctx_weirdo - pmhc_ctx)", "pmhc_ctx_weirdo", "pmhc_ctx"),
        ):
            if with_ not in gaps.columns:
                continue
            diff = (gaps[with_] - gaps[base_]).to_numpy()
            effects.append({"model": label_, "mean_gap": diff.mean(),
                            "low": np.percentile(diff[boot].mean(axis=1), 2.5),
                            "high": np.percentile(diff[boot].mean(axis=1), 97.5),
                            "studies_improved": int((diff > 0).sum()), "studies": len(gaps)})
        extra = pd.DataFrame(effects)
        table = pd.concat([table, extra], ignore_index=True)
        table.to_csv(args.output / f"grouped_cv_{label}_m{args.margin_scale:g}.csv", index=False)
        gaps.to_csv(args.output / f"grouped_cv_{label}_m{args.margin_scale:g}_per_study.csv")
        print(f"== config {label}, margin scale {args.margin_scale:g} (gap = within-study AUC minus the pMHC core)")
        print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
