#!/usr/bin/env python3
"""Freeze the label-free rank references used by ``weirdo.pmhc_weirdo_score``.

Stores 1,001-point quantile tables (no labels, no outcomes) for MHCflurry 2.3
presentation, negated NetMHCpan BA rank, and the shipped ensemble's
``virus_foreign_like`` head, computed over the pooled evaluation peptides.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_lodo_weirdo_models import BA_RANK, PRESENTATION, load  # noqa: E402
from scripts.evaluate_self_relative_face_weirdness import human_surprise  # noqa: E402

HEAD = "virus_foreign_like"
POINTS = 1_001


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(values: np.ndarray) -> dict:
    """Quantile positions and values; duplicate values keep the highest probability."""
    probs = np.linspace(0.0, 1.0, POINTS)
    quantiles = np.quantile(values, probs)
    values_unique, last = np.unique(quantiles[::-1], return_index=True)
    probs_unique = probs[::-1][last]
    return {"values": values_unique.tolist(), "probs": probs_unique.tolist()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal", type=Path, required=True)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--weirdo", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True, help="swissprot-sequences.csv")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame, _ = load(args.internal, args.external, args.weirdo)
    residue_surprise = human_surprise(pd.read_csv(args.sequences, usecols=["sequence", "category"]))
    face = np.array([np.mean([residue_surprise[a] for a in p[2:-1]]) for p in frame["peptide"]])
    reference = {
        "version": "pmhc-rank-reference-v1",
        "n_peptides": int(len(frame)),
        "weirdo_model": "peptide-content-v1-ensemble",
        "weirdo_head": HEAD,
        "label_free": True,
        "sources_sha256": {p.name: digest(p) for p in (args.internal, args.external, args.weirdo, args.sequences)},
        "presentation": table(frame[PRESENTATION].to_numpy(float)),
        "negative_ba_rank": table(-frame[BA_RANK].to_numpy(float)),
        "weirdo": table(frame[HEAD].to_numpy(float)),
        "residue_surprise": {k: float(v) for k, v in residue_surprise.items()},
        "face_surprise": table(face),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, separators=(",", ":")))
    print(args.output, args.output.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
