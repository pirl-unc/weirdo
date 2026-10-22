#!/usr/bin/env python3
"""Protein-family-aware k-mer splits with per-category labels from SwissProt.

The ten WEIRDO categories are the UniProt taxonomic divisions of SwissProt
(archaea, bacteria, fungi, human, invertebrates, mammals, plants, rodents,
vertebrates, viruses). A k-mer is labelled positive for a category when it
occurs in at least one SwissProt protein of that division, so labels are
multi-hot: a conserved k-mer is positive for several categories at once.

Leakage control
---------------
Overlapping k-mers of one protein and k-mers of orthologous proteins share most
of their residues, so a random split of k-mers would let a model memorize what
it is later tested on. Proteins are therefore split by the mnemonic prefix of the
UniProt entry name (``RS19`` in ``RS19_ECOLI``), and the prefix is hashed into
train / validation / test folds. This groups identical names, not sequence
homology clusters; homologs with different names can cross folds. Only k-mers
that occur in proteins of exactly one fold are kept, so no identical k-mer
crosses folds and each kept k-mer's labels come entirely from its own fold.

Build the cached splits::

    python scripts/category_data.py build \
        --sequences data/swissprot-sequences.csv --out data/category-splits
"""

import argparse
import json
import time
import zlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from weirdo.kmer_table import (  # noqa: F401  (re-exported for the tests and other scripts)
    AMINO_ACIDS,
    BITS_PER_RESIDUE,
    MAX_K,
    decode_kmers,
    encode_residues,
    masks_to_matrix,
    unique_with_masks,
    window_codes,
)
from weirdo.uniprot import SWISSPROT_DIVISIONS, read_sequence_table

CATEGORIES = list(SWISSPROT_DIVISIONS)
SPLIT_NAMES = ("train", "val", "test")
DEFAULT_SALT = "category-predictor-v1"
DEFAULT_TRAIN_PERCENT = 70
DEFAULT_VAL_PERCENT = 15
N_BUCKETS = 20  # family buckets inside each fold, for cluster bootstrap intervals


def family_key(identifier: str) -> str:
    """Mnemonic prefix of a UniProt entry name: the gene/protein family key."""
    return identifier.split("_", 1)[0]


def split_of_family(
    key: str,
    salt: str = DEFAULT_SALT,
    train_percent: int = DEFAULT_TRAIN_PERCENT,
    val_percent: int = DEFAULT_VAL_PERCENT,
) -> int:
    """Deterministic split index (0 train, 1 val, 2 test) for a family key."""
    bucket = zlib.crc32((salt + ":" + key).encode("utf-8")) % 100
    if bucket < train_percent:
        return 0
    if bucket < train_percent + val_percent:
        return 1
    return 2


def bucket_of_family(key: str, salt: str = DEFAULT_SALT, n_buckets: int = N_BUCKETS) -> int:
    """Deterministic bucket (0..n_buckets-1) of a family key within its fold.

    Buckets partition each fold's protein families, so resampling buckets (rather
    than individual k-mers) respects the correlation among k-mers of one family.
    """
    return zlib.crc32((salt + ":bucket:" + key).encode("utf-8")) % n_buckets


def read_sequences(path: str):
    """Read ``identifier,sequence,category`` rows (also the older layout with an index column)."""
    frame = read_sequence_table(path)
    unknown = sorted(set(frame["category"]) - set(CATEGORIES))
    if unknown:
        raise ValueError(f"Unknown categories in {path}: {unknown}")
    return frame


def build_universes(
    frame,
    k: int = 8,
    salt: str = DEFAULT_SALT,
    train_percent: int = DEFAULT_TRAIN_PERCENT,
    val_percent: int = DEFAULT_VAL_PERCENT,
    verbose: bool = True,
    include_all: bool = True,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, int]]:
    """Unique k-mers with category masks, per fold and over all proteins.

    Returns ``(universes, shared_counts)``. ``universes`` maps
    ``"train"|"val"|"test"|"all"`` to ``{"kmers": uint64, "masks": uint16}``; the
    fold universes hold only k-mers found in that fold's proteins and in no other
    fold's, and ``shared_counts`` is how many k-mers each fold lost that way. The
    ``"all"`` universe (every protein) is the largest sort; pass
    ``include_all=False`` to skip it.
    """
    def log(message: str) -> None:
        if verbose:
            print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    sequences = frame["sequence"].tolist()
    fold_of_protein = np.array(
        [split_of_family(family_key(i), salt, train_percent, val_percent) for i in frame["identifier"]],
        dtype=np.uint8,
    )
    category_index = {name: i for i, name in enumerate(CATEGORIES)}
    bit_of_protein = np.array(
        [1 << category_index[c] for c in frame["category"]], dtype=np.uint16
    )
    bucket_of_protein = np.array(
        [bucket_of_family(family_key(i), salt) for i in frame["identifier"]], dtype=np.uint8
    )
    log(f"{len(sequences):,} proteins; fold sizes "
        f"{[int((fold_of_protein == f).sum()) for f in range(3)]}")

    codes, starts = encode_residues(sequences)
    log(f"{len(codes):,} residue codes")
    packed, valid = window_codes(codes, k)
    protein_of_position = np.repeat(
        np.arange(len(sequences), dtype=np.int32), np.fromiter((len(s) for s in sequences), dtype=np.int64) + 1
    )[:len(packed)]
    del codes
    packed = packed[valid]
    protein_of_window = protein_of_position[valid]
    del protein_of_position, valid
    log(f"{len(packed):,} canonical {k}-mer windows")

    fold_of_window = fold_of_protein[protein_of_window]
    bit_of_window = bit_of_protein[protein_of_window]
    bucket_of_window = bucket_of_protein[protein_of_window]
    del protein_of_window

    universes: Dict[str, Dict[str, np.ndarray]] = {}
    for fold, name in enumerate(SPLIT_NAMES):
        selected = fold_of_window == fold
        kmers, masks, buckets = unique_with_masks(
            packed[selected], bit_of_window[selected], bucket_of_window[selected]
        )
        universes[name] = {"kmers": kmers, "masks": masks, "groups": buckets}
        log(f"fold {name}: {len(kmers):,} unique {k}-mers")
    if include_all:
        all_kmers, all_masks = unique_with_masks(packed, bit_of_window)
        universes["all"] = {"kmers": all_kmers, "masks": all_masks}
        log(f"all folds: {len(all_kmers):,} unique {k}-mers")
    del packed, fold_of_window, bit_of_window, bucket_of_window

    # Keep only k-mers that occur in exactly one fold. Every fold is compared with
    # the other folds' unfiltered universes, so the exclusion is symmetric.
    shared_flags = {}
    for name in SPLIT_NAMES:
        in_other = np.zeros(len(universes[name]["kmers"]), dtype=bool)
        for other in SPLIT_NAMES:
            if other != name:
                in_other |= np.isin(
                    universes[name]["kmers"], universes[other]["kmers"], assume_unique=True
                )
        shared_flags[name] = in_other
    shared_counts = {}
    for name in SPLIT_NAMES:
        flags = shared_flags[name]
        shared_counts[name] = int(flags.sum())
        universes[name] = {key: values[~flags] for key, values in universes[name].items()}
        log(f"fold {name}: dropped {shared_counts[name]:,} k-mers also present in another fold; "
            f"{len(universes[name]['kmers']):,} remain")
    return universes, shared_counts


def draw_sample(
    universe: Dict[str, np.ndarray], n: Optional[int], rng: np.random.RandomState
) -> Dict[str, np.ndarray]:
    """Uniform sample without replacement, keeping every array of the universe aligned.

    Returns all rows when ``n`` is None or not smaller than the universe.
    """
    total = len(universe["kmers"])
    if n is None or n >= total:
        return dict(universe)
    chosen = np.sort(rng.choice(total, size=n, replace=False))
    return {key: values[chosen] for key, values in universe.items()}


def draw_disjoint_sample(
    universe: Dict[str, np.ndarray],
    taken: Dict[str, np.ndarray],
    n: int,
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    """Sample ``n`` rows of ``universe`` that are not in ``taken`` (compared by k-mer)."""
    available = np.flatnonzero(~np.isin(universe["kmers"], taken["kmers"], assume_unique=True))
    chosen = np.sort(rng.choice(available, size=min(n, len(available)), replace=False))
    return {key: values[chosen] for key, values in universe.items()}


def verify_against_kmer_table(
    universe: Dict[str, np.ndarray], table_csv: str, k: int = 8, nrows: int = 300_000
) -> Dict[str, float]:
    """Compare rebuilt labels with the first rows of the shipped k-mer table.

    The table (``data/swissprot-8mers.csv``) has one row per unique k-mer and one
    True/False column per category. Returns the fraction of its k-mers found in
    the rebuilt universe and, among those, the fraction whose label vector matches.
    """
    import pandas as pd

    table = pd.read_csv(table_csv, nrows=nrows, index_col=0)
    flags = table[CATEGORIES].to_numpy(dtype=bool)
    table_masks = (flags.astype(np.uint16) << np.arange(len(CATEGORIES), dtype=np.uint16)).sum(axis=1).astype(np.uint16)
    codes, _ = encode_residues(table["seq"].tolist())
    packed, valid = window_codes(codes, k)
    # One k-mer per row, separated by NUL: windows start every k + 1 characters.
    row_codes = packed[np.arange(len(table)) * (k + 1)]
    kmers = universe["kmers"]
    position = np.clip(np.searchsorted(kmers, row_codes), 0, len(kmers) - 1)
    found = kmers[position] == row_codes
    matches = universe["masks"][position] == table_masks
    return {
        "rows": float(len(table)),
        "found_fraction": float(found.mean()),
        "label_match_fraction_of_found": float(matches[found].mean()) if found.any() else float("nan"),
    }


def load_split(
    directory: str,
    name: str,
    n: Optional[int] = None,
    seed: int = 0,
    return_groups: bool = False,
):
    """Peptides and multi-hot labels of a cached split, in a seeded random order.

    The cached arrays are stored in lexicographic k-mer order, so taking a prefix
    would over-represent k-mers that start with early residues. This shuffles
    first and then keeps ``n`` rows. With ``return_groups`` a third value holds
    each k-mer's family bucket (fold splits only) for cluster bootstrapping.
    """
    archive = np.load(Path(directory) / "splits.npz")
    kmers, masks = archive[f"{name}_kmers"], archive[f"{name}_masks"]
    order = np.random.RandomState(seed).permutation(len(kmers))
    if n is not None:
        order = order[:n]
    peptides, labels = kmers[order].tolist(), masks_to_matrix(masks[order])
    if not return_groups:
        return peptides, labels
    key = f"{name}_groups"
    if key not in archive.files:
        raise KeyError(f"{directory} has no family buckets for split '{name}'")
    return peptides, labels, archive[key][order]


def build_and_cache(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame = read_sequences(args.sequences)
    needs_all = args.n_all != 0 or bool(args.verify_table)
    universes, shared_counts = build_universes(
        frame, k=args.k, salt=args.salt, include_all=needs_all
    )
    verification = None
    if args.verify_table:
        verification = verify_against_kmer_table(universes["all"], args.verify_table, k=args.k)
        print("table verification:", json.dumps(verification), flush=True)
    rng = np.random.RandomState(args.seed)
    sizes = {"train": args.n_train, "val": args.n_val, "test": args.n_test, "all": args.n_all}
    held_out_size = getattr(args, "n_train_heldout", 0)
    arrays = {}
    meta = {
        "k": args.k, "salt": args.salt, "seed": args.seed, "categories": CATEGORIES,
        "sequences": str(args.sequences), "proteins": int(len(frame)),
        "shared_across_folds": shared_counts,
        "universe_sizes": {n: int(len(universes[n]["kmers"])) for n in universes},
        "sample_sizes": {}, "prevalence": {}, "table_verification": verification,
    }
    samples = {}
    for name in (*SPLIT_NAMES, "all"):
        if sizes[name] == 0 or name not in universes:
            continue
        samples[name] = draw_sample(universes[name], sizes[name], rng)
    if held_out_size > 0 and "train" in samples:
        # Random k-mers from the training families that the model never trains on:
        # compared with the family-held-out test fold this measures how much a
        # naive random split would overstate performance.
        held_rng = np.random.RandomState(args.seed + 1)
        samples["train_heldout"] = draw_disjoint_sample(
            universes["train"], samples["train"], held_out_size, held_rng
        )
    for name, sample in samples.items():
        arrays[f"{name}_kmers"] = np.array(decode_kmers(sample["kmers"], args.k))
        arrays[f"{name}_masks"] = sample["masks"]
        if "groups" in sample:
            arrays[f"{name}_groups"] = sample["groups"]
        meta["sample_sizes"][name] = int(len(sample["kmers"]))
        prevalence = (
            [round(float(p), 5) for p in masks_to_matrix(sample["masks"]).mean(axis=0)]
            if len(sample["masks"]) else [None] * len(CATEGORIES)
        )
        meta["prevalence"][name] = dict(zip(CATEGORIES, prevalence))
    np.savez_compressed(out / "splits.npz", **arrays)
    with open(out / "splits.json", "w") as handle:
        json.dump(meta, handle, indent=2)
    print(json.dumps(meta["sample_sizes"]), json.dumps(meta["shared_across_folds"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser(
        "build",
        help="Build and cache sampled train/val/test/all k-mer sets (a size of 0 skips a set)",
    )
    build.add_argument("--sequences", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--k", type=int, default=8)
    build.add_argument("--salt", default=DEFAULT_SALT)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--n-train", type=int, default=3_000_000)
    build.add_argument("--n-val", type=int, default=300_000)
    build.add_argument("--n-test", type=int, default=600_000)
    build.add_argument("--n-all", type=int, default=4_000_000)
    build.add_argument("--n-train-heldout", type=int, default=0,
                       help="Also sample k-mers of the training families that are not in the "
                            "training sample (leakage diagnostic)")
    build.add_argument("--verify-table", default=None,
                       help="Optional path to swissprot-8mers.csv to cross-check rebuilt labels")
    build.set_defaults(func=build_and_cache)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
