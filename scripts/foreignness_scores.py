"""Foreignness-score engine shared by the benchmark scripts.

Every definition of "distance to self" benchmarked in this repository is a
choice along a few axes, and this module implements them with one engine so the
comparison is between choices and not between code paths:

* the **reference** the peptide is compared with: every window of the human
  proteome, the windows compatible with the query allele's anchor motif, or a
  set of eluted self ligands;
* the **positions** compared: the TCR-facing face (P3 through the penultimate
  residue) or the whole peptide;
* the **substitution matrix**: BLOSUM62/50, PMBEC, a physicochemical distance
  built from the package's property tables, or plain mismatch counting;
* the **statistic**: the exact nearest neighbour or a soft maximum
  ``log(sum(exp(beta * similarity))) / beta`` over all reference windows.

It also provides k-mer rarity scores and an anchor-motif estimator driven by
MHCflurry. Nothing here touches T-cell outcomes.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from weirdo import amino_acid_properties as properties
from weirdo.amino_acid_alphabet import canonical_amino_acid_letters
from weirdo.blosum import blosum50_dict, blosum62_dict
from weirdo.pmbec import pmbec_dict

AA = "".join(sorted(canonical_amino_acid_letters))
AA_INDEX = {aa: i for i, aa in enumerate(AA)}
N_AA = len(AA)
SEPARATOR = 255
MIN_LENGTH, MAX_LENGTH = 8, 14
POSITION_CHOICES = ("face", "whole")
MATRIX_CHOICES = ("blosum62", "blosum50", "pmbec", "physchem", "hamming")


def tcr_face(peptide: str) -> str:
    """P3 through the penultimate residue, excluding the common anchors."""
    return peptide[2:-1]


def similarity_matrix(name: str) -> np.ndarray:
    """A 20x20 float32 similarity matrix in :data:`AA` order, higher = closer."""
    if name in ("blosum62", "blosum50"):
        table = blosum62_dict if name == "blosum62" else blosum50_dict
        return np.asarray(
            [[table[a][b] for b in AA] for a in AA], dtype=np.float32)
    if name == "pmbec":
        return np.asarray(
            [[pmbec_dict[a][b] for b in AA] for a in AA], dtype=np.float32)
    if name == "hamming":
        return np.eye(N_AA, dtype=np.float32)
    if name == "physchem":
        # Negative Euclidean distance in a standardized space of the package's
        # hydropathy, volume and polarity tables.
        columns = np.asarray([
            [table[aa] for aa in AA]
            for table in (
                properties.hydropathy, properties.volume,
                properties.polarity)], dtype=float).T
        columns = (columns - columns.mean(axis=0)) / columns.std(axis=0)
        distance = np.linalg.norm(
            columns[:, None, :] - columns[None, :, :], axis=2)
        return (-distance).astype(np.float32)
    raise ValueError(f"unknown matrix {name!r}; choose from {MATRIX_CHOICES}")


def encode(sequence: str) -> np.ndarray:
    return np.asarray([AA_INDEX[aa] for aa in sequence], dtype=np.uint8)


def pair_code(second: np.ndarray, last: np.ndarray) -> np.ndarray:
    """Index into a flat 20x20 table of (P2, P-omega) anchor pairs."""
    return second.astype(np.int32) * N_AA + last.astype(np.int32)


class ProteomeIndex:
    """Every canonical window of a protein collection, addressable by length.

    Proteins are joined into one stream with a separator that can never match,
    so no window spans a protein boundary or an ambiguous residue.
    """

    def __init__(self, proteins: list[str]):
        lookup = np.full(256, SEPARATOR, dtype=np.uint8)
        for aa, index in AA_INDEX.items():
            lookup[ord(aa)] = index
        joined = "*".join(proteins).encode("ascii")
        self.stream = lookup[np.frombuffer(joined, dtype=np.uint8)]
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def from_swissprot(cls, path: Path, category: str = "human"):
        with Path(path).open(newline="") as stream:
            return cls([
                row["sequence"] for row in csv.DictReader(stream)
                if row["category"] == category])

    def _windows(self, length: int) -> tuple[np.ndarray, np.ndarray]:
        """Start offsets of every valid window and its (P2, P-omega) code."""
        if length not in self._cache:
            n = len(self.stream) - length + 1
            if n <= 0:
                starts = np.empty(0, dtype=np.int64)
                codes = np.empty(0, dtype=np.int32)
            else:
                window = np.lib.stride_tricks.sliding_window_view(
                    self.stream, length)
                valid = ~(window == SEPARATOR).any(axis=1)
                starts = np.flatnonzero(valid)
                codes = pair_code(window[valid][:, 1], window[valid][:, -1])
            self._cache[length] = (starts, codes)
        return self._cache[length]

    def count(self, length: int) -> int:
        return len(self._windows(length)[0])

    def blocks(self, length: int, group=None, allowed_pairs=None,
               chunk: int = 50_000):
        """Yield ``(n, length)`` uint8 window blocks, optionally anchor-filtered.

        ``allowed_pairs`` is a boolean array over the 400 (P2, P-omega) pairs; a
        window is kept only if its anchor pair is allowed.
        """
        starts, codes = self._windows(length)
        if allowed_pairs is not None:
            starts = starts[np.asarray(allowed_pairs)[codes]]
        offsets = np.arange(length)
        for first in range(0, len(starts), chunk):
            yield self.stream[starts[first:first + chunk, None] + offsets]


class PeptideSetIndex:
    """A reference made of whole peptides, optionally split by group (allele).

    Only peptides of exactly the query length are compared, so a ligand is never
    treated as a source of sub-windows.
    """

    def __init__(self, peptides_by_group: dict, fallback_minimum: int = 200):
        self._by_group_length: dict = defaultdict(list)
        self._pooled: dict[int, list[np.ndarray]] = defaultdict(list)
        self.fallback_minimum = fallback_minimum
        seen = set()
        for group, peptides in peptides_by_group.items():
            for peptide in peptides:
                if not MIN_LENGTH <= len(peptide) <= MAX_LENGTH:
                    continue
                if set(peptide) - set(AA):
                    continue
                self._by_group_length[(group, len(peptide))].append(
                    encode(peptide))
                if peptide not in seen:
                    seen.add(peptide)
                    self._pooled[len(peptide)].append(encode(peptide))
        self.fallback_groups: set = set()

    def count(self, length: int, group=None) -> int:
        if group is None:
            return len(self._pooled.get(length, ()))
        return len(self._by_group_length.get((group, length), ()))

    def blocks(self, length: int, group=None, allowed_pairs=None,
               chunk: int = 50_000):
        members = (
            self._by_group_length.get((group, length), ())
            if group is not None else self._pooled.get(length, ()))
        if group is not None and len(members) < self.fallback_minimum:
            # Too few allele-specific ligands of this length for a meaningful
            # nearest neighbour: fall back to all ligands, and say so.
            self.fallback_groups.add((group, length))
            members = self._pooled.get(length, ())
        if len(members):
            array = np.stack(members)
            for first in range(0, len(array), chunk):
                yield array[first:first + chunk]


def nearest_self_distance(
        reference, peptides: list[str], *, groups: list | None = None,
        positions: str = "face", matrix: str = "blosum62",
        beta: float | None = 1.0, allowed_pairs: dict | None = None,
        query_batch: int = 256, chunk: int = 50_000,
        return_windows: bool = False) -> dict:
    """Exact and soft distance from each peptide to its nearest reference window.

    ``distance = (self_similarity - best_similarity) / n_positions``, so 0 means
    the compared positions occur verbatim in the reference. ``groups`` (one per
    peptide) selects a per-allele reference or anchor filter; ``allowed_pairs``
    maps a group to the boolean (P2, P-omega) table its windows must satisfy.
    Peptides with no reference window of their length get NaN. With
    ``return_windows`` the nearest window itself is also returned (``"window"``);
    for a point-mutated peptide under the ``hamming`` matrix that is its
    wild-type counterpart.
    """
    if positions not in POSITION_CHOICES:
        raise ValueError(f"positions must be one of {POSITION_CHOICES}")
    similarity = similarity_matrix(matrix)
    n = len(peptides)
    nearest = np.full(n, np.nan)
    soft = np.full(n, np.nan)
    windows: list = [None] * n
    by_key = defaultdict(list)
    for i, peptide in enumerate(peptides):
        by_key[(len(peptide), None if groups is None else groups[i])].append(i)
    for (length, group), members in by_key.items():
        if not MIN_LENGTH <= length <= MAX_LENGTH:
            continue
        columns = (np.arange(2, length - 1) if positions == "face"
                   else np.arange(length))
        if not len(columns):
            continue
        encoded = np.stack([encode(peptides[i])[columns] for i in members])
        scoring = np.empty((len(members), len(columns) * N_AA),
                           dtype=np.float32)
        for slot in range(len(columns)):
            scoring[:, slot * N_AA:(slot + 1) * N_AA] = (
                similarity[encoded[:, slot]])
        self_similarity = similarity[encoded, encoded].sum(axis=1)
        best = np.full(len(members), -np.inf, dtype=np.float64)
        best_block = [None] * len(members)
        log_sum = np.full(len(members), -np.inf)
        offsets = np.arange(len(columns)) * N_AA
        filter_pairs = (
            None if allowed_pairs is None else allowed_pairs.get(group))
        seen_any = False
        for block in reference.blocks(length, group, filter_pairs, chunk):
            if not len(block):
                continue
            seen_any = True
            hot = np.zeros((len(block), len(columns) * N_AA),
                           dtype=np.float32)
            hot[np.arange(len(block))[:, None],
                offsets[None, :] + block[:, columns]] = 1
            for first in range(0, len(members), query_batch):
                part = slice(first, first + query_batch)
                sims = scoring[part] @ hot.T
                top = sims.max(axis=1).astype(np.float64)
                if return_windows:
                    which = sims.argmax(axis=1)
                    for slot in np.flatnonzero(top > best[part]):
                        best_block[first + slot] = block[which[slot]]
                best[part] = np.maximum(best[part], top)
                if beta is not None:
                    shifted = np.exp(beta * (sims - top[:, None]))
                    log_sum[part] = np.logaddexp(
                        log_sum[part],
                        beta * top + np.log(
                            shifted.sum(axis=1, dtype=np.float64)))
        if not seen_any:
            continue
        for slot, i in enumerate(members):
            nearest[i] = (self_similarity[slot] - best[slot]) / len(columns)
            if return_windows and best_block[slot] is not None:
                windows[i] = "".join(AA[a] for a in best_block[slot])
            if beta is not None:
                soft[i] = (self_similarity[slot] - log_sum[slot] / beta
                           ) / len(columns)
    result = {"nearest": nearest, "soft": soft}
    if return_windows:
        result["window"] = windows
    return result


def sample_windows(stream: np.ndarray, length: int, n: int,
                   rng: np.random.Generator) -> list[str]:
    """Uniformly random valid windows, by rejection, without indexing them all."""
    result: list[str] = []
    high = len(stream) - length
    offsets = np.arange(length)
    while len(result) < n:
        starts = rng.integers(0, high, size=max(2 * (n - len(result)), 64))
        block = stream[starts[:, None] + offsets]
        good = block[~(block == SEPARATOR).any(axis=1)]
        result.extend(
            "".join(AA[a] for a in row) for row in good[:n - len(result)])
    return result


def _kmer_codes(stream: np.ndarray, k: int) -> np.ndarray:
    """Base-20 codes of every canonical k-window of a joined stream."""
    n = len(stream) - k + 1
    if n <= 0:
        return np.empty(0, dtype=np.uint64)
    valid = np.ones(n, dtype=bool)
    codes = np.zeros(n, dtype=np.uint64)
    for offset in range(k):
        part = stream[offset:offset + n]
        valid &= part != SEPARATOR
        codes = codes * np.uint64(N_AA) + part.astype(np.uint64)
    return codes[valid]


def _face_kmer_codes(face: str, k: int) -> np.ndarray:
    return _kmer_codes(encode(face), k)


class KmerIndex:
    """Presence and frequency of every k-mer of a protein collection."""

    def __init__(self, proteome: ProteomeIndex, k_values=(3, 4, 5, 6)):
        self.k_values = tuple(k_values)
        self.unique: dict[int, np.ndarray] = {}
        self.counts: dict[int, np.ndarray] = {}
        self.total: dict[int, int] = {}
        for k in self.k_values:
            codes = _kmer_codes(proteome.stream, k)
            self.unique[k], self.counts[k] = np.unique(
                codes, return_counts=True)
            self.total[k] = len(codes)

    def _lookup(self, k: int, codes: np.ndarray) -> np.ndarray:
        position = np.searchsorted(self.unique[k], codes)
        position = np.minimum(position, len(self.unique[k]) - 1)
        found = self.unique[k][position] == codes
        return np.where(found, self.counts[k][position], 0)

    def absent_fraction(self, peptides: list[str], k: int) -> np.ndarray:
        """Fraction of TCR-face k-mers that never occur in the reference."""
        result = np.full(len(peptides), np.nan)
        for i, peptide in enumerate(peptides):
            codes = _face_kmer_codes(tcr_face(peptide), k)
            if len(codes):
                result[i] = float(np.mean(self._lookup(k, codes) == 0))
        return result

    def surprisal(self, peptides: list[str], k: int,
                  pseudocount: float = 0.5) -> np.ndarray:
        """Mean smoothed -log frequency of the TCR-face k-mers."""
        result = np.full(len(peptides), np.nan)
        denominator = self.total[k] + pseudocount * (N_AA ** k)
        for i, peptide in enumerate(peptides):
            codes = _face_kmer_codes(tcr_face(peptide), k)
            if len(codes):
                result[i] = float(np.mean(-np.log(
                    (self._lookup(k, codes) + pseudocount) / denominator)))
        return result


def allele_anchor_pairs(
        alleles: list[str], affinity_predictor, proteome: ProteomeIndex,
        top_pairs: int = 24, n_backgrounds: int = 16,
        seed: int = 0) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    """Estimate each allele's compatible (P2, P-omega) anchor pairs.

    For every pair of anchor residues, predict binding with MHCflurry across
    several random human backgrounds and keep the ``top_pairs`` pairs with the
    lowest median predicted affinity. This is a motif filter, not a binder
    call: it asks which self windows could be presented by the allele at all.
    Returns the boolean (400,) tables and a readable list of the kept pairs.
    """
    rng = np.random.default_rng(seed)
    starts, _ = proteome._windows(9)
    chosen = rng.choice(starts, size=n_backgrounds, replace=False)
    backgrounds = [
        "".join(AA[i] for i in proteome.stream[s:s + 9]) for s in chosen]
    probes, pair_index = [], []
    for background in backgrounds:
        for second in range(N_AA):
            for last in range(N_AA):
                probes.append(
                    background[0] + AA[second] + background[2:8] + AA[last])
                pair_index.append(second * N_AA + last)
    pair_index = np.asarray(pair_index)
    tables, readable = {}, {}
    for allele in alleles:
        affinity = np.asarray(
            affinity_predictor.predict(probes, allele=allele), dtype=float)
        median = np.empty(N_AA * N_AA)
        for code in range(N_AA * N_AA):
            median[code] = np.median(affinity[pair_index == code])
        keep = np.argsort(median)[:top_pairs]
        table = np.zeros(N_AA * N_AA, dtype=bool)
        table[keep] = True
        tables[allele] = table
        readable[allele] = [
            AA[c // N_AA] + "_" + AA[c % N_AA] for c in keep[:8]]
    return tables, readable
