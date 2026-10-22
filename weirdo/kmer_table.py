"""K-mer tables of SwissProt: which taxonomic divisions contain each k-mer.

A k-mer is *present* in a division when at least one SwissProt protein of that division
contains it. :func:`build_kmer_table` builds the reference table that
:class:`weirdo.scorers.SwissProtReference` reads, from the protein sequences that
:mod:`weirdo.uniprot` downloads, with bounded memory.

Residues are packed five bits each into a ``uint64`` (k up to 12), so a k-mer is one integer
and sorting integers sorts the k-mers alphabetically.
"""

import os
import time
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from .uniprot import SWISSPROT_DIVISIONS

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
BITS_PER_RESIDUE = 5
MAX_K = 12  # 12 residues x 5 bits = 60 bits fit in a uint64
_NOT_A_RESIDUE = 31

_RESIDUE_CODE = np.full(256, _NOT_A_RESIDUE, dtype=np.uint8)
for _index, _aa in enumerate(AMINO_ACIDS):
    _RESIDUE_CODE[ord(_aa)] = _index


def encode_residues(sequences: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Concatenate sequences (NUL separated) into residue codes.

    Returns the code array (0-19 for the 20 canonical residues, 31 for anything else
    including the separators) and the start offset of each sequence.
    """
    lengths = np.fromiter((len(s) for s in sequences), dtype=np.int64, count=len(sequences))
    starts = np.concatenate([[0], np.cumsum(lengths + 1)[:-1]]).astype(np.int64)
    joined = "\x00".join(sequences).encode("ascii", errors="replace")
    codes = _RESIDUE_CODE[np.frombuffer(joined, dtype=np.uint8)]
    return codes, starts


def window_codes(codes: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Packed k-mer integers for every window and a mask of fully canonical windows."""
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be in 1..{MAX_K}, got {k}")
    n_windows = len(codes) - k + 1
    if n_windows <= 0:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=bool)
    packed = np.zeros(n_windows, dtype=np.uint64)
    shift = np.uint64(BITS_PER_RESIDUE)
    for offset in range(k):
        packed = (packed << shift) | codes[offset:offset + n_windows].astype(np.uint64)
    bad = np.concatenate([[0], np.cumsum(codes >= 20, dtype=np.int64)])
    valid = (bad[k:k + n_windows] - bad[:n_windows]) == 0
    return packed, valid


def unique_with_masks(
    kmers: np.ndarray, bits: np.ndarray, groups: Optional[np.ndarray] = None
):
    """Unique k-mers and the bitwise OR of their category bits.

    Returns ``(kmers, masks)``; when per-window ``groups`` (e.g. family buckets) are
    given, returns ``(kmers, masks, groups)`` where a k-mer found in several groups keeps
    the smallest group id.
    """
    if len(kmers) == 0:
        empty = (kmers.copy(), bits.copy())
        return empty if groups is None else (*empty, groups.copy())
    order = np.argsort(kmers, kind="stable")
    sorted_kmers = kmers[order]
    sorted_bits = bits[order]
    first = np.flatnonzero(np.concatenate([[True], sorted_kmers[1:] != sorted_kmers[:-1]]))
    unique, masks = sorted_kmers[first], np.bitwise_or.reduceat(sorted_bits, first)
    if groups is None:
        return unique, masks
    return unique, masks, np.minimum.reduceat(groups[order], first)


def decode_kmers(packed: np.ndarray, k: int) -> List[str]:
    """Inverse of :func:`window_codes` for canonical k-mers."""
    return np.ascontiguousarray(_decode_letters(packed, k)).view("S%d" % k).ravel().astype(str).tolist()


def _decode_letters(packed: np.ndarray, k: int) -> np.ndarray:
    """(n, k) array of ASCII letters for canonical packed k-mers."""
    shifts = (BITS_PER_RESIDUE * (k - 1 - np.arange(k))).astype(np.uint64)
    index = ((packed[:, None] >> shifts[None, :]) & np.uint64(31)).astype(np.uint8)
    return np.frombuffer(AMINO_ACIDS.encode("ascii"), dtype=np.uint8)[index]


def masks_to_matrix(masks: np.ndarray, n_categories: int = len(SWISSPROT_DIVISIONS)) -> np.ndarray:
    """Multi-hot float32 matrix (n, n_categories) from category bitmasks."""
    shifts = np.arange(n_categories, dtype=np.uint16)
    return ((masks[:, None] >> shifts[None, :]) & 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Building the table
# ---------------------------------------------------------------------------

def _row_tails(categories: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Text after the k-mer for every category bitmask: ``,False,True,...,<count>\\n``."""
    n = len(categories)
    texts = []
    for mask in range(1 << n):
        flags = [b"True" if (mask >> i) & 1 else b"False" for i in range(n)]
        texts.append(b"," + b",".join(flags) + b"," + str(bin(mask).count("1")).encode() + b"\n")
    width = max(len(text) for text in texts)
    table = np.zeros((len(texts), width), dtype=np.uint8)
    lengths = np.zeros(len(texts), dtype=np.int64)
    for mask, text in enumerate(texts):
        table[mask, :len(text)] = np.frombuffer(text, dtype=np.uint8)
        lengths[mask] = len(text)
    return table, lengths


def _csv_rows(
    packed: np.ndarray, masks: np.ndarray, k: int, tails: np.ndarray, tail_lengths: np.ndarray
) -> bytes:
    """CSV text of table rows (``kmer,flags...,count``) for a block of k-mers."""
    block = np.concatenate([_decode_letters(packed, k), tails[masks]], axis=1)
    keep = np.arange(block.shape[1])[None, :] < (k + tail_lengths[masks])[:, None]
    return block[keep].tobytes()


def _iter_chunks(lengths: np.ndarray, chunk_residues: int) -> Iterator[Tuple[int, int]]:
    """Ranges of consecutive proteins holding about ``chunk_residues`` residues each."""
    start, total = 0, 0
    for index, length in enumerate(lengths):
        total += int(length) + 1
        if total >= chunk_residues:
            yield start, index + 1
            start, total = index + 1, 0
    if start < len(lengths):
        yield start, len(lengths)


def iter_kmer_masks(
    sequences: Sequence[str],
    bits: np.ndarray,
    k: int = 8,
    memory_passes: int = 4,
    chunk_residues: int = 16_000_000,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Unique canonical k-mers with category bitmasks, in sorted order, a block at a time.

    Parameters
    ----------
    sequences : sequence of str
        Protein sequences.
    bits : np.ndarray
        One category bit (``1 << category_index``) per sequence.
    k : int
        K-mer size (1 to 12).
    memory_passes : int
        The k-mers are split by their first residue and processed in this many passes over
        the sequences; more passes use less memory (about ``1 / memory_passes`` of the
        windows at a time) but take longer.
    chunk_residues : int
        Residues encoded at a time.

    Yields
    ------
    kmers, masks : np.ndarray
        ``uint64`` packed k-mers (ascending across the whole iteration) and ``uint16``
        bitmasks of the categories that contain them. A k-mer appears once, with the
        OR of the bits of every sequence that contains it.
    """
    if not 1 <= k <= MAX_K:
        raise ValueError(f"k must be in 1..{MAX_K}, got {k}")
    if memory_passes < 1:
        raise ValueError("memory_passes must be at least 1")
    bits = np.asarray(bits, dtype=np.uint16)
    if len(bits) != len(sequences):
        raise ValueError("bits needs one entry per sequence")
    lengths = np.fromiter((len(s) for s in sequences), dtype=np.int64, count=len(sequences))
    chunks = list(_iter_chunks(lengths, chunk_residues))
    top_shift = np.uint64(BITS_PER_RESIDUE * (k - 1))
    residue_groups = np.array_split(np.arange(len(AMINO_ACIDS)), min(memory_passes, len(AMINO_ACIDS)))

    for group in residue_groups:
        wanted = np.zeros(32, dtype=bool)
        wanted[group] = True
        found_kmers: List[np.ndarray] = []
        found_bits: List[np.ndarray] = []
        for first, last in chunks:
            codes, _ = encode_residues(sequences[first:last])
            packed, valid = window_codes(codes, k)
            if len(packed) == 0:
                continue
            owner = np.repeat(np.arange(first, last, dtype=np.int64), lengths[first:last] + 1)[:len(packed)]
            select = valid & wanted[(packed >> top_shift).astype(np.uint8) & 31]
            found_kmers.append(packed[select])
            found_bits.append(bits[owner[select]])
        if not found_kmers:
            continue
        kmers, masks = unique_with_masks(np.concatenate(found_kmers), np.concatenate(found_bits))
        del found_kmers, found_bits
        # Whole first-residue bins are contiguous in sorted order, so blocks stay ascending.
        for residue in group:
            low = np.uint64(int(residue) << (BITS_PER_RESIDUE * (k - 1)))
            high = np.uint64((int(residue) + 1) << (BITS_PER_RESIDUE * (k - 1)))
            start, stop = np.searchsorted(kmers, [low, high])
            if stop > start:
                yield kmers[start:stop], masks[start:stop]


def build_kmer_table(
    sequences: Union[str, os.PathLike, "object"],
    output_path: Union[str, os.PathLike],
    k: int = 8,
    categories: Optional[Sequence[str]] = None,
    memory_passes: int = 4,
    rows_per_block: int = 1_000_000,
    progress: Optional[Callable[[str], None]] = None,
) -> int:
    """Write the SwissProt k-mer reference table.

    The table has one row per unique k-mer made only of the 20 canonical residues:
    ``seq``, one ``True``/``False`` column per category and ``label_count``, sorted by
    k-mer. This is the layout :class:`weirdo.scorers.SwissProtReference` reads.

    Parameters
    ----------
    sequences : path or DataFrame
        A table with ``sequence`` and ``category`` columns, e.g. the file written by
        :func:`weirdo.uniprot.build_sequence_table`.
    output_path : path
        Where to write the CSV (written to a temporary file and moved into place).
    k : int, default=8
        K-mer size.
    categories : sequence of str, optional
        Category names, in column order (default: the ten SwissProt divisions).
    memory_passes : int, default=4
        See :func:`iter_kmer_masks`.
    rows_per_block : int
        Rows formatted at a time when writing.
    progress : callable, optional
        Called with status messages.

    Returns
    -------
    n_rows : int
        Number of k-mers written.
    """
    from .uniprot import read_sequence_table

    categories = list(categories or SWISSPROT_DIVISIONS)
    if len(categories) > 16:
        raise ValueError("at most 16 categories are supported")
    log = progress or (lambda message: None)
    frame = read_sequence_table(sequences) if not hasattr(sequences, "columns") else sequences
    unknown = sorted(set(frame["category"]) - set(categories))
    if unknown:
        raise ValueError(f"Sequences have categories that are not in `categories`: {unknown}")
    index = {name: i for i, name in enumerate(categories)}
    bits = np.array([1 << index[c] for c in frame["category"]], dtype=np.uint16)
    sequence_list = frame["sequence"].tolist()
    log(f"{len(sequence_list):,} proteins")

    tails, tail_lengths = _row_tails(categories)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".partial")
    n_rows = 0
    started = time.time()
    try:
        with open(temporary, "wb") as handle:
            handle.write(("seq," + ",".join(categories) + ",label_count\n").encode("ascii"))
            for kmers, masks in iter_kmer_masks(sequence_list, bits, k, memory_passes):
                for start in range(0, len(kmers), rows_per_block):
                    handle.write(_csv_rows(
                        kmers[start:start + rows_per_block], masks[start:start + rows_per_block],
                        k, tails, tail_lengths,
                    ))
                n_rows += len(kmers)
                log(f"{n_rows:,} {k}-mers written ({time.time() - started:.0f}s)")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return n_rows
