"""Peptide features for the MLP scorers.

Each peptide is described by a vector of composition and physicochemical features:

- amino acid property statistics (12 properties x mean/std/min/max = 48)
- structural and charge features (27)
- amino acid composition (20)
- sequence-level statistics (12)
- reduced-alphabet compositions (80)
- dipeptide summary statistics and dipeptide composition (5 + 400, optional)
- positional one-hot encoding of the k-mer (``k * 21``, optional)

``extract_features`` defines the features for a single peptide.
``extract_features_batch`` computes the same values for many peptides at once and is
what training and prediction use; the test-suite checks that the two agree.
"""

import functools
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..reduced_alphabet import alphabets as REDUCED_ALPHABETS


# Amino acid to index mapping
AA_TO_IDX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,  # Unknown/padding
}
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
NUM_AMINO_ACIDS = 21

# Amino acid categories for derived features
POSITIVE_CHARGED = set('KRH')  # Basic residues (H partially charged at pH 7)
NEGATIVE_CHARGED = set('DE')   # Acidic residues
HYDROPHOBIC = set('AILMFVPWG')
AROMATIC = set('FWY')
ALIPHATIC = set('AVILM')
POLAR_UNCHARGED = set('STNQ')
TINY = set('AGS')
SMALL = set('AGSCTDNPV')
DISORDER_PROMOTING = set('AEGRQSKP')  # Disorder-promoting residues
ORDER_PROMOTING = set('WFYILMVC')     # Order-promoting residues

# Chou-Fasman secondary structure propensities
HELIX_PROPENSITY = {
    'A': 1.42, 'C': 0.70, 'D': 1.01, 'E': 1.51, 'F': 1.13,
    'G': 0.57, 'H': 1.00, 'I': 1.08, 'K': 1.16, 'L': 1.21,
    'M': 1.45, 'N': 0.67, 'P': 0.57, 'Q': 1.11, 'R': 0.98,
    'S': 0.77, 'T': 0.83, 'V': 1.06, 'W': 1.08, 'Y': 0.69,
}
SHEET_PROPENSITY = {
    'A': 0.83, 'C': 1.19, 'D': 0.54, 'E': 0.37, 'F': 1.38,
    'G': 0.75, 'H': 0.87, 'I': 1.60, 'K': 0.74, 'L': 1.30,
    'M': 1.05, 'N': 0.89, 'P': 0.55, 'Q': 1.10, 'R': 0.93,
    'S': 0.75, 'T': 1.19, 'V': 1.70, 'W': 1.37, 'Y': 1.47,
}
TURN_PROPENSITY = {
    'A': 0.66, 'C': 1.19, 'D': 1.46, 'E': 0.74, 'F': 0.60,
    'G': 1.56, 'H': 0.95, 'I': 0.47, 'K': 1.01, 'L': 0.59,
    'M': 0.60, 'N': 1.56, 'P': 1.52, 'Q': 0.98, 'R': 0.95,
    'S': 1.43, 'T': 0.96, 'V': 0.50, 'W': 0.96, 'Y': 1.14,
}


def _get_aa_properties() -> Dict[str, Dict[str, float]]:
    """Load all amino acid property dictionaries."""
    from ..amino_acid_properties import (
        accessible_surface_area,
        accessible_surface_area_folded,
        hydropathy,
        hydrophilicity,
        local_flexibility,
        mass,
        pK_side_chain,
        polarity,
        prct_exposed_residues,
        refractivity,
        solvent_exposed_area,
        volume,
    )
    return {
        'accessible_surface_area': accessible_surface_area,
        'accessible_surface_area_folded': accessible_surface_area_folded,
        'hydropathy': hydropathy,
        'hydrophilicity': hydrophilicity,
        'local_flexibility': local_flexibility,
        'mass': mass,
        'pK_side_chain': pK_side_chain,
        'polarity': polarity,
        'prct_exposed_residues': prct_exposed_residues,
        'refractivity': refractivity,
        'solvent_exposed_area': solvent_exposed_area,
        'volume': volume,
    }


def _compute_property_features(peptide: str, properties: Dict[str, Dict[str, float]]) -> np.ndarray:
    """Compute aggregate statistics from amino acid properties.

    For each property, compute: mean, std, min, max over the peptide.
    Returns array of shape (n_properties * 4,).
"""
    features = []

    for prop_name, prop_dict in properties.items():
        # Get property values for each residue
        values = [prop_dict[aa] for aa in peptide if aa in prop_dict]

        if values:
            features.extend([
                np.mean(values),
                np.std(values),
                np.min(values),
                np.max(values),
            ])
        else:
            # Unknown amino acids - use zeros
            features.extend([0.0, 0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)


def _compute_composition_features(peptide: str) -> np.ndarray:
    """Compute amino acid composition (frequencies).

    Returns array of shape (20,) with frequency of each amino acid.
    """
    counts = np.zeros(20, dtype=np.float32)
    for aa in peptide:
        idx = AMINO_ACIDS.find(aa)
        if idx >= 0:
            counts[idx] += 1

    # Normalize to frequencies
    if len(peptide) > 0:
        counts /= len(peptide)

    return counts


def _compute_dipeptide_features(peptide: str) -> np.ndarray:
    """Compute dipeptide composition (frequencies of AA pairs).

    Returns array of shape (400,) with frequency of each dipeptide.
    """
    counts = np.zeros(400, dtype=np.float32)  # 20 * 20

    for i in range(len(peptide) - 1):
        aa1_idx = AMINO_ACIDS.find(peptide[i])
        aa2_idx = AMINO_ACIDS.find(peptide[i + 1])
        if aa1_idx >= 0 and aa2_idx >= 0:
            counts[aa1_idx * 20 + aa2_idx] += 1

    # Normalize to frequencies
    n_dipeptides = max(1, len(peptide) - 1)
    counts /= n_dipeptides

    return counts


def _compute_structural_features(peptide: str) -> np.ndarray:
    """Compute structural and physicochemical category features.

    Returns array with:
    - Secondary structure propensities (helix, sheet, turn) - 12 features (3 props × 4 stats)
    - Category fractions (9 features: charged+/-, hydrophobic, aromatic, etc.)
    - Charge features (4 features: net charge, charge density, transitions, clusters)
    - Disorder features (2 features: disorder/order promoting ratios)

    Total: 27 features
    """
    n = len(peptide) if peptide else 1
    features = []

    # Secondary structure propensities - mean, std, min, max for each
    for prop_dict in [HELIX_PROPENSITY, SHEET_PROPENSITY, TURN_PROPENSITY]:
        values = [prop_dict.get(aa, 1.0) for aa in peptide if aa in prop_dict]
        if values:
            features.extend([np.mean(values), np.std(values), np.min(values), np.max(values)])
        else:
            features.extend([1.0, 0.0, 1.0, 1.0])

    # Category fractions (9 features)
    features.append(sum(1 for aa in peptide if aa in POSITIVE_CHARGED) / n)  # Positive charged
    features.append(sum(1 for aa in peptide if aa in NEGATIVE_CHARGED) / n)  # Negative charged
    features.append(sum(1 for aa in peptide if aa in HYDROPHOBIC) / n)       # Hydrophobic
    features.append(sum(1 for aa in peptide if aa in AROMATIC) / n)          # Aromatic
    features.append(sum(1 for aa in peptide if aa in ALIPHATIC) / n)         # Aliphatic
    features.append(sum(1 for aa in peptide if aa in POLAR_UNCHARGED) / n)   # Polar uncharged
    features.append(sum(1 for aa in peptide if aa in TINY) / n)              # Tiny
    features.append(sum(1 for aa in peptide if aa in SMALL) / n)             # Small
    features.append(sum(1 for aa in peptide if aa == 'C') / n)               # Cysteine (viral)

    # Charge features (4 features)
    pos_count = sum(1 for aa in peptide if aa in POSITIVE_CHARGED)
    neg_count = sum(1 for aa in peptide if aa in NEGATIVE_CHARGED)
    net_charge = pos_count - neg_count
    features.append(net_charge / n)  # Net charge per residue

    # Charge transitions (+ to - or - to +)
    transitions = 0
    for i in range(len(peptide) - 1):
        curr_pos = peptide[i] in POSITIVE_CHARGED
        curr_neg = peptide[i] in NEGATIVE_CHARGED
        next_pos = peptide[i+1] in POSITIVE_CHARGED
        next_neg = peptide[i+1] in NEGATIVE_CHARGED
        if (curr_pos and next_neg) or (curr_neg and next_pos):
            transitions += 1
    features.append(transitions / max(1, n - 1))  # Charge transitions

    # Charge clustering - max consecutive same-sign charges
    max_cluster = 0
    current_cluster = 0
    current_sign = None
    for aa in peptide:
        if aa in POSITIVE_CHARGED:
            sign = '+'
        elif aa in NEGATIVE_CHARGED:
            sign = '-'
        else:
            sign = None
        if sign and sign == current_sign:
            current_cluster += 1
        elif sign:
            current_cluster = 1
            current_sign = sign
        else:
            current_cluster = 0
            current_sign = None
        max_cluster = max(max_cluster, current_cluster)
    features.append(max_cluster / n)  # Max charge cluster size

    # Arginine depletion (viruses often have less R) - R/(R+K) ratio
    r_count = sum(1 for aa in peptide if aa == 'R')
    k_count = sum(1 for aa in peptide if aa == 'K')
    if r_count + k_count > 0:
        features.append(r_count / (r_count + k_count))
    else:
        features.append(0.5)  # Neutral when no R or K present

    # Disorder features (2 features)
    disorder_promoting = sum(1 for aa in peptide if aa in DISORDER_PROMOTING)
    order_promoting = sum(1 for aa in peptide if aa in ORDER_PROMOTING)
    features.append(disorder_promoting / n)  # Disorder-promoting fraction
    features.append(order_promoting / n)     # Order-promoting fraction

    return np.array(features, dtype=np.float32)


def _build_reduced_alphabet_index():
    """Build reduced alphabet group indices in a stable order."""
    alphabet_order = list(REDUCED_ALPHABETS.keys())
    alphabet_groups = {}
    for name in alphabet_order:
        mapping = REDUCED_ALPHABETS[name]
        groups: List[str] = []
        for aa in AMINO_ACIDS:
            rep = mapping.get(aa)
            if rep is None:
                continue
            if rep not in groups:
                groups.append(rep)
        alphabet_groups[name] = {
            'groups': groups,
            'rep_to_idx': {rep: idx for idx, rep in enumerate(groups)},
        }
    return alphabet_order, alphabet_groups


REDUCED_ALPHABET_ORDER, REDUCED_ALPHABET_GROUPS = _build_reduced_alphabet_index()


def _compute_sequence_stats(peptide: str) -> np.ndarray:
    """Compute sequence-level non-positional statistics."""
    if not peptide:
        return np.zeros(12, dtype=np.float32)

    n = len(peptide)
    log_len = np.log1p(n)
    sqrt_len = np.sqrt(n)

    counts = np.zeros(20, dtype=np.float32)
    unknown = 0
    for aa in peptide:
        idx = AMINO_ACIDS.find(aa)
        if idx >= 0:
            counts[idx] += 1
        else:
            unknown += 1

    total = counts.sum()
    if total > 0:
        freqs = counts / total
        nonzero = freqs[freqs > 0]
        entropy = -np.sum(nonzero * np.log(nonzero))
        entropy_norm = entropy / np.log(20)
        effective = np.exp(entropy) / 20.0
        gini = 1.0 - np.sum(freqs ** 2)
        max_freq = float(freqs.max())
        top2 = float(np.sort(freqs)[-2:].sum())
        unique_frac = float(np.count_nonzero(counts) / 20.0)
    else:
        entropy_norm = 0.0
        effective = 0.0
        gini = 0.0
        max_freq = 0.0
        top2 = 0.0
        unique_frac = 0.0

    # Run-length and repeat statistics
    max_run = 1
    repeats = 0
    current_run = 1
    for i in range(1, n):
        if peptide[i] == peptide[i - 1]:
            repeats += 1
            current_run += 1
        else:
            current_run = 1
        if current_run > max_run:
            max_run = current_run

    max_run_frac = max_run / n
    repeat_frac = repeats / max(1, n - 1)
    frac_unknown = unknown / n

    return np.array([
        n,
        log_len,
        sqrt_len,
        frac_unknown,
        unique_frac,
        max_run_frac,
        repeat_frac,
        entropy_norm,
        effective,
        max_freq,
        top2,
        gini,
    ], dtype=np.float32)


def _compute_reduced_alphabet_features(peptide: str) -> np.ndarray:
    """Compute reduced alphabet composition features."""
    if not peptide:
        total_features = sum(
            len(REDUCED_ALPHABET_GROUPS[name]['groups'])
            for name in REDUCED_ALPHABET_ORDER
        )
        return np.zeros(total_features, dtype=np.float32)

    features: List[np.ndarray] = []
    for name in REDUCED_ALPHABET_ORDER:
        mapping = REDUCED_ALPHABETS[name]
        groups = REDUCED_ALPHABET_GROUPS[name]['groups']
        rep_to_idx = REDUCED_ALPHABET_GROUPS[name]['rep_to_idx']
        counts = np.zeros(len(groups), dtype=np.float32)
        total = 0
        for aa in peptide:
            rep = mapping.get(aa)
            if rep is None:
                continue
            counts[rep_to_idx[rep]] += 1
            total += 1
        if total > 0:
            counts /= total
        features.append(counts)

    return np.concatenate(features) if features else np.array([], dtype=np.float32)


def _compute_dipeptide_summary(dipeptide_freqs: np.ndarray) -> np.ndarray:
    """Compute summary statistics from dipeptide frequencies."""
    if dipeptide_freqs.size == 0:
        return np.zeros(5, dtype=np.float32)

    total = dipeptide_freqs.sum()
    if total > 0:
        probs = dipeptide_freqs / total
        nonzero = probs[probs > 0]
        entropy = -np.sum(nonzero * np.log(nonzero))
        entropy_norm = entropy / np.log(probs.size)
        gini = 1.0 - np.sum(probs ** 2)
        max_freq = float(probs.max())
        top2 = float(np.sort(probs)[-2:].sum()) if probs.size >= 2 else max_freq
        homodipep = float(np.trace(probs.reshape(20, 20)))
    else:
        entropy_norm = 0.0
        gini = 0.0
        max_freq = 0.0
        top2 = 0.0
        homodipep = 0.0

    return np.array(
        [entropy_norm, gini, max_freq, top2, homodipep],
        dtype=np.float32,
    )


def extract_features(
    peptide: str,
    k: int = 8,
    use_dipeptides: bool = True,
    positional: bool = False,
) -> np.ndarray:
    """Extract all features from a peptide.

    Features include:
    - Amino acid property statistics (12 props × 4 stats = 48 features)
    - Structural/physicochemical features (27 features)
    - Amino acid composition (20 features)
    - Dipeptide composition (400 features, optional)
    - Dipeptide summary statistics (5 features, optional)
    - Sequence-level statistics (12 features)
    - Reduced alphabet compositions (80 features)
    - Positional one-hot encoding of the k-mer (k x 21 features, optional)

    Parameters
    ----------
    peptide : str
        Peptide sequence.
    k : int
        K-mer size of the positional block (only used when ``positional=True``).
    use_dipeptides : bool
        Include dipeptide composition features.
    positional : bool
        Append the positional one-hot block.

    Returns
    -------
    features : np.ndarray
        Feature vector.
    """
    properties = _get_aa_properties()

    feature_parts = [
        _compute_property_features(peptide, properties),  # 48 features (12 props × 4 stats)
        _compute_structural_features(peptide),             # 27 features
        _compute_composition_features(peptide),            # 20 features
        _compute_sequence_stats(peptide),                  # 12 features
        _compute_reduced_alphabet_features(peptide),       # 80 features
    ]

    if use_dipeptides:
        dipep_freqs = _compute_dipeptide_features(peptide)
        feature_parts.append(_compute_dipeptide_summary(dipep_freqs))  # 5 features
        feature_parts.append(dipep_freqs)  # 400 features

    if positional:
        feature_parts.append(_compute_kmer_onehot(peptide, k))  # k * 21 features

    return np.concatenate(feature_parts)


# ---------------------------------------------------------------------------
# Optional positional block
# ---------------------------------------------------------------------------

def _compute_kmer_onehot(peptide: str, k: int) -> np.ndarray:
    """Positional one-hot encoding averaged over the k-mers of ``peptide``.

    Returns an array of shape (k * 21,): for each of the ``k`` positions, the share
    of k-mer windows that carry each of the 20 amino acids (or ``X`` for any other
    character) there. Peptides shorter than ``k`` are padded with ``X``.
    """
    if len(peptide) < k:
        peptide = peptide + 'X' * (k - len(peptide))
    n_kmers = len(peptide) - k + 1
    onehot = np.zeros(k * NUM_AMINO_ACIDS, dtype=np.float32)
    for start in range(n_kmers):
        for position, aa in enumerate(peptide[start:start + k]):
            onehot[position * NUM_AMINO_ACIDS + AA_TO_IDX.get(aa, 20)] += 1
    onehot /= n_kmers
    return onehot


# ---------------------------------------------------------------------------
# Batch extraction
#
# The functions above define each feature for one peptide. The functions below
# compute the same values for many peptides at once with array operations, which
# is what training and prediction use: it is much faster than looping over
# peptides in Python and needs no worker processes. The test-suite checks that
# both paths agree.
# ---------------------------------------------------------------------------

N_BASE_FEATURES_NO_DIPEPTIDES = 48 + 27 + 20 + 12 + 80
N_DIPEPTIDE_FEATURES = 5 + 400
DEFAULT_CHUNK_SIZE = 16384


def feature_count(use_dipeptides: bool = True, positional: bool = False, k: int = 8) -> int:
    """Number of features produced for the given options."""
    count = N_BASE_FEATURES_NO_DIPEPTIDES
    if use_dipeptides:
        count += N_DIPEPTIDE_FEATURES
    if positional:
        count += k * NUM_AMINO_ACIDS
    return count


def _byte_table(mapping: Dict[str, float]) -> np.ndarray:
    """Lookup table over the 256 byte values, NaN where ``mapping`` has no entry."""
    table = np.full(256, np.nan, dtype=np.float64)
    for key, value in mapping.items():
        if len(key) == 1 and ord(key) < 256:
            table[ord(key)] = value
    return table


def _membership_table(residues) -> np.ndarray:
    """Boolean lookup table over the 256 byte values."""
    table = np.zeros(256, dtype=bool)
    for residue in residues:
        table[ord(residue)] = True
    return table


@functools.lru_cache(maxsize=1)
def _batch_tables() -> SimpleNamespace:
    """Lookup tables shared by all batch extractions (built once)."""
    residue_class = np.full(256, 20, dtype=np.uint8)  # 0-19 amino acids, 20 anything else
    for index, aa in enumerate(AMINO_ACIDS):
        residue_class[ord(aa)] = index
    reduced = []
    for name in REDUCED_ALPHABET_ORDER:
        mapping = REDUCED_ALPHABETS[name]
        rep_to_idx = REDUCED_ALPHABET_GROUPS[name]['rep_to_idx']
        table = np.full(256, -1, dtype=np.int16)
        for aa, rep in mapping.items():
            if len(aa) == 1 and ord(aa) < 256:
                table[ord(aa)] = rep_to_idx[rep]
        reduced.append((table, len(REDUCED_ALPHABET_GROUPS[name]['groups'])))
    return SimpleNamespace(
        properties=np.stack([_byte_table(d) for d in _get_aa_properties().values()]),
        propensities=np.stack([
            _byte_table(d) for d in (HELIX_PROPENSITY, SHEET_PROPENSITY, TURN_PROPENSITY)
        ]),
        fraction_sets=np.stack([
            _membership_table(residues) for residues in (
                POSITIVE_CHARGED, NEGATIVE_CHARGED, HYDROPHOBIC, AROMATIC, ALIPHATIC,
                POLAR_UNCHARGED, TINY, SMALL, 'C',
            )
        ]),
        positive=_membership_table(POSITIVE_CHARGED),
        negative=_membership_table(NEGATIVE_CHARGED),
        disorder=_membership_table(DISORDER_PROMOTING),
        order=_membership_table(ORDER_PROMOTING),
        residue_class=residue_class,
        reduced=reduced,
    )


def _encode_equal_length(peptides: Sequence[str], length: int) -> np.ndarray:
    """(n_peptides, length) array of byte codes; characters beyond Latin-1 become '?'."""
    joined = ''.join(peptides).encode('latin-1', errors='replace')
    return np.frombuffer(joined, dtype=np.uint8).reshape(len(peptides), length)


def _row_stats(values: np.ndarray, default: Tuple[float, float, float, float]) -> np.ndarray:
    """mean, std, min and max of the non-NaN entries of each row.

    Rows without any value get ``default``.
    """
    valid = ~np.isnan(values)
    count = valid.sum(axis=1)
    safe_count = np.where(count > 0, count, 1)
    mean = np.where(valid, values, 0.0).sum(axis=1) / safe_count
    deviation = np.where(valid, values - mean[:, None], 0.0)
    std = np.sqrt((deviation * deviation).sum(axis=1) / safe_count)
    low = np.where(valid, values, np.inf).min(axis=1)
    high = np.where(valid, values, -np.inf).max(axis=1)
    stats = np.stack([mean, std, low, high], axis=1)
    stats[count == 0] = default
    return stats


def _count_matrix(classes: np.ndarray, n_classes: int, valid: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-row counts of each class in an (n_rows, length) integer array."""
    n_rows = classes.shape[0]
    flat = np.arange(n_rows, dtype=np.int64)[:, None] * n_classes + classes
    if valid is not None:
        flat = flat[valid]
    return np.bincount(flat.ravel(), minlength=n_rows * n_classes).reshape(n_rows, n_classes)


def _property_block(codes: np.ndarray, tables: SimpleNamespace) -> np.ndarray:
    return np.concatenate(
        [_row_stats(table[codes], (0.0, 0.0, 0.0, 0.0)) for table in tables.properties], axis=1
    )


def _structural_block(codes: np.ndarray, tables: SimpleNamespace) -> np.ndarray:
    n_rows, length = codes.shape
    parts = [
        _row_stats(table[codes], (1.0, 0.0, 1.0, 1.0)) for table in tables.propensities
    ]
    fractions = np.stack(
        [table[codes].sum(axis=1) / length for table in tables.fraction_sets], axis=1
    )
    positive = tables.positive[codes]
    negative = tables.negative[codes]
    net_charge = (positive.sum(axis=1) - negative.sum(axis=1)) / length
    transitions = (
        (positive[:, :-1] & negative[:, 1:]) | (negative[:, :-1] & positive[:, 1:])
    ).sum(axis=1) / max(1, length - 1)

    sign = positive.astype(np.int8) - negative.astype(np.int8)
    run = np.abs(sign[:, 0]).astype(np.int32)
    longest = run.copy()
    for position in range(1, length):
        same_sign = (sign[:, position] != 0) & (sign[:, position] == sign[:, position - 1])
        run = np.where(same_sign, run + 1, np.abs(sign[:, position]))
        longest = np.maximum(longest, run)

    arginine = (codes == ord('R')).sum(axis=1)
    lysine = (codes == ord('K')).sum(axis=1)
    arginine_ratio = np.where(
        arginine + lysine > 0, arginine / np.maximum(arginine + lysine, 1), 0.5
    )
    parts.append(fractions)
    parts.append(np.stack([
        net_charge,
        transitions,
        longest / length,
        arginine_ratio,
        tables.disorder[codes].sum(axis=1) / length,
        tables.order[codes].sum(axis=1) / length,
    ], axis=1))
    return np.concatenate(parts, axis=1)


def _sequence_stats_block(codes: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The 12 sequence-level statistics; ``counts`` has 21 columns (20 amino acids, other)."""
    n_rows, length = codes.shape
    known = counts[:, :20].astype(np.float64)
    unknown = counts[:, 20]
    total = known.sum(axis=1)
    present = total > 0
    freqs = known / np.where(present, total, 1.0)[:, None]
    log_freqs = np.where(freqs > 0, np.log(np.where(freqs > 0, freqs, 1.0)), 0.0)
    entropy = -(freqs * log_freqs).sum(axis=1)

    same = codes[:, 1:] == codes[:, :-1]
    run = np.ones(n_rows, dtype=np.int32)
    longest = run.copy()
    for position in range(1, length):
        run = np.where(same[:, position - 1], run + 1, 1)
        longest = np.maximum(longest, run)
    repeats = same.sum(axis=1)

    return np.stack([
        np.full(n_rows, float(length)),
        np.full(n_rows, np.log1p(length)),
        np.full(n_rows, np.sqrt(length)),
        unknown / length,
        np.where(present, (known > 0).sum(axis=1) / 20.0, 0.0),
        longest / length,
        repeats / max(1, length - 1),
        np.where(present, entropy / np.log(20), 0.0),
        np.where(present, np.exp(entropy) / 20.0, 0.0),
        np.where(present, freqs.max(axis=1), 0.0),
        np.where(present, np.partition(freqs, 18, axis=1)[:, 18:].sum(axis=1), 0.0),
        np.where(present, 1.0 - (freqs * freqs).sum(axis=1), 0.0),
    ], axis=1)


def _reduced_alphabet_block(codes: np.ndarray, tables: SimpleNamespace) -> np.ndarray:
    parts = []
    for table, n_groups in tables.reduced:
        groups = table[codes]
        valid = groups >= 0
        counts = _count_matrix(groups.astype(np.int64), n_groups, valid)
        parts.append(counts / np.maximum(valid.sum(axis=1), 1)[:, None])
    return np.concatenate(parts, axis=1)


def _dipeptide_blocks(classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Dipeptide summary statistics (5 columns) and dipeptide frequencies (400 columns)."""
    n_rows, length = classes.shape
    first = classes[:, :-1].astype(np.int64)
    second = classes[:, 1:].astype(np.int64)
    valid = (first < 20) & (second < 20)
    counts = _count_matrix(first * 20 + second, 400, valid)
    freqs = counts.astype(np.float32) / np.float32(max(1, length - 1))

    total = freqs.sum(axis=1)
    present = total > 0
    probs = freqs / np.where(present, total, 1.0).astype(np.float32)[:, None]
    log_probs = np.where(probs > 0, np.log(np.where(probs > 0, probs, 1.0)), 0.0)
    entropy = -(probs * log_probs).sum(axis=1)
    summary = np.stack([
        np.where(present, entropy / np.log(400), 0.0),
        np.where(present, 1.0 - (probs * probs).sum(axis=1), 0.0),
        np.where(present, probs.max(axis=1), 0.0),
        np.where(present, np.partition(probs, 398, axis=1)[:, 398:].sum(axis=1), 0.0),
        np.where(present, probs[:, ::21].sum(axis=1), 0.0),  # trace of the 20 x 20 matrix
    ], axis=1)
    return summary, freqs


def _positional_block(classes: np.ndarray, k: int) -> np.ndarray:
    """Positional one-hot block for equal-length peptides (see ``_compute_kmer_onehot``)."""
    n_rows, length = classes.shape
    if length < k:
        classes = np.concatenate(
            [classes, np.full((n_rows, k - length), 20, dtype=classes.dtype)], axis=1
        )
        length = k
    n_kmers = length - k + 1
    block = np.empty((n_rows, k, NUM_AMINO_ACIDS), dtype=np.float64)
    for position in range(k):
        window = classes[:, position:position + n_kmers].astype(np.int64)
        block[:, position, :] = _count_matrix(window, NUM_AMINO_ACIDS)
    return (block / n_kmers).reshape(n_rows, k * NUM_AMINO_ACIDS)


def _features_equal_length(
    codes: np.ndarray,
    use_dipeptides: bool,
    positional_k: Optional[int],
) -> np.ndarray:
    """Feature matrix for an (n_rows, length) byte-code array, ``length >= 1``."""
    tables = _batch_tables()
    n_rows, length = codes.shape
    classes = tables.residue_class[codes]
    counts = _count_matrix(classes.astype(np.int64), 21)
    blocks = [
        _property_block(codes, tables),
        _structural_block(codes, tables),
        counts[:, :20] / length,
        _sequence_stats_block(codes, counts),
        _reduced_alphabet_block(codes, tables),
    ]
    if use_dipeptides:
        summary, freqs = _dipeptide_blocks(classes)
        blocks.extend([summary, freqs])
    if positional_k is not None:
        blocks.append(_positional_block(classes, positional_k))
    return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


def extract_features_batch(
    peptides: Sequence[str],
    k: int = 8,
    use_dipeptides: bool = True,
    positional: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    """Feature matrix for many peptides, identical to stacking :func:`extract_features`.

    Peptides of equal length are processed together with array operations, so the
    cost per peptide is small; peptides may differ in length. Memory use is bounded
    by ``chunk_size`` rows at a time plus the returned matrix.

    Parameters
    ----------
    peptides : sequence of str
        Peptide sequences.
    k : int
        K-mer size of the positional block (only used when ``positional=True``).
    use_dipeptides : bool
        Include the dipeptide summary and composition features.
    positional : bool
        Append the positional one-hot block (``k * 21`` features).
    chunk_size : int
        Rows processed per step.

    Returns
    -------
    features : np.ndarray
        Array of shape (n_peptides, n_features) and dtype float32.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    peptides = peptides if isinstance(peptides, list) else list(peptides)
    out = np.empty((len(peptides), feature_count(use_dipeptides, positional, k)), dtype=np.float32)
    if not peptides:
        return out
    lengths = np.fromiter((len(p) for p in peptides), dtype=np.int64, count=len(peptides))
    for length in np.unique(lengths):
        rows = np.flatnonzero(lengths == length)
        if length == 0:
            for row in rows:
                out[row] = extract_features(peptides[row], k, use_dipeptides, positional)
            continue
        for start in range(0, len(rows), chunk_size):
            block_rows = rows[start:start + chunk_size]
            codes = _encode_equal_length([peptides[row] for row in block_rows], int(length))
            out[block_rows] = _features_equal_length(
                codes, use_dipeptides, k if positional else None
            )
    return out


def get_feature_names(use_dipeptides: bool = True, positional: bool = False, k: int = 8) -> List[str]:
    """Names of the features, in column order."""
    properties = list(_get_aa_properties().keys())
    names: List[str] = []

    # Property statistics (12 props x 4 stats = 48 features)
    for prop in properties:
        for stat in ['mean', 'std', 'min', 'max']:
            names.append(f'{prop}_{stat}')

    # Structural features (27 features)
    for struct in ['helix', 'sheet', 'turn']:
        for stat in ['mean', 'std', 'min', 'max']:
            names.append(f'{struct}_propensity_{stat}')
    names.extend([
        'frac_positive_charged', 'frac_negative_charged', 'frac_hydrophobic',
        'frac_aromatic', 'frac_aliphatic', 'frac_polar_uncharged',
        'frac_tiny', 'frac_small', 'frac_cysteine',
        'net_charge_per_residue', 'charge_transitions', 'max_charge_cluster',
        'arginine_ratio',  # R/(R+K) - lower in viruses
        'frac_disorder_promoting', 'frac_order_promoting',
    ])

    # Amino acid composition (20 features)
    for aa in AMINO_ACIDS:
        names.append(f'aa_freq_{aa}')

    # Sequence statistics (12 features)
    names.extend([
        'seq_length',
        'seq_log_length',
        'seq_sqrt_length',
        'frac_unknown',
        'unique_frac',
        'max_run_frac',
        'repeat_frac',
        'entropy_aa',
        'effective_aa',
        'max_aa_freq',
        'top2_aa_freq',
        'gini_aa',
    ])

    # Reduced alphabet compositions (80 features)
    for name in REDUCED_ALPHABET_ORDER:
        groups = REDUCED_ALPHABET_GROUPS[name]['groups']
        for rep in groups:
            names.append(f'{name}_freq_{rep}')

    if use_dipeptides:
        # Dipeptide summary (5 features)
        names.extend([
            'dipep_entropy',
            'dipep_gini',
            'dipep_max_freq',
            'dipep_top2_freq',
            'dipep_homodimer_frac',
        ])
        # Dipeptide composition (400 features)
        for aa1 in AMINO_ACIDS:
            for aa2 in AMINO_ACIDS:
                names.append(f'dipep_{aa1}{aa2}')

    if positional:
        # Positional one-hot (k * 21 features)
        for position in range(k):
            for aa in AMINO_ACIDS + 'X':
                names.append(f'kmer_pos{position + 1}_{aa}')

    return names
