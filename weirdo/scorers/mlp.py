"""MLP-based origin scorer.

Neural network model for learning category probabilities from labeled data.
Uses rich peptide features including amino acid properties and composition
statistics.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .trainable import TrainableScorer
from .registry import register_scorer
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


def extract_features(peptide: str, k: int = 8, use_dipeptides: bool = True) -> np.ndarray:
    """Extract all features from a peptide.

    Features include:
    - Amino acid property statistics (12 props × 4 stats = 48 features)
    - Structural/physicochemical features (27 features)
    - Amino acid composition (20 features)
    - Dipeptide composition (400 features, optional)
    - Dipeptide summary statistics (5 features, optional)
    - Sequence-level statistics (12 features)
    - Reduced alphabet compositions (80 features)

    Parameters
    ----------
    peptide : str
        Peptide sequence.
    k : int
        Unused; retained for backward compatibility.
    use_dipeptides : bool
        Include dipeptide composition features.

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

    return np.concatenate(feature_parts)


@register_scorer('mlp', description='MLP foreignness scorer with rich peptide features')
class MLPScorer(TrainableScorer):
    """MLP-based origin scorer using rich peptide features.

    Combines multiple feature types:
    - Amino acid properties (hydropathy, mass, polarity, etc.)
    - Amino acid composition (single AA frequencies)
    - Dipeptide composition (AA pair frequencies)
    - Sequence-level statistics (entropy, repeats, complexity)
    - Reduced alphabet compositions (Murphy/GBMR/SDM, etc.)

    All features are normalized using StandardScaler before training.

    Parameters
    ----------
    k : int, default=8
        K-mer size used to window long peptides for aggregation.
    hidden_layer_sizes : tuple of int, default=(256, 128, 64)
        Sizes of hidden layers.
    activation : str, default='relu'
        Activation function: 'relu', 'tanh', 'logistic'.
    alpha : float, default=0.0001
        L2 regularization strength.
    max_iter : int, default=200
        Maximum training iterations.
    early_stopping : bool, default=True
        Use early stopping with validation split.
    use_dipeptides : bool, default=True
        Include dipeptide composition features.
    batch_size : int, default=256
        Batch size for training.

    Example
    -------
    >>> from weirdo.scorers import MLPScorer
    >>>
    >>> scorer = MLPScorer(hidden_layer_sizes=(256, 128))
    >>> scorer.train(peptides, labels, target_categories=['human', 'viruses'])
    >>> scores = scorer.score(['MTMDKSEL', 'XXXXXXXX'])
    """

    def __init__(
        self,
        k: int = 8,
        hidden_layer_sizes: Tuple[int, ...] = (256, 128, 64),
        activation: str = 'relu',
        alpha: float = 0.0001,
        learning_rate_init: float = 0.001,
        max_iter: int = 200,
        early_stopping: bool = True,
        use_dipeptides: bool = True,
        batch_size: int = 256,
        random_state: Optional[int] = None,
        **kwargs
    ):
        super().__init__(k=k, batch_size=batch_size, **kwargs)
        self._params.update({
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'alpha': alpha,
            'learning_rate_init': learning_rate_init,
            'max_iter': max_iter,
            'early_stopping': early_stopping,
            'use_dipeptides': use_dipeptides,
            'random_state': random_state,
        })
        self._model: Optional[MLPRegressor] = None
        self._scaler: Optional[StandardScaler] = None
        self._target_categories: Optional[List[str]] = None

    def _extract_features(self, peptides: Sequence[str]) -> np.ndarray:
        """Extract features from a list of peptides."""
        return np.array([
            extract_features(p, self.k, self._params['use_dipeptides'])
            for p in peptides
        ])

    def _predict_raw_kmers(self, kmers: Sequence[str]) -> np.ndarray:
        """Predict raw model outputs for k-mers (no aggregation)."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before scoring.")
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model is not initialized. Train or load a model first.")

        X = self._extract_features(kmers)
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def train(
        self,
        peptides: Sequence[str],
        labels: Any,
        val_peptides: Optional[Sequence[str]] = None,
        val_labels: Optional[Any] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        verbose: bool = True,
        target_categories: Optional[List[str]] = None,
        plot_loss: Union[bool, str, Path, None] = None,
        **kwargs
    ) -> 'MLPScorer':
        """Train the MLP on labeled peptide data.

        Parameters
        ----------
        peptides : sequence of str
            Training peptide sequences.
        labels : sequence of float or 2D array
            Target labels in [0, 1]. Can be 1D (single foreignness score)
            or 2D (multi-label with one column per category).
        val_peptides : sequence of str, optional
            Validation peptides for external validation-loss reporting.
            If provided, val_labels must also be provided.
        val_labels : sequence of float or 2D array, optional
            Validation labels.
        epochs : int, optional
            Maximum training iterations (maps to max_iter). Defaults to max_iter.
        learning_rate : float, optional
            Initial learning rate. Defaults to learning_rate_init if not provided.
        verbose : bool, default=True
            Print training progress.
        target_categories : list of str, optional
            Names of target categories (for multi-label training).
            E.g., ['human', 'viruses', 'bacteria', 'mammals'].
        plot_loss : bool, str, or Path, optional
            Save loss curve plot. If True, saves to 'loss_curve.png' in
            current directory. If a path, saves to that location.

        Returns
        -------
        self : MLPScorer
        """
        peptides = list(peptides)
        if not peptides:
            raise ValueError("Training data cannot be empty.")

        y = np.array(labels)
        if len(peptides) != len(y):
            raise ValueError(
                f"Length mismatch: {len(peptides)} peptides but {len(y)} labels."
            )

        if (val_peptides is None) != (val_labels is None):
            raise ValueError("Provide both val_peptides and val_labels together.")

        if val_peptides is not None and val_labels is not None:
            val_peptides = list(val_peptides)
            y_val = np.array(val_labels)
            if len(val_peptides) != len(y_val):
                raise ValueError(
                    f"Length mismatch: {len(val_peptides)} val peptides but {len(y_val)} val labels."
                )
        else:
            y_val = None

        self._target_categories = target_categories
        if epochs is None:
            epochs = self._params['max_iter']
        if learning_rate is None:
            learning_rate = self._params['learning_rate_init']
        # Extract features
        X = self._extract_features(peptides)
        if target_categories is not None:
            if y.ndim == 1 and len(target_categories) != 1:
                raise ValueError("target_categories length must match label dimensions.")
            if y.ndim == 2 and y.shape[1] != len(target_categories):
                raise ValueError("target_categories length must match label dimensions.")

        # Scale features to zero mean, unit variance
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Disable early stopping if dataset is too small
        use_early_stopping = self._params['early_stopping']
        if use_early_stopping and len(peptides) < 20:
            use_early_stopping = False
            if verbose:
                print("Note: Early stopping disabled (dataset too small)")

        # Create and train model
        self._model = MLPRegressor(
            hidden_layer_sizes=self._params['hidden_layer_sizes'],
            activation=self._params['activation'],
            alpha=self._params['alpha'],
            learning_rate_init=learning_rate,
            max_iter=epochs,
            early_stopping=use_early_stopping,
            validation_fraction=0.1 if use_early_stopping else 0.0,
            n_iter_no_change=10,
            random_state=self._params['random_state'],
            verbose=verbose,
        )

        self._model.fit(X_scaled, y)

        self._is_trained = True
        self._is_fitted = True

        # Save training metadata
        self._metadata['n_train'] = len(peptides)
        self._metadata['n_features'] = X.shape[1]
        self._metadata['n_epochs'] = self._model.n_iter_
        self._metadata['final_train_loss'] = float(self._model.loss_)
        if (
            hasattr(self._model, 'best_validation_score_')
            and self._model.best_validation_score_ is not None
        ):
            self._metadata['best_val_score'] = float(self._model.best_validation_score_)

        if val_peptides is not None and y_val is not None:
            X_val = self._extract_features(val_peptides)
            X_val_scaled = self._scaler.transform(X_val)
            y_val_pred = self._model.predict(X_val_scaled)
            y_val_pred_arr = np.array(y_val_pred)
            y_val_arr = np.array(y_val)
            if y_val_pred_arr.ndim == 1:
                y_val_pred_arr = y_val_pred_arr.reshape(-1, 1)
            if y_val_arr.ndim == 1:
                y_val_arr = y_val_arr.reshape(-1, 1)
            if y_val_pred_arr.shape != y_val_arr.shape:
                raise ValueError(
                    "Validation labels shape does not match predictions: "
                    f"pred={y_val_pred_arr.shape}, labels={y_val_arr.shape}"
                )
            val_loss = float(np.mean((y_val_pred_arr - y_val_arr) ** 2))
            self._metadata['final_val_loss'] = val_loss
            self._metadata['best_val_loss'] = val_loss

        self._training_history = [
            {'epoch': i + 1, 'loss': loss}
            for i, loss in enumerate(self._model.loss_curve_)
        ]

        if verbose:
            print(f"\nTraining complete:")
            print(f"  Features: {X.shape[1]}")
            print(f"  Iterations: {self._model.n_iter_}")
            print(f"  Final loss: {self._model.loss_:.4f}")

        # Save loss curve plot if requested
        if plot_loss:
            self._save_loss_plot(plot_loss, verbose=verbose)

        return self

    def _save_loss_plot(
        self,
        path: Union[bool, str, Path],
        verbose: bool = True,
    ) -> Path:
        """Save loss curve plot to file.

        Parameters
        ----------
        path : bool, str, or Path
            If True, saves to 'loss_curve.png'. If str/Path, saves to that location.
        verbose : bool
            Print save location.

        Returns
        -------
        save_path : Path
            Path where plot was saved.
        """
        import matplotlib.pyplot as plt

        if not self._is_trained or self._model is None:
            raise RuntimeError("Model must be trained before saving loss plot.")

        # Determine save path
        if path is True:
            save_path = Path('loss_curve.png')
        else:
            save_path = Path(path)

        # Get loss curve
        loss_curve = self._model.loss_curve_

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        epochs = range(1, len(loss_curve) + 1)
        ax.plot(epochs, loss_curve, 'b-', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title(f'MLPScorer Training Loss ({self._metadata.get("n_train", "?")} samples)', fontsize=14)
        ax.grid(True, alpha=0.3)

        # Use log scale if loss spans multiple orders of magnitude
        if len(loss_curve) > 1 and loss_curve[0] / max(loss_curve[-1], 1e-10) > 10:
            ax.set_yscale('log')

        # Add annotations
        ax.annotate(
            f'Start: {loss_curve[0]:.3f}',
            xy=(1, loss_curve[0]),
            xytext=(len(loss_curve) * 0.1, loss_curve[0] * 1.2),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
        )
        ax.annotate(
            f'End: {loss_curve[-1]:.4f}',
            xy=(len(loss_curve), loss_curve[-1]),
            xytext=(len(loss_curve) * 0.7, loss_curve[-1] * 2),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
        )

        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        if verbose:
            print(f"  Loss plot saved to: {save_path}")

        return save_path

    def score(
        self,
        peptides: Union[str, Sequence[str]],
        aggregate: str = 'mean',
        pathogen_categories: Optional[List[str]] = None,
        self_categories: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Score peptides for foreignness.

        For variable-length peptides, scores are computed per k-mer and aggregated.

        Parameters
        ----------
        peptides : str or sequence of str
            Peptide(s) to score.
        aggregate : str, default='mean'
            How to aggregate k-mer probabilities: 'mean', 'max', 'min'.
        pathogen_categories : list of str, optional
            Categories considered "foreign" (default: ['bacteria', 'viruses']).
        self_categories : list of str, optional
            Categories considered "self" (default: ['human', 'rodents', 'mammals']).

        Returns
        -------
        scores : np.ndarray
            Foreignness scores (higher = more foreign).
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before scoring.")

        if self._target_categories is None:
            probs = self.predict_proba(peptides, aggregate=aggregate)
            if probs.shape[1] > 1:
                raise RuntimeError(
                    "Model has multiple outputs. Use predict_proba() or train with "
                    "target_categories to compute foreignness."
                )
            return probs.ravel()

        return self.foreignness(
            peptides,
            pathogen_categories=pathogen_categories,
            self_categories=self_categories,
            aggregate=aggregate,
        )

    def _score_batch_impl(self, batch: List[str]) -> np.ndarray:
        """Score a batch of peptides."""
        return self.score(batch)

    def predict_proba(
        self,
        peptides: Union[str, Sequence[str]],
        aggregate: str = 'mean',
    ) -> np.ndarray:
        """Predict category probabilities using sigmoid activation.

        Parameters
        ----------
        peptides : str or sequence of str
            Peptide(s) to predict.
        aggregate : str, default='mean'
            How to aggregate k-mer probabilities for long peptides: 'mean', 'max', 'min'.

        Returns
        -------
        probs : np.ndarray
            Probabilities for each category, shape (n_peptides, n_categories).
            Values are in [0, 1] via sigmoid transformation.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction.")

        peptides = self._ensure_list(peptides)
        results: List[np.ndarray] = []

        for peptide in peptides:
            kmers = self._extract_kmers(peptide)
            raw = self._predict_raw_kmers(kmers)
            probs = 1 / (1 + np.exp(-raw))
            if probs.ndim == 1:
                probs = probs.reshape(-1, 1)

            if aggregate == 'mean':
                agg_probs = probs.mean(axis=0)
            elif aggregate == 'max':
                agg_probs = probs.max(axis=0)
            elif aggregate == 'min':
                agg_probs = probs.min(axis=0)
            else:
                raise ValueError(f"Unknown aggregate method: {aggregate}")

            results.append(agg_probs)

        return np.vstack(results)

    def foreignness(
        self,
        peptides: Union[str, Sequence[str]],
        pathogen_categories: Optional[List[str]] = None,
        self_categories: Optional[List[str]] = None,
        aggregate: str = 'mean',
    ) -> np.ndarray:
        """Compute foreignness score from category probabilities.

        Foreignness = max(pathogens) / (max(pathogens) + max(self))

        Parameters
        ----------
        peptides : str or sequence of str
            Peptide(s) to score.
        pathogen_categories : list of str, optional
            Categories considered "foreign" (default: ['bacteria', 'viruses']).
        self_categories : list of str, optional
            Categories considered "self" (default: ['human', 'rodents', 'mammals']).
        aggregate : str, default='mean'
            How to aggregate k-mer probabilities for long peptides.

        Returns
        -------
        foreignness : np.ndarray
            Foreignness scores in [0, 1]. Higher = more foreign.
        """
        if self._target_categories is None:
            raise RuntimeError(
                "Model must be trained with target_categories to use foreignness(). "
                "Use score() for single-output models."
            )

        if pathogen_categories is None:
            pathogen_categories = ['bacteria', 'viruses']
        if self_categories is None:
            self_categories = ['human', 'rodents', 'mammals']

        # Get category indices
        pathogen_idx = [
            self._target_categories.index(cat)
            for cat in pathogen_categories
            if cat in self._target_categories
        ]
        self_idx = [
            self._target_categories.index(cat)
            for cat in self_categories
            if cat in self._target_categories
        ]

        if not pathogen_idx:
            raise ValueError(
                f"No pathogen categories found. Available: {self._target_categories}"
            )
        if not self_idx:
            raise ValueError(
                f"No self categories found. Available: {self._target_categories}"
            )

        # Get probabilities
        probs = self.predict_proba(peptides, aggregate=aggregate)

        # Compute foreignness: max(pathogens) / (max(pathogens) + max(self))
        max_pathogen = probs[:, pathogen_idx].max(axis=1)
        max_self = probs[:, self_idx].max(axis=1)

        # Avoid division by zero
        denominator = max_pathogen + max_self
        foreignness = np.where(
            denominator > 0,
            max_pathogen / denominator,
            0.5  # Neutral when both are zero
        )

        return foreignness

    @property
    def target_categories(self) -> Optional[List[str]]:
        """Get target category names (if trained with multi-label)."""
        return self._target_categories

    def predict_dataframe(
        self,
        peptides: Sequence[str],
        pathogen_categories: Optional[List[str]] = None,
        self_categories: Optional[List[str]] = None,
        aggregate: str = 'mean',
    ) -> 'pd.DataFrame':
        """Predict category probabilities and foreignness for variable-length peptides.

        For peptides longer than k, breaks into overlapping k-mers and aggregates.

        Parameters
        ----------
        peptides : sequence of str
            Peptide sequences (can be variable length).
        pathogen_categories : list of str, optional
            Categories considered "foreign" (default: ['bacteria', 'viruses']).
        self_categories : list of str, optional
            Categories considered "self" (default: ['human', 'rodents', 'mammals']).
        aggregate : str, default='mean'
            How to aggregate k-mer scores for long peptides: 'mean', 'max', 'min'.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns:
            - 'peptide': input peptide sequence
            - One column per target category (probabilities)
            - 'foreignness': foreignness score
        """
        import pandas as pd

        if self._target_categories is None:
            raise RuntimeError(
                "Model must be trained with target_categories to use predict_dataframe()."
            )

        if pathogen_categories is None:
            pathogen_categories = ['bacteria', 'viruses']
        if self_categories is None:
            self_categories = ['human', 'rodents', 'mammals']

        # Get category indices for foreignness calculation
        pathogen_idx = [
            self._target_categories.index(cat)
            for cat in pathogen_categories
            if cat in self._target_categories
        ]
        self_idx = [
            self._target_categories.index(cat)
            for cat in self_categories
            if cat in self._target_categories
        ]

        if not pathogen_idx:
            raise ValueError(
                f"No pathogen categories found. Available: {self._target_categories}"
            )
        if not self_idx:
            raise ValueError(
                f"No self categories found. Available: {self._target_categories}"
            )

        peptides = list(peptides)
        probs = self.predict_proba(peptides, aggregate=aggregate)
        results = []
        for peptide, row_probs in zip(peptides, probs):
            max_pathogen = row_probs[pathogen_idx].max()
            max_self = row_probs[self_idx].max()
            denom = max_pathogen + max_self
            foreignness = max_pathogen / denom if denom > 0 else 0.5

            row = {'peptide': peptide}
            for cat, p in zip(self._target_categories, row_probs):
                row[cat] = float(p)
            row['foreignness'] = float(foreignness)
            results.append(row)

        # Create DataFrame with consistent column order
        columns = ['peptide'] + self._target_categories + ['foreignness']
        return pd.DataFrame(results, columns=columns)

    def features_dataframe(
        self,
        peptides: Sequence[str],
        aggregate: str = 'mean',
        include_peptide: bool = True,
    ) -> 'pd.DataFrame':
        """Extract features for peptides as a DataFrame.

        For peptides longer than k, breaks into overlapping k-mers and aggregates.

        Parameters
        ----------
        peptides : sequence of str
            Peptide sequences (can be variable length).
        aggregate : str, default='mean'
            How to aggregate k-mer features for long peptides: 'mean', 'max', 'min'.
        include_peptide : bool, default=True
            Include peptide sequence as first column.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with 592 feature columns (+ peptide column if include_peptide=True).
            Features: 48 AA properties, 27 structural, 20 AA composition,
            12 sequence stats, 80 reduced alphabet frequencies, 5 dipeptide
            summaries, 400 dipeptides (if enabled).
        """
        import pandas as pd

        feature_names = self.get_feature_names()
        results = []

        for peptide in peptides:
            if len(peptide) < self.k:
                # Pad short peptides
                kmers = [peptide + 'X' * (self.k - len(peptide))]
            elif len(peptide) == self.k:
                kmers = [peptide]
            else:
                # Extract overlapping k-mers
                kmers = [peptide[i:i+self.k] for i in range(len(peptide) - self.k + 1)]

            # Extract features for all k-mers
            kmer_features = np.array([
                extract_features(kmer, self.k, self._params['use_dipeptides'])
                for kmer in kmers
            ])

            # Aggregate across k-mers
            if aggregate == 'mean':
                features = kmer_features.mean(axis=0)
            elif aggregate == 'max':
                features = kmer_features.max(axis=0)
            elif aggregate == 'min':
                features = kmer_features.min(axis=0)
            else:
                raise ValueError(f"Unknown aggregate method: {aggregate}")

            if include_peptide:
                row = {'peptide': peptide}
                row.update(dict(zip(feature_names, features)))
            else:
                row = dict(zip(feature_names, features))
            results.append(row)

        # Create DataFrame with consistent column order
        if include_peptide:
            columns = ['peptide'] + feature_names
        else:
            columns = feature_names
        return pd.DataFrame(results, columns=columns)

    def _save_model(self, path: Path) -> None:
        """Save model weights."""
        model_data = {
            'model': self._model,
            'scaler': self._scaler,
            'target_categories': self._target_categories,
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def _load_model(self, path: Path) -> None:
        """Load model weights."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self._model = model_data['model']
        self._scaler = model_data['scaler']
        self._target_categories = model_data.get('target_categories')

    def get_feature_names(self) -> List[str]:
        """Get names of all features used by the model.

        Returns
        -------
        names : list of str
            Feature names.
        """
        properties = list(_get_aa_properties().keys())
        names = []

        # Property statistics (12 props × 4 stats = 48 features)
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

        # Dipeptide summary (5 features)
        if self._params.get('use_dipeptides', True):
            names.extend([
                'dipep_entropy',
                'dipep_gini',
                'dipep_max_freq',
                'dipep_top2_freq',
                'dipep_homodimer_frac',
            ])

        # Dipeptide composition (400 features)
        if self._params.get('use_dipeptides', True):
            for aa1 in AMINO_ACIDS:
                for aa2 in AMINO_ACIDS:
                    names.append(f'dipep_{aa1}{aa2}')

        return names
