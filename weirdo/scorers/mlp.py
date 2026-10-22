"""MLP-based origin scorer.

Neural network model for learning category probabilities from labeled data.
Uses rich peptide features including amino acid properties and composition
statistics.
"""

import copy
import hashlib
import pickle
import time
import warnings
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

from .trainable import TrainableScorer
from .registry import register_scorer
from .peptide_features import (  # noqa: F401  (re-exported for backward compatibility)
    AA_TO_IDX,
    AMINO_ACIDS,
    NUM_AMINO_ACIDS,
    REDUCED_ALPHABET_GROUPS,
    REDUCED_ALPHABET_ORDER,
    _compute_composition_features,
    _compute_dipeptide_features,
    _compute_property_features,
    _get_aa_properties,
    extract_features,
    extract_features_batch,
    feature_count,
    get_feature_names as _feature_names,
)


# Default category groups for the pathogen-versus-self foreignness summary.
DEFAULT_PATHOGEN_CATEGORIES = ('bacteria', 'viruses')
DEFAULT_SELF_CATEGORIES = ('human', 'rodents', 'mammals')

# How the probabilities inside the pathogen group and inside the self group are
# combined before they are compared. ``combine`` may also be a callable that maps an
# array of shape (n_peptides, n_categories_in_group) to one value per peptide.
DEFAULT_COMBINE = 'max'
NAMED_COMBINERS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'max': lambda values: values.max(axis=1),
    'sum': lambda values: values.sum(axis=1),
}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 0.5 * (1.0 + np.tanh(0.5 * z))


_NUMPY_ACTIVATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'relu': lambda z: np.maximum(z, 0.0),
    'tanh': np.tanh,
    'logistic': _sigmoid,
    'identity': lambda z: z,
}


class _NumpyScaler:
    """Inference-only standardizer equivalent to a fitted ``StandardScaler``."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray):
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.scale_ = np.asarray(scale, dtype=np.float32)
        self.n_features_in_ = int(self.mean_.shape[0])

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=np.float32) - self.mean_) / self.scale_


class _NumpyMLP:
    """Inference-only feed-forward network with sigmoid outputs.

    Holds the weights of a trained ``MLPClassifier`` in the same ``coefs_`` /
    ``intercepts_`` layout so a model can be stored without pickling
    scikit-learn objects and evaluated with NumPy alone.
    """

    def __init__(self, coefs: Sequence[np.ndarray], intercepts: Sequence[np.ndarray],
                 activation: str = 'relu'):
        if activation not in _NUMPY_ACTIVATIONS:
            raise ValueError(
                f"Unsupported activation '{activation}'. "
                f"Available: {sorted(_NUMPY_ACTIVATIONS)}"
            )
        if len(coefs) != len(intercepts) or not coefs:
            raise ValueError("coefs and intercepts must be non-empty and equally long.")
        self.coefs_ = [np.asarray(w, dtype=np.float32) for w in coefs]
        self.intercepts_ = [np.asarray(b, dtype=np.float32) for b in intercepts]
        self.activation = activation
        self.n_outputs_ = int(self.coefs_[-1].shape[1])

    def _logits(self, X: np.ndarray) -> np.ndarray:
        hidden = np.asarray(X, dtype=np.float32)
        activate = _NUMPY_ACTIVATIONS[self.activation]
        last = len(self.coefs_) - 1
        for index, (weights, bias) in enumerate(zip(self.coefs_, self.intercepts_)):
            hidden = hidden @ weights + bias
            if index < last:
                hidden = activate(hidden)
        return hidden

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self._logits(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)


@register_scorer('mlp', description='MLP foreignness scorer with rich peptide features')
class MLPScorer(TrainableScorer):
    """MLP-based origin scorer using rich peptide features.

    Combines multiple feature types:
    - Amino acid properties (hydropathy, mass, polarity, etc.)
    - Amino acid composition (single AA frequencies)
    - Dipeptide composition (AA pair frequencies)
    - Sequence-level statistics (entropy, repeats, complexity)
    - Reduced alphabet compositions (Murphy/GBMR/SDM, etc.)
    - Optionally, a positional one-hot encoding of the k-mer (``positional=True``)

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
    random_state : int, optional
        Seed for weight initialization and batch shuffling.
    positional : bool, default=False
        Append a position-specific one-hot encoding of the k-mer (``k * 21``
        features: for each position, which amino acid is there). This lets the
        network learn position-dependent patterns, such as an anchor residue at
        position 2 of a peptide, but it also makes memorising individual k-mers
        easy; the default composition-only features generalise to unseen
        sequences better. Peptides longer than ``k`` are scored per k-mer, as
        usual. Models trained with ``positional=True`` store the option and load
        with it.
    loss : {'auto', 'bce', 'squared'}, default='auto'
        Training objective. ``'bce'`` trains one independent sigmoid output per
        label with binary cross-entropy, so outputs are probabilities
        (``P(label = 1 | peptide)``); it needs binary 0/1 labels. ``'squared'``
        fits scores by least squares (the previous behaviour, kept for scalar
        scores and for loading older models). ``'auto'`` uses ``'bce'`` for
        multi-label binary targets and ``'squared'`` otherwise.

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
        loss: str = 'auto',
        positional: bool = False,
        **kwargs
    ):
        if loss not in ('auto', 'bce', 'squared'):
            raise ValueError(
                f"loss must be 'auto', 'bce' or 'squared', got {loss!r}."
            )
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
            'loss': loss,
            'positional': bool(positional),
        })
        self._model: Optional[Union[MLPClassifier, MLPRegressor, _NumpyMLP]] = None
        self._scaler: Optional[Union[StandardScaler, _NumpyScaler]] = None
        self._target_categories: Optional[List[str]] = None

    def _extract_features(self, peptides: Sequence[str]) -> np.ndarray:
        """Extract features from a list of peptides."""
        return extract_features_batch(
            peptides,
            self.k,
            use_dipeptides=self._params['use_dipeptides'],
            positional=self._params['positional'],
        )

    @property
    def _is_logistic(self) -> bool:
        """True when the network has sigmoid outputs trained with cross-entropy."""
        return isinstance(self._model, (MLPClassifier, _NumpyMLP))

    @staticmethod
    def _resolve_loss(loss: str, y: np.ndarray) -> str:
        """Pick the training objective for an in-memory label array."""
        n_labels = 1 if y.ndim == 1 else int(y.shape[1])
        binary = bool(np.isin(y, (0.0, 1.0)).all())
        if loss == 'squared':
            return 'squared'
        if loss == 'bce':
            if not binary:
                raise ValueError(
                    "loss='bce' needs binary 0/1 labels; use loss='squared' for "
                    "continuous scores."
                )
            return 'bce'
        return 'bce' if (n_labels >= 2 and binary) else 'squared'

    def _make_network(self, loss_kind: str, learning_rate: float, n_iter_no_change: int):
        """Create the (incrementally trained) scikit-learn network."""
        network = MLPClassifier if loss_kind == 'bce' else MLPRegressor
        return network(
            hidden_layer_sizes=self._params['hidden_layer_sizes'],
            activation=self._params['activation'],
            alpha=self._params['alpha'],
            learning_rate_init=learning_rate,
            max_iter=1,
            early_stopping=False,
            n_iter_no_change=n_iter_no_change,
            random_state=self._params['random_state'],
            shuffle=False,
            warm_start=True,
            verbose=False,
        )

    def _partial_fit_batch(self, X_scaled: np.ndarray, y_batch: np.ndarray) -> None:
        """One incremental optimizer step on a scaled feature batch."""
        y_fit = y_batch.ravel() if (y_batch.ndim == 2 and y_batch.shape[1] == 1) else y_batch
        if self._is_logistic:
            if not np.isin(y_fit, (0.0, 1.0)).all():
                raise ValueError(
                    "Cross-entropy training needs binary 0/1 labels; pass "
                    "loss='squared' for continuous targets."
                )
            classes = np.arange(y_fit.shape[1]) if y_fit.ndim == 2 else np.array([0.0, 1.0])
            self._model.partial_fit(X_scaled, y_fit, classes=classes)
        else:
            self._model.partial_fit(X_scaled, y_fit)

    def _batch_loss(self) -> float:
        """Loss of the last optimizer step, averaged over samples and labels.

        scikit-learn's multi-label ``MLPClassifier`` sums the cross-entropy over its
        outputs; dividing by their number puts it on the scale of the validation
        log loss.
        """
        loss = float(self._model.loss_)
        if isinstance(self._model, MLPClassifier):
            loss /= max(int(self._model.n_outputs_), 1)
        return loss

    def _model_outputs(self, X_scaled: np.ndarray) -> np.ndarray:
        """Network outputs: probabilities for sigmoid heads, raw scores otherwise."""
        if self._is_logistic:
            outputs = np.asarray(self._model.predict_proba(X_scaled), dtype=np.float32)
            if isinstance(self._model, MLPClassifier) and self._model.n_outputs_ == 1:
                outputs = outputs[:, 1:2]  # scikit-learn returns [P(0), P(1)] for one label
            return outputs
        return np.asarray(self._model.predict(X_scaled), dtype=np.float32)

    def _extract_matrix(self, peptides: Sequence[str], chunk_size: int = 65536) -> np.ndarray:
        """Feature matrix for ``peptides``.

        The matrix is filled chunk by chunk, so memory use stays close to the size
        of the result (no second copy is built).
        """
        peptides = list(peptides)
        features = np.empty((len(peptides), len(self.get_feature_names())), dtype=np.float32)
        for start in range(0, len(peptides), chunk_size):
            features[start:start + chunk_size] = self._extract_features(
                peptides[start:start + chunk_size]
            )
        return features

    def _fit_scaler_and_scale(self, features: np.ndarray, chunk_rows: int = 262144) -> None:
        """Fit ``self._scaler`` on ``features`` in chunks, then standardize them in place."""
        self._scaler = StandardScaler()
        for start in range(0, len(features), chunk_rows):
            self._scaler.partial_fit(features[start:start + chunk_rows])
        self._scale_in_place(features, chunk_rows)

    def _scale_in_place(self, features: np.ndarray, chunk_rows: int = 262144) -> None:
        """Standardize ``features`` with the fitted scaler without making a copy."""
        mean = np.asarray(self._scaler.mean_, dtype=np.float32)
        scale = np.asarray(self._scaler.scale_, dtype=np.float32)
        for start in range(0, len(features), chunk_rows):
            block = features[start:start + chunk_rows]
            block -= mean
            block /= scale

    @staticmethod
    def _iter_cached_batches(
        X_scaled: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool,
        rng: np.random.RandomState,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (scaled features, labels) batches from an in-memory matrix."""
        order = rng.permutation(len(X_scaled)) if shuffle else np.arange(len(X_scaled))
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            yield X_scaled[index], np.asarray(labels[index], dtype=np.float32)

    def _snapshot_weights(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        return copy.deepcopy(self._model.coefs_), copy.deepcopy(self._model.intercepts_)

    def _restore_weights(self, snapshot: Tuple[List[np.ndarray], List[np.ndarray]]) -> None:
        self._model.coefs_, self._model.intercepts_ = snapshot

    def _feature_signature(self) -> str:
        """Short hash of the ordered feature names the network was trained on."""
        joined = "\n".join(self.get_feature_names())
        return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]

    def _predict_raw_kmers(self, kmers: Sequence[str]) -> np.ndarray:
        """Predict model outputs for k-mers (no aggregation).

        Probabilities for sigmoid-head models, raw scores for least-squares models.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before scoring.")
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model is not initialized. Train or load a model first.")

        X = self._extract_matrix(kmers)
        X_scaled = self._scaler.transform(X).astype(np.float32, copy=False)
        return self._model_outputs(X_scaled)

    def _predict_kmer_probabilities(self, kmers: Sequence[str]) -> np.ndarray:
        """Per-k-mer values in [0, 1], shape (n_kmers, n_outputs)."""
        outputs = self._predict_raw_kmers(kmers)
        probs = outputs if self._is_logistic else _sigmoid(outputs)
        return probs.reshape(-1, 1) if probs.ndim == 1 else probs

    def _iter_feature_batches(
        self,
        peptides: Sequence[str],
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool,
        rng: np.random.RandomState,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield feature/label batches without materializing the full feature matrix."""
        n_samples = len(peptides)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if shuffle:
            order = np.arange(n_samples)
            rng.shuffle(order)
            for start in range(0, n_samples, batch_size):
                batch_idx = order[start:start + batch_size]
                batch_peptides = [peptides[i] for i in batch_idx]
                X_batch = self._extract_features(batch_peptides).astype(np.float32, copy=False)
                y_batch = np.asarray(labels[batch_idx], dtype=np.float32)
                yield X_batch, y_batch
            return

        for start in range(0, n_samples, batch_size):
            stop = min(start + batch_size, n_samples)
            batch_peptides = peptides[start:stop]
            X_batch = self._extract_features(batch_peptides).astype(np.float32, copy=False)
            y_batch = np.asarray(labels[start:stop], dtype=np.float32)
            yield X_batch, y_batch

    def _predict_batches(self, peptides: Sequence[str], batch_size: int) -> np.ndarray:
        """Predict in batches to avoid building a full feature matrix."""
        if self._model is None or self._scaler is None:
            raise RuntimeError("Model is not initialized. Train or load a model first.")

        outputs: List[np.ndarray] = []
        for start in range(0, len(peptides), batch_size):
            stop = min(start + batch_size, len(peptides))
            X_batch = self._extract_features(peptides[start:stop]).astype(np.float32, copy=False)
            X_scaled = self._scaler.transform(X_batch).astype(np.float32, copy=False)
            outputs.append(self._model_outputs(X_scaled))

        if not outputs:
            return np.array([], dtype=np.float32)
        if outputs[0].ndim == 1:
            return np.concatenate(outputs, axis=0)
        return np.vstack(outputs)

    def _predict_scaled(self, X_scaled: np.ndarray, batch_size: int) -> np.ndarray:
        """Predict from an already scaled feature matrix, in batches."""
        outputs = [
            self._model_outputs(X_scaled[start:start + batch_size])
            for start in range(0, len(X_scaled), batch_size)
        ]
        if not outputs:
            return np.array([], dtype=np.float32)
        if outputs[0].ndim == 1:
            return np.concatenate(outputs, axis=0)
        return np.vstack(outputs)

    def _validation_metrics(
        self,
        val_peptides: Sequence[str],
        val_labels: np.ndarray,
        batch_size: int,
        features: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """Validation (loss, score) in batches.

        Sigmoid-head models report the mean binary cross-entropy and its negation
        (higher is better); least-squares models report MSE and R^2. ``features``
        may hold already scaled features for ``val_peptides``.
        """
        if features is not None:
            y_pred = np.asarray(self._predict_scaled(features, batch_size), dtype=np.float32)
        else:
            y_pred = np.asarray(self._predict_batches(val_peptides, batch_size), dtype=np.float32)
        y_true = np.asarray(val_labels, dtype=np.float32)

        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_pred.shape != y_true.shape:
            raise ValueError(
                "Validation labels shape does not match predictions: "
                f"pred={y_pred.shape}, labels={y_true.shape}"
            )

        if self._is_logistic:
            clipped = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
            bce = float(np.mean(-(y_true * np.log(clipped) + (1.0 - y_true) * np.log1p(-clipped))))
            return bce, -bce

        mse = float(np.mean((y_pred - y_true) ** 2))

        y_mean = y_true.mean(axis=0)
        ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
        ss_tot = np.sum((y_true - y_mean) ** 2, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            r2_each = np.where(ss_tot > 0, 1.0 - (ss_res / ss_tot), 0.0)
        r2 = float(np.mean(r2_each))

        return mse, r2

    def _normalize_label_row(self, label: Any) -> np.ndarray:
        """Convert a single label row to 1D float32 array."""
        arr = np.asarray(label, dtype=np.float32)
        if arr.ndim == 0:
            return arr.reshape(1)
        if arr.ndim != 1:
            raise ValueError(f"Each label row must be scalar or 1D, got shape {arr.shape}.")
        return arr

    def _iter_feature_batches_from_rows(
        self,
        row_iterator_factory: Callable[[], Iterator[Tuple[str, Any]]],
        batch_size: int,
        expected_label_dim: int,
        max_rows: Optional[int] = None,
        skip_rows: int = 0,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield feature/label batches from a row iterator factory."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if expected_label_dim <= 0:
            raise ValueError("expected_label_dim must be positive.")
        if max_rows is not None and max_rows <= 0:
            raise ValueError("max_rows must be positive when provided.")
        if skip_rows < 0:
            raise ValueError("skip_rows must be non-negative.")

        peptides_batch: List[str] = []
        labels_batch: List[np.ndarray] = []
        rows_emitted = 0

        row_iter = iter(row_iterator_factory())
        skipped = 0
        while skipped < skip_rows:
            try:
                next(row_iter)
            except StopIteration:
                return
            skipped += 1

        while max_rows is None or rows_emitted < max_rows:
            try:
                peptide, label = next(row_iter)
            except StopIteration:
                break
            label_row = self._normalize_label_row(label)
            if label_row.shape[0] != expected_label_dim:
                raise ValueError(
                    "Inconsistent label dimensions in row iterator: "
                    f"expected {expected_label_dim}, got {label_row.shape[0]}"
                )

            peptides_batch.append(peptide)
            labels_batch.append(label_row)
            rows_emitted += 1

            if len(peptides_batch) >= batch_size:
                X_batch = self._extract_features(peptides_batch)
                y_batch = np.stack(labels_batch).astype(np.float32, copy=False)
                yield X_batch, y_batch
                peptides_batch = []
                labels_batch = []

        if peptides_batch:
            X_batch = self._extract_features(peptides_batch)
            y_batch = np.stack(labels_batch).astype(np.float32, copy=False)
            yield X_batch, y_batch

    def train_streaming(
        self,
        row_iterator_factory: Callable[[], Iterator[Tuple[str, Any]]],
        target_categories: Optional[List[str]] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        verbose: bool = True,
        plot_loss: Union[bool, str, Path, None] = None,
        **kwargs,
    ) -> 'MLPScorer':
        """Train using a row iterator factory without materializing all samples.

        Parameters
        ----------
        row_iterator_factory : callable
            Callable that returns a fresh iterator yielding ``(peptide, label)``
            pairs for each pass.
        target_categories : list of str, optional
            Names of target categories (for multi-label training).
        epochs : int, optional
            Number of epochs.
        learning_rate : float, optional
            Initial learning rate.
        verbose : bool, default=True
            Print training progress.
        plot_loss : bool, str, or Path, optional
            Save loss curve plot.
        batch_size : int, optional
            Number of rows per update step.
        progress_every_samples : int, optional
            Emit progress logs every N processed rows during scaler fit and each
            epoch. Default is 250000.
        total_rows_hint : int, optional
            Estimated total stream size for scaler-pass ETA reporting.
        scaler_max_rows : int, optional
            Cap rows used to fit scaler. If <= 0, uses full stream.
        train_max_rows_per_epoch : int, optional
            Cap streamed rows consumed per epoch. If <= 0, uses full stream.
        epoch_shuffle : bool, optional
            Shuffle streamed epoch windows deterministically from ``random_state``.
            When `train_max_rows_per_epoch` is set, each epoch samples a different
            deterministic offset into the stream.
        epoch_shuffle_max_skip_rows : int, optional
            Cap on per-epoch skip distance used for deterministic shuffling.
            If <= 0, defaults to `train_max_rows_per_epoch`.
        shuffle_in_batch : bool, optional
            Shuffle rows within each streamed batch deterministically per epoch.
        parallel_workers, parallel_chunksize : int, optional
            Deprecated and ignored. Features are computed with vectorised array
            operations in the calling process, which needs no worker processes.

        Returns
        -------
        self : MLPScorer

        Notes
        -----
        The objective follows the scorer's ``loss`` setting; with ``'auto'``,
        multi-label rows are trained with sigmoid outputs and binary
        cross-entropy and every streamed label must then be 0 or 1.
        """
        batch_size = int(kwargs.pop('batch_size', self.batch_size))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        progress_every_samples = int(kwargs.pop('progress_every_samples', 250000))
        if progress_every_samples <= 0:
            raise ValueError("progress_every_samples must be positive.")
        total_rows_hint_arg = kwargs.pop('total_rows_hint', None)
        scaler_max_rows_arg = int(kwargs.pop('scaler_max_rows', 0))
        train_max_rows_per_epoch_arg = int(kwargs.pop('train_max_rows_per_epoch', 0))
        epoch_shuffle = bool(kwargs.pop('epoch_shuffle', True))
        epoch_shuffle_max_skip_rows_arg = int(kwargs.pop('epoch_shuffle_max_skip_rows', 0))
        shuffle_in_batch = bool(kwargs.pop('shuffle_in_batch', True))
        parallel_workers = int(kwargs.pop('parallel_workers', 1))
        parallel_chunksize = int(kwargs.pop('parallel_chunksize', 256))
        total_rows_hint: Optional[int]
        if total_rows_hint_arg is None:
            total_rows_hint = None
        else:
            total_rows_hint = int(total_rows_hint_arg)
            if total_rows_hint <= 0:
                raise ValueError("total_rows_hint must be positive when provided.")
        scaler_max_rows = scaler_max_rows_arg if scaler_max_rows_arg > 0 else None
        train_max_rows_per_epoch = (
            train_max_rows_per_epoch_arg if train_max_rows_per_epoch_arg > 0 else None
        )
        if epoch_shuffle_max_skip_rows_arg < 0:
            raise ValueError("epoch_shuffle_max_skip_rows must be non-negative.")
        if parallel_workers <= 0:
            raise ValueError("parallel_workers must be positive.")
        if parallel_chunksize <= 0:
            raise ValueError("parallel_chunksize must be positive.")
        if parallel_workers != 1 or parallel_chunksize != 256:
            warnings.warn(
                "parallel_workers and parallel_chunksize are ignored: features are "
                "computed with vectorised array operations in the calling process.",
                DeprecationWarning,
                stacklevel=2,
            )

        if epochs is None:
            epochs = self._params['max_iter']
        if learning_rate is None:
            learning_rate = self._params['learning_rate_init']

        # Probe one row to infer label dimensionality.
        probe_iterator = row_iterator_factory()
        try:
            probe_item = next(probe_iterator)
        except StopIteration:
            raise ValueError("Training data cannot be empty.")
        if not isinstance(probe_item, tuple) or len(probe_item) != 2:
            raise ValueError(
                "row_iterator_factory must yield (peptide, label) pairs."
            )
        _, probe_label = probe_item
        probe_row = self._normalize_label_row(probe_label)
        label_dim = int(probe_row.shape[0])
        # Later rows are checked batch by batch (see _partial_fit_batch).
        loss_kind = self._resolve_loss(self._params['loss'], probe_row.reshape(1, -1))

        if target_categories is not None:
            if label_dim != len(target_categories):
                raise ValueError(
                    "target_categories length must match label dimensions: "
                    f"{len(target_categories)} != {label_dim}"
                )
        elif label_dim > 1:
            raise ValueError(
                "Multi-label streaming training requires target_categories."
            )

        self._target_categories = target_categories

        # Fit scaler incrementally with one pass over the stream.
        self._scaler = StandardScaler()
        n_train = 0
        n_features = 0
        scaler_start = time.time()
        next_scaler_log = progress_every_samples
        scaler_target_rows = scaler_max_rows or total_rows_hint
        for X_batch, _ in self._iter_feature_batches_from_rows(
            row_iterator_factory=row_iterator_factory,
            batch_size=batch_size,
            expected_label_dim=label_dim,
            max_rows=scaler_max_rows,
        ):
            self._scaler.partial_fit(X_batch)
            n_train += X_batch.shape[0]
            n_features = X_batch.shape[1]

            if verbose and n_train >= next_scaler_log:
                elapsed = max(time.time() - scaler_start, 1e-9)
                rows_per_sec = n_train / elapsed
                message = (
                    f"[scaler] rows={n_train:,} "
                    f"rate={rows_per_sec:,.0f} rows/s"
                )
                if scaler_target_rows is not None:
                    pct = (100.0 * n_train) / max(float(scaler_target_rows), 1.0)
                    remaining = max(scaler_target_rows - n_train, 0)
                    eta_sec = remaining / max(rows_per_sec, 1e-9)
                    message += f" ({pct:.2f}% est, ETA {eta_sec / 60.0:.1f} min)"
                print(message)
                while n_train >= next_scaler_log:
                    next_scaler_log += progress_every_samples

        if n_train == 0:
            raise ValueError("Training data cannot be empty.")
        if verbose:
            elapsed = max(time.time() - scaler_start, 1e-9)
            rows_per_sec = n_train / elapsed
            print(
                f"[scaler] complete rows={n_train:,} "
                f"in {elapsed / 60.0:.2f} min ({rows_per_sec:,.0f} rows/s)"
            )

        # Incremental MLP training over multiple streaming passes.
        self._model = self._make_network(
            loss_kind, learning_rate, int(kwargs.pop('n_iter_no_change', 10))
        )

        self._training_history = []
        epoch_row_target = train_max_rows_per_epoch or total_rows_hint
        epoch_rows_history: List[int] = []
        epoch_skip_rows: List[int] = []
        base_seed = self._params['random_state']
        if base_seed is None:
            base_seed = 0
        sampled_total_rows = (
            total_rows_hint
            if total_rows_hint is not None
            else (n_train if scaler_max_rows is None else None)
        )
        for epoch_idx in range(int(epochs)):
            batch_losses: List[float] = []
            epoch_num = epoch_idx + 1
            epoch_rows = 0
            epoch_start = time.time()
            next_epoch_log = progress_every_samples
            epoch_rng = np.random.RandomState(base_seed + epoch_num)
            skip_rows = 0
            if epoch_shuffle and train_max_rows_per_epoch is not None:
                max_skip_cap = (
                    epoch_shuffle_max_skip_rows_arg
                    if epoch_shuffle_max_skip_rows_arg > 0
                    else train_max_rows_per_epoch
                )
                if (
                    sampled_total_rows is not None
                    and sampled_total_rows > train_max_rows_per_epoch
                ):
                    max_skip = min(
                        sampled_total_rows - train_max_rows_per_epoch,
                        max_skip_cap,
                    )
                    skip_rows = int(epoch_rng.randint(0, max_skip + 1))
                elif sampled_total_rows is None:
                    skip_rows = int(epoch_rng.randint(0, max_skip_cap + 1))
            epoch_skip_rows.append(skip_rows)
            for X_batch, y_batch in self._iter_feature_batches_from_rows(
                row_iterator_factory=row_iterator_factory,
                batch_size=batch_size,
                expected_label_dim=label_dim,
                max_rows=train_max_rows_per_epoch,
                skip_rows=skip_rows,
            ):
                if shuffle_in_batch and X_batch.shape[0] > 1:
                    perm = epoch_rng.permutation(X_batch.shape[0])
                    X_batch = X_batch[perm]
                    y_batch = y_batch[perm]
                X_scaled = self._scaler.transform(X_batch).astype(np.float32, copy=False)
                self._partial_fit_batch(X_scaled, y_batch)
                batch_losses.append(self._batch_loss())
                epoch_rows += X_batch.shape[0]

                if verbose and epoch_rows >= next_epoch_log:
                    elapsed = max(time.time() - epoch_start, 1e-9)
                    rows_per_sec = epoch_rows / elapsed
                    if epoch_row_target is not None:
                        remaining = max(epoch_row_target - epoch_rows, 0)
                        eta_sec = remaining / max(rows_per_sec, 1e-9)
                        pct = (100.0 * epoch_rows) / max(float(epoch_row_target), 1.0)
                        print(
                            f"[epoch {epoch_num}] rows={epoch_rows:,}/{epoch_row_target:,} "
                            f"({pct:.1f}%) rate={rows_per_sec:,.0f} rows/s "
                            f"ETA {eta_sec / 60.0:.1f} min"
                        )
                    else:
                        print(
                            f"[epoch {epoch_num}] rows={epoch_rows:,} "
                            f"rate={rows_per_sec:,.0f} rows/s"
                        )
                    while epoch_rows >= next_epoch_log:
                        next_epoch_log += progress_every_samples

            if not batch_losses:
                raise ValueError("Training data cannot be empty.")

            epoch_loss = float(np.mean(batch_losses))
            self._training_history.append({'epoch': epoch_num, 'loss': epoch_loss})
            epoch_rows_history.append(epoch_rows)

            if verbose:
                epoch_minutes = (time.time() - epoch_start) / 60.0
                print(
                    f"Iteration {epoch_num}, loss = {epoch_loss:.8f} "
                    f"(epoch_time={epoch_minutes:.2f} min, rows={epoch_rows:,})"
                )

        self._is_trained = True
        self._is_fitted = True

        final_train_loss = float(self._training_history[-1]['loss'])
        n_train_effective = int(epoch_rows_history[0]) if epoch_rows_history else 0
        self._metadata['n_train'] = n_train_effective
        self._metadata['n_features'] = n_features
        self._metadata['n_epochs'] = len(self._training_history)
        self._metadata['final_train_loss'] = final_train_loss
        self._metadata['loss'] = loss_kind
        self._metadata['scaler_rows'] = n_train
        self._metadata['scaler_max_rows'] = scaler_max_rows_arg
        self._metadata['train_max_rows_per_epoch'] = train_max_rows_per_epoch_arg
        self._metadata['total_rows_hint'] = total_rows_hint
        self._metadata['epoch_rows'] = epoch_rows_history
        self._metadata['epoch_skip_rows'] = epoch_skip_rows
        self._metadata['epoch_shuffle'] = epoch_shuffle
        self._metadata['epoch_shuffle_max_skip_rows'] = epoch_shuffle_max_skip_rows_arg
        self._metadata['shuffle_in_batch'] = shuffle_in_batch

        if verbose:
            print("\nTraining complete:")
            print(f"  Features: {n_features}")
            print(f"  n_train (per-epoch): {n_train_effective}")
            print(f"  Scaler rows: {n_train}")
            print(f"  Iterations: {len(self._training_history)}")
            print(f"  Final loss: {final_train_loss:.4f}")

        if plot_loss:
            self._save_loss_plot(plot_loss, verbose=verbose)

        return self

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
        batch_size : int, optional
            Rows per optimizer step (default: the scorer's ``batch_size``).
        cache_features : bool, default=False
            Extract and scale all features once and keep them in memory instead
            of recomputing them every epoch. Much faster for large training
            sets; needs about ``n_samples * n_features * 4`` bytes.
        restore_best : bool, default=True
            For sigmoid-head models with early stopping, restore the weights of
            the epoch with the lowest validation loss rather than keeping the
            weights of the last epoch.

        Returns
        -------
        self : MLPScorer

        Notes
        -----
        With binary multi-label targets and ``loss='auto'`` the network is trained
        with one independent sigmoid output per label and binary cross-entropy, so
        ``predict_proba`` returns probabilities. Pass external ``val_peptides`` /
        ``val_labels`` that are not near-duplicates of the training peptides for
        early stopping to be meaningful.
        """
        peptides = list(peptides)
        if not peptides:
            raise ValueError("Training data cannot be empty.")

        y = np.asarray(labels, dtype=np.float32)
        if len(peptides) != len(y):
            raise ValueError(
                f"Length mismatch: {len(peptides)} peptides but {len(y)} labels."
            )
        if y.ndim not in (1, 2):
            raise ValueError(f"Labels must be 1D or 2D, got shape {y.shape}.")

        if (val_peptides is None) != (val_labels is None):
            raise ValueError("Provide both val_peptides and val_labels together.")

        if val_peptides is not None and val_labels is not None:
            val_peptides = list(val_peptides)
            y_val = np.asarray(val_labels, dtype=np.float32)
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
        loss_kind = self._resolve_loss(self._params['loss'], y)

        if target_categories is not None:
            if y.ndim == 1 and len(target_categories) != 1:
                raise ValueError("target_categories length must match label dimensions.")
            if y.ndim == 2 and y.shape[1] != len(target_categories):
                raise ValueError("target_categories length must match label dimensions.")

        batch_size = int(kwargs.pop('batch_size', self.batch_size))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        shuffle_batches = bool(kwargs.pop('shuffle_batches', True))
        tol = float(kwargs.pop('tol', 1e-4))
        n_iter_no_change = int(kwargs.pop('n_iter_no_change', 10))
        cache_features = bool(kwargs.pop('cache_features', False))
        restore_best = bool(kwargs.pop('restore_best', True))

        # Disable automatic validation split when too small.
        use_early_stopping = self._params['early_stopping']
        if use_early_stopping and len(peptides) < 20:
            use_early_stopping = False
            if verbose:
                print("Note: Early stopping disabled (dataset too small)")

        train_peptides = peptides
        y_train = y

        # Emulate sklearn early stopping using a held-out split when external
        # validation data is not provided.
        if use_early_stopping and y_val is None:
            n_val = max(1, int(round(0.1 * len(peptides))))
            if n_val >= len(peptides):
                use_early_stopping = False
            else:
                split_rng = np.random.RandomState(self._params['random_state'])
                indices = np.arange(len(peptides))
                split_rng.shuffle(indices)
                val_idx = indices[:n_val]
                train_idx = indices[n_val:]
                val_peptides = [peptides[i] for i in val_idx]
                y_val = np.asarray(y[val_idx], dtype=np.float32)
                train_peptides = [peptides[i] for i in train_idx]
                y_train = np.asarray(y[train_idx], dtype=np.float32)

        rng = np.random.RandomState(self._params['random_state'])

        # Fit the scaler: once on a cached matrix, or incrementally over batches.
        X_train_cached: Optional[np.ndarray] = None
        X_val_cached: Optional[np.ndarray] = None
        self._scaler = StandardScaler()
        if cache_features:
            X_train_cached = self._extract_matrix(train_peptides)
            self._fit_scaler_and_scale(X_train_cached)
            if val_peptides is not None and y_val is not None:
                X_val_cached = self._extract_matrix(val_peptides)
                self._scale_in_place(X_val_cached)
        else:
            for X_batch, _ in self._iter_feature_batches(
                train_peptides,
                y_train,
                batch_size=batch_size,
                shuffle=False,
                rng=rng,
            ):
                self._scaler.partial_fit(X_batch)

        # Train incrementally with partial_fit to avoid building a full feature matrix.
        self._model = self._make_network(loss_kind, learning_rate, n_iter_no_change)

        self._training_history = []
        best_val_score: Optional[float] = None
        best_val_loss: Optional[float] = None
        final_val_loss: Optional[float] = None
        best_epoch: Optional[int] = None
        best_weights = None
        no_improve_count = 0

        for epoch_idx in range(int(epochs)):
            batch_losses: List[float] = []

            if X_train_cached is not None:
                scaled_batches = self._iter_cached_batches(
                    X_train_cached, y_train, batch_size, shuffle_batches, rng
                )
            else:
                scaled_batches = (
                    (self._scaler.transform(X_raw).astype(np.float32, copy=False), y_raw)
                    for X_raw, y_raw in self._iter_feature_batches(
                        train_peptides,
                        y_train,
                        batch_size=batch_size,
                        shuffle=shuffle_batches,
                        rng=rng,
                    )
                )
            for X_scaled, y_batch in scaled_batches:
                self._partial_fit_batch(X_scaled, y_batch)
                batch_losses.append(self._batch_loss())

            if not batch_losses:
                raise ValueError("Training data cannot be empty.")

            epoch_loss = float(np.mean(batch_losses))
            epoch_num = epoch_idx + 1
            self._training_history.append({'epoch': epoch_num, 'loss': epoch_loss})

            val_score: Optional[float] = None
            if val_peptides is not None and y_val is not None:
                final_val_loss, val_score = self._validation_metrics(
                    val_peptides=val_peptides,
                    val_labels=y_val,
                    batch_size=batch_size,
                    features=X_val_cached,
                )
                self._training_history[-1]['val_loss'] = float(final_val_loss)

            if verbose:
                print(f"Iteration {epoch_num}, loss = {epoch_loss:.8f}")
                if val_score is not None:
                    print(f"Validation score: {val_score:.6f}")
                    if self._is_logistic:
                        print(f"Validation log loss: {final_val_loss:.6f}")

            if use_early_stopping and val_score is not None and final_val_loss is not None:
                if best_val_score is None or val_score > best_val_score + tol:
                    best_val_score = float(val_score)
                    best_val_loss = float(final_val_loss)
                    best_epoch = epoch_num
                    if self._is_logistic and restore_best:
                        best_weights = self._snapshot_weights()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                    if no_improve_count >= n_iter_no_change:
                        if verbose:
                            print(
                                "Validation score did not improve more than "
                                f"tol={tol:.6f} for {n_iter_no_change} consecutive epochs. "
                                "Stopping."
                            )
                        break

        if best_weights is not None:
            self._restore_weights(best_weights)
            if verbose and best_epoch != len(self._training_history):
                print(f"Restored weights from epoch {best_epoch} (lowest validation loss).")

        self._is_trained = True
        self._is_fitted = True

        n_features = int(getattr(self._scaler, 'n_features_in_', 0))
        n_epochs_run = len(self._training_history)
        final_train_loss = float(self._training_history[-1]['loss'])

        # Save training metadata
        self._metadata['n_train'] = len(peptides)
        self._metadata['n_features'] = n_features
        self._metadata['n_epochs'] = n_epochs_run
        self._metadata['final_train_loss'] = final_train_loss
        self._metadata['loss'] = loss_kind
        if best_epoch is not None:
            self._metadata['best_epoch'] = best_epoch
            self._metadata['weights_restored_from_best_epoch'] = best_weights is not None
        if best_val_score is not None:
            self._metadata['best_val_score'] = float(best_val_score)

        if val_peptides is not None and y_val is not None and final_val_loss is not None:
            self._metadata['final_val_loss'] = float(final_val_loss)
            self._metadata['best_val_loss'] = float(
                best_val_loss if best_val_loss is not None else final_val_loss
            )

        if verbose:
            print("\nTraining complete:")
            print(f"  Features: {n_features}")
            print(f"  Iterations: {n_epochs_run}")
            print(f"  Final loss: {final_train_loss:.4f}")

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

        # Get loss curve from epoch-level training history.
        loss_curve = [entry['loss'] for entry in self._training_history]
        if not loss_curve:
            raise RuntimeError("No training history is available to plot.")

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
        combine: Union[str, Callable[[np.ndarray], np.ndarray]] = DEFAULT_COMBINE,
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
        combine : {'max', 'sum'} or callable, default='max'
            How to combine the probabilities within each group; see
            :meth:`foreignness`.

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
            combine=combine,
        )

    def _score_batch_impl(self, batch: List[str]) -> np.ndarray:
        """Score a batch of peptides."""
        return self.score(batch)

    def predict_proba(
        self,
        peptides: Union[str, Sequence[str]],
        aggregate: str = 'mean',
    ) -> np.ndarray:
        """Predict category probabilities with one independent sigmoid per category.

        For models trained with binary cross-entropy (the default for binary
        multi-label targets) column ``j`` estimates ``P(label j = 1 | peptide)``
        under the training distribution. The columns are not mutually exclusive
        and do not sum to one: a conserved peptide can belong to several
        categories at once.

        Peptides longer than ``k`` are split into overlapping k-mers whose
        probabilities are aggregated; peptides shorter than ``k`` are padded with
        ``X`` and are not reliable.

        Parameters
        ----------
        peptides : str or sequence of str
            Peptide(s) to predict.
        aggregate : str, default='mean'
            How to aggregate k-mer probabilities for long peptides: 'mean', 'max', 'min'.

        Returns
        -------
        probs : np.ndarray
            Probabilities for each category, shape (n_peptides, n_categories),
            in [0, 1].

        Notes
        -----
        Models trained with ``loss='squared'`` (including models saved by older
        versions) fit 0/1 labels by least squares and pass the result through a
        sigmoid: a perfect fit of the 0/1 targets would output only 0.50 or 0.73,
        so these values are *not* calibrated probabilities. A warning is issued
        for such multi-output models.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction.")
        if aggregate not in ('mean', 'max', 'min'):
            raise ValueError(f"Unknown aggregate method: {aggregate}")

        peptides = self._ensure_list(peptides)
        n_outputs = int(self._model.coefs_[-1].shape[1])
        if self._target_categories is not None and not self._is_logistic and n_outputs > 1:
            warnings.warn(
                "This model was trained with squared loss on binary labels; its "
                "outputs are a sigmoid of a regression value and are not calibrated "
                "probabilities. Retrain with the default loss='auto' (binary "
                "cross-entropy) to obtain calibrated sigmoid outputs.",
                UserWarning,
                stacklevel=2,
            )
        if not peptides:
            return np.zeros((0, n_outputs), dtype=np.float64)

        windows: List[str] = []
        counts: List[int] = []
        for peptide in peptides:
            kmers = self._extract_kmers(peptide)
            windows.extend(kmers)
            counts.append(len(kmers))

        step = 65536
        probs = np.vstack([
            self._predict_kmer_probabilities(windows[start:start + step])
            for start in range(0, len(windows), step)
        ]).astype(np.float64, copy=False)

        counts_array = np.asarray(counts)
        starts = np.concatenate([[0], np.cumsum(counts_array)[:-1]])
        if aggregate == 'mean':
            return np.add.reduceat(probs, starts, axis=0) / counts_array[:, None]
        if aggregate == 'max':
            return np.maximum.reduceat(probs, starts, axis=0)
        return np.minimum.reduceat(probs, starts, axis=0)

    def _group_indices(
        self,
        pathogen_categories: Optional[Sequence[str]],
        self_categories: Optional[Sequence[str]],
    ) -> Tuple[List[int], List[int]]:
        """Column indices of the pathogen and self category groups."""
        if self._target_categories is None:
            raise RuntimeError(
                "Model must be trained with target_categories to use foreignness(). "
                "Use score() for single-output models."
            )
        if pathogen_categories is None:
            pathogen_categories = DEFAULT_PATHOGEN_CATEGORIES
        if self_categories is None:
            self_categories = DEFAULT_SELF_CATEGORIES

        available = list(self._target_categories)
        pathogen_idx = [
            available.index(cat) for cat in dict.fromkeys(pathogen_categories)
            if cat in available
        ]
        self_idx = [
            available.index(cat) for cat in dict.fromkeys(self_categories)
            if cat in available
        ]
        if not pathogen_idx:
            raise ValueError(f"No pathogen categories found. Available: {available}")
        if not self_idx:
            raise ValueError(f"No self categories found. Available: {available}")
        return pathogen_idx, self_idx

    @staticmethod
    def _combine_category_scores(
        probs: np.ndarray,
        pathogen_idx: Sequence[int],
        self_idx: Sequence[int],
        combine: Union[str, Callable[[np.ndarray], np.ndarray]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pathogen value, self value and ``pathogen / (pathogen + self)``."""
        if callable(combine):
            reduce = combine
        elif combine in NAMED_COMBINERS:
            reduce = NAMED_COMBINERS[combine]
        else:
            raise ValueError(
                f"Unknown combine method: {combine!r}. Use one of "
                f"{sorted(NAMED_COMBINERS)} or a callable."
            )
        pathogen_value = np.asarray(reduce(probs[:, list(pathogen_idx)]), dtype=np.float64)
        self_value = np.asarray(reduce(probs[:, list(self_idx)]), dtype=np.float64)
        denominator = pathogen_value + self_value
        foreignness = np.where(
            denominator > 0,
            pathogen_value / np.where(denominator > 0, denominator, 1.0),
            0.5,  # Neutral when both are zero
        )
        return pathogen_value, self_value, foreignness

    def foreignness(
        self,
        peptides: Union[str, Sequence[str]],
        pathogen_categories: Optional[List[str]] = None,
        self_categories: Optional[List[str]] = None,
        aggregate: str = 'mean',
        combine: Union[str, Callable[[np.ndarray], np.ndarray]] = DEFAULT_COMBINE,
    ) -> np.ndarray:
        """Compute foreignness from category probabilities.

        ``foreignness = pathogen / (pathogen + self)``, where ``pathogen`` and
        ``self`` combine the probabilities of the categories in each group:

        - ``combine='max'`` (default): the largest probability in each group, i.e.
          ``max(pathogens) / (max(pathogens) + max(self))``.
        - ``combine='sum'``: the sum of the probabilities in each group. The sum of
          independent sigmoid outputs is the expected number of categories in the
          group that contain the peptide, so this compares the expected number of
          pathogen categories with the expected number of self categories (the
          groups differ in size, which shifts the neutral point).
        - a callable receiving an array of shape (n_peptides, n_categories_in_group)
          and returning one value per peptide.

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
        combine : {'max', 'sum'} or callable, default='max'
            How to combine probabilities within each group.

        Returns
        -------
        foreignness : np.ndarray
            Scores in [0, 1] (0.5 when both groups are 0). Higher = more foreign.
        """
        pathogen_idx, self_idx = self._group_indices(pathogen_categories, self_categories)
        probs = self.predict_proba(peptides, aggregate=aggregate)
        return self._combine_category_scores(probs, pathogen_idx, self_idx, combine)[2]

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
        combine: Union[str, Callable[[np.ndarray], np.ndarray]] = DEFAULT_COMBINE,
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
        combine : {'max', 'sum'} or callable, default='max'
            How to combine probabilities within the pathogen and self groups; see
            :meth:`foreignness`.

        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns:
            - 'peptide': input peptide sequence
            - One column per target category (probabilities)
            - 'pathogen_<combine>' and 'self_<combine>': the group values, e.g.
              'pathogen_max' / 'self_max' (a callable ``combine`` gives
              'pathogen_value' / 'self_value')
            - 'foreignness': pathogen / (pathogen + self)
        """
        import pandas as pd

        if self._target_categories is None:
            raise RuntimeError(
                "Model must be trained with target_categories to use predict_dataframe()."
            )

        pathogen_idx, self_idx = self._group_indices(pathogen_categories, self_categories)
        peptides = list(peptides)
        probs = self.predict_proba(peptides, aggregate=aggregate)
        pathogen_value, self_value, foreignness = self._combine_category_scores(
            probs, pathogen_idx, self_idx, combine
        )

        frame = pd.DataFrame(probs, columns=list(self._target_categories))
        frame.insert(0, 'peptide', peptides)
        label = combine if isinstance(combine, str) else 'value'
        frame[f'pathogen_{label}'] = pathogen_value
        frame[f'self_{label}'] = self_value
        frame['foreignness'] = foreignness
        return frame

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
            kmer_features = self._extract_features(kmers)

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
        """Save model weights.

        Sigmoid-head models are written to ``path`` as a pickle-free ``.npz``
        archive holding the weights, the scaler statistics, the category names and
        a hash of the feature definitions, so they load without scikit-learn
        internals. Least-squares models keep the legacy pickle format.
        """
        if self._is_logistic:
            arrays: Dict[str, np.ndarray] = {
                'format_version': np.array(1),
                'activation': np.array(self._model.activation),
                'n_layers': np.array(len(self._model.coefs_)),
                'has_categories': np.array(self._target_categories is not None),
                'target_categories': np.array(self._target_categories or [], dtype=str),
                'scaler_mean': np.asarray(self._scaler.mean_, dtype=np.float32),
                'scaler_scale': np.asarray(self._scaler.scale_, dtype=np.float32),
                'feature_signature': np.array(self._feature_signature()),
            }
            for index, (weights, bias) in enumerate(
                zip(self._model.coefs_, self._model.intercepts_)
            ):
                arrays[f'coef_{index}'] = np.asarray(weights, dtype=np.float32)
                arrays[f'intercept_{index}'] = np.asarray(bias, dtype=np.float32)
            with open(path, 'wb') as handle:
                np.savez_compressed(handle, **arrays)
            return

        model_data = {
            'model': self._model,
            'scaler': self._scaler,
            'target_categories': self._target_categories,
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def _load_model(self, path: Path) -> None:
        """Load model weights (portable ``.npz`` archive or legacy pickle)."""
        if zipfile.is_zipfile(path):
            with np.load(path, allow_pickle=False) as archive:
                version = int(archive['format_version'])
                if version != 1:
                    raise ValueError(f"Unsupported model format version: {version}")
                n_layers = int(archive['n_layers'])
                coefs = [archive[f'coef_{index}'] for index in range(n_layers)]
                intercepts = [archive[f'intercept_{index}'] for index in range(n_layers)]
                activation = str(archive['activation'])
                scaler_mean = archive['scaler_mean']
                scaler_scale = archive['scaler_scale']
                signature = str(archive['feature_signature'])
                categories = (
                    [str(name) for name in archive['target_categories']]
                    if bool(archive['has_categories']) else None
                )
            if signature != self._feature_signature():
                raise ValueError(
                    "The feature definitions of this WEIRDO version differ from those the "
                    "model was trained with (feature signature mismatch); retrain the "
                    "model or install the WEIRDO version that created it."
                )
            self._model = _NumpyMLP(coefs, intercepts, activation)
            self._scaler = _NumpyScaler(scaler_mean, scaler_scale)
            self._target_categories = categories
            return

        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        trained_on = getattr(model_data['scaler'], 'n_features_in_', None)
        expected = len(self.get_feature_names())
        if trained_on is not None and int(trained_on) != expected:
            raise ValueError(
                f"This model was trained on {int(trained_on)} features, but this version "
                f"of WEIRDO builds {expected} for its configuration "
                f"(use_dipeptides={self._params['use_dipeptides']}, "
                f"positional={self._params['positional']}, k={self.k}). Models saved by "
                "development versions with a different feature layout cannot be used "
                "to score peptides; retrain the model (see MLPScorer.train)."
            )
        self._model = model_data['model']
        self._scaler = model_data['scaler']
        self._target_categories = model_data.get('target_categories')

    def get_feature_names(self) -> List[str]:
        """Get names of all features used by the model.

        Returns
        -------
        names : list of str
            Feature names, in column order.
        """
        return _feature_names(
            use_dipeptides=self._params.get('use_dipeptides', True),
            positional=self._params.get('positional', False),
            k=self.k,
        )
