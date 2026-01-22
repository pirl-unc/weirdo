"""MLP-based foreignness scorer.

Neural network model for learning foreignness from labeled data.
Uses scikit-learn's MLPRegressor for simplicity.
"""

import pickle
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .trainable import TrainableScorer
from .registry import register_scorer


# Amino acid to index mapping
AA_TO_IDX = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,  # Unknown/padding
}
NUM_AMINO_ACIDS = 21


def _kmer_to_onehot(kmer: str) -> np.ndarray:
    """Convert a k-mer to one-hot encoding.

    Returns flattened array of shape (k * 21,).
    """
    k = len(kmer)
    onehot = np.zeros(k * NUM_AMINO_ACIDS, dtype=np.float32)
    for i, aa in enumerate(kmer):
        idx = AA_TO_IDX.get(aa, 20)
        onehot[i * NUM_AMINO_ACIDS + idx] = 1.0
    return onehot


def _peptide_to_features(peptide: str, k: int) -> np.ndarray:
    """Convert peptide to feature vector by averaging k-mer one-hot encodings."""
    if len(peptide) < k:
        peptide = peptide + 'X' * (k - len(peptide))

    # Extract k-mers
    kmers = [peptide[i:i+k] for i in range(len(peptide) - k + 1)]

    # Average one-hot encodings
    features = np.mean([_kmer_to_onehot(kmer) for kmer in kmers], axis=0)
    return features


@register_scorer('mlp', description='MLP foreignness scorer using scikit-learn')
class MLPScorer(TrainableScorer):
    """MLP-based foreignness scorer using scikit-learn.

    Uses one-hot encoded amino acid features and a multi-layer perceptron
    to predict foreignness scores from peptide sequences.

    Parameters
    ----------
    k : int, default=8
        K-mer size for decomposing peptides.
    hidden_layer_sizes : tuple of int, default=(256, 128, 64)
        Sizes of hidden layers.
    activation : str, default='relu'
        Activation function: 'relu', 'tanh', 'logistic'.
    alpha : float, default=0.0001
        L2 regularization strength.
    learning_rate_init : float, default=0.001
        Initial learning rate.
    max_iter : int, default=200
        Maximum training iterations.
    early_stopping : bool, default=True
        Use early stopping with validation split.
    batch_size : int, default=256
        Batch size for training.
    aggregate : str, default='mean'
        How to aggregate k-mer scores: 'mean', 'max', 'min'.

    Example
    -------
    >>> from weirdo.scorers import MLPScorer
    >>>
    >>> # Create and train
    >>> scorer = MLPScorer(hidden_layer_sizes=(256, 128))
    >>> scorer.train(peptides, labels)
    >>>
    >>> # Score new peptides
    >>> scores = scorer.score(['MTMDKSEL', 'XXXXXXXX'])
    >>>
    >>> # Save and load
    >>> scorer.save('my_model')
    >>> loaded = MLPScorer.load('my_model')
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
        batch_size: int = 256,
        aggregate: str = 'mean',
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
            'aggregate': aggregate,
            'random_state': random_state,
        })
        self._model: Optional[MLPRegressor] = None
        self._scaler: Optional[StandardScaler] = None

    def train(
        self,
        peptides: Sequence[str],
        labels: Sequence[float],
        val_peptides: Optional[Sequence[str]] = None,
        val_labels: Optional[Sequence[float]] = None,
        epochs: int = 200,
        learning_rate: float = 0.001,
        verbose: bool = True,
        **kwargs
    ) -> 'MLPScorer':
        """Train the MLP on labeled peptide data.

        Parameters
        ----------
        peptides : sequence of str
            Training peptide sequences.
        labels : sequence of float
            Target foreignness scores.
        val_peptides : sequence of str, optional
            Not used (sklearn handles validation internally).
        val_labels : sequence of float, optional
            Not used.
        epochs : int, default=200
            Maximum training iterations (maps to max_iter).
        learning_rate : float, default=0.001
            Initial learning rate.
        verbose : bool, default=True
            Print training progress.

        Returns
        -------
        self : MLPScorer
        """
        # Convert peptides to features
        X = np.array([_peptide_to_features(p, self.k) for p in peptides])
        y = np.array(labels)

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Create and train model
        self._model = MLPRegressor(
            hidden_layer_sizes=self._params['hidden_layer_sizes'],
            activation=self._params['activation'],
            alpha=self._params['alpha'],
            learning_rate_init=learning_rate,
            max_iter=epochs,
            early_stopping=self._params['early_stopping'],
            validation_fraction=0.1 if self._params['early_stopping'] else 0.0,
            n_iter_no_change=10,
            random_state=self._params['random_state'],
            verbose=verbose,
        )

        self._model.fit(X_scaled, y)

        self._is_trained = True
        self._is_fitted = True

        # Save training metadata
        self._metadata['n_train'] = len(peptides)
        self._metadata['n_iter'] = self._model.n_iter_
        self._metadata['loss'] = float(self._model.loss_)
        if hasattr(self._model, 'best_loss_') and self._model.best_loss_ is not None:
            self._metadata['best_loss'] = float(self._model.best_loss_)

        self._training_history = [
            {'epoch': i + 1, 'loss': loss}
            for i, loss in enumerate(self._model.loss_curve_)
        ]

        return self

    def score(self, peptides: Union[str, Sequence[str]]) -> np.ndarray:
        """Score peptides for foreignness.

        Parameters
        ----------
        peptides : str or sequence of str
            Peptide(s) to score.

        Returns
        -------
        scores : np.ndarray
            Foreignness scores (higher = more foreign).
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before scoring.")

        peptides = self._ensure_list(peptides)

        # Convert to features
        X = np.array([_peptide_to_features(p, self.k) for p in peptides])
        X_scaled = self._scaler.transform(X)

        # Predict
        scores = self._model.predict(X_scaled)

        return np.array(scores)

    def _score_batch_impl(self, batch: List[str]) -> np.ndarray:
        """Score a batch of peptides."""
        return self.score(batch)

    def _save_model(self, path: Path) -> None:
        """Save model weights."""
        model_data = {
            'model': self._model,
            'scaler': self._scaler,
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

    def _load_model(self, path: Path) -> None:
        """Load model weights."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self._model = model_data['model']
        self._scaler = model_data['scaler']
