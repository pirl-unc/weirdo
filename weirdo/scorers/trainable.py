"""Base class for trainable (ML-based) scorers.

Provides common infrastructure for training, saving, and loading models.
"""

import json
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .base import BatchScorer


class TrainableScorer(BatchScorer):
    """Base class for trainable foreignness scorers.

    Extends BatchScorer with training, saving, and loading capabilities.
    Subclasses implement specific model architectures (MLP, etc.).

    Parameters
    ----------
    k : int, default=8
        K-mer size for decomposing peptides.
    batch_size : int, default=256
        Batch size for training and inference.
    """

    def __init__(
        self,
        k: int = 8,
        batch_size: int = 256,
        **kwargs
    ):
        super().__init__(batch_size=batch_size, **kwargs)
        self._params.update({'k': k})
        self._model = None
        self._is_trained = False
        self._training_history: List[Dict[str, float]] = []
        self._metadata: Dict[str, Any] = {}

    @property
    def k(self) -> int:
        """Get k-mer size."""
        return self._params['k']

    @property
    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained

    @property
    def training_history(self) -> List[Dict[str, float]]:
        """Get training history (loss per epoch)."""
        return self._training_history.copy()

    @abstractmethod
    def train(
        self,
        peptides: Sequence[str],
        labels: Sequence[float],
        val_peptides: Optional[Sequence[str]] = None,
        val_labels: Optional[Sequence[float]] = None,
        epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        verbose: bool = True,
    ) -> 'TrainableScorer':
        """Train the model on labeled data.

        Parameters
        ----------
        peptides : sequence of str
            Training peptide sequences.
        labels : sequence of float or 2D array
            Target labels. For multi-label classification, use a 2D array.
        val_peptides : sequence of str, optional
            Validation peptides for early stopping.
        val_labels : sequence of float, optional
            Validation labels.
        epochs : int, optional
            Number of training epochs. Defaults to model's max_iter.
        learning_rate : float, optional
            Learning rate for optimizer. Defaults to model's learning_rate_init.
        verbose : bool, default=True
            Print training progress.

        Returns
        -------
        self : TrainableScorer
            Returns self for method chaining.
        """
        pass

    @abstractmethod
    def _save_model(self, path: Path) -> None:
        """Save model weights to path (implemented by subclass)."""
        pass

    @abstractmethod
    def _load_model(self, path: Path) -> None:
        """Load model weights from path (implemented by subclass)."""
        pass

    def save(self, path: Union[str, Path]) -> None:
        """Save trained model to disk.

        Creates a directory containing:
        - model.pt: Model weights
        - config.json: Model configuration
        - metadata.json: Training metadata

        Parameters
        ----------
        path : str or Path
            Directory path to save model.
        """
        if not self._is_trained:
            raise RuntimeError("Model must be trained before saving.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        self._save_model(path / 'model.pt')

        # Save configuration
        config = {
            'scorer_type': self.__class__.__name__,
            'params': self._params,
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)

        # Save metadata
        metadata = {
            'training_history': self._training_history,
            **self._metadata,
        }
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TrainableScorer':
        """Load a trained model from disk.

        Parameters
        ----------
        path : str or Path
            Directory path containing saved model.

        Returns
        -------
        scorer : TrainableScorer
            Loaded model ready for inference.
        """
        path = Path(path)

        # Load configuration
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)

        # Create instance with saved params
        instance = cls(**config['params'])

        # Load model weights
        instance._load_model(path / 'model.pt')
        instance._is_trained = True
        instance._is_fitted = True

        # Load metadata
        if (path / 'metadata.json').exists():
            with open(path / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            instance._training_history = metadata.get('training_history', [])
            instance._metadata = metadata

        return instance

    def fit(self, reference=None) -> 'TrainableScorer':
        """For compatibility with BaseScorer interface.

        Trainable scorers use train() instead of fit().
        If already trained, this is a no-op.
        """
        if self._is_trained:
            self._is_fitted = True
            return self
        raise RuntimeError(
            "Trainable scorers must be trained with train() or loaded with load(). "
            "Use scorer.train(peptides, labels) to train a new model."
        )

    def _extract_kmers(self, peptide: str) -> List[str]:
        """Extract overlapping k-mers from peptide."""
        k = self.k
        if len(peptide) < k:
            # Pad short peptides
            peptide = peptide + 'X' * (k - len(peptide))
        return [peptide[i:i+k] for i in range(len(peptide) - k + 1)]

    def _peptide_to_indices(self, peptide: str) -> List[List[int]]:
        """Convert peptide to list of k-mer amino acid indices.

        Returns list of k-mers, each as list of AA indices (0-20).
        """
        AA_TO_IDX = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
            'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
            'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
            'X': 20,  # Unknown/padding
        }
        kmers = self._extract_kmers(peptide)
        return [
            [AA_TO_IDX.get(aa, 20) for aa in kmer]
            for kmer in kmers
        ]
