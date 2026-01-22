"""MLP-based foreignness scorer.

Neural network model for learning foreignness from labeled data.
Uses modern components: residual connections, layer normalization,
GELU activations, and optionally gated linear units.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .trainable import TrainableScorer
from .registry import register_scorer

# Lazy import torch to make it optional
_torch = None
_nn = None


def _ensure_torch():
    """Lazy import PyTorch."""
    global _torch, _nn
    if _torch is None:
        try:
            import torch
            import torch.nn as nn
            _torch = torch
            _nn = nn
        except ImportError:
            raise ImportError(
                "PyTorch is required for MLP scorers. "
                "Install with: pip install torch"
            )
    return _torch, _nn


def _build_mlp(
    vocab_size: int = 21,
    embedding_dim: int = 64,
    k: int = 8,
    hidden_dim: int = 256,
    num_layers: int = 3,
    dropout: float = 0.1,
):
    """Build a modern MLP model.

    Architecture:
    - Embedding layer for amino acids
    - Positional embeddings
    - Mean pooling over k-mer positions
    - Stack of residual blocks with LayerNorm + GELU + Dropout
    - Final output head
    """
    torch, nn = _ensure_torch()

    class ResidualBlock(nn.Module):
        """Pre-norm residual block with GELU activation."""

        def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
            super().__init__()
            self.norm = nn.LayerNorm(dim)
            self.fc1 = nn.Linear(dim, dim * expansion)
            self.fc2 = nn.Linear(dim * expansion, dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            residual = x
            x = self.norm(x)
            x = self.fc1(x)
            x = nn.functional.gelu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return residual + x

    class ForeignnessMLP(nn.Module):
        """MLP for predicting peptide foreignness scores."""

        def __init__(self):
            super().__init__()

            # Amino acid embeddings
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

            # Positional embeddings for k-mer positions
            self.pos_embedding = nn.Embedding(k, embedding_dim)

            # Project to hidden dimension
            self.input_proj = nn.Linear(embedding_dim, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim)

            # Residual blocks
            self.blocks = nn.ModuleList([
                ResidualBlock(hidden_dim, expansion=4, dropout=dropout)
                for _ in range(num_layers)
            ])

            # Output head
            self.output_norm = nn.LayerNorm(hidden_dim)
            self.output = nn.Linear(hidden_dim, 1)

            # Initialize weights
            self._init_weights()

        def _init_weights(self):
            """Initialize weights with small values for stable training."""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, x):
            """Forward pass.

            Args:
                x: (batch, k) tensor of amino acid indices

            Returns:
                (batch,) tensor of foreignness scores
            """
            batch_size, seq_len = x.shape

            # Token + position embeddings
            positions = _torch.arange(seq_len, device=x.device).unsqueeze(0)
            x = self.embedding(x) + self.pos_embedding(positions)

            # Mean pool over positions
            x = x.mean(dim=1)

            # Project and normalize
            x = self.input_proj(x)
            x = self.input_norm(x)

            # Apply residual blocks
            for block in self.blocks:
                x = block(x)

            # Output
            x = self.output_norm(x)
            x = self.output(x)

            return x.squeeze(-1)

    return ForeignnessMLP()


@register_scorer('mlp', description='MLP foreignness scorer with modern architecture')
class MLPScorer(TrainableScorer):
    """MLP-based foreignness scorer.

    Uses learned amino acid embeddings and a modern MLP architecture
    with residual connections, layer normalization, and GELU activations
    to predict foreignness scores from peptide sequences.

    Parameters
    ----------
    k : int, default=8
        K-mer size for decomposing peptides.
    embedding_dim : int, default=64
        Dimension of amino acid embeddings.
    hidden_dim : int, default=256
        Hidden dimension for MLP layers.
    num_layers : int, default=3
        Number of residual blocks.
    dropout : float, default=0.1
        Dropout rate for regularization.
    batch_size : int, default=256
        Batch size for training and inference.
    aggregate : str, default='mean'
        How to aggregate k-mer scores: 'mean', 'max', 'min'.

    Example
    -------
    >>> from weirdo.scorers import MLPScorer
    >>>
    >>> # Create and train
    >>> scorer = MLPScorer(hidden_dim=256, num_layers=3)
    >>> scorer.train(peptides, labels, epochs=100)
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
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
        batch_size: int = 256,
        aggregate: str = 'mean',
        **kwargs
    ):
        super().__init__(k=k, batch_size=batch_size, **kwargs)
        self._params.update({
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'aggregate': aggregate,
        })
        self._model = None
        self._device = None

    def _build_model(self):
        """Build the model architecture."""
        torch, nn = _ensure_torch()
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = _build_mlp(
            vocab_size=21,
            embedding_dim=self._params['embedding_dim'],
            k=self.k,
            hidden_dim=self._params['hidden_dim'],
            num_layers=self._params['num_layers'],
            dropout=self._params['dropout'],
        ).to(self._device)

    def train(
        self,
        peptides: Sequence[str],
        labels: Sequence[float],
        val_peptides: Optional[Sequence[str]] = None,
        val_labels: Optional[Sequence[float]] = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        patience: int = 15,
        warmup_epochs: int = 5,
        verbose: bool = True,
    ) -> 'MLPScorer':
        """Train the MLP on labeled peptide data.

        Uses AdamW optimizer with cosine annealing learning rate schedule
        and optional warmup.

        Parameters
        ----------
        peptides : sequence of str
            Training peptide sequences.
        labels : sequence of float
            Target foreignness scores.
        val_peptides : sequence of str, optional
            Validation peptides for early stopping.
        val_labels : sequence of float, optional
            Validation labels.
        epochs : int, default=100
            Maximum training epochs.
        learning_rate : float, default=1e-3
            Peak learning rate.
        weight_decay : float, default=0.01
            L2 regularization strength.
        patience : int, default=15
            Early stopping patience (epochs without improvement).
        warmup_epochs : int, default=5
            Number of warmup epochs for learning rate.
        verbose : bool, default=True
            Print training progress.

        Returns
        -------
        self : MLPScorer
        """
        torch, nn = _ensure_torch()

        # Build model
        self._build_model()

        # Prepare data
        X_train = self._prepare_data(peptides)
        y_train = torch.tensor(labels, dtype=torch.float32, device=self._device)

        if val_peptides is not None and val_labels is not None:
            X_val = self._prepare_data(val_peptides)
            y_val = torch.tensor(val_labels, dtype=torch.float32, device=self._device)
            use_validation = True
        else:
            use_validation = False

        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )

        # Cosine annealing with warmup
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.1 + 0.9 * (1 + np.cos(np.pi * progress)) / 2

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        self._training_history = []

        for epoch in range(epochs):
            # Training phase
            self._model.train()
            total_loss = 0.0
            n_batches = 0

            # Shuffle data
            perm = torch.randperm(len(X_train), device=self._device)
            X_train_shuffled = X_train[perm]
            y_train_shuffled = y_train[perm]

            for i in range(0, len(X_train), self._params['batch_size']):
                batch_X = X_train_shuffled[i:i+self._params['batch_size']]
                batch_y = y_train_shuffled[i:i+self._params['batch_size']]

                optimizer.zero_grad()
                pred = self._model(batch_X)
                loss = criterion(pred, batch_y)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)

                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_loss = total_loss / max(1, n_batches)

            # Validation phase
            if use_validation:
                self._model.eval()
                with torch.no_grad():
                    val_pred = self._model(X_val)
                    val_loss = criterion(val_pred, y_val).item()

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
                else:
                    patience_counter += 1

                self._training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'lr': scheduler.get_last_lr()[0],
                })

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val: {val_loss:.4f}")

                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                self._training_history.append({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'lr': scheduler.get_last_lr()[0],
                })

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}")

        # Restore best model if we used validation
        if best_state is not None:
            self._model.load_state_dict({k: v.to(self._device) for k, v in best_state.items()})

        self._is_trained = True
        self._is_fitted = True

        # Save training metadata
        self._metadata['n_train'] = len(peptides)
        self._metadata['n_epochs'] = len(self._training_history)
        self._metadata['final_train_loss'] = self._training_history[-1]['train_loss']
        if use_validation:
            self._metadata['best_val_loss'] = best_val_loss

        return self

    def _prepare_data(self, peptides: Sequence[str]) -> 'torch.Tensor':
        """Convert peptides to tensor of k-mer indices."""
        torch, _ = _ensure_torch()

        all_kmers = []
        for peptide in peptides:
            kmer_indices = self._peptide_to_indices(peptide)
            if kmer_indices:
                # Use first k-mer for each peptide
                all_kmers.append(kmer_indices[0])
            else:
                # Padding for short peptides
                all_kmers.append([20] * self.k)

        return torch.tensor(all_kmers, dtype=torch.long, device=self._device)

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

        torch, _ = _ensure_torch()
        peptides = self._ensure_list(peptides)

        aggregate = self._params['aggregate']
        scores = []

        self._model.eval()
        with torch.no_grad():
            for peptide in peptides:
                kmer_indices = self._peptide_to_indices(peptide)
                if not kmer_indices:
                    scores.append(float('inf'))
                    continue

                # Score all k-mers
                X = torch.tensor(kmer_indices, dtype=torch.long, device=self._device)
                kmer_scores = self._model(X).cpu().numpy()

                # Aggregate k-mer scores
                if aggregate == 'mean':
                    scores.append(float(np.mean(kmer_scores)))
                elif aggregate == 'max':
                    scores.append(float(np.max(kmer_scores)))
                elif aggregate == 'min':
                    scores.append(float(np.min(kmer_scores)))
                else:
                    scores.append(float(np.mean(kmer_scores)))

        return np.array(scores)

    def _score_batch_impl(self, batch: List[str]) -> np.ndarray:
        """Score a batch of peptides."""
        return self.score(batch)

    def _save_model(self, path: Path) -> None:
        """Save model weights."""
        torch, _ = _ensure_torch()
        torch.save(self._model.state_dict(), path)

    def _load_model(self, path: Path) -> None:
        """Load model weights."""
        torch, _ = _ensure_torch()
        self._build_model()
        self._model.load_state_dict(torch.load(path, map_location=self._device, weights_only=True))
        self._model.eval()
