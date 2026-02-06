"""Model manager for trained foreignness scorers.

Handles saving, loading, and listing trained ML models.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Default model storage directory
DEFAULT_MODEL_DIR = Path.home() / '.weirdo' / 'models'


class ModelInfo:
    """Information about a saved model."""

    def __init__(
        self,
        name: str,
        scorer_type: str,
        path: Path,
        created: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.scorer_type = scorer_type
        self.path = path
        self.created = created
        self.params = params or {}
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"ModelInfo(name='{self.name}', type='{self.scorer_type}')"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'scorer_type': self.scorer_type,
            'path': str(self.path),
            'created': self.created,
            'params': self.params,
            'metadata': self.metadata,
        }


class ModelManager:
    """Manager for trained foreignness scoring models.

    Handles saving, loading, and listing trained models in a
    centralized directory structure.

    Parameters
    ----------
    model_dir : str or Path, optional
        Directory for storing models. Defaults to ~/.weirdo/models.

    Example
    -------
    >>> mm = ModelManager()
    >>> mm.list_models()  # List available models
    >>> model = mm.load('my-mlp')  # Load a saved model
    >>> mm.save(scorer, 'my-new-model')  # Save a trained model
    """

    def __init__(self, model_dir: Optional[Union[str, Path]] = None):
        self._model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self._model_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_dir(self) -> Path:
        """Get model storage directory."""
        return self._model_dir

    def list_models(self) -> List[ModelInfo]:
        """List all available models.

        Returns
        -------
        models : list of ModelInfo
            Information about each saved model.
        """
        models = []

        if not self._model_dir.exists():
            return models

        for path in sorted(self._model_dir.iterdir()):
            if path.is_dir() and (path / 'config.json').exists():
                try:
                    info = self._load_model_info(path)
                    models.append(info)
                except Exception:
                    # Skip corrupted models
                    pass

        return models

    def _load_model_info(self, path: Path) -> ModelInfo:
        """Load model info from a model directory."""
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)

        metadata = {}
        if (path / 'metadata.json').exists():
            with open(path / 'metadata.json', 'r') as f:
                metadata = json.load(f)

        # Get creation time
        created = None
        stat = path.stat()
        created = datetime.fromtimestamp(stat.st_mtime).isoformat()

        return ModelInfo(
            name=path.name,
            scorer_type=config.get('scorer_type', 'unknown'),
            path=path,
            created=created,
            params=config.get('params', {}),
            metadata=metadata,
        )

    def get_model_info(self, name: str) -> Optional[ModelInfo]:
        """Get info for a specific model.

        Parameters
        ----------
        name : str
            Model name.

        Returns
        -------
        info : ModelInfo or None
            Model info if found, None otherwise.
        """
        path = self._model_dir / name
        if path.exists() and (path / 'config.json').exists():
            return self._load_model_info(path)
        return None

    def load(self, name: str) -> 'TrainableScorer':
        """Load a trained model by name.

        Parameters
        ----------
        name : str
            Model name (directory name in model storage).

        Returns
        -------
        scorer : TrainableScorer
            Loaded model ready for inference.
        """
        from .scorers.trainable import TrainableScorer
        from .scorers.mlp import MLPScorer

        path = self._model_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {name}")

        # Load config to determine scorer type
        with open(path / 'config.json', 'r') as f:
            config = json.load(f)

        scorer_type = config.get('scorer_type', '')

        # Map to scorer class
        scorer_classes = {
            'MLPScorer': MLPScorer,
        }

        if scorer_type not in scorer_classes:
            raise ValueError(
                f"Unknown scorer type: {scorer_type}. "
                f"Available: {list(scorer_classes.keys())}"
            )

        return scorer_classes[scorer_type].load(path)

    def save(
        self,
        scorer: 'TrainableScorer',
        name: str,
        overwrite: bool = False,
    ) -> Path:
        """Save a trained model.

        Parameters
        ----------
        scorer : TrainableScorer
            Trained model to save.
        name : str
            Name for the saved model.
        overwrite : bool, default=False
            Whether to overwrite existing model with same name.

        Returns
        -------
        path : Path
            Path where model was saved.
        """
        path = self._model_dir / name

        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Model already exists: {name}. "
                "Use overwrite=True to replace."
            )

        scorer.save(path)
        return path

    def delete(self, name: str) -> bool:
        """Delete a saved model.

        Parameters
        ----------
        name : str
            Model name to delete.

        Returns
        -------
        deleted : bool
            True if model was deleted.
        """
        import shutil
        path = self._model_dir / name

        if path.exists():
            shutil.rmtree(path)
            return True
        return False

    def print_models(self) -> None:
        """Print formatted list of available models."""
        models = self.list_models()

        if not models:
            print("No trained models found.")
            print(f"Model directory: {self._model_dir}")
            return

        print(f"Trained models ({len(models)}):")
        print("-" * 60)

        for model in models:
            print(f"\n  {model.name}")
            print(f"    Type: {model.scorer_type}")
            if model.created:
                print(f"    Created: {model.created[:19]}")
            if 'n_train' in model.metadata:
                print(f"    Training samples: {model.metadata['n_train']}")
            if 'n_epochs' in model.metadata:
                print(f"    Epochs trained: {model.metadata['n_epochs']}")
            if 'final_train_loss' in model.metadata:
                print(f"    Final train loss: {model.metadata['final_train_loss']:.4f}")
            if 'final_val_loss' in model.metadata:
                print(f"    Final val loss: {model.metadata['final_val_loss']:.4f}")
            if 'best_val_loss' in model.metadata:
                print(f"    Best val loss: {model.metadata['best_val_loss']:.4f}")
            if 'best_val_score' in model.metadata:
                print(f"    Best val score: {model.metadata['best_val_score']:.4f}")
            if 'k' in model.params:
                print(f"    K-mer size: {model.params['k']}")

        print(f"\nModel directory: {self._model_dir}")


# Singleton instance
_model_manager: Optional[ModelManager] = None


def get_model_manager(model_dir: Optional[Union[str, Path]] = None) -> ModelManager:
    """Get the model manager instance.

    Parameters
    ----------
    model_dir : str or Path, optional
        Custom model directory.

    Returns
    -------
    manager : ModelManager
    """
    global _model_manager
    if _model_manager is None or model_dir is not None:
        _model_manager = ModelManager(model_dir)
    return _model_manager


def list_models(model_dir: Optional[Union[str, Path]] = None) -> List[ModelInfo]:
    """List all available trained models.

    Parameters
    ----------
    model_dir : str or Path, optional
        Custom model directory.

    Returns
    -------
    models : list of ModelInfo
    """
    return get_model_manager(model_dir).list_models()


def load_model(name: str, model_dir: Optional[Union[str, Path]] = None) -> 'TrainableScorer':
    """Load a trained model by name.

    Parameters
    ----------
    name : str
        Model name.
    model_dir : str or Path, optional
        Custom model directory.

    Returns
    -------
    scorer : TrainableScorer
        Loaded model.
    """
    return get_model_manager(model_dir).load(name)


def save_model(
    scorer: 'TrainableScorer',
    name: str,
    model_dir: Optional[Union[str, Path]] = None,
    overwrite: bool = False,
) -> Path:
    """Save a trained model.

    Parameters
    ----------
    scorer : TrainableScorer
        Trained model.
    name : str
        Model name.
    model_dir : str or Path, optional
        Custom model directory.
    overwrite : bool, default=False
        Overwrite existing model.

    Returns
    -------
    path : Path
    """
    return get_model_manager(model_dir).save(scorer, name, overwrite)
