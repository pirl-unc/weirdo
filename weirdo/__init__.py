from .amino_acid_alphabet import (
    AminoAcid,
    canonical_amino_acids,
    canonical_amino_acid_letters,
    extended_amino_acids,
    extended_amino_acid_letters,
    amino_acid_letter_indices,
    amino_acid_name_indices,
)
from .peptide_vectorizer import PeptideVectorizer
from .distances import hamming

# High-level scoring API
from .api import (
    score_peptide,
    score_peptides,
    create_scorer,
    clear_cache,
    get_available_presets,
    get_preset_info,
    # Model management
    list_models,
    load_model,
    save_model,
    get_available_scorers,
)

# Data management
from .data_manager import (
    DataManager,
    get_data_manager,
    ensure_data_available,
)

# Model management
from .model_manager import (
    ModelManager,
    get_model_manager,
    ModelInfo,
)

# Scorer classes (for advanced usage)
from .scorers import (
    BaseScorer,
    BatchScorer,
    BaseReference,
    StreamingReference,
    TrainableScorer,
    SwissProtReference,
    ScorerConfig,
    register_scorer,
    register_reference,
)

# ML scorer
from .scorers import MLPScorer

__version__ = "2.1.1"

__all__ = [
    # Amino acid data
    "AminoAcid",
    "canonical_amino_acids",
    "canonical_amino_acid_letters",
    "extended_amino_acids",
    "extended_amino_acid_letters",
    "amino_acid_letter_indices",
    "amino_acid_name_indices",
    # Vectorization
    "PeptideVectorizer",
    # Distances
    "hamming",
    # High-level scoring API
    "score_peptide",
    "score_peptides",
    "create_scorer",
    "clear_cache",
    "get_available_presets",
    "get_preset_info",
    "get_available_scorers",
    # Model management
    "list_models",
    "load_model",
    "save_model",
    "ModelManager",
    "get_model_manager",
    "ModelInfo",
    # Scorer classes
    "BaseScorer",
    "BatchScorer",
    "BaseReference",
    "StreamingReference",
    "TrainableScorer",
    "SwissProtReference",
    "ScorerConfig",
    "register_scorer",
    "register_reference",
    # ML scorer
    "MLPScorer",
    # Data management
    "DataManager",
    "get_data_manager",
    "ensure_data_available",
]
