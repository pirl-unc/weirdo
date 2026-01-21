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
)

# Data management
from .data_manager import (
    DataManager,
    get_data_manager,
    ensure_data_available,
)

# Scorer classes (for advanced usage)
from .scorers import (
    BaseScorer,
    BatchScorer,
    BaseReference,
    StreamingReference,
    FrequencyScorer,
    SimilarityScorer,
    SwissProtReference,
    ScorerConfig,
    register_scorer,
    register_reference,
)

__version__ = "1.1.0"

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
    # Scorer classes
    "BaseScorer",
    "BatchScorer",
    "BaseReference",
    "StreamingReference",
    "FrequencyScorer",
    "SimilarityScorer",
    "SwissProtReference",
    "ScorerConfig",
    "register_scorer",
    "register_reference",
    # Data management
    "DataManager",
    "get_data_manager",
    "ensure_data_available",
]
