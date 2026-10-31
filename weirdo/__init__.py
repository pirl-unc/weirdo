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
from .peptide_content import MICROBIAL_CATEGORIES, summarize_peptide_content
from .score_policies import (
    DEFAULT_PEPTIDE_CONTENT_SCORE_POLICY,
    ScorePolicy,
    apply_score_policy,
    get_score_policy,
    list_score_policies,
)
from .peptide_content_models import (
    STANDARD_SCORE_CATEGORIES,
    STANDARD_SCORE_NAMES,
    TorchPeptideContentModel,
    geometric_mean_scores,
    load_peptide_content_model,
    predict_taxonomy_scores,
    presentation_geometric_mean_scores,
    select_best_peptide_content_model,
    standard_peptide_content_scores,
)

# Rank-normalized pMHC + Weirdo score
from .pmhc_score import pmhc_weirdo_score
from .reference_proteomes import (
    REFERENCE_PROTEOMES,
    DownloadedReferenceProteome,
    ReferenceProteome,
    ReferenceProteomeManifest,
    ReferenceProteomeManifestEntry,
    fetch_reference_proteome,
    get_reference_proteome,
    iter_fasta_sequences,
    list_reference_proteome_manifests,
    list_reference_proteomes,
    load_reference_proteome_manifest,
    reference_proteome_manifest_path,
    reference_proteome_from_uniprot,
    reference_proteome_metadata_url,
)

# High-level scoring API
from .api import (
    score_peptide,
    score_peptides,
    predict_categories,
    predict_peptide_content,
    create_scorer,
    clear_cache,
    get_available_presets,
    get_preset_info,
    # Model management
    list_models,
    list_bundled_models,
    load_model,
    save_model,
    list_pretrained_models,
    download_pretrained_model,
    download_model_from_url,
    get_available_scorers,
)

from .released import (
    DEFAULT_PRETRAINED_MODEL, DEFAULT_SCORE_POLICY, PretrainedModel,
    get_model_manifest, load_pretrained_model,
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

__version__ = "2.5.0"

__all__ = [
    "pmhc_weirdo_score",
    "STANDARD_SCORE_CATEGORIES",
    "STANDARD_SCORE_NAMES",
    "TorchPeptideContentModel",
    "select_best_peptide_content_model",
    "geometric_mean_scores",
    "load_peptide_content_model",
    "predict_taxonomy_scores",
    "presentation_geometric_mean_scores",
    "standard_peptide_content_scores",
    "DEFAULT_PRETRAINED_MODEL", "DEFAULT_SCORE_POLICY", "PretrainedModel",
    "DEFAULT_PEPTIDE_CONTENT_SCORE_POLICY", "ScorePolicy",
    "apply_score_policy", "get_score_policy", "list_score_policies",
    "get_model_manifest", "load_pretrained_model",
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
    "MICROBIAL_CATEGORIES",
    "summarize_peptide_content",
    "ReferenceProteome",
    "ReferenceProteomeManifest",
    "ReferenceProteomeManifestEntry",
    "DownloadedReferenceProteome",
    "REFERENCE_PROTEOMES",
    "list_reference_proteomes",
    "list_reference_proteome_manifests",
    "load_reference_proteome_manifest",
    "reference_proteome_manifest_path",
    "get_reference_proteome",
    "reference_proteome_from_uniprot",
    "reference_proteome_metadata_url",
    "fetch_reference_proteome",
    "iter_fasta_sequences",
    # High-level scoring API
    "score_peptide",
    "score_peptides",
    "predict_categories",
    "predict_peptide_content",
    "create_scorer",
    "clear_cache",
    "get_available_presets",
    "get_preset_info",
    "get_available_scorers",
    # Model management
    "list_models",
    "list_bundled_models",
    "load_model",
    "save_model",
    "list_pretrained_models",
    "download_pretrained_model",
    "download_model_from_url",
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
