"""Uniform inference and pre-response selection for peptide-content models."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


STANDARD_SCORE_CATEGORIES = (
    "viruses",
    "bacteria",
    "archaea",
    "fungi",
    "protists",
    "mammals",
    "human",
)

MICROBIAL_SCORE_CATEGORIES = (
    "viruses",
    "bacteria",
    "archaea",
    "fungi",
    "protists",
)

STANDARD_SCORE_NAMES = (
    "p_viruses",
    "p_bacteria",
    "microbial_like",
    "p_mammals",
    "one_minus_mammals",
    "one_minus_human",
    "virus_mammal_like",
    "virus_foreign_like",
    "microbial_foreign_like",
)


def geometric_mean_scores(*scores: np.ndarray) -> np.ndarray:
    """Element-wise geometric mean for one or more bounded ranking scores.

    This is the sole combination rule used for Weirdo score components and for
    combining a Weirdo score with MHCflurry presentation.  Inputs broadcast by
    NumPy's normal rules and must be finite values in ``[0, 1]``.
    """
    if not scores:
        raise ValueError("at least one score is required")
    arrays = np.broadcast_arrays(*(np.asarray(score, dtype=float) for score in scores))
    if any(not np.isfinite(array).all() for array in arrays):
        raise ValueError("scores must be finite values in [0, 1]")
    if any(((array < 0) | (array > 1)).any() for array in arrays):
        raise ValueError("scores must be finite values in [0, 1]")
    product = np.ones(arrays[0].shape, dtype=float)
    for array in arrays:
        product *= array
    return np.power(product, 1.0 / len(arrays))


def presentation_geometric_mean_scores(
    presentation_score: np.ndarray,
    weirdo_scores: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Combine MHCflurry presentation with each named Weirdo score.

    No labels are used and no adapter is fitted.  Returned names are stable and
    make the combination explicit.
    """
    return {
        f"presentation_geomean__{name}": geometric_mean_scores(
            presentation_score, score
        )
        for name, score in weirdo_scores.items()
    }


def standard_peptide_content_scores(
    probabilities: np.ndarray,
    categories: Sequence[str],
    *,
    required_scores: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Return the fixed Weirdo score family used by response benchmarks.

    These are deterministic transforms of frozen taxonomy heads. No response
    model is involved here. Category heads are independent sigmoids and the
    derived values are ranking scores rather than biological probabilities.
    """
    values = np.asarray(probabilities, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    names = tuple(categories)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("probability columns must match categories")
    if len(set(names)) != len(names):
        raise ValueError("categories must be unique")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    index = {name: position for position, name in enumerate(names)}
    result: dict[str, np.ndarray] = {}
    if "viruses" in index:
        result["p_viruses"] = values[:, index["viruses"]]
    if "bacteria" in index:
        result["p_bacteria"] = values[:, index["bacteria"]]
    if set(MICROBIAL_SCORE_CATEGORIES) <= set(index):
        result["microbial_like"] = values[
            :, [index[name] for name in MICROBIAL_SCORE_CATEGORIES]
        ].max(axis=1)
    if "mammals" in index:
        result["p_mammals"] = values[:, index["mammals"]]
        result["one_minus_mammals"] = 1.0 - result["p_mammals"]
    if "human" in index:
        result["one_minus_human"] = 1.0 - values[:, index["human"]]
    if {"p_viruses", "p_mammals"} <= set(result):
        result["virus_mammal_like"] = geometric_mean_scores(
            result["p_viruses"], result["p_mammals"]
        )
    if {"p_viruses", "one_minus_mammals"} <= set(result):
        result["virus_foreign_like"] = geometric_mean_scores(
            result["p_viruses"], result["one_minus_mammals"]
        )
    if {"microbial_like", "one_minus_mammals"} <= set(result):
        result["microbial_foreign_like"] = geometric_mean_scores(
            result["microbial_like"], result["one_minus_mammals"]
        )
    result = {name: result[name] for name in STANDARD_SCORE_NAMES if name in result}
    if required_scores is not None:
        unknown = sorted(set(required_scores) - set(STANDARD_SCORE_NAMES))
        if unknown:
            raise ValueError(f"unknown standard Weirdo scores: {unknown}")
        missing = [name for name in required_scores if name not in result]
        if missing:
            raise ValueError(
                f"model categories cannot produce requested Weirdo scores: {missing}"
            )
        result = {name: result[name] for name in required_scores}
    return result


def _masked_pool(values, mask):
    import torch

    weights = mask.unsqueeze(-1).to(values.dtype)
    mean = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
    maximum = values.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(dim=1).values
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=1)


def _encode_peptides(peptides: list[str], max_length: int = 15):
    import torch

    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    indices = {amino_acid: index + 1 for index, amino_acid in enumerate(alphabet)}
    tokens = torch.zeros((len(peptides), max_length), dtype=torch.long)
    mask = torch.zeros((len(peptides), max_length), dtype=torch.bool)
    for row, peptide in enumerate(peptides):
        length = min(len(peptide), max_length)
        tokens[row, :length] = torch.tensor([indices.get(aa, 0) for aa in peptide[:length]])
        mask[row, :length] = True
    return tokens, mask


def _logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(values / (1.0 - values))


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _build_torch_model(architecture: str, n_outputs: int, config: dict[str, Any], model_dir: Path):
    import torch
    from torch import nn

    if architecture == "cnn":
        class CNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(21, 48, padding_idx=0)
                self.convolutions = nn.ModuleList(
                    nn.Conv1d(48, 64, kernel_size=kernel, padding=kernel // 2)
                    for kernel in (3, 5, 7)
                )
                self.head = nn.Sequential(
                    nn.Linear(64 * 3 * 2 + 1, 192), nn.GELU(), nn.Dropout(0.15), nn.Linear(192, n_outputs)
                )

            def forward(self, tokens, mask):
                embedded = self.embedding(tokens).transpose(1, 2)
                pooled = [
                    _masked_pool(torch.nn.functional.gelu(layer(embedded)).transpose(1, 2), mask)
                    for layer in self.convolutions
                ]
                length = mask.sum(dim=1, keepdim=True).to(embedded.dtype) / 15.0
                return self.head(torch.cat((*pooled, length), dim=1))

        return CNN(), None

    if architecture == "transformer":
        class Transformer(nn.Module):
            def __init__(self):
                super().__init__()
                width = 96
                self.embedding = nn.Embedding(21, width, padding_idx=0)
                self.position = nn.Parameter(torch.zeros(1, 15, width))
                nn.init.normal_(self.position, std=0.02)
                layer = nn.TransformerEncoderLayer(
                    d_model=width, nhead=4, dim_feedforward=256, dropout=0.1,
                    activation="gelu", batch_first=True, norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=3)
                self.head = nn.Sequential(
                    nn.LayerNorm(width * 2 + 1), nn.Linear(width * 2 + 1, 192),
                    nn.GELU(), nn.Dropout(0.15), nn.Linear(192, n_outputs),
                )

            def forward(self, tokens, mask):
                values = self.embedding(tokens) + self.position[:, : tokens.shape[1]]
                values = self.encoder(values, src_key_padding_mask=~mask)
                pooled = _masked_pool(values, mask)
                length = mask.sum(dim=1, keepdim=True).to(values.dtype) / 15.0
                return self.head(torch.cat((pooled, length), dim=1))

        return Transformer(), None

    if architecture in ("esm_frozen", "esm_finetune"):
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
        encoder_config = AutoConfig.from_pretrained(model_dir / "encoder_config")

        class ESMHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = AutoModel.from_config(encoder_config)
                width = int(self.encoder.config.hidden_size)
                self.head = nn.Sequential(
                    nn.LayerNorm(width * 2 + 1), nn.Linear(width * 2 + 1, 256),
                    nn.GELU(), nn.Dropout(0.15), nn.Linear(256, n_outputs),
                )
                self.freeze_encoder = architecture == "esm_frozen"
                if self.freeze_encoder:
                    for parameter in self.encoder.parameters():
                        parameter.requires_grad = False

            def forward(self, input_ids, attention_mask, special_tokens_mask):
                values = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
                residue_mask = attention_mask.bool() & ~special_tokens_mask.bool()
                pooled = _masked_pool(values, residue_mask)
                length = residue_mask.sum(dim=1, keepdim=True).to(values.dtype) / 15.0
                return self.head(torch.cat((pooled, length), dim=1))

        return ESMHead(), tokenizer
    raise ValueError(f"Unknown peptide-content architecture: {architecture!r}")


@dataclass
class TorchPeptideContentModel:
    """A common ``predict_proba`` interface for CNN, Transformer, and ESM artifacts."""

    architecture: str
    target_categories: tuple[str, ...]
    model: Any
    tokenizer: Any
    device: Any
    batch_size: int
    model_dir: Path

    def predict_proba(self, peptides: Iterable[str]) -> np.ndarray:
        import torch

        peptides = list(peptides)
        invalid = [peptide for peptide in peptides if not 8 <= len(peptide) <= 15]
        if invalid:
            raise ValueError("peptide-content models support complete peptides of length 8--15")
        self.model.eval()
        probabilities = []
        with torch.no_grad():
            for start in range(0, len(peptides), self.batch_size):
                batch = peptides[start : start + self.batch_size]
                if self.architecture.startswith("esm"):
                    encoded = self.tokenizer(
                        batch, padding=True, return_tensors="pt", return_special_tokens_mask=True
                    )
                    features = {
                        key: encoded[key].to(self.device)
                        for key in ("input_ids", "attention_mask", "special_tokens_mask")
                    }
                else:
                    tokens, mask = _encode_peptides(batch)
                    features = {"tokens": tokens.to(self.device), "mask": mask.to(self.device)}
                probabilities.append(torch.sigmoid(self.model(**features)).cpu().numpy())
        return np.vstack(probabilities) if probabilities else np.empty((0, len(self.target_categories)))


@dataclass
class StackedPeptideContentModel:
    """Calibrated stack of peptide-only taxonomy models.

    The stack consumes only member logits and peptide length, so it does not add
    source context or MHC information that would be unavailable at inference.
    Coefficients are fitted on held-out validation protein families and frozen in
    the artifact.
    """

    target_categories: tuple[str, ...]
    member_names: tuple[str, ...]
    members: tuple[Any, ...]
    coefficients: np.ndarray
    intercepts: np.ndarray
    length_center: float
    length_scale: float
    model_dir: Path

    def __post_init__(self) -> None:
        expected = (len(self.target_categories), len(self.members) + 1)
        if self.coefficients.shape != expected:
            raise ValueError(
                f"Stack coefficients have shape {self.coefficients.shape}; expected {expected}"
            )
        if self.intercepts.shape != (len(self.target_categories),):
            raise ValueError("Stack intercepts must have one value per category")
        if len(self.member_names) != len(self.members):
            raise ValueError("Stack member names and models must have equal length")
        if not np.isfinite(self.length_scale) or self.length_scale <= 0:
            raise ValueError("Stack length_scale must be finite and positive")
        for name, member in zip(self.member_names, self.members):
            categories = tuple(getattr(member, "target_categories", ()))
            if categories != self.target_categories:
                raise ValueError(
                    f"Stack member {name!r} categories do not match the ensemble"
                )

    @property
    def architecture(self) -> str:
        return "stacked_ensemble"

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "members": self.member_names,
            "whole_peptide": True,
            "length_range": (8, 15),
        }

    def predict_proba(self, peptides: Iterable[str]) -> np.ndarray:
        peptides = list(peptides)
        invalid = [peptide for peptide in peptides if not 8 <= len(peptide) <= 15]
        if invalid:
            raise ValueError("peptide-content models support complete peptides of length 8--15")
        if not peptides:
            return np.empty((0, len(self.target_categories)))
        member_logits = np.stack(
            [_logit(member.predict_proba(peptides)) for member in self.members],
            axis=2,
        )
        model_weights = self.coefficients[:, : len(self.members)]
        combined = np.einsum("ncm,cm->nc", member_logits, model_weights)
        normalized_length = (
            np.asarray([len(peptide) for peptide in peptides], dtype=np.float64)
            - self.length_center
        ) / self.length_scale
        combined += normalized_length[:, None] * self.coefficients[:, -1]
        combined += self.intercepts
        return _sigmoid(combined).astype(np.float32)


def _contained_model_path(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute():
        raise ValueError("Stack member paths must be relative to the ensemble artifact")
    resolved = (root / value).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("Stack member path escapes the ensemble artifact") from error
    return resolved


def _load_stacked_model(
    candidate: Path,
    config: dict[str, Any],
    *,
    device: str | None,
    batch_size: int | None,
) -> StackedPeptideContentModel:
    categories = tuple(config.get("categories", ()))
    member_specs = config.get("members", ())
    if not member_specs:
        raise ValueError("Stacked peptide-content config has no members")
    member_names = tuple(spec["name"] for spec in member_specs)
    members = tuple(
        load_peptide_content_model(
            _contained_model_path(candidate, spec["path"]),
            device=device,
            batch_size=batch_size,
        )
        for spec in member_specs
    )
    stacking = config.get("stacking", {})
    coefficients_by_category = stacking.get("coefficients", {})
    coefficients = np.asarray(
        [
            [
                coefficients_by_category[category][name]
                for name in (*member_names, "length")
            ]
            for category in categories
        ],
        dtype=np.float64,
    )
    intercepts = np.asarray(
        [coefficients_by_category[category]["intercept"] for category in categories],
        dtype=np.float64,
    )
    return StackedPeptideContentModel(
        target_categories=categories,
        member_names=member_names,
        members=members,
        coefficients=coefficients,
        intercepts=intercepts,
        length_center=float(stacking.get("length_center", 11.5)),
        length_scale=float(stacking.get("length_scale", 3.5)),
        model_dir=candidate.resolve(),
    )


def load_peptide_content_model(
    model: str | Path = "peptide-content-v1",
    *,
    model_dir: str | Path | None = None,
    device: str | None = None,
    batch_size: int | None = None,
):
    """Load any Weirdo peptide-content artifact behind one inference API.

    ``model`` may be a bundled/user model name or a directory containing an
    architecture ``config.json`` and weights. Returned models expose
    ``target_categories`` and ``predict_proba(peptides)``.
    """
    candidate = Path(model).expanduser() if isinstance(model, (str, Path)) else None
    if candidate is None or not candidate.is_dir():
        from .model_manager import load_model

        return load_model(str(model), str(model_dir) if model_dir is not None else None)
    config_path = candidate / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing peptide-content config: {config_path}")
    config = json.loads(config_path.read_text())
    architecture = config.get("architecture")
    if architecture is None and config.get("scorer_type") == "MLPScorer":
        from .scorers import MLPScorer

        return MLPScorer.load(candidate)
    categories = tuple(config.get("categories", ()))
    if not categories:
        raise ValueError(f"Peptide-content config has no categories: {config_path}")
    if architecture == "stacked_ensemble":
        return _load_stacked_model(
            candidate, config, device=device, batch_size=batch_size
        )
    import torch

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch_model, tokenizer = _build_torch_model(architecture, len(categories), config, candidate)
    torch_model.load_state_dict(torch.load(candidate / "weights.pt", map_location="cpu", weights_only=True))
    torch_model.to(selected_device)
    return TorchPeptideContentModel(
        architecture=architecture,
        target_categories=categories,
        model=torch_model,
        tokenizer=tokenizer,
        device=selected_device,
        batch_size=batch_size or (1024 if architecture.startswith("esm") else 4096),
        model_dir=candidate.resolve(),
    )


def select_best_peptide_content_model(
    model_dirs: Iterable[str | Path],
    *,
    metric: str = "microbial_foreign_like_auc",
    required_categories: Sequence[str] = STANDARD_SCORE_CATEGORIES,
) -> tuple[Path, list[dict[str, Any]]]:
    """Select a model by frozen taxonomy evaluation, before response scoring."""
    records = []
    for value in model_dirs:
        directory = Path(value)
        config_path = directory / "config.json"
        evaluation_path = directory / "evaluation.json"
        record: dict[str, Any] = {"path": str(directory), "eligible": False}
        if not config_path.is_file() or not evaluation_path.is_file():
            record["reason"] = "missing config.json or evaluation.json"
            records.append(record)
            continue
        config = json.loads(config_path.read_text())
        evaluation = json.loads(evaluation_path.read_text())
        categories = tuple(config.get("categories") or evaluation.get("categories", ()))
        missing = sorted(set(required_categories) - set(categories))
        score = evaluation.get("summary", {}).get(metric)
        record.update(
            architecture=config.get("architecture", "composition_mlp"),
            categories=list(categories),
            metric=metric,
            value=score,
            missing_categories=missing,
        )
        if missing:
            record["reason"] = "missing required output heads"
        elif score is None or not np.isfinite(score):
            record["reason"] = "missing finite selection metric"
        else:
            record["eligible"] = True
        records.append(record)
    eligible = [record for record in records if record["eligible"]]
    if not eligible:
        raise ValueError("No peptide-content model satisfies the selection policy")
    winner = max(eligible, key=lambda record: (record["value"], record["path"]))
    for record in records:
        record["selected"] = record is winner
    return Path(winner["path"]), records


def predict_taxonomy_scores(
    peptides: str | Sequence[str],
    model: str | Path = "peptide-content-v1",
    *,
    model_dir: str | Path | None = None,
    score_names: Sequence[str] | None = None,
):
    """Per-category probabilities plus the standard composite Weirdo scores.

    Returns a frame with ``peptide``, one probability column per model category and
    the composite scores of :data:`STANDARD_SCORE_NAMES` the model's heads support
    (for example ``virus_foreign_like``). Composite scores are geometric means and
    are bounded ranking scores, not probabilities. Complete 8--15mers only. This
    complements ``predict_peptide_content``, which applies a versioned score policy.
    """
    import pandas as pd

    if isinstance(peptides, str):
        peptides = [peptides]
    peptides = list(peptides)
    invalid = sorted({len(p) for p in peptides if not 8 <= len(p) <= 15})
    if invalid:
        raise ValueError(f"predict_taxonomy_scores supports peptides of length 8--15; received lengths {invalid}")
    scorer = load_peptide_content_model(model, model_dir=model_dir)
    categories = getattr(scorer, "target_categories", None)
    if not categories:
        raise ValueError("predict_taxonomy_scores needs a model trained with target_categories")
    probabilities = np.asarray(scorer.predict_proba(peptides), dtype=float)
    frame = pd.DataFrame(probabilities, columns=list(categories))
    frame.insert(0, "peptide", peptides)
    for name, values in standard_peptide_content_scores(
        probabilities, categories, required_scores=score_names
    ).items():
        frame[name] = values
    return frame

