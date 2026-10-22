"""Offline, versioned inference with the model distributed in the wheel."""

import copy
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .scorers.mlp import MLPScorer


DEFAULT_PRETRAINED_MODEL = "swissprot-categories-v1"
DEFAULT_SCORE_POLICY = "human-self-complement-v1"
MODEL_DIRECTORY = Path(__file__).parent / "pretrained" / "swissprot-categories"
MANIFEST_SHA256 = "716e931ecb88ab6f042e01c886ac4e929bccf1c8d8c762e3f20ee0b9623fd5f9"
CANONICAL_RESIDUES = frozenset("ACDEFGHIKLMNPQRSTVWY")


def get_model_manifest(model=DEFAULT_PRETRAINED_MODEL, model_dir=None):
    """Read the released model's identity and evaluation contract offline.

    Parameters
    ----------
    model : str
        Immutable model name. Unknown names fail rather than selecting another model.
    model_dir : path-like, optional
        Explicit copy of the released model directory. No user cache is searched.

    Returns
    -------
    dict
        A new manifest dictionary, including its SHA-256 ``artifact_id``.
        Reading this manifest does not download or deserialize weights. Use
        :func:`load_pretrained_model` to verify all artifact checksums.
    """
    if model != DEFAULT_PRETRAINED_MODEL:
        raise ValueError(f"Unknown released model {model!r}; available: {DEFAULT_PRETRAINED_MODEL}")
    directory = Path(model_dir) if model_dir is not None else MODEL_DIRECTORY
    path = directory / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Released WEIRDO model is unavailable at {directory}; reinstall weirdo.")
    artifact_id = hashlib.sha256(path.read_bytes()).hexdigest()
    if artifact_id != MANIFEST_SHA256:
        raise ValueError("Released model manifest checksum mismatch")
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("model_id") != model:
        raise ValueError("Unsupported WEIRDO model manifest schema or model identity")
    manifest["artifact_id"] = artifact_id
    return manifest


class PretrainedModel:
    """A verified category model with an explicit human-reference score policy.

    Construct directly or using :func:`load_pretrained_model` with the same
    arguments and validation. ``score`` returns
    ``1 - P(human category)`` under the training distribution. Higher means less
    human-like by this model, not more likely to elicit an immune response.
    Only human reference taxon 9606 is supported: the rodent category cannot
    identify mouse-specific self. Ranking with this score remains experimental.
    """

    def __init__(self, model=DEFAULT_PRETRAINED_MODEL, *, model_dir=None,
                 score_policy=DEFAULT_SCORE_POLICY, host_taxon_id=9606):
        if score_policy != DEFAULT_SCORE_POLICY:
            raise ValueError(f"Unsupported score policy: {score_policy!r}")
        if type(host_taxon_id) is not int or host_taxon_id != 9606:
            raise ValueError("Only human host_taxon_id=9606 is supported by this score policy")
        manifest = get_model_manifest(model, model_dir)
        directory = Path(model_dir) if model_dir is not None else MODEL_DIRECTORY
        expected = {"config.json", "model.pt", "metadata.json"}
        if set(manifest.get("files", {})) != expected:
            raise ValueError("Model manifest must identify config, weights and metadata")
        for name, checksum in manifest["files"].items():
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(f"Missing model artifact: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Model artifact checksum mismatch: {name}")
        if not zipfile.is_zipfile(directory / "model.pt"):
            raise ValueError("Released models require portable NumPy weights, never pickle")
        feature_path = Path(__file__).parent / "scorers" / "peptide_features.py"
        if hashlib.sha256(feature_path.read_bytes()).hexdigest() != manifest["feature_definition_sha256"]:
            raise ValueError("Released model feature implementation checksum mismatch")
        scorer = MLPScorer.load(directory)
        if scorer.target_categories != manifest["categories"] or scorer.k != 8:
            raise ValueError("Model categories or window length disagree with the manifest")
        smoke = manifest["smoke"]
        if not np.allclose(scorer.predict_proba([smoke["peptide"]])[0], smoke["category_values"], rtol=0, atol=smoke["atol"]):
            raise ValueError("Released model smoke prediction does not match its manifest")
        self._scorer = scorer
        self._manifest = copy.deepcopy(manifest)
        self._score_policy = score_policy
        self._host_taxon_id = 9606
        self._package_version = version("weirdo")

    @property
    def manifest(self):
        """Return a copy of the verified model's manifest."""
        return copy.deepcopy(self._manifest)

    @property
    def score_policy(self):
        """Versioned definition of the scalar score."""
        return self._score_policy

    @property
    def host_taxon_id(self):
        """NCBI taxonomy identifier of the supported reference host."""
        return self._host_taxon_id

    @property
    def prediction_identity(self):
        """Exact artifact, feature, input, aggregation, host and policy cache key."""
        return ":".join([
            "weirdo", self._package_version,
            self._manifest["model_id"], self._manifest["artifact_id"],
            self._score_policy, str(self._host_taxon_id),
        ])

    def predict_categories(self, peptides, batch_size=1024):
        """Return category outputs in input order, preserving duplicate peptides.

        Parameters
        ----------
        peptides : str or iterable of str
            Canonical amino-acid sequences of length 8 through 11. Lowercase is
            normalized to uppercase; whitespace, ambiguous residues, empty
            sequences, non-strings and unsupported lengths raise ``ValueError``.
            An empty iterable returns a typed empty DataFrame.
        batch_size : int
            Maximum number of distinct peptides evaluated per inference batch.

        Returns
        -------
        pandas.DataFrame
            Normalized ``peptide`` and one float column per model category.
            Longer peptides average their overlapping 8-mer category outputs.
            ``attrs`` records the exact prediction identity and policy. Outputs
            describe the training population, not immune-response probabilities.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(peptides, str):
            peptides = [peptides]
        try:
            peptides = list(peptides)
        except TypeError as exc:
            raise ValueError("peptides must be a string or iterable of strings") from exc
        normalized = []
        for index, peptide in enumerate(peptides):
            if not isinstance(peptide, str) or not peptide.isascii():
                raise ValueError(f"Peptide at index {index} must be an ASCII string")
            peptide = peptide.upper()
            if not 8 <= len(peptide) <= 11 or not set(peptide) <= CANONICAL_RESIDUES:
                raise ValueError(f"Peptide at index {index} must contain 8–11 canonical amino acids")
            normalized.append(peptide)
        unique = list(dict.fromkeys(normalized))
        categories = self._manifest["categories"]
        values = np.empty((len(unique), len(categories)), dtype=np.float64)
        for start in range(0, len(unique), batch_size):
            batch = unique[start:start + batch_size]
            values[start:start + len(batch)] = self._scorer.predict_proba(batch, aggregate="mean")
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Model returned invalid category outputs")
        lookup = {peptide: index for index, peptide in enumerate(unique)}
        indices = np.array([lookup[peptide] for peptide in normalized], dtype=np.intp)
        frame = pd.DataFrame(values[indices], columns=categories)
        frame.insert(0, "peptide", pd.Series(normalized, dtype="str"))
        frame.attrs.update(
            prediction_identity=self.prediction_identity,
            model_id=self._manifest["model_id"],
            artifact_id=self._manifest["artifact_id"],
            score_policy=self._score_policy,
            host_taxon_id=self._host_taxon_id,
        )
        return frame

    def score(self, peptides, batch_size=1024):
        """Return human-category complement scores, preserving order and duplicates.

        Parameters
        ----------
        peptides : str or iterable of str
            Inputs accepted by :meth:`predict_categories`; empty input returns
            an empty float array. Invalid peptides fail before inference.
        batch_size : int
            Maximum distinct peptides per inference batch.

        Returns
        -------
        numpy.ndarray
            ``1 - P(human category)`` in [0, 1]. This descriptive score has no
            independently established benefit for immune-response ranking.
        """
        return 1.0 - self.predict_categories(peptides, batch_size)["human"].to_numpy()


def load_pretrained_model(
    model=DEFAULT_PRETRAINED_MODEL, *, model_dir=None,
    score_policy=DEFAULT_SCORE_POLICY, host_taxon_id=9606,
):
    """Verify and load the bundled, portable model without network or cache writes.

    Parameters
    ----------
    model : str
        Immutable released model name, by default ``swissprot-categories-v1``.
    model_dir : path-like, optional
        Explicit artifact directory; defaults to the copy shipped in the wheel.
    score_policy : str
        ``human-self-complement-v1``: one minus the human category output.
        Unknown policies fail rather than silently substituting a definition.
    host_taxon_id : int
        Human (9606) only. Other hosts require a separately validated policy.

    Returns
    -------
    PretrainedModel
        Ready to call ``score`` or ``predict_categories`` offline. Missing files,
        bad checksums, pickle weights and incompatible schemas fail clearly.
        User-installed models with the same name never override this artifact.
    """
    return PretrainedModel(model, model_dir=model_dir, score_policy=score_policy, host_taxon_id=host_taxon_id)
