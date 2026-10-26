import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_peptide_content_immunogenicity.py"
SPEC = importlib.util.spec_from_file_location("evaluate_peptide_content", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_derive_scores_uses_independent_heads():
    probabilities = np.zeros((2, len(module.CATEGORIES)))
    index = {name: i for i, name in enumerate(module.CATEGORIES)}
    probabilities[0, index["viruses"]] = 0.8
    probabilities[0, index["mammals"]] = 0.25
    probabilities[1, index["bacteria"]] = 0.9
    probabilities[1, index["fungi"]] = 0.7
    probabilities[1, index["mammals"]] = 1.0
    scores = module.derive_scores(probabilities)
    np.testing.assert_allclose(scores["microbial_like"], [0.8, 0.9])
    np.testing.assert_allclose(scores["microbial_foreign_like"], [0.6, 0.0])
    np.testing.assert_allclose(scores["virus_foreign_like"], [0.6, 0.0])


def test_design_matrix_adds_requested_fixed_effects():
    frame = pd.DataFrame(
        {
            "mhcflurry_presentation_score": [0.1, 0.2, 0.8, 0.9],
            "mhcflurry_affinity": [1000.0, 500.0, 50.0, 10.0],
            "hla": ["A", "A", "B", "B"],
            "cohort": ["c1", "c2", "c1", "c2"],
            "source_species": ["s1", "s1", "s2", "s2"],
        }
    )
    without_species = module.design_matrix(frame, False)
    with_species = module.design_matrix(frame, True)
    affinity = module.design_matrix(frame, False, "affinity")
    assert without_species.shape[0] == len(frame)
    assert with_species.shape[1] >= without_species.shape[1]
    assert np.all(np.diff(affinity[:, 0]) > 0)


def test_registered_transfer_excludes_discovery_peptide_from_validation():
    rows = []
    for role, offset in (("discovery", 0), ("validation", 100)):
        for index in range(24):
            rows.append(
                {
                    "role": role,
                    "peptide": f"PEPTIDE{index + offset:03d}",
                    "peptide_cluster": f"cluster-{index}",
                    "outcome": "Positive" if index % 2 else "Negative",
                    "mhcflurry_presentation_score": 0.1 + index / 100,
                    "mhcflurry_affinity": 1000 - index,
                    "hla": "HLA-A*01:01",
                    "cohort": f"cohort-{role}",
                    "cnn__microbial_foreign_like": index / 24,
                    "cnn__microbial_like": index / 24,
                    "cnn__virus_foreign_like": index / 24,
                    "cnn__one_minus_mammals": index / 24,
                }
            )
    # Deliberately duplicate one peptide into validation to exercise the guardrail.
    rows[-1]["peptide"] = rows[0]["peptide"]
    results = module.evaluate_registered_transfer(
        pd.DataFrame(rows), ["cnn"], ablation_architecture="cnn", bootstrap_draws=10
    )
    assert len(results) == len(module.SCORE_NAMES)
    assert all(row["evaluation_protocol"] == "discovery_to_registered_validation" for row in results)
    assert all(row["dropped_validation_peptide_overlap"] == 1 for row in results)


def test_load_predictor_runs_a_self_contained_torch_artifact(tmp_path):
    torch = pytest.importorskip("torch")

    artifact = tmp_path / "cnn"
    artifact.mkdir()
    (artifact / "config.json").write_text(
        json.dumps(
            {
                "architecture": "cnn",
                "categories": list(module.CATEGORIES),
                "length_range": [8, 15],
            }
        )
    )
    model = module.training.build_torch_model("cnn", len(module.CATEGORIES))
    torch.save(model.state_dict(), artifact / "weights.pt")

    architecture, predict = module._load_predictor(artifact)
    probabilities = predict(["SIINFEKL", "GILGFVFTL"])
    assert architecture == "cnn"
    assert probabilities.shape == (2, len(module.CATEGORIES))
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
