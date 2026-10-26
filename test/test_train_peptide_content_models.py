import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_peptide_content_models.py"
SPEC = importlib.util.spec_from_file_location("train_peptide_content_models", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)

REFERENCE_DATA_SCRIPT = Path(__file__).parents[1] / "scripts" / "reference_proteome_data.py"


def test_encode_variable_length_peptides():
    tokens, mask = module.encode_peptides(["SIINFEKL", "ABCDEFGHIJKLMNO"])
    assert tokens.shape == (2, 15)
    assert mask.sum(dim=1).tolist() == [8, 15]
    assert tokens[0, 8:].eq(0).all()


def test_microbe_times_one_minus_mammal_summary():
    n_categories = len(module.CATEGORIES)
    labels = np.zeros((4, n_categories), dtype=np.float32)
    probabilities = np.zeros_like(labels)
    index = {name: i for i, name in enumerate(module.CATEGORIES)}
    labels[:2, index["bacteria"]] = 1
    labels[2:, index["mammals"]] = 1
    probabilities[:2, index["bacteria"]] = 0.9
    probabilities[2:, index["mammals"]] = 0.9
    summary = module.evaluate_summary_only(probabilities, labels)
    assert summary["auc"] == 1.0


def test_cnn_accepts_8_to_15mers():
    model = module.build_torch_model("cnn", len(module.CATEGORIES))
    tokens, mask = module.encode_peptides(["SIINFEKL", "ABCDEFGHIJKLMNO"])
    result = model(tokens, mask)
    assert result.shape == (2, len(module.CATEGORIES))


def test_collator_is_pickle_safe_for_spawned_data_loader_workers():
    collator = module.make_collate("cnn")
    assert "<locals>" not in type(collator).__qualname__
    features, labels = collator(
        [("SIINFEKL", np.zeros(len(module.CATEGORIES), dtype=np.float32))]
    )
    assert features["tokens"].shape == (1, 15)
    assert labels.shape == (1, len(module.CATEGORIES))


def test_predict_torch_weights_validation_loss_by_elements():
    import torch

    class FixedModel(torch.nn.Module):
        def forward(self, tokens):
            return tokens[:, :1].float()

    loader = [
        ({"tokens": torch.tensor([[0.0]])}, torch.tensor([[1.0]])),
        ({"tokens": torch.tensor([[2.0], [2.0], [2.0]])}, torch.tensor([[1.0], [1.0], [1.0]])),
    ]
    _, _, observed = module.predict_torch(FixedModel(), loader, torch.device("cpu"))
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    expected = torch.cat(
        (
            criterion(torch.tensor([[0.0]]), torch.tensor([[1.0]])).reshape(-1),
            criterion(torch.tensor([[2.0], [2.0], [2.0]]), torch.ones(3, 1)).reshape(-1),
        )
    ).mean()
    assert observed == pytest.approx(float(expected))


def test_split_fingerprint_changes_with_split_contents(tmp_path):
    first = tmp_path / "split-a.txt"
    second = tmp_path / "split-b.txt"
    first.write_text("train")
    second.write_text("validation")
    directory = tmp_path / "splits"
    directory.mkdir()
    (directory / "train.txt").write_text(first.read_text())
    before = module._sha256_path(directory)
    (directory / "train.txt").write_text("changed")
    after = module._sha256_path(directory)
    assert before != after


def test_training_script_accepts_the_complete_reference_split_builder():
    default_module = module.pcd
    try:
        module.set_data_module(module.load_data_module(REFERENCE_DATA_SCRIPT))
        assert module.CATEGORIES[0:2] == ("viruses", "riboviria")
        assert module.MICROBIAL_CATEGORIES == ("viruses",)
        values = np.zeros((2, len(module.CATEGORIES)), dtype=np.float32)
        labels = np.zeros_like(values)
        index = {name: position for position, name in enumerate(module.CATEGORIES)}
        values[0, index["viruses"]] = labels[0, index["viruses"]] = 0.9
        values[1, index["mammals"]] = labels[1, index["mammals"]] = 0.9
        assert module.evaluate_summary_only(values, labels)["auc"] == 1.0
    finally:
        module.set_data_module(default_module)
