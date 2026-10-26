import numpy as np
import pytest

from weirdo.peptide_content import summarize_peptide_content


CATEGORIES = ("viruses", "bacteria", "archaea", "fungi", "protists", "mammals", "human")


def test_summary_uses_max_and_does_not_reward_multiple_microbe_heads():
    one_high = np.array([[0.9, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1]])
    all_high = np.array([[0.9, 0.9, 0.9, 0.9, 0.9, 0.2, 0.1]])
    first = summarize_peptide_content(one_high, CATEGORIES)
    second = summarize_peptide_content(all_high, CATEGORIES)
    assert first["microbial_like"] == pytest.approx([0.9])
    assert second["microbial_like"] == pytest.approx([0.9])
    assert first["microbial_foreign_like"] == pytest.approx([0.72])
    assert second["microbial_foreign_like"] == pytest.approx([0.72])


def test_summary_accepts_a_caller_defined_self_bundle():
    values = np.array([[0.8, 0.1, 0.1, 0.1, 0.1, 0.2, 0.95]])
    broad = summarize_peptide_content(values, CATEGORIES)
    human = summarize_peptide_content(
        values, CATEGORIES, self_categories=("mammals", "human")
    )
    assert broad["self_like"] == pytest.approx([0.2])
    assert human["self_like"] == pytest.approx([0.95])
    assert broad["microbial_foreign_like"] == pytest.approx([0.64])
    assert human["microbial_foreign_like"] == pytest.approx([0.04])


def test_summary_rejects_unknown_categories():
    with pytest.raises(ValueError, match="Unknown self_categories"):
        summarize_peptide_content(
            np.zeros((1, len(CATEGORIES))), CATEGORIES, self_categories=("dog",)
        )
