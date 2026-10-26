import numpy as np
import pytest

from weirdo import (
    DEFAULT_PEPTIDE_CONTENT_SCORE_POLICY,
    DEFAULT_SCORE_POLICY,
    apply_score_policy,
    get_score_policy,
    list_score_policies,
)


CATEGORIES = [
    "viruses",
    "bacteria",
    "archaea",
    "fungi",
    "protists",
    "mammals",
    "human",
    "mouse",
    "dog",
]


def test_registry_exposes_complete_json_compatible_definitions():
    policies = list_score_policies()
    names = {policy["policy_id"] for policy in policies}
    assert DEFAULT_SCORE_POLICY in names
    assert DEFAULT_PEPTIDE_CONTENT_SCORE_POLICY in names
    assert "human-microbial-content-v1" in names
    assert "mouse-microbial-content-v1" in names
    assert {policy["model_id"] for policy in list_score_policies("peptide-content-v1")} == {
        "peptide-content-v1"
    }
    policies[0]["self_categories"].append("changed")
    assert "changed" not in get_score_policy(policies[0]["policy_id"]).self_categories


def test_human_and_mouse_policies_apply_different_self_bundles():
    values = np.asarray([[0.8, 0.2, 0.1, 0.1, 0.1, 0.3, 0.9, 0.1, 0.2]])
    human = apply_score_policy(values, CATEGORIES, "human-microbial-content-v1")
    mouse = apply_score_policy(values, CATEGORIES, "mouse-microbial-content-v1")
    assert human["source_like"] == pytest.approx([0.8])
    assert human["self_like"] == pytest.approx([0.9])
    assert human["score"] == pytest.approx([0.08])
    assert mouse["self_like"] == pytest.approx([0.3])
    assert mouse["score"] == pytest.approx([0.56])


def test_exact_self_override_is_part_of_content_policy():
    values = np.asarray(
        [
            [0.8, 0.2, 0.1, 0.1, 0.1, 0.3, 0.9, 0.1, 0.2],
            [0.7, 0.2, 0.1, 0.1, 0.1, 0.4, 0.2, 0.1, 0.2],
        ]
    )
    result = apply_score_policy(
        values,
        CATEGORIES,
        "human-microbial-content-v1",
        exact_self_hits=[True, False],
    )
    assert result["exact_self_hit"].tolist() == [True, False]
    assert result["self_like"] == pytest.approx([1.0, 0.4])
    assert result["score"] == pytest.approx([0.0, 0.42])


def test_released_human_complement_policy_has_no_pathogen_numerator():
    values = np.asarray([[0.8, 0.25]])
    result = apply_score_policy(
        values,
        ["viruses", "human"],
        DEFAULT_SCORE_POLICY,
    )
    assert "source_like" not in result
    assert result["score"] == pytest.approx([0.75])
    with pytest.raises(ValueError, match="does not define an exact-self override"):
        apply_score_policy(
            values,
            ["viruses", "human"],
            DEFAULT_SCORE_POLICY,
            exact_self_hits=[True],
        )


def test_unknown_or_incompatible_policy_inputs_fail_clearly():
    with pytest.raises(ValueError, match="Unknown score policy"):
        get_score_policy("unknown")
    with pytest.raises(ValueError, match="missing self categories"):
        apply_score_policy(
            np.asarray([[0.5]]), ["viruses"], "human-microbial-content-v1"
        )
    with pytest.raises(ValueError, match="one value per peptide"):
        apply_score_policy(
            np.zeros((1, len(CATEGORIES))),
            CATEGORIES,
            "human-microbial-content-v1",
            exact_self_hits=[],
        )
