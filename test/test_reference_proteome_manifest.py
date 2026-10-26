"""Invariant checks for the frozen reference-proteome training selection."""

from pathlib import Path
import re


MANIFEST = (
    Path(__file__).parents[1]
    / "weirdo"
    / "reference_proteome_manifests"
    / "reference-proteomes-v1.yaml"
)


def test_reference_proteome_seed_has_diverse_mammals_and_over_500_viruses():
    """Keep the checked-in training seed meaningful without requiring PyYAML."""
    text = MANIFEST.read_text(encoding="utf-8")
    mammal_block = text.split("\nviruses:\n", maxsplit=1)[0]
    mammal_ids = re.findall(r"^    proteome_id: (UP\d{9})$", mammal_block, flags=re.M)
    assert len(mammal_ids) >= 30
    assert len(mammal_ids) == len(set(mammal_ids))
    mammal_leaves = re.findall(r"^    taxonomy_leaf: (\w+)$", mammal_block, flags=re.M)
    assert len(mammal_leaves) == len(mammal_ids)
    assert set(mammal_leaves) == {"primates", "glires", "carnivora", "other_mammals"}

    virus_block = text.split("\nviruses:\n", maxsplit=1)[1]
    groups = re.findall(
        r"^  (riboviria|duplodnaviria|varidnaviria|monodnaviria):\n"
        r"(.*?)(?=^  (?:riboviria|duplodnaviria|varidnaviria|monodnaviria):\n|\Z)",
        virus_block,
        flags=re.M | re.S,
    )
    assert {name for name, _ in groups} == {
        "riboviria",
        "duplodnaviria",
        "varidnaviria",
        "monodnaviria",
    }
    by_realm = {
        name: re.findall(r"^  - (UP\d{9})$", entries, flags=re.M)
        for name, entries in groups
    }
    assert {name: len(ids) for name, ids in by_realm.items()} == {
        "riboviria": 128,
        "duplodnaviria": 128,
        "varidnaviria": 128,
        "monodnaviria": 128,
    }
    virus_ids = [proteome_id for ids in by_realm.values() for proteome_id in ids]
    assert len(virus_ids) >= 500
    assert len(virus_ids) == len(set(virus_ids))
