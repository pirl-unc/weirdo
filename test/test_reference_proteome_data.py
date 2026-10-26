"""Unit tests for full-reference variable-length split construction."""

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "reference_proteome_data.py"
SPEC = importlib.util.spec_from_file_location("reference_proteome_data", SCRIPT)
reference_data = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reference_data
SPEC.loader.exec_module(reference_data)


def test_nested_mammal_and_viral_labels_are_independent_memberships():
    index = {name: position for position, name in enumerate(reference_data.CATEGORIES)}
    dog = reference_data.Protein(
        "DOG1", "SIINFEKL", "UP000805418", ("mammals", "carnivora", "dog"), 0
    )
    virus = reference_data.Protein(
        "VIRUS1", "GILGFVFT", "UP000000277", ("viruses", "riboviria"), 1
    )
    dog_mask = reference_data.label_mask(dog, index)
    virus_mask = reference_data.label_mask(virus, index)
    for label in dog.labels:
        assert dog_mask & (1 << index[label])
    for label in virus.labels:
        assert virus_mask & (1 << index[label])
    assert not dog_mask & (1 << index["viruses"])
    assert not virus_mask & (1 << index["mammals"])


def test_sampled_reference_peptides_retain_source_split_bits():
    proteins = [
        reference_data.Protein(
            "DOG1", "ASIINFEKLG", "UP000805418", ("mammals", "carnivora", "dog"), 0
        ),
        reference_data.Protein(
            "VIRUS1", "VSIINFEKLM", "UP000000277", ("viruses", "riboviria"), 1
        ),
    ]
    peptides, folds, _ = reference_data.sample_candidates(
        proteins,
        lengths=(8,),
        per_stratum={"train": 10, "val": 10, "test": 10},
        seed=0,
    )
    assert peptides
    assert set(folds.tolist()) <= {1, 2, 3}


def test_source_aliases_do_not_create_undefined_category_heads():
    index = {name: position for position, name in enumerate(reference_data.CATEGORIES)}
    rat = reference_data.Protein(
        "RAT1", "SIINFEKL", "UP000002494", ("mammals", "glires", "rat"), 0
    )
    mask = reference_data.label_mask(rat, index)
    assert mask & (1 << index["mammals"])
    assert mask & (1 << index["glires"])
    assert mask.bit_count() == 2


def test_load_excluded_peptides_can_select_registered_validation(tmp_path):
    source = tmp_path / "assays.csv"
    source.write_text(
        "role,peptide\n"
        "discovery,SIINFEKL\n"
        "validation,GILGFVFTL\n"
        "validation,not-a-peptide\n"
    )
    excluded = reference_data.load_excluded_peptides(source, role="validation")
    assert excluded == {"GILGFVFTL"}
