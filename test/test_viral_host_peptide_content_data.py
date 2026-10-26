import importlib.util
import sys
from pathlib import Path


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "viral_host_peptide_content_data",
    SCRIPTS / "viral_host_peptide_content_data.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TinyTaxonomy:
    parents = {1: 1, 2: 1, 40674: 2, 9606: 40674, 33208: 2, 7227: 33208}

    def resolve(self, taxon_id):
        return taxon_id

    def is_mammal(self, taxon_id):
        return taxon_id in {40674, 9606}

    def contains_mammals(self, taxon_id):
        return taxon_id in {1, 2, 40674}


def protein(identifier, virus_taxon, hosts):
    return MODULE.base.Protein(
        identifier, "SIINFEKL", "viruses", virus_taxon, "viruses", 0,
        host_taxon_ids=tuple(hosts),
    )


def test_host_assignments_aggregate_by_virus_taxon_and_keep_unknowns():
    proteins = [
        protein("A_ONE", 10, (7227,)),
        protein("B_ONE", 10, (9606,)),
        protein("C_TWO", 20, (7227,)),
        protein("D_THREE", 30, ()),
        protein("E_FOUR", 40, (2,)),
    ]
    assignments, reasons = MODULE.viral_host_assignments(proteins, TinyTaxonomy())
    assert assignments == {
        10: "mammal_host_viruses",
        20: "other_host_viruses",
        30: None,
        40: None,
    }
    assert reasons == {
        "explicit_mammal": 1,
        "explicit_non_mammal": 1,
        "missing_host": 1,
        "broad_host_contains_mammals": 1,
    }


def test_nested_host_category_is_added_to_membership_mask():
    index = {name: i for i, name in enumerate(MODULE.CATEGORIES)}
    value = protein("A_ONE", 10, (9606,))
    value = MODULE.replace(value, extra_categories=("mammal_host_viruses",))
    mask = MODULE.base.label_mask(value, index)
    assert mask & (1 << index["viruses"])
    assert mask & (1 << index["mammal_host_viruses"])
    assert not mask & (1 << index["other_host_viruses"])
