import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "peptide_content_data.py"
SPEC = importlib.util.spec_from_file_location("peptide_content_data", SCRIPT)
pcd = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pcd
SPEC.loader.exec_module(pcd)


def test_coarse_taxonomy_mapping():
    assert pcd.coarse_category("human", ("Eukaryota", "Metazoa")) == "mammals"
    assert pcd.coarse_category("rodents", ("Eukaryota", "Mammalia")) == "mammals"
    assert pcd.coarse_category("mammals", ("Eukaryota", "Mammalia")) == "mammals"
    assert pcd.coarse_category("invertebrates", ("Eukaryota", "Metazoa")) == "invertebrates"
    assert pcd.coarse_category("invertebrates", ("Eukaryota", "Apicomplexa")) == "protists"
    assert pcd.coarse_category("vertebrates", ("Vertebrata", "Aves")) == "non_mammalian_tetrapods"
    assert pcd.coarse_category("vertebrates", ("Vertebrata", "Squamata")) == "non_mammalian_tetrapods"
    assert pcd.coarse_category("vertebrates", ("Vertebrata", "Amphibia")) == "non_mammalian_tetrapods"
    assert pcd.coarse_category("vertebrates", ("Vertebrata", "Actinopterygii")) == "fish"


def test_species_heads_are_nested_under_mammals():
    index = {name: i for i, name in enumerate(pcd.CATEGORIES)}
    human = pcd.Protein("P_HUMAN", "SIINFEKL", "human", 9606, "mammals", 0)
    dog = pcd.Protein("P_CANLF", "SIINFEKL", "mammals", 9615, "mammals", 0)
    human_mask = pcd.label_mask(human, index)
    dog_mask = pcd.label_mask(dog, index)
    assert human_mask & (1 << index["mammals"])
    assert human_mask & (1 << index["human"])
    assert not human_mask & (1 << index["dog"])
    assert dog_mask & (1 << index["mammals"])
    assert dog_mask & (1 << index["dog"])


def test_taxonomy_summary_counts_unique_taxa_not_proteins():
    proteins = [
        pcd.Protein("ONE_A", "SIINFEKL", "bacteria", 1, "bacteria", 0),
        pcd.Protein("TWO_A", "GILGFVFTL", "bacteria", 1, "bacteria", 0),
        pcd.Protein("THREE_B", "NLVPMVATV", "bacteria", 2, "bacteria", 1),
        pcd.Protein("P_HUMAN", "LLFGYPVYV", "human", 9606, "mammals", 2),
    ]
    summary = pcd.summarize(proteins)
    assert summary["categories"]["bacteria"] == {
        "taxa": 2,
        "proteins": 3,
        "amino_acids": 26,
    }
    assert summary["species_heads"]["human"]["taxa"] == 1


def test_sampled_shared_peptide_keeps_all_source_fold_bits():
    proteins = [
        pcd.Protein("ONE_A", "ASIINFEKLG", "bacteria", 1, "bacteria", 0),
        pcd.Protein("TWO_B", "VSIINFEKLM", "human", 9606, "mammals", 1),
    ]
    peptides, folds, _ = pcd.sample_candidates(
        proteins,
        lengths=(8,),
        per_stratum={"train": 10, "val": 10, "test": 10},
        seed=0,
    )
    assert len(peptides) >= 2
    assert set(folds.tolist()) <= {1, 2, 3}
