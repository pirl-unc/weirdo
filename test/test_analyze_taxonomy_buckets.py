import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "analyze_taxonomy_buckets.py"
SPEC = importlib.util.spec_from_file_location("analyze_taxonomy_buckets", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def test_proposed_bacterial_buckets_are_ordered_from_specific_to_fallback():
    classify = analysis.proposed_bacterial_bucket
    assert classify(("Pseudomonadota", "Gammaproteobacteria", "Enterobacterales")) == (
        "enterobacterales"
    )
    assert classify(("Pseudomonadota", "Gammaproteobacteria", "Vibrionales")) == (
        "other_gammaproteobacteria"
    )
    assert classify(("Pseudomonadota", "Alphaproteobacteria")) == (
        "alphaproteobacteria"
    )
    assert classify(("Pseudomonadota", "Betaproteobacteria")) == (
        "other_pseudomonadota"
    )
    assert classify(("Bacillota", "Bacilli")) == "bacilli"
    assert classify(("Actinomycetota",)) == "actinomycetota"
    assert classify(("Bacteroidota",)) == "other_bacteria"


def test_proposed_mammal_buckets_are_mutually_exclusive():
    classify = analysis.proposed_mammal_bucket
    assert classify(("Mammalia", "Primates")) == "primates"
    assert classify(("Mammalia", "Glires", "Rodentia")) == "glires"
    assert classify(("Mammalia", "Laurasiatheria", "Carnivora")) == "carnivora"
    assert classify(("Mammalia", "Laurasiatheria", "Artiodactyla")) == "other_mammals"
    assert classify(("Mammalia", "Metatheria")) == "other_mammals"
