import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_foreignness_definitions.py"
SPEC = importlib.util.spec_from_file_location("audit_foreignness_definitions", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_score_definitions_distinguish_human_and_mammal_complements():
    frame = pd.DataFrame(
        {
            "esm_finetune__p_viruses": [0.8, 0.2],
            "esm_finetune__p_mammals": [0.1, 0.4],
            "esm_finetune__p_human": [0.2, 0.3],
            "esm_finetune__microbial_foreign_like": [0.72, 0.12],
            "esm_finetune__microbial_like": [0.8, 0.2],
        }
    )
    scores = module._score_definitions(frame)
    np.testing.assert_allclose(scores["one_minus_mammals"], [0.9, 0.6])
    np.testing.assert_allclose(scores["one_minus_human"], [0.8, 0.7])
    np.testing.assert_allclose(scores["virus_vs_mammals"], [0.72, 0.12])
    np.testing.assert_allclose(scores["virus_vs_human"], [0.64, 0.14])
