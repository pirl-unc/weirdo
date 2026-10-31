import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_human_self_neighborhood_features.py"
)
SPEC = importlib.util.spec_from_file_location("human_self_neighborhood", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_load_uniprot_sequences_filters_noncanonical(tmp_path):
    path = tmp_path / "human.dat"
    path.write_text(
        "ID   ONE\nSQ   SEQUENCE\n     ACDE FGHI\n//\n"
        "ID   TWO\nSQ   SEQUENCE\n     ABCD\n//\n"
    )
    assert MODULE.load_uniprot_sequences(path) == ["ACDEFGHI"]


def test_binary_one_hot_hamming_is_twice_residue_mismatches():
    values = np.vstack(
        [MODULE.encode_sequence("ACDEF"), MODULE.encode_sequence("ACDGF")]
    )
    packed, dimensions = MODULE.binary_one_hot(values)
    assert dimensions % 8 == 0
    xor = np.unpackbits(np.bitwise_xor(packed[0], packed[1])).sum()
    assert xor == 2


def test_candidate_statistics_separates_face_and_anchor_changes():
    queries = np.vstack([MODULE.encode_sequence("AACDEFGY")])
    candidates = np.stack(
        [
            np.vstack(
                [
                    MODULE.encode_sequence("AACDEFGY"),
                    MODULE.encode_sequence("ATCDEFGF"),
                    MODULE.encode_sequence("AACKKFGY"),
                ]
            )
        ]
    )
    features, weighted, similarity = MODULE.candidate_statistics(queries, candidates)
    assert features.loc[0, "human_self_face_mismatch"] == 0
    assert features.loc[0, "human_self_full_mismatch"] == 0
    assert features.loc[0, "human_self_anchor_matched_face_mismatch"] == 0
    assert weighted[0, 1] == 0.5
    assert weighted[0, 2] == 2.0
    assert similarity[0, 0] > similarity[0, 2]
