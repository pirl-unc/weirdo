"""Tests for the SwissProt k-mer reference table builder."""

import csv

import numpy as np
import pandas as pd
import pytest

from weirdo.kmer_table import (
    AMINO_ACIDS,
    build_kmer_table,
    decode_kmers,
    encode_residues,
    iter_kmer_masks,
    masks_to_matrix,
    unique_with_masks,
    window_codes,
)
from weirdo.scorers.swissprot import SwissProtReference
from weirdo.uniprot import SWISSPROT_DIVISIONS

CATEGORIES = list(SWISSPROT_DIVISIONS)


def random_proteins(n=60, seed=0, alphabet=AMINO_ACIDS):
    rng = np.random.RandomState(seed)
    sequences, categories = [], []
    for _ in range(n):
        length = int(rng.randint(1, 40))
        sequences.append("".join(rng.choice(list(alphabet), length)))
        categories.append(CATEGORIES[int(rng.randint(len(CATEGORIES)))])
    return pd.DataFrame({"identifier": [f"P{i}_X" for i in range(n)], "sequence": sequences,
                         "category": categories})


def brute_force(frame, k):
    """{kmer: set of categories}, ignoring k-mers that contain a non-canonical residue."""
    expected = {}
    for sequence, category in zip(frame["sequence"], frame["category"]):
        for start in range(len(sequence) - k + 1):
            kmer = sequence[start:start + k]
            if set(kmer) <= set(AMINO_ACIDS):
                expected.setdefault(kmer, set()).add(category)
    return expected


def read_table(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


class TestPrimitives:
    def test_pack_and_decode_round_trip(self):
        codes, starts = encode_residues(["ACDEFGHIK", "LMNPQRSTVWY"])
        packed, valid = window_codes(codes, 4)
        assert valid.sum() == (9 - 3) + (11 - 3)
        assert decode_kmers(packed[valid][:2], 4) == ["ACDE", "CDEF"]
        assert starts.tolist() == [0, 10]

    def test_non_canonical_residues_and_separators_invalidate_windows(self):
        codes, _ = encode_residues(["ACDXEFG", "HIK"])
        packed, valid = window_codes(codes, 3)
        kmers = decode_kmers(packed[valid], 3)
        assert kmers == ["ACD", "EFG", "HIK"]  # nothing spans the X or the protein boundary

    def test_packed_order_is_alphabetical(self):
        codes, _ = encode_residues(["WYACDA"])
        packed, valid = window_codes(codes, 2)
        kmers = decode_kmers(np.sort(packed[valid]), 2)
        assert kmers == sorted(kmers)

    def test_k_limits(self):
        codes, _ = encode_residues(["ACDEFGHIKLMNPQ"])
        with pytest.raises(ValueError):
            window_codes(codes, 0)
        with pytest.raises(ValueError):
            window_codes(codes, 13)
        assert window_codes(codes, 12)[0].dtype == np.uint64

    def test_unique_with_masks_ors_the_category_bits(self):
        kmers = np.array([5, 3, 5, 3, 9], dtype=np.uint64)
        bits = np.array([1, 2, 4, 2, 8], dtype=np.uint16)
        unique, masks = unique_with_masks(kmers, bits)
        assert unique.tolist() == [3, 5, 9] and masks.tolist() == [2, 5, 8]
        matrix = masks_to_matrix(masks, 4)
        assert matrix.tolist() == [[0, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]]


class TestBuildTable:
    @pytest.mark.parametrize("k", [3, 5, 8])
    def test_table_equals_the_brute_force_definition(self, tmp_path, k):
        frame = random_proteins(80, seed=k, alphabet=AMINO_ACIDS + "XUB")
        n = build_kmer_table(frame, tmp_path / "table.csv", k=k)
        expected = brute_force(frame, k)
        rows = read_table(tmp_path / "table.csv")
        assert n == len(rows) == len(expected)
        assert [row["seq"] for row in rows] == sorted(expected)  # sorted, each once
        for row in rows:
            present = {name for name in CATEGORIES if row[name] == "True"}
            assert present == expected[row["seq"]]
            assert int(row["label_count"]) == len(present)
            assert all(row[name] in ("True", "False") for name in CATEGORIES)

    def test_the_layout_matches_the_reference_reader(self, tmp_path):
        frame = pd.DataFrame({
            "identifier": ["A_X", "B_X", "C_X"],
            "sequence": ["MTMDKSELV", "MTMDKSELA", "GGGGGGGGG"],
            "category": ["human", "viruses", "bacteria"],
        })
        build_kmer_table(frame, tmp_path / "t.csv", k=8)
        header = (tmp_path / "t.csv").read_text().splitlines()[0]
        assert header == "seq," + ",".join(CATEGORIES) + ",label_count"
        reference = SwissProtReference(data_path=str(tmp_path / "t.csv"), k=8).load()
        assert reference.contains("MTMDKSEL") and reference.contains("GGGGGGGG")
        assert not reference.contains("AAAAAAAA")
        assert reference.get_kmer_categories("MTMDKSEL")["human"] is True
        assert reference.get_kmer_categories("MTMDKSEL")["viruses"] is True
        assert reference.get_kmer_categories("MTMDKSEL")["bacteria"] is False
        human_only = SwissProtReference(categories=["human"], data_path=str(tmp_path / "t.csv"), k=8).load()
        assert human_only.contains("MTMDKSEL") and not human_only.contains("GGGGGGGG")

    def test_memory_passes_and_chunking_do_not_change_the_result(self, tmp_path):
        frame = random_proteins(120, seed=7)
        for index, passes in enumerate((1, 3, 20)):
            build_kmer_table(frame, tmp_path / f"t{index}.csv", k=4, memory_passes=passes)
        reference = (tmp_path / "t0.csv").read_bytes()
        assert (tmp_path / "t1.csv").read_bytes() == reference
        assert (tmp_path / "t2.csv").read_bytes() == reference

        sequences = frame["sequence"].tolist()
        bits = np.array([1 << CATEGORIES.index(c) for c in frame["category"]], dtype=np.uint16)
        whole = [(k.copy(), m.copy()) for k, m in iter_kmer_masks(sequences, bits, 4, 1, 10**9)]
        small = [(k.copy(), m.copy()) for k, m in iter_kmer_masks(sequences, bits, 4, 2, 50)]
        np.testing.assert_array_equal(np.concatenate([k for k, _ in whole]), np.concatenate([k for k, _ in small]))
        np.testing.assert_array_equal(np.concatenate([m for _, m in whole]), np.concatenate([m for _, m in small]))

    def test_blocks_are_written_in_order_for_large_row_counts(self, tmp_path):
        frame = random_proteins(200, seed=5)
        build_kmer_table(frame, tmp_path / "a.csv", k=3, rows_per_block=7)
        build_kmer_table(frame, tmp_path / "b.csv", k=3, rows_per_block=10**6)
        assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()

    def test_reads_a_sequence_file(self, tmp_path):
        frame = random_proteins(20, seed=2)
        frame.to_csv(tmp_path / "sequences.csv", index=False)
        n = build_kmer_table(tmp_path / "sequences.csv", tmp_path / "t.csv", k=3)
        assert n == len(brute_force(frame, 3))

    def test_custom_categories_and_no_leftovers(self, tmp_path):
        frame = pd.DataFrame({"identifier": ["a", "b"], "sequence": ["ACDEF", "ACDEG"], "category": ["x", "y"]})
        build_kmer_table(frame, tmp_path / "t.csv", k=4, categories=["x", "y"])
        rows = read_table(tmp_path / "t.csv")
        assert [(r["seq"], r["x"], r["y"], r["label_count"]) for r in rows] == [
            ("ACDE", "True", "True", "2"), ("CDEF", "True", "False", "1"), ("CDEG", "False", "True", "1"),
        ]
        assert [p.name for p in tmp_path.iterdir()] == ["t.csv"]

    def test_bad_input_is_rejected_without_leaving_a_file(self, tmp_path):
        frame = pd.DataFrame({"identifier": ["a"], "sequence": ["ACDEFGHI"], "category": ["martians"]})
        with pytest.raises(ValueError, match="martians"):
            build_kmer_table(frame, tmp_path / "t.csv", k=8)
        with pytest.raises(ValueError, match="at most 16"):
            build_kmer_table(frame, tmp_path / "t.csv", k=8, categories=[str(i) for i in range(17)])
        assert list(tmp_path.iterdir()) == []

    def test_sequences_shorter_than_k_give_an_empty_table(self, tmp_path):
        frame = pd.DataFrame({"identifier": ["a"], "sequence": ["ACD"], "category": ["human"]})
        assert build_kmer_table(frame, tmp_path / "t.csv", k=8) == 0
        assert (tmp_path / "t.csv").read_text().splitlines() == ["seq," + ",".join(CATEGORIES) + ",label_count"]
