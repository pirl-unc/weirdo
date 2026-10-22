"""Tests for the family-aware SwissProt k-mer split builder (scripts/category_data.py)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import category_data as cd  # noqa: E402


def _windows(sequences, k=8):
    codes, _ = cd.encode_residues(sequences)
    packed, valid = cd.window_codes(codes, k)
    return cd.decode_kmers(packed[valid], k)


def _family_in_fold(fold, taken=()):
    """A family key that the default salt assigns to ``fold``."""
    for index in range(10_000):
        key = f"F{index}"
        if key not in taken and cd.split_of_family(key) == fold:
            return key
    raise AssertionError("no family key found")


class TestEncoding:
    def test_windows_do_not_cross_proteins_or_noncanonical_residues(self):
        windows = _windows(["ACDEFGHIKL", "ACDEFGHIKX", "MMMMMMMMMM"])
        assert windows == [
            "ACDEFGHI", "CDEFGHIK", "DEFGHIKL",   # protein 1
            "ACDEFGHI", "CDEFGHIK",               # protein 2: DEFGHIKX contains X
            "MMMMMMMM", "MMMMMMMM", "MMMMMMMM",   # protein 3
        ]

    @pytest.mark.parametrize("k", [1, 5, 12])
    def test_round_trip_for_other_k(self, k):
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        windows = _windows([sequence], k=k)
        assert windows == [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

    def test_rejects_unsupported_k(self):
        codes, _ = cd.encode_residues(["ACDEFGHIKLMNPQ"])
        with pytest.raises(ValueError):
            cd.window_codes(codes, 13)
        with pytest.raises(ValueError):
            cd.window_codes(codes, 0)

    def test_sequences_shorter_than_k_have_no_windows(self):
        assert _windows(["ACDEF"]) == []


class TestMasks:
    def test_unique_with_masks_ors_category_bits(self):
        codes, _ = cd.encode_residues(["ACDEFGHIKL", "ACDEFGHIKX", "MMMMMMMMMM"])
        packed, valid = cd.window_codes(codes, 8)
        bits = np.array([1, 1, 1, 2, 2, 4, 4, 4], dtype=np.uint16)
        kmers, masks = cd.unique_with_masks(packed[valid], bits)
        table = dict(zip(cd.decode_kmers(kmers, 8), masks.tolist()))
        assert table == {"ACDEFGHI": 3, "CDEFGHIK": 3, "DEFGHIKL": 1, "MMMMMMMM": 4}

    def test_unique_with_masks_handles_empty_input(self):
        kmers, masks = cd.unique_with_masks(np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.uint16))
        assert len(kmers) == 0 and len(masks) == 0

    def test_masks_to_matrix(self):
        matrix = cd.masks_to_matrix(np.array([3, 4, 0], dtype=np.uint16), 3)
        assert matrix.tolist() == [[1, 1, 0], [0, 0, 1], [0, 0, 0]]
        assert matrix.dtype == np.float32


class TestFamilySplit:
    def test_family_key_is_the_entry_name_mnemonic(self):
        assert cd.family_key("RS19_ECOLI") == "RS19"
        assert cd.family_key("1433B_HUMAN") == "1433B"
        assert cd.family_key("NOUNDERSCORE") == "NOUNDERSCORE"

    def test_split_is_deterministic_and_roughly_proportional(self):
        keys = [f"K{i:05d}" for i in range(20_000)]
        first = np.array([cd.split_of_family(k) for k in keys])
        second = np.array([cd.split_of_family(k) for k in keys])
        assert (first == second).all()
        fractions = np.bincount(first, minlength=3) / len(keys)
        assert fractions == pytest.approx([0.70, 0.15, 0.15], abs=0.02)

    def test_salt_changes_the_assignment(self):
        keys = [f"K{i:04d}" for i in range(2000)]
        default = np.array([cd.split_of_family(k) for k in keys])
        other = np.array([cd.split_of_family(k, salt="another-salt") for k in keys])
        assert (default != other).any()

    def test_orthologs_of_one_gene_share_a_split(self):
        splits = {cd.split_of_family(cd.family_key(name))
                  for name in ("RS19_ECOLI", "RS19_BACSU", "RS19_HUMAN", "RS19_ARATH")}
        assert len(splits) == 1


class TestUniverses:
    @staticmethod
    def _frame():
        train_key = _family_in_fold(0)
        val_key = _family_in_fold(1)
        test_key = _family_in_fold(2)
        return pd.DataFrame({
            "identifier": [f"{train_key}_ECOLI", f"{val_key}_9VIRU", f"{test_key}_HUMAN"],
            "sequence": ["ACDEFGHIKLMN", "ACDEFGHIWWWWYYYY", "IWWWWYYYYPPPP"],
            "category": ["bacteria", "viruses", "human"],
        })

    def test_fold_universes_are_exclusive_and_labelled_from_their_own_fold(self):
        universes, shared = cd.build_universes(self._frame(), verbose=False)

        def kmer_table(name):
            entry = universes[name]
            return dict(zip(cd.decode_kmers(entry["kmers"], 8), entry["masks"].tolist()))

        bacteria = 1 << cd.CATEGORIES.index("bacteria")
        viruses = 1 << cd.CATEGORIES.index("viruses")
        human = 1 << cd.CATEGORIES.index("human")
        # ACDEFGHI is in train and val, IWWWWYYY and WWWWYYYY in val and test:
        # all three are removed from every fold that contains them.
        assert set(kmer_table("train")) == {"CDEFGHIK", "DEFGHIKL", "EFGHIKLM", "FGHIKLMN"}
        assert set(kmer_table("val")) == {
            "CDEFGHIW", "DEFGHIWW", "EFGHIWWW", "FGHIWWWW", "GHIWWWWY", "HIWWWWYY"}
        assert set(kmer_table("test")) == {"WWWYYYYP", "WWYYYYPP", "WYYYYPPP", "YYYYPPPP"}
        assert set(kmer_table("train")) .isdisjoint(kmer_table("val"))
        assert set(kmer_table("val")).isdisjoint(kmer_table("test"))
        assert set(kmer_table("train")).isdisjoint(kmer_table("test"))
        assert shared == {"train": 1, "val": 3, "test": 2}
        for name in ("train", "val", "test"):  # every kept k-mer carries a family bucket
            assert len(universes[name]["groups"]) == len(universes[name]["kmers"])
            assert universes[name]["groups"].max() < cd.N_BUCKETS

        assert set(kmer_table("train").values()) == {bacteria}
        assert set(kmer_table("val").values()) == {viruses}
        assert set(kmer_table("test").values()) == {human}
        # The global universe keeps the shared k-mers with the union of their labels.
        everything = kmer_table("all")
        assert everything["ACDEFGHI"] == bacteria | viruses
        assert everything["IWWWWYYY"] == viruses | human
        assert len(everything) == 4 + 1 + 6 + 2 + 4

    def test_draw_sample_is_a_reproducible_subset(self):
        universe = {
            "kmers": np.arange(100, dtype=np.uint64),
            "masks": np.arange(100, dtype=np.uint16),
            "groups": (np.arange(100) % 7).astype(np.uint8),
        }
        first = cd.draw_sample(universe, 10, np.random.RandomState(3))
        second = cd.draw_sample(universe, 10, np.random.RandomState(3))
        assert len(first["kmers"]) == 10 and (first["kmers"] == second["kmers"]).all()
        assert (np.diff(first["kmers"].astype(np.int64)) > 0).all()
        assert (first["kmers"] == first["masks"]).all()  # masks stay aligned with k-mers
        assert (first["groups"] == first["kmers"] % 7).all()  # and so do the buckets
        assert len(cd.draw_sample(universe, 1000, np.random.RandomState(3))["kmers"]) == 100
        assert len(cd.draw_sample(universe, None, np.random.RandomState(3))["kmers"]) == 100

    def test_disjoint_sample_avoids_taken_kmers(self):
        universe = {
            "kmers": np.arange(100, dtype=np.uint64),
            "masks": np.arange(100, dtype=np.uint16),
            "groups": np.zeros(100, dtype=np.uint8),
        }
        taken = cd.draw_sample(universe, 40, np.random.RandomState(0))
        extra = cd.draw_disjoint_sample(universe, taken, 30, np.random.RandomState(1))
        assert len(extra["kmers"]) == 30
        assert set(extra["kmers"].tolist()).isdisjoint(taken["kmers"].tolist())
        assert (extra["kmers"] == extra["masks"]).all()
        # asking for more than is left returns what is left
        assert len(cd.draw_disjoint_sample(universe, taken, 1000, np.random.RandomState(1))["kmers"]) == 60

    def test_family_buckets_are_deterministic_and_balanced(self):
        keys = [f"K{i:05d}" for i in range(20_000)]
        buckets = np.array([cd.bucket_of_family(k) for k in keys])
        assert (buckets == np.array([cd.bucket_of_family(k) for k in keys])).all()
        assert buckets.min() == 0 and buckets.max() == cd.N_BUCKETS - 1
        assert np.bincount(buckets).min() > 0.8 * len(keys) / cd.N_BUCKETS

    def test_unique_with_masks_keeps_the_smallest_group(self):
        kmers = np.array([5, 5, 9, 2], dtype=np.uint64)
        bits = np.array([1, 2, 4, 8], dtype=np.uint16)
        groups = np.array([7, 3, 5, 1], dtype=np.uint8)
        unique, masks, buckets = cd.unique_with_masks(kmers, bits, groups)
        assert unique.tolist() == [2, 5, 9]
        assert masks.tolist() == [8, 3, 4]
        assert buckets.tolist() == [1, 3, 5]


class TestInputsAndCache:
    def test_read_sequences_validates_columns_and_categories(self, tmp_path):
        good = tmp_path / "good.csv"
        pd.DataFrame({"identifier": ["A_B"], "sequence": ["ACDEFGHI"], "category": ["human"]}).to_csv(good)
        assert len(cd.read_sequences(str(good))) == 1

        missing = tmp_path / "missing.csv"
        pd.DataFrame({"identifier": ["A_B"], "sequence": ["ACDEFGHI"]}).to_csv(missing)
        with pytest.raises(ValueError, match="lacks columns"):
            cd.read_sequences(str(missing))

        unknown = tmp_path / "unknown.csv"
        pd.DataFrame({"identifier": ["A_B"], "sequence": ["ACDEFGHI"], "category": ["aliens"]}).to_csv(unknown)
        with pytest.raises(ValueError, match="Unknown categories"):
            cd.read_sequences(str(unknown))

    def test_verify_against_kmer_table(self, tmp_path):
        codes, _ = cd.encode_residues(["ACDEFGHI", "KLMNPQRS"])
        packed, valid = cd.window_codes(codes, 8)
        kmers, masks = cd.unique_with_masks(packed[valid], np.array([3, 4], dtype=np.uint16))
        universe = {"kmers": kmers, "masks": masks}

        def row(index, seq, positive):
            flags = {name: False for name in cd.CATEGORIES}
            for name in positive:
                flags[name] = True
            return {"": index, "seq": seq, **flags, "label_count": len(positive)}

        # ACDEFGHI is stored with archaea+bacteria (mask 3); KLMNPQRS with fungi (mask 4).
        table = pd.DataFrame([
            row(0, "ACDEFGHI", ["archaea", "bacteria"]),
            row(1, "KLMNPQRS", ["human"]),   # wrong labels on purpose
            row(2, "WWWWWWWW", ["viruses"]),  # absent from the universe
        ]).set_index("")
        path = tmp_path / "table.csv"
        table.to_csv(path)
        result = cd.verify_against_kmer_table(universe, str(path), nrows=10)
        assert result["rows"] == 3
        assert result["found_fraction"] == pytest.approx(2 / 3)
        assert result["label_match_fraction_of_found"] == pytest.approx(0.5)

    def test_zero_sized_sets_are_skipped(self, tmp_path):
        keys = [_family_in_fold(0), _family_in_fold(1), _family_in_fold(2)]
        frame = pd.DataFrame({
            "identifier": [f"{keys[0]}_X", f"{keys[1]}_Y", f"{keys[2]}_Z"],
            "sequence": ["ACDEFGHIKLMNPQ", "STVWYACDEFGHIK", "MNPQRSTVWYACDE"],
            "category": ["archaea", "fungi", "plants"],
        })
        frame.to_csv(tmp_path / "seqs.csv")
        args = argparse.Namespace(
            sequences=str(tmp_path / "seqs.csv"), out=str(tmp_path / "out"), k=9, salt=cd.DEFAULT_SALT,
            seed=1, n_train=0, n_val=0, n_test=10, n_all=0, n_train_heldout=0, verify_table=None,
        )
        cd.build_and_cache(args)
        archive = np.load(tmp_path / "out" / "splits.npz")
        assert set(archive.files) == {"test_kmers", "test_masks", "test_groups"}
        assert all(len(kmer) == 9 for kmer in archive["test_kmers"])

    def test_load_split_shuffles_before_taking_a_prefix(self, tmp_path):
        kmers = np.array(["AAAAAAAA", "CCCCCCCC", "DDDDDDDD", "EEEEEEEE", "FFFFFFFF", "GGGGGGGG"])
        masks = np.array([1, 2, 4, 8, 16, 32], dtype=np.uint16)
        groups = np.array([0, 1, 2, 3, 4, 5], dtype=np.uint8)
        np.savez(tmp_path / "splits.npz", train_kmers=kmers, train_masks=masks, train_groups=groups)
        peptides, labels = cd.load_split(str(tmp_path), "train", n=4, seed=3)
        assert len(peptides) == 4 and labels.shape == (4, len(cd.CATEGORIES))
        assert peptides != sorted(peptides)  # not the lexicographic prefix
        for peptide, row in zip(peptides, labels):  # labels stay attached to their k-mer
            assert row[int(np.log2(masks[list(kmers).index(peptide)]))] == 1.0
        again, _ = cd.load_split(str(tmp_path), "train", n=4, seed=3)
        assert again == peptides
        assert len(cd.load_split(str(tmp_path), "train")[0]) == 6
        _, _, buckets = cd.load_split(str(tmp_path), "train", n=4, seed=3, return_groups=True)
        assert [list(kmers).index(p) for p in peptides] == buckets.tolist()
        with pytest.raises(KeyError):
            cd.load_split(str(tmp_path), "all", return_groups=True)

    def test_build_and_cache_writes_arrays_and_metadata(self, tmp_path):
        keys = [_family_in_fold(0), _family_in_fold(1, ), _family_in_fold(2)]
        frame = pd.DataFrame({
            "identifier": [f"{keys[0]}_X", f"{keys[1]}_Y", f"{keys[2]}_Z"],
            "sequence": ["ACDEFGHIKLMNPQ", "STVWYACDEFGHIK", "MNPQRSTVWYACDE"],
            "category": ["archaea", "fungi", "plants"],
        })
        sequences = tmp_path / "seqs.csv"
        frame.to_csv(sequences)
        args = argparse.Namespace(
            sequences=str(sequences), out=str(tmp_path / "out"), k=8, salt=cd.DEFAULT_SALT,
            seed=1, n_train=3, n_val=3, n_test=100, n_all=7, n_train_heldout=2, verify_table=None,
        )
        cd.build_and_cache(args)
        archive = np.load(tmp_path / "out" / "splits.npz")
        expected = {f"{n}_{part}" for n in ("train", "val", "test", "all", "train_heldout")
                    for part in ("kmers", "masks")}
        expected |= {f"{n}_groups" for n in ("train", "val", "test", "train_heldout")}  # folds only
        assert set(archive.files) == expected
        assert set(archive["train_heldout_kmers"].tolist()).isdisjoint(archive["train_kmers"].tolist())
        assert len(archive["train_heldout_kmers"]) == 2
        assert archive["val_groups"].max() < cd.N_BUCKETS
        assert len(archive["train_kmers"]) == 3 and len(archive["val_kmers"]) == 3
        assert len(archive["all_kmers"]) == 7
        assert all(len(kmer) == 8 for kmer in archive["train_kmers"])
        meta = json.loads((tmp_path / "out" / "splits.json").read_text())
        assert meta["categories"] == cd.CATEGORIES
        assert meta["sample_sizes"]["train"] == 3
        assert meta["sample_sizes"]["train_heldout"] == 2
        assert set(meta["prevalence"]["all"]) == set(cd.CATEGORIES)
