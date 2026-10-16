"""Brute-force checks for the foreignness-score engine."""

import numpy as np
import pytest

from scripts.foreignness_scores import (
    AA,
    AA_INDEX,
    KmerIndex,
    PeptideSetIndex,
    ProteomeIndex,
    allele_anchor_pairs,
    nearest_self_distance,
    similarity_matrix,
    tcr_face,
)

PROTEINS = [
    "MKTAYIAKQRQISFVKSHFSRQ",
    "GGSSTTNNQQAAKLVTGGCYNVA",
    "XXMKTAYXXLLLLLLLLLL",  # ambiguous residues must never be matched
]


def _brute_windows(length):
    result = []
    for protein in PROTEINS:
        for i in range(len(protein) - length + 1):
            window = protein[i:i + length]
            if set(window) <= set(AA):
                result.append(window)
    return result


def _brute(query, windows, positions, matrix, beta):
    columns = list(range(2, len(query) - 1)) if positions == "face" else list(
        range(len(query)))
    s = similarity_matrix(matrix).astype(float)
    sims = np.asarray([
        sum(s[AA_INDEX[query[c]], AA_INDEX[w[c]]] for c in columns)
        for w in windows])
    own = sum(s[AA_INDEX[query[c]], AA_INDEX[query[c]]] for c in columns)
    nearest = (own - sims.max()) / len(columns)
    soft = (own - np.log(np.exp(beta * sims).sum()) / beta) / len(columns)
    return nearest, soft


def test_windows_never_span_proteins_or_ambiguous_residues():
    index = ProteomeIndex(PROTEINS)
    seen = {"".join(AA[i] for i in w) for block in index.blocks(9) for w in block}
    assert seen == set(_brute_windows(9))
    assert index.count(9) == len(_brute_windows(9))


@pytest.mark.parametrize("positions", ["face", "whole"])
@pytest.mark.parametrize(
    "matrix", ["blosum62", "blosum50", "pmbec", "physchem", "hamming"])
def test_nearest_and_soft_match_brute_force_across_chunks(positions, matrix):
    index = ProteomeIndex(PROTEINS)
    queries = ["MKTAYIAKQ", "AAKLVSGGCYN", "LLMKTAYIAK", "GGSSTTNN"]
    for chunk in (1, 5, 1000):
        result = nearest_self_distance(
            index, queries, positions=positions, matrix=matrix, beta=1.0,
            chunk=chunk, query_batch=2)
        for i, query in enumerate(queries):
            nearest, soft = _brute(
                query, _brute_windows(len(query)), positions, matrix, 1.0)
            assert result["nearest"][i] == pytest.approx(nearest, abs=1e-4)
            assert result["soft"][i] == pytest.approx(soft, abs=1e-4)


def test_exact_window_has_zero_distance_and_an_unseen_length_is_nan():
    index = ProteomeIndex(PROTEINS)
    exact = "MKTAYIAKQ"
    result = nearest_self_distance(index, [exact, "ACDEFGHIKLMNPQ"[:13]])
    assert result["nearest"][0] == pytest.approx(0.0, abs=1e-5)
    # The proteins are long enough for 13-mer windows, so it is defined.
    assert not np.isnan(result["nearest"][1])
    tiny = ProteomeIndex(["ACDEFGHIK"])
    assert np.isnan(nearest_self_distance(tiny, ["ACDEFGHIKL"])["nearest"][0])


def test_allele_anchor_filter_restricts_the_reference():
    index = ProteomeIndex(PROTEINS)
    query = "MKTAYIAKQ"
    # Allow only windows whose P2 is K and whose P-omega is Q.
    table = np.zeros(400, dtype=bool)
    table[AA_INDEX["K"] * 20 + AA_INDEX["Q"]] = True
    filtered = nearest_self_distance(
        index, [query], groups=["A"], allowed_pairs={"A": table})
    allowed = [
        w for w in _brute_windows(9) if w[1] == "K" and w[-1] == "Q"]
    assert allowed
    nearest, _ = _brute(query, allowed, "face", "blosum62", 1.0)
    assert filtered["nearest"][0] == pytest.approx(nearest, abs=1e-4)
    unfiltered = nearest_self_distance(index, [query])
    assert filtered["nearest"][0] >= unfiltered["nearest"][0]
    # A pair no window satisfies leaves nothing to compare with.
    nothing = np.zeros(400, dtype=bool)
    assert np.isnan(nearest_self_distance(
        index, [query], groups=["A"], allowed_pairs={"A": nothing}
    )["nearest"][0])


def test_peptide_set_reference_compares_same_length_ligands_only():
    ligands = {
        "A*02:01": ["MKTAYIAKQ", "GGSSTTNNQ", "LLLLLLLLLL"],
        "B*07:02": ["PPPPPPPPP"],
    }
    index = PeptideSetIndex(ligands, fallback_minimum=2)
    result = nearest_self_distance(
        index, ["MKTAYIAKQ", "MKTAYIAKQR"], groups=["A*02:01", "A*02:01"])
    assert result["nearest"][0] == pytest.approx(0.0, abs=1e-5)
    # The 10-mer is compared with the single 10-mer ligand, not with a window.
    expected, _ = _brute("MKTAYIAKQR", ["LLLLLLLLLL"], "face", "blosum62", 1.0)
    assert result["nearest"][1] == pytest.approx(expected, abs=1e-4)
    # An allele with too few ligands falls back to the pooled set and is logged.
    fallback = nearest_self_distance(index, ["MKTAYIAKQ"], groups=["B*07:02"])
    assert fallback["nearest"][0] == pytest.approx(0.0, abs=1e-5)
    assert ("B*07:02", 9) in index.fallback_groups


def test_kmer_absence_and_surprisal_match_brute_force():
    index = ProteomeIndex(PROTEINS)
    kmers = KmerIndex(index, k_values=(3, 4))
    corpus = [w for p in PROTEINS for w in (
        p[i:i + 4] for i in range(len(p) - 3)) if set(w) <= set(AA)]
    queries = ["MKTAYIAKQ", "WWWWWWWWW", "ACDEFGHIK"]
    absent = kmers.absent_fraction(queries, 4)
    for i, query in enumerate(queries):
        face = tcr_face(query)
        windows = [face[j:j + 4] for j in range(len(face) - 3)]
        expected = np.mean([w not in set(corpus) for w in windows])
        assert absent[i] == pytest.approx(expected)
    surprisal = kmers.surprisal(queries, 4)
    assert surprisal[1] > surprisal[0]
    assert np.isnan(kmers.absent_fraction(["ACDE"], 4)[0])  # face too short


def test_anchor_pairs_keep_the_pairs_the_predictor_prefers():
    index = ProteomeIndex(PROTEINS * 5)

    class Fake:
        def predict(self, peptides, allele=None):
            return [10.0 if (p[1] == "L" and p[-1] == "V") else 20000.0
                    for p in peptides]

    tables, readable = allele_anchor_pairs(
        ["A*02:01"], Fake(), index, top_pairs=3, n_backgrounds=4)
    assert tables["A*02:01"][AA_INDEX["L"] * 20 + AA_INDEX["V"]]
    assert tables["A*02:01"].sum() == 3
    assert readable["A*02:01"][0] == "L_V"
