from weirdo.amino_acid_alphabet import (
    canonical_amino_acids,
    canonical_amino_acid_letters,
    extended_amino_acids,
    extended_amino_acid_letters,
    index_to_letter,
    peptide_to_short_amino_acid_names,
)


def test_canonical_amino_acids():
    assert len(canonical_amino_acids) == 20


def test_canonical_amino_acids_letters():
    assert len(canonical_amino_acid_letters) == 20
    assert "X" not in canonical_amino_acid_letters
    expected_letters = [aa.letter for aa in canonical_amino_acids]
    assert expected_letters == canonical_amino_acid_letters


def test_extended_amino_acids():
    assert len(extended_amino_acids) > 20


def test_extended_amino_acids_letters():
    assert len(extended_amino_acid_letters) > 20
    assert "X" in extended_amino_acid_letters
    assert "J" in extended_amino_acid_letters
    expected_letters = [aa.letter for aa in extended_amino_acids]
    assert expected_letters == extended_amino_acid_letters


def test_index_to_letter_returns_letter():
    assert index_to_letter(0) == "A"
    assert index_to_letter(1) == "R"


def test_peptide_to_short_amino_acid_names_returns_short_names():
    assert peptide_to_short_amino_acid_names("AR") == ["Ala", "Arg"]
