# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Re-exports from :mod:`pepdata.amino_acid_alphabet` for backwards compatibility."""

from pepdata.amino_acid_alphabet import (
    AminoAcid,
    amino_acid_letter_indices,
    amino_acid_letter_pairs,
    amino_acid_name_indices,
    amino_acid_pair_positions,
    canonical_amino_acid_letters,
    canonical_amino_acids,
    canonical_amino_acids_with_unknown,
    dict_to_amino_acid_matrix,
    extended_amino_acid_letters,
    extended_amino_acids,
    extended_amino_acids_with_unknown_names,
    index_to_full_name,
    index_to_letter,
    index_to_short_name,
    letter_to_index,
    letter_to_short_name,
    modified_amino_acids,
    peptide_to_indices,
    peptide_to_short_amino_acid_names,
    rare_amino_acids,
    wildcard_amino_acids,
)

__all__ = [
    "AminoAcid",
    "amino_acid_letter_indices",
    "amino_acid_letter_pairs",
    "amino_acid_name_indices",
    "amino_acid_pair_positions",
    "canonical_amino_acid_letters",
    "canonical_amino_acids",
    "canonical_amino_acids_with_unknown",
    "dict_to_amino_acid_matrix",
    "extended_amino_acid_letters",
    "extended_amino_acids",
    "extended_amino_acids_with_unknown_names",
    "index_to_full_name",
    "index_to_letter",
    "index_to_short_name",
    "letter_to_index",
    "letter_to_short_name",
    "modified_amino_acids",
    "peptide_to_indices",
    "peptide_to_short_amino_acid_names",
    "rare_amino_acids",
    "wildcard_amino_acids",
]
