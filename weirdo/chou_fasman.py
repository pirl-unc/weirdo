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

"""Re-exports from :mod:`pepdata.chou_fasman` for backwards compatibility."""

from pepdata.amino_acid_alphabet import canonical_amino_acids
from pepdata.chou_fasman import (
    alpha_helix_score,
    beta_sheet_score,
    chou_fasman_table,
    parse_chou_fasman,
    turn_score,
)

AMINO_ACID_NAME_TO_LETTER = {
    aa.full_name: aa.letter for aa in canonical_amino_acids
}

__all__ = [
    "AMINO_ACID_NAME_TO_LETTER",
    "alpha_helix_score",
    "beta_sheet_score",
    "chou_fasman_table",
    "parse_chou_fasman",
    "turn_score",
]
