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

"""Re-exports from :mod:`pepdata.residue_contact_energies` for backwards compatibility."""

from pepdata.residue_contact_energies import (
    CONTACT_TABLE_AMINO_ACID_ORDER,
    coil_vs_helix_array,
    coil_vs_helix_dict,
    coil_vs_strand_array,
    coil_vs_strand_dict,
    helix_vs_coil_array,
    helix_vs_coil_dict,
    helix_vs_strand_array,
    helix_vs_strand_dict,
    parse_interaction_table,
    strand_vs_coil_array,
    strand_vs_coil_dict,
    strand_vs_helix_array,
    strand_vs_helix_dict,
    transpose_interaction_dict,
)

# Backward-compatible aliases matching historical public names used in docs.
strand_vs_coil = strand_vs_coil_dict
coil_vs_strand = coil_vs_strand_dict
helix_vs_strand = helix_vs_strand_dict
strand_vs_helix = strand_vs_helix_dict
helix_vs_coil = helix_vs_coil_dict
coil_vs_helix = coil_vs_helix_dict

__all__ = [
    "CONTACT_TABLE_AMINO_ACID_ORDER",
    "parse_interaction_table",
    "transpose_interaction_dict",
    "strand_vs_coil",
    "coil_vs_strand",
    "helix_vs_strand",
    "strand_vs_helix",
    "helix_vs_coil",
    "coil_vs_helix",
    "strand_vs_coil_dict",
    "coil_vs_strand_dict",
    "helix_vs_strand_dict",
    "strand_vs_helix_dict",
    "helix_vs_coil_dict",
    "coil_vs_helix_dict",
    "strand_vs_coil_array",
    "coil_vs_strand_array",
    "helix_vs_strand_array",
    "strand_vs_helix_array",
    "helix_vs_coil_array",
    "coil_vs_helix_array",
]
