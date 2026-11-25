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

"""Re-exports from :mod:`pepdata.amino_acid_properties` for backwards compatibility."""

from pepdata.amino_acid_properties import (
    CRASP_PROPERTIES_PATH,
    aa_dict_to_positional_list,
    accessible_surface_area,
    accessible_surface_area_folded,
    crasp_properties,
    hydropathy,
    hydrophilicity,
    local_flexibility,
    mass,
    pK_side_chain,
    parse_property_table,
    polarity,
    prct_exposed_residues,
    read_crasp_properties,
    refractivity,
    solvent_exposed_area,
    volume,
)

__all__ = [
    "CRASP_PROPERTIES_PATH",
    "aa_dict_to_positional_list",
    "accessible_surface_area",
    "accessible_surface_area_folded",
    "crasp_properties",
    "hydropathy",
    "hydrophilicity",
    "local_flexibility",
    "mass",
    "pK_side_chain",
    "parse_property_table",
    "polarity",
    "prct_exposed_residues",
    "read_crasp_properties",
    "refractivity",
    "solvent_exposed_area",
    "volume",
]
