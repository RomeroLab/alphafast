/*
 * Copyright 2024 DeepMind Technologies Limited
 *
 * AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
 * this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
 *
 * To request access to the AlphaFold 3 model parameters, follow the process set
 * out at https://github.com/google-deepmind/alphafold3. You may only use these
 * if received directly from Google. Use is subject to terms of use available at
 * https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
 */

#ifndef ALPHAFOLD3_SRC_ALPHAFOLD3_STRUCTURE_PYTHON_MMCIF_STRUCT_CONN_H_
#define ALPHAFOLD3_SRC_ALPHAFOLD3_STRUCTURE_PYTHON_MMCIF_STRUCT_CONN_H_

#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "alphafold3/parsers/cpp/cif_dict_lib.h"

namespace alphafold3 {

// Returns a pair of atom indices for each row in the bonds table (aka
// _struct_conn). The indices are simple 0-based indexes into the columns of
// the _atom_site table in the input mmCIF, and do not necessarily correspond
// to the values in _atom_site.id, or any other column.
absl::StatusOr<std::pair<std::vector<std::size_t>, std::vector<std::size_t>>>
GetBondAtomIndices(const CifDict& mmcif, absl::string_view model_id);

}  // namespace alphafold3

#endif  // ALPHAFOLD3_SRC_ALPHAFOLD3_STRUCTURE_PYTHON_MMCIF_STRUCT_CONN_H_
