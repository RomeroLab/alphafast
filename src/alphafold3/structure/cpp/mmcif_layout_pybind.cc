// Copyright 2024 DeepMind Technologies Limited
//
// AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
// this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
//
// To request access to the AlphaFold 3 model parameters, follow the process set
// out at https://github.com/google-deepmind/alphafold3. You may only use these
// if received directly from Google. Use is subject to terms of use available at
// https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

#include "alphafold3/structure/cpp/mmcif_layout.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "pybind11/stl.h"

namespace alphafold3 {

namespace py = pybind11;

void RegisterModuleMmcifLayout(pybind11::module m) {
  py::class_<MmcifLayout>(m, "MmcifLayout")
      .def("__str__", &MmcifLayout::ToDebugString)
      .def("num_models", &MmcifLayout::num_models)
      .def("num_chains", &MmcifLayout::num_chains)
      .def("num_residues", &MmcifLayout::num_residues)
      .def("num_atoms", &MmcifLayout::num_atoms)
      .def("residue_range", &MmcifLayout::residue_range, py::arg("chain_index"))
      .def("atom_range", &MmcifLayout::atom_range, py::arg("residue_index"))
      .def("chains", &MmcifLayout::chains,
           py::doc("Returns a list of indices one past the last residue of "
                   "each chain."))
      .def(
          "chain_starts", &MmcifLayout::chain_starts,
          py::doc("Returns a list of indices of the first atom of each chain."))
      .def("residues", &MmcifLayout::residues,
           py::doc("Returns a list of indices one past the last atom of each "
                   "residue."))
      .def("residue_starts", &MmcifLayout::residue_starts,
           py::doc(
               "Returns a list of indices of the first atom of each residue."))
      .def("model_offset", &MmcifLayout::model_offset,
           py::doc("Returns the first atom index that is part of the specified "
                   "model."));

  m.def("from_mmcif", &MmcifLayout::Create, py::arg("mmcif"),
        py::arg("model_id") = "");
}

}  // namespace alphafold3
