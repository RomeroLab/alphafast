# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Structure module initialization."""

# pylint: disable=g-importing-member
from alphafold3.structure.bioassemblies import BioassemblyData
from alphafold3.structure.bonds import Bonds
from alphafold3.structure.chemical_components import ChemCompEntry
from alphafold3.structure.chemical_components import ChemicalComponentsData
from alphafold3.structure.chemical_components import get_data_for_ccd_components
from alphafold3.structure.chemical_components import populate_missing_ccd_data
from alphafold3.structure.mmcif import BondParsingError
from alphafold3.structure.parsing import BondAtomId
from alphafold3.structure.parsing import from_atom_arrays
from alphafold3.structure.parsing import from_mmcif
from alphafold3.structure.parsing import from_parsed_mmcif
from alphafold3.structure.parsing import from_res_arrays
from alphafold3.structure.parsing import from_sequences_and_bonds
from alphafold3.structure.parsing import ModelID
from alphafold3.structure.parsing import NoAtomsError
from alphafold3.structure.parsing import SequenceFormat
from alphafold3.structure.structure import ARRAY_FIELDS
from alphafold3.structure.structure import AuthorNamingScheme
from alphafold3.structure.structure import Bond
from alphafold3.structure.structure import CascadeDelete
from alphafold3.structure.structure import concat
from alphafold3.structure.structure import enumerate_residues
from alphafold3.structure.structure import fix_non_standard_polymer_residues
from alphafold3.structure.structure import GLOBAL_FIELDS
from alphafold3.structure.structure import make_empty_structure
from alphafold3.structure.structure import MissingAtomError
from alphafold3.structure.structure import MissingAuthorResidueIdError
from alphafold3.structure.structure import multichain_residue_index
from alphafold3.structure.structure import stack
from alphafold3.structure.structure import Structure
from alphafold3.structure.structure_tables import Atoms
from alphafold3.structure.structure_tables import Chains
from alphafold3.structure.structure_tables import Residues
