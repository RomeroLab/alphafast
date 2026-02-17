# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Some methods to compute metrics for PTMs."""

import collections
from collections.abc import Mapping
import dataclasses

from alphafold3 import structure
from alphafold3.constants import mmcif_names
from alphafold3.model.atom_layout import atom_layout
import numpy as np


@dataclasses.dataclass(frozen=True)
class ResIdMapping:
  old_res_ids: np.ndarray
  new_res_ids: np.ndarray


def _count_symmetric_chains(struc: structure.Structure) -> Mapping[str, int]:
  """Returns a dict with each chain ID and count."""
  chain_res_name_sequence_from_chain_id = struc.chain_res_name_sequence(
      include_missing_residues=True, fix_non_standard_polymer_res=False
  )
  counts_for_chain_res_name_sequence = collections.Counter(
      chain_res_name_sequence_from_chain_id.values()
  )
  chain_symmetric_count = {}
  for chain_id, chain_res_name in chain_res_name_sequence_from_chain_id.items():
    chain_symmetric_count[chain_id] = counts_for_chain_res_name_sequence[
        chain_res_name
    ]
  return chain_symmetric_count


def has_nonsymmetric_bonds_on_symmetric_polymer_chains(
    struc: structure.Structure, polymer_ligand_bonds: atom_layout.AtomLayout
) -> bool:
  """Returns true if nonsymmetric bonds found on polymer chains."""
  try:
    _get_polymer_dim(polymer_ligand_bonds)
  except ValueError:
    return True
  if _has_non_polymer_ligand_ptm_bonds(polymer_ligand_bonds):
    return True
  if _has_multiple_polymers_bonded_to_one_ligand(polymer_ligand_bonds):
    return True
  combined_struc, _ = _combine_polymer_ligand_ptm_chains(
      struc, polymer_ligand_bonds
  )
  struc = struc.filter(chain_type=mmcif_names.POLYMER_CHAIN_TYPES)
  combined_struc = combined_struc.filter(
      chain_type=mmcif_names.POLYMER_CHAIN_TYPES
  )
  return _count_symmetric_chains(struc) != _count_symmetric_chains(
      combined_struc
  )


def _has_non_polymer_ligand_ptm_bonds(
    polymer_ligand_bonds: atom_layout.AtomLayout,
):
  """Checks if all bonds are between a polymer chain and a ligand chain type."""
  for start_chain_type, end_chain_type in polymer_ligand_bonds.chain_type:
    if (
        start_chain_type in mmcif_names.POLYMER_CHAIN_TYPES
        and end_chain_type in mmcif_names.LIGAND_CHAIN_TYPES
    ):
      continue
    elif (
        start_chain_type in mmcif_names.LIGAND_CHAIN_TYPES
        and end_chain_type in mmcif_names.POLYMER_CHAIN_TYPES
    ):
      continue
    else:
      return True
  return False


def _combine_polymer_ligand_ptm_chains(
    struc: structure.Structure,
    polymer_ligand_bonds: atom_layout.AtomLayout,
) -> tuple[structure.Structure, dict[tuple[str, str], ResIdMapping]]:
  """Combines the ptm polymer-ligand chains together.

  This will prevent them from being permuted away from each other when chains
  are matched to the ground truth. This function also returns the res_id mapping
  from the separate ligand res_ids to their res_ids in the combined
  polymer-ligand chain; this information is needed to later separate the
  combined polymer-ligand chain.

  Args:
    struc: Structure to be modified.
    polymer_ligand_bonds: AtomLayout with polymer-ligand bond info.

  Returns:
    A tuple of a Structure with each ptm polymer-ligand chain relabelled as one
    chain and a dict from bond chain pair to the res_id mapping.
  """
  if not _has_only_single_bond_from_each_chain(polymer_ligand_bonds):
    if _has_multiple_ligands_bonded_to_one_polymer(polymer_ligand_bonds):
      # For structures where a polymer chain is connected to multiple ligands,
      # we need to sort the multiple bonds from the same chain by res_id to
      # ensure that the combined polymer-ligand chain will always be the same
      # when you have repeated symmetric polymer-ligand chains.
      polymer_ligand_bonds = (
          _sort_polymer_ligand_bonds_by_polymer_chain_and_res_id(
              polymer_ligand_bonds
          )
      )
    else:
      raise ValueError(
          'Code cannot handle multiple bonds from one chain unless'
          ' its several ligands bonded to a polymer.'
      )
  res_id_mappings_for_bond_chain_pair = dict()
  for (start_chain_id, end_chain_id), (start_chain_type, end_chain_type) in zip(
      polymer_ligand_bonds.chain_id, polymer_ligand_bonds.chain_type
  ):
    poly_info, ligand_info = _get_polymer_and_ligand_chain_ids_and_types(
        start_chain_id, end_chain_id, start_chain_type, end_chain_type
    )
    polymer_chain_id, polymer_chain_type = poly_info
    ligand_chain_id, _ = ligand_info

    # Join the ligand chain to the polymer chain.
    ligand_res_ids = struc.filter(chain_id=ligand_chain_id).res_id
    new_res_ids = ligand_res_ids + len(struc.all_residues[polymer_chain_id])
    res_id_mappings_for_bond_chain_pair[(polymer_chain_id, ligand_chain_id)] = (
        ResIdMapping(old_res_ids=ligand_res_ids, new_res_ids=new_res_ids)
    )
    chain_groups = []
    chain_group_ids = []
    chain_group_types = []
    for chain_id, chain_type in zip(
        struc.chains_table.id, struc.chains_table.type
    ):
      if chain_id == ligand_chain_id:
        continue
      elif chain_id == polymer_chain_id:
        chain_groups.append([polymer_chain_id, ligand_chain_id])
        chain_group_ids.append(polymer_chain_id)
        chain_group_types.append(polymer_chain_type)
      else:
        chain_groups.append([chain_id])
        chain_group_ids.append(chain_id)
        chain_group_types.append(chain_type)

    struc = struc.merge_chains(
        chain_groups=chain_groups,
        chain_group_ids=chain_group_ids,
        chain_group_types=chain_group_types,
    )

  return struc, res_id_mappings_for_bond_chain_pair


def _has_only_single_bond_from_each_chain(
    polymer_ligand_bonds: atom_layout.AtomLayout,
) -> bool:
  """Checks that there is at most one bond from each chain."""
  chain_ids = []
  for chains in polymer_ligand_bonds.chain_id:
    chain_ids.extend(chains)
  if len(chain_ids) != len(set(chain_ids)):
    return False
  return True


def _get_polymer_and_ligand_chain_ids_and_types(
    start_chain_id: str,
    end_chain_id: str,
    start_chain_type: str,
    end_chain_type: str,
) -> tuple[tuple[str, str], tuple[str, str]]:
  """Finds polymer and ligand chain ids from chain types."""
  if (
      start_chain_type in mmcif_names.POLYMER_CHAIN_TYPES
      and end_chain_type in mmcif_names.LIGAND_CHAIN_TYPES
  ):
    return (start_chain_id, start_chain_type), (end_chain_id, end_chain_type)
  elif (
      start_chain_type in mmcif_names.LIGAND_CHAIN_TYPES
      and end_chain_type in mmcif_names.POLYMER_CHAIN_TYPES
  ):
    return (end_chain_id, end_chain_type), (start_chain_id, start_chain_type)
  else:
    raise ValueError(
        'This code only handles PTM-bonds from polymer chain to ligands.'
    )


def _get_polymer_dim(polymer_ligand_bonds: atom_layout.AtomLayout) -> int:
  """Gets polymer dimension from the polymer-ligand bond layout."""
  start_chain_types = []
  end_chain_types = []
  for start_chain_type, end_chain_type in polymer_ligand_bonds.chain_type:
    start_chain_types.append(start_chain_type)
    end_chain_types.append(end_chain_type)
  if set(start_chain_types).issubset(
      set(mmcif_names.POLYMER_CHAIN_TYPES)
  ) and set(end_chain_types).issubset(set(mmcif_names.LIGAND_CHAIN_TYPES)):
    return 0
  elif set(start_chain_types).issubset(mmcif_names.LIGAND_CHAIN_TYPES) and set(
      end_chain_types
  ).issubset(set(mmcif_names.POLYMER_CHAIN_TYPES)):
    return 1
  else:
    raise ValueError(
        'Polymer and ligand dimensions are not consistent within the structure.'
    )


def _has_multiple_ligands_bonded_to_one_polymer(polymer_ligand_bonds):
  """Checks if there are multiple ligands bonded to one polymer."""
  polymer_dim = _get_polymer_dim(polymer_ligand_bonds)
  polymer_chain_ids = [
      chains[polymer_dim] for chains in polymer_ligand_bonds.chain_id
  ]
  if len(polymer_chain_ids) != len(set(polymer_chain_ids)):
    return True
  return False


def _has_multiple_polymers_bonded_to_one_ligand(polymer_ligand_bonds):
  """Checks if there are multiple polymer chains bonded to one ligand."""
  polymer_dim = _get_polymer_dim(polymer_ligand_bonds)
  ligand_dim = 1 - polymer_dim
  ligand_chain_ids = [
      chains[ligand_dim] for chains in polymer_ligand_bonds.chain_id
  ]
  if len(ligand_chain_ids) != len(set(ligand_chain_ids)):
    return True
  return False


def _sort_polymer_ligand_bonds_by_polymer_chain_and_res_id(
    polymer_ligand_bonds,
):
  """Sorts bonds by res_id (for when a polymer chain has multiple bonded ligands)."""

  polymer_dim = _get_polymer_dim(polymer_ligand_bonds)

  polymer_chain_ids = [
      chains[polymer_dim] for chains in polymer_ligand_bonds.chain_id
  ]
  polymer_res_ids = [res[polymer_dim] for res in polymer_ligand_bonds.res_id]

  polymer_chain_and_res_id = zip(polymer_chain_ids, polymer_res_ids)
  sorted_indices = [
      idx
      for idx, _ in sorted(
          enumerate(polymer_chain_and_res_id), key=lambda x: x[1]
      )
  ]
  return polymer_ligand_bonds[sorted_indices]
