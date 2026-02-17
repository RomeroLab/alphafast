# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Functions relating to spatial locations of atoms within a structure."""

from collections.abc import Collection, Sequence

from alphafold3 import structure
from alphafold3.structure import mmcif
import numpy as np
import scipy


def _make_atom_has_clash_mask(
    kd_query_result: np.ndarray,
    struc: structure.Structure,
    ignore_chains: Collection[str],
) -> np.ndarray:
  """Returns a boolean NumPy array representing whether each atom has a clash.

  Args:
    kd_query_result: NumPy array containing N-atoms arrays, each array
      containing indices to atoms that clash with the N'th atom.
    struc: Structure over which clashes were detected.
    ignore_chains: Collection of chains that should not be considered clashing.
      A boolean NumPy array of length N atoms.
  """
  atom_is_clashing = np.zeros((struc.num_atoms,), dtype=bool)
  for atom_index, clashes in enumerate(kd_query_result):
    chain_i = struc.chain_id[atom_index]
    if chain_i in ignore_chains:
      continue
    islig_i = struc.is_ligand_mask[atom_index]
    for clashing_atom_index in clashes:
      chain_c = struc.chain_id[clashing_atom_index]
      if chain_c in ignore_chains:
        continue
      islig_c = struc.is_ligand_mask[clashing_atom_index]
      if (
          clashing_atom_index == atom_index
          or chain_i == chain_c
          or islig_i != islig_c
      ):
        # Ignore clashes within chain or between ligand and polymer.
        continue
      atom_is_clashing[atom_index] = True
  return atom_is_clashing


def find_clashing_chains(
    struc: structure.Structure,
    clash_thresh_angstrom: float = 1.7,
    clash_thresh_fraction: float = 0.3,
) -> Sequence[str]:
  """Finds chains that clash with others.

  Clashes are defined by polymer backbone atoms and all ligand atoms.
  Ligand-polymer clashes are not dropped.

  Will not find clashes if all coordinates are 0. Coordinates are all 0s if
  the structure is generated from sequences only, as done for inference in
  dendro for example.

  Args:
    struc: The structure defining the chains and atom positions.
    clash_thresh_angstrom: Below this distance, atoms are considered clashing.
    clash_thresh_fraction: Chains with more than this fraction of their atoms
      considered clashing will be dropped. This value should be in the range (0,
      1].

  Returns:
    A sequence of chain ids for chains that clash.

  Raises:
    ValueError: If `clash_thresh_fraction` is not in range (0,1].
  """
  if not 0 < clash_thresh_fraction <= 1:
    raise ValueError('clash_thresh_fraction must be in range (0,1]')

  struc_backbone = struc.filter_polymers_to_single_atom_per_res()
  if struc_backbone.num_chains == 0:
    return []

  # If the coordinates are all 0, do not search for clashes.
  if not np.any(struc_backbone.coords):
    return []

  coord_kdtree = scipy.spatial.cKDTree(struc_backbone.coords)

  # For each atom coordinate, find all atoms within the clash thresh radius.
  clashing_per_atom = coord_kdtree.query_ball_point(
      struc_backbone.coords, r=clash_thresh_angstrom
  )
  chain_ids = struc_backbone.chains
  if struc_backbone.atom_occupancy is not None:
    chain_occupancy = np.array([
        np.mean(struc_backbone.atom_occupancy[start:end])
        for start, end in struc_backbone.iter_chain_ranges()
    ])
  else:
    chain_occupancy = None

  # Remove chains until no more significant clashing.
  chains_to_remove = set()
  for _ in range(len(chain_ids)):
    # Calculate maximally clashing.
    atom_has_clash = _make_atom_has_clash_mask(
        clashing_per_atom, struc_backbone, chains_to_remove
    )
    clashes_per_chain = np.array([
        atom_has_clash[start:end].mean()
        for start, end in struc_backbone.iter_chain_ranges()
    ])
    max_clash = np.max(clashes_per_chain)
    if max_clash <= clash_thresh_fraction:
      # None of the remaining chains exceed the clash fraction threshold, so
      # we can exit.
      break

    # Greedily remove worst with the lowest occupancy.
    most_clashes = np.nonzero(clashes_per_chain == max_clash)[0]
    if chain_occupancy is not None:
      occupancy_clashing = chain_occupancy[most_clashes]
      last_lowest_occupancy = (
          len(occupancy_clashing) - np.argmin(occupancy_clashing[::-1]) - 1
      )
      worst_and_last = most_clashes[last_lowest_occupancy]
    else:
      worst_and_last = most_clashes[-1]

    chains_to_remove.add(chain_ids[worst_and_last])

  return sorted(chains_to_remove, key=mmcif.str_id_to_int_id)
