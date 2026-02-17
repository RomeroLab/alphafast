# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Interface and implementations for fetching MSA data."""

from collections.abc import Sequence
from typing import Protocol, TypeAlias

from alphafold3.data import msa
from alphafold3.data import msa_config


MsaErrors: TypeAlias = Sequence[tuple[msa_config.RunConfig, str]]


class MsaProvider(Protocol):
  """Interface for providing Multiple Sequence Alignments."""

  def __call__(
      self,
      query_sequence: str,
      chain_polymer_type: str,
  ) -> tuple[msa.Msa, MsaErrors]:
    """Retrieve MSA for the given polymer query_sequence.

    Args:
      query_sequence: The residue sequence of the polymer to search for.
      chain_polymer_type: The polymer type of the query_sequence. This must
        match the chain_polymer_type of the provider.

    Returns:
      A tuple containing the MSA and MsaErrors. MsaErrors is a Sequence
      containing a tuple for each msa_query that failed. Each tuple contains
      the failing query and the associated error message.
    """


class EmptyMsaProvider:
  """MSA provider that returns just the query sequence, useful for testing."""

  def __init__(self, chain_polymer_type: str):
    self._chain_polymer_type = chain_polymer_type

  def __call__(
      self, query_sequence: str, chain_polymer_type: str
  ) -> tuple[msa.Msa, MsaErrors]:
    """Returns an MSA containing just the query sequence, never errors."""
    if chain_polymer_type != self._chain_polymer_type:
      raise ValueError(
          f'EmptyMsaProvider of type {self._chain_polymer_type} called with '
          f'sequence of {chain_polymer_type=}, {query_sequence=}.'
      )
    return (
        msa.Msa.from_empty(
            query_sequence=query_sequence,
            chain_poly_type=self._chain_polymer_type,
        ),
        (),
    )
