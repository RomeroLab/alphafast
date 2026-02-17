# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Interface and implementations for fetching templates data."""

from collections.abc import Mapping
import datetime
from typing import Any, Protocol, TypeAlias


TemplateFeatures: TypeAlias = Mapping[str, Any]


class TemplateFeatureProvider(Protocol):
  """Interface for providing Template Features."""

  def __call__(
      self,
      sequence: str,
      release_date: datetime.date | None,
      include_ligand_features: bool = True,
  ) -> TemplateFeatures:
    """Retrieve template features for the given sequence and release_date.

    Args:
      sequence: The residue sequence of the query.
      release_date: The release_date of the template query, this is used to
        filter templates for training, ensuring that they do not leak structure
        information from the future.
      include_ligand_features: Whether to include ligand features.

    Returns:
      Template features: A mapping of template feature labels to features, which
        may be numpy arrays, bytes objects, or for the special case of label
        `ligand_features`, a nested feature map of labels to numpy arrays.

    Raises:
      TemplateRetrievalError if the template features were not found.
    """
