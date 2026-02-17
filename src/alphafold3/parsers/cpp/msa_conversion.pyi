# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Type annotations for Python bindings for `msa_conversion`.

The type annotations in this file were modified from the automatically generated
stubgen output.
"""

from collections.abc import Iterable


def align_sequence_to_gapless_query(
    sequence: str | bytes,
    query_sequence: str | bytes,
) -> str: ...


def convert_a3m_to_stockholm(a3m_sequences: Iterable[str]) -> list[str]: ...
