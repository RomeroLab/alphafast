# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

from collections.abc import Callable
from alphafold3.cpp import cif_dict


def get_internal_to_author_chain_id_map(
    mmcif: cif_dict.CifDict
) -> dict[str,str]: ...


def get_or_infer_type_symbol(
    mmcif: cif_dict.CifDict,
    atom_id_to_type_symbol: Callable[[str, str], str],
) -> list[str]: ...
