# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
# Modifications Copyright 2026 Romero Lab, Duke University

"""Functions for getting MSA and calculating alignment features."""

from collections.abc import MutableMapping, Sequence
from concurrent import futures
import string
from typing import Self

from absl import logging
from alphafold3.constants import mmcif_names
from alphafold3.data import msa_config
from alphafold3.data import msa_features
from alphafold3.data import parsers
from alphafold3.data.tools import msa_tool
import numpy as np


class Error(Exception):
    """Error indicatating a problem with MSA Search."""


def _featurize(seq: str, chain_poly_type: str) -> str | list[int]:
    if mmcif_names.is_standard_polymer_type(chain_poly_type):
        featurized_seqs, _ = msa_features.extract_msa_features(
            msa_sequences=[seq], chain_poly_type=chain_poly_type
        )
        return featurized_seqs[0].tolist()
    # For anything else simply require an identical match.
    return seq


def sequences_are_feature_equivalent(
    sequence1: str,
    sequence2: str,
    chain_poly_type: str,
) -> bool:
    feat1 = _featurize(sequence1, chain_poly_type)
    feat2 = _featurize(sequence2, chain_poly_type)
    return feat1 == feat2


class Msa:
    """Multiple Sequence Alignment container with methods for manipulating it."""

    def __init__(
        self,
        query_sequence: str,
        chain_poly_type: str,
        sequences: Sequence[str],
        descriptions: Sequence[str],
        deduplicate: bool = True,
    ):
        """Raw constructor, prefer using the from_{a3m,multiple_msas} class methods.

        The first sequence must be equal (in featurised form) to the query sequence.
        If sequences/descriptions are empty, they will be initialised to the query.

        Args:
          query_sequence: The sequence that was used to search for MSA.
          chain_poly_type: Polymer type of the query sequence, see mmcif_names.
          sequences: The sequences returned by the MSA search tool.
          descriptions: Metadata for the sequences returned by the MSA search tool.
          deduplicate: If True, the MSA sequences will be deduplicated in the input
            order. Lowercase letters (insertions) are ignored when deduplicating.
        """
        if len(sequences) != len(descriptions):
            raise ValueError("The number of sequences and descriptions must match.")

        self.query_sequence = query_sequence
        self.chain_poly_type = chain_poly_type

        if not deduplicate:
            self.sequences = sequences
            self.descriptions = descriptions
        else:
            self.sequences = []
            self.descriptions = []
            # A replacement table that removes all lowercase characters.
            deletion_table = str.maketrans("", "", string.ascii_lowercase)
            unique_sequences = set()
            for seq, desc in zip(sequences, descriptions, strict=True):
                # Using string.translate is faster than re.sub('[a-z]+', '').
                sequence_no_deletions = seq.translate(deletion_table)
                if sequence_no_deletions not in unique_sequences:
                    unique_sequences.add(sequence_no_deletions)
                    self.sequences.append(seq)
                    self.descriptions.append(desc)

        # Make sure the MSA always has at least the query.
        self.sequences = self.sequences or [query_sequence]
        self.descriptions = self.descriptions or ["Original query"]

        # Check if the 1st MSA sequence matches the query sequence. Since it may be
        # mutated by the search tool (jackhmmer) check using the featurized version.
        if not sequences_are_feature_equivalent(
            self.sequences[0], query_sequence, chain_poly_type
        ):
            raise ValueError(
                f"First MSA sequence {self.sequences[0]} is not the {query_sequence=}"
            )

    @classmethod
    def from_multiple_msas(cls, msas: Sequence[Self], deduplicate: bool = True) -> Self:
        """Initializes the MSA from multiple MSAs.

        Args:
          msas: A sequence of Msa objects representing individual MSAs produced by
            different tools/dbs.
          deduplicate: If True, the MSA sequences will be deduplicated in the input
            order. Lowercase letters (insertions) are ignored when deduplicating.

        Returns:
          An Msa object created by merging multiple MSAs.
        """
        if not msas:
            raise ValueError("At least one MSA must be provided.")

        query_sequence = msas[0].query_sequence
        chain_poly_type = msas[0].chain_poly_type
        sequences = []
        descriptions = []

        for msa in msas:
            if msa.query_sequence != query_sequence:
                raise ValueError(
                    f"Query sequences must match: {[m.query_sequence for m in msas]}"
                )
            if msa.chain_poly_type != chain_poly_type:
                raise ValueError(
                    f"Chain poly types must match: {[m.chain_poly_type for m in msas]}"
                )
            sequences.extend(msa.sequences)
            descriptions.extend(msa.descriptions)

        return cls(
            query_sequence=query_sequence,
            chain_poly_type=chain_poly_type,
            sequences=sequences,
            descriptions=descriptions,
            deduplicate=deduplicate,
        )

    @classmethod
    def from_multiple_a3ms(
        cls, a3ms: Sequence[str], chain_poly_type: str, deduplicate: bool = True
    ) -> Self:
        """Initializes the MSA from multiple A3M strings.

        Args:
          a3ms: A sequence of A3M strings representing individual MSAs produced by
            different tools/dbs.
          chain_poly_type: Polymer type of the query sequence, see mmcif_names.
          deduplicate: If True, the MSA sequences will be deduplicated in the input
            order. Lowercase letters (insertions) are ignored when deduplicating.

        Returns:
          An Msa object created by merging multiple A3Ms.
        """
        if not a3ms:
            raise ValueError("At least one A3M must be provided.")

        query_sequence = None
        all_sequences = []
        all_descriptions = []

        for a3m in a3ms:
            sequences, descriptions = parsers.parse_fasta(a3m)
            if query_sequence is None:
                query_sequence = sequences[0]

            if sequences[0] != query_sequence:
                raise ValueError(
                    f"Query sequences must match: {sequences[0]=} != {query_sequence=}"
                )
            all_sequences.extend(sequences)
            all_descriptions.extend(descriptions)

        return cls(
            query_sequence=query_sequence,
            chain_poly_type=chain_poly_type,
            sequences=all_sequences,
            descriptions=all_descriptions,
            deduplicate=deduplicate,
        )

    @classmethod
    def from_a3m(
        cls,
        query_sequence: str,
        chain_poly_type: str,
        a3m: str,
        max_depth: int | None = None,
        deduplicate: bool = True,
    ) -> Self:
        """Parses the single A3M and builds the Msa object."""
        sequences, descriptions = parsers.parse_fasta(a3m)

        if max_depth is not None and 0 < max_depth < len(sequences):
            logging.info(
                "MSA cropped from depth of %d to %d for %s.",
                len(sequences),
                max_depth,
                query_sequence,
            )
            sequences = sequences[:max_depth]
            descriptions = descriptions[:max_depth]

        return cls(
            query_sequence=query_sequence,
            chain_poly_type=chain_poly_type,
            sequences=sequences,
            descriptions=descriptions,
            deduplicate=deduplicate,
        )

    @classmethod
    def from_empty(cls, query_sequence: str, chain_poly_type: str) -> Self:
        """Creates an empty Msa containing just the query sequence."""
        return cls(
            query_sequence=query_sequence,
            chain_poly_type=chain_poly_type,
            sequences=[],
            descriptions=[],
            deduplicate=False,
        )

    @property
    def depth(self) -> int:
        return len(self.sequences)

    def __repr__(self) -> str:
        return f"Msa({self.depth} sequences, {self.chain_poly_type})"

    def to_a3m(self) -> str:
        """Returns the MSA in the A3M format."""
        a3m_lines = []
        for desc, seq in zip(self.descriptions, self.sequences, strict=True):
            a3m_lines.append(f">{desc}")
            a3m_lines.append(seq)
        return "\n".join(a3m_lines) + "\n"

    def featurize(self) -> MutableMapping[str, np.ndarray]:
        """Featurises the MSA and returns a map of feature names to features.

        Returns:
          A dictionary mapping feature names to values.

        Raises:
          msa.Error:
            * If the sequences in the MSA don't have the same length after deletions
              (lower case letters) are removed.
            * If the MSA contains an unknown amino acid code.
            * If there are no sequences after aligning.
        """
        try:
            msa, deletion_matrix = msa_features.extract_msa_features(
                msa_sequences=self.sequences, chain_poly_type=self.chain_poly_type
            )
        except ValueError as e:
            raise Error(f"Error extracting MSA or deletion features: {e}") from e

        if msa.shape == (0, 0):
            raise Error(f"Empty MSA feature for {self}")

        species_ids = msa_features.extract_species_ids(self.descriptions)

        return {
            "msa_species_identifiers": np.array(species_ids, dtype=object),
            "num_alignments": np.array(self.depth, dtype=np.int32),
            "msa": msa,
            "deletion_matrix": deletion_matrix,
        }


def get_msa_tool(
    msa_tool_config: msa_config.MmseqsConfig,
) -> msa_tool.MsaTool:
    """Returns the requested MSA tool."""

    match msa_tool_config:
        case msa_config.MmseqsConfig():
            from alphafold3.data.tools import mmseqs

            return mmseqs.Mmseqs(
                binary_path=msa_tool_config.binary_path,
                database_path=msa_tool_config.database_config.path,
                e_value=msa_tool_config.e_value,
                sensitivity=msa_tool_config.sensitivity,
                max_sequences=msa_tool_config.max_sequences,
                gpu_enabled=msa_tool_config.gpu_enabled,
                gpu_device=msa_tool_config.gpu_device,
                threads=msa_tool_config.threads,
            )
        case _:
            raise ValueError(f"Unknown MSA tool: {msa_tool_config}.")


def get_msa(
    target_sequence: str,
    run_config: msa_config.RunConfig,
    chain_poly_type: str,
    deduplicate: bool = False,
) -> Msa:
    """Computes the MSA for a given query sequence.

    Args:
      target_sequence: The target amino-acid sequence.
      run_config: MSA run configuration.
      chain_poly_type: The type of chain for which to get an MSA.
      deduplicate: If True, the MSA sequences will be deduplicated in the input
        order. Lowercase letters (insertions) are ignored when deduplicating.

    Returns:
      Aligned MSA sequences.
    """

    return Msa.from_a3m(
        query_sequence=target_sequence,
        chain_poly_type=chain_poly_type,
        a3m=get_msa_tool(run_config.config).query(target_sequence).a3m,
        max_depth=run_config.crop_size,
        deduplicate=deduplicate,
    )


def get_msa_pipelined(
    target_sequence: str,
    run_config: msa_config.RunConfig,
    chain_poly_type: str,
    executor: futures.ThreadPoolExecutor,
    deduplicate: bool = False,
) -> futures.Future["Msa"]:
    """Computes the MSA for a given query sequence using pipelined execution.

    This function runs the GPU search synchronously but submits post-processing
    to the executor, allowing overlap between GPU searches and CPU post-processing
    across multiple databases.

    Only works with MmseqsConfig. For other tools, use get_msa() instead.

    Args:
        target_sequence: The target amino-acid sequence.
        run_config: MSA run configuration (must be MmseqsConfig).
        chain_poly_type: The type of chain for which to get an MSA.
        executor: ThreadPoolExecutor for running post-processing tasks.
        deduplicate: If True, the MSA sequences will be deduplicated.

    Returns:
        Future that resolves to aligned MSA sequences.

    Raises:
        ValueError: If run_config is not MmseqsConfig.
    """
    if not isinstance(run_config.config, msa_config.MmseqsConfig):
        raise ValueError(
            f"get_msa_pipelined only supports MmseqsConfig, got {type(run_config.config)}"
        )

    from alphafold3.data.tools import mmseqs

    tool = mmseqs.Mmseqs(
        binary_path=run_config.config.binary_path,
        database_path=run_config.config.database_config.path,
        e_value=run_config.config.e_value,
        sensitivity=run_config.config.sensitivity,
        max_sequences=run_config.config.max_sequences,
        gpu_enabled=run_config.config.gpu_enabled,
        gpu_device=run_config.config.gpu_device,
        threads=run_config.config.threads,
    )

    # Get the future from pipelined query (GPU search runs now, post-processing async)
    msa_result_future = tool.query_pipelined(target_sequence, executor)

    # Wrap the future to convert MsaToolResult to Msa
    def convert_result(result_future: futures.Future[msa_tool.MsaToolResult]) -> Msa:
        msa_result = result_future.result()
        return Msa.from_a3m(
            query_sequence=target_sequence,
            chain_poly_type=chain_poly_type,
            a3m=msa_result.a3m,
            max_depth=run_config.crop_size,
            deduplicate=deduplicate,
        )

    # Submit the conversion to run after the msa_result completes
    return executor.submit(convert_result, msa_result_future)


def get_msa_shared_db_pipelined(
    target_sequence: str,
    query_db: str,
    run_configs: list[msa_config.RunConfig],
    chain_poly_type: str,
    executor: futures.ThreadPoolExecutor,
    deduplicate: bool = False,
) -> list[futures.Future["Msa"]]:
    """Get MSAs for a sequence across multiple databases using a shared query DB.

    This function uses a pre-created query database to avoid redundant createdb
    calls when searching the same sequence across multiple target databases.
    GPU searches run sequentially, but CPU post-processing runs in parallel.

    Args:
        target_sequence: The target amino-acid sequence.
        query_db: Path to pre-created MMseqs2 query database.
        run_configs: List of RunConfig for each database to search.
        chain_poly_type: The type of chain for which to get an MSA.
        executor: ThreadPoolExecutor for running post-processing tasks.
        deduplicate: If True, the MSA sequences will be deduplicated.

    Returns:
        List of Futures, one per database, in the same order as run_configs.

    Raises:
        ValueError: If any run_config is not MmseqsConfig.
    """
    from alphafold3.data.tools import mmseqs

    msa_futures = []

    for run_config in run_configs:
        if not isinstance(run_config.config, msa_config.MmseqsConfig):
            raise ValueError(
                f"get_msa_shared_db_pipelined only supports MmseqsConfig, "
                f"got {type(run_config.config)}"
            )

        tool = mmseqs.Mmseqs(
            binary_path=run_config.config.binary_path,
            database_path=run_config.config.database_config.path,
            e_value=run_config.config.e_value,
            sensitivity=run_config.config.sensitivity,
            max_sequences=run_config.config.max_sequences,
            gpu_enabled=run_config.config.gpu_enabled,
            gpu_device=run_config.config.gpu_device,
            threads=run_config.config.threads,
        )

        # Search using shared query DB (GPU search runs now, post-processing async)
        result_futures = tool.search_with_query_db(
            query_db=query_db,
            sequences={"query": target_sequence},
            executor=executor,
        )

        # Get the future for our single query
        msa_result_future = result_futures["query"]

        # Wrap to convert MsaToolResult to Msa
        crop_size = run_config.crop_size

        def convert_result(
            future: futures.Future,
            seq: str = target_sequence,
            crop: int | None = crop_size,
        ) -> Msa:
            msa_result = future.result()
            return Msa.from_a3m(
                query_sequence=seq,
                chain_poly_type=chain_poly_type,
                a3m=msa_result.a3m,
                max_depth=crop,
                deduplicate=deduplicate,
            )

        msa_futures.append(executor.submit(convert_result, msa_result_future))

    return msa_futures
