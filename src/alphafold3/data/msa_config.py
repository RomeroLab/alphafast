# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Genetic search config settings for data pipelines."""

import dataclasses
import datetime
from typing import Self
from alphafold3.constants import mmcif_names


def _validate_chain_poly_type(chain_poly_type: str) -> None:
    if chain_poly_type not in mmcif_names.STANDARD_POLYMER_CHAIN_TYPES:
        raise ValueError(
            "chain_poly_type must be one of"
            f" {mmcif_names.STANDARD_POLYMER_CHAIN_TYPES}: {chain_poly_type}"
        )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class DatabaseConfig:
    """Configuration for a database."""

    name: str
    path: str


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class JackhmmerConfig:
    """Configuration for a jackhmmer run.

    Attributes:
        binary_path: Path to the binary of the msa tool.
        database_config: Database configuration.
        n_cpu: An integer with the number of CPUs to use.
        n_iter: An integer with the number of database search iterations.
        e_value: e-value for the database lookup.
        z_value: The Z-value representing the database size in number of sequences
          for E-value and domain E-value calculation. Must be set for sharded
          databases.
        dom_z_value: The Z-value representing the database size in number of
          sequences for domain E-value calculation. Must be set for sharded
          databases.
        max_sequences: Max sequences to return in MSA.
        max_parallel_shards: If given, the maximum number of shards to search
          against in parallel. If None, one Jackhmmer instance will be run per
          shard. Only applicable if the database is sharded.
    """

    binary_path: str
    database_config: DatabaseConfig
    n_cpu: int
    n_iter: int
    e_value: float
    z_value: int | None
    dom_z_value: int | None
    max_sequences: int
    max_parallel_shards: int | None = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class NhmmerConfig:
    """Configuration for a nhmmer run.

    Attributes:
        binary_path: Path to the binary of the msa tool.
        hmmalign_binary_path: Path to the hmmalign binary.
        hmmbuild_binary_path: Path to the hmmbuild binary.
        database_config: Database configuration.
        n_cpu: An integer with the number of CPUs to use.
        e_value: e-value for the database lookup.
        z_value: The Z-value representing the database size in megabases for
          E-value calculation. Allows fractional values. Must be set for sharded
          databases.
        max_sequences: Max sequences to return in MSA.
        alphabet: The alphabet when building a profile with hmmbuild.
        max_parallel_shards: If given, the maximum number of shards to search
          against in parallel. If None, one Nhmmer instance will be run per shard.
          Only applicable if the database is sharded.
    """

    binary_path: str
    hmmalign_binary_path: str
    hmmbuild_binary_path: str
    database_config: DatabaseConfig
    n_cpu: int
    e_value: float
    z_value: float | None
    max_sequences: int
    alphabet: str | None
    max_parallel_shards: int | None = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class MmseqsConfig:
    """Configuration for MMseqs2 sequence search.

    Attributes:
      binary_path: Path to the mmseqs binary.
      database_config: Target database configuration.
      e_value: E-value threshold for the search.
      sensitivity: Sensitivity parameter (-s flag). Range 1-7.5, higher values
        find more remote homologs but are slower.
      max_sequences: Maximum number of sequences to return in the MSA.
      gpu_enabled: Whether to use GPU acceleration (--gpu 1 flag).
      gpu_device: Specific GPU device to use (via CUDA_VISIBLE_DEVICES). If None,
        uses all available GPUs.
      threads: Number of CPU threads for non-GPU parts of the search.
      temp_dir: Directory for temporary files. If None, uses system default.
        Set to fast local storage on HPC clusters for better performance.
    """

    binary_path: str
    database_config: DatabaseConfig
    e_value: float = 1e-4
    sensitivity: float = 7.5
    max_sequences: int = 5000
    gpu_enabled: bool = True
    gpu_device: int | None = None
    threads: int = 8
    temp_dir: str | None = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class MmseqsTemplateConfig:
    """Configuration for MMseqs2 template search.

    Attributes:
      binary_path: Path to the mmseqs binary.
      database_config: Target database configuration (pdb_seqres).
      e_value: E-value threshold for the search. Using high value (100) to match
        hmmsearch behavior for broad template discovery.
      sensitivity: Sensitivity parameter (-s flag). Range 1-7.5, higher values
        find more remote homologs but are slower.
      max_hits: Maximum number of template hits to return.
      gpu_enabled: Whether to use GPU acceleration (--gpu 1 flag).
      gpu_device: Specific GPU device to use (via CUDA_VISIBLE_DEVICES). If None,
        uses all available GPUs.
      threads: Number of CPU threads for non-GPU parts of the search.
      temp_dir: Directory for temporary files. If None, uses system default.
        Set to fast local storage on HPC clusters for better performance.
    """

    binary_path: str
    database_config: DatabaseConfig
    e_value: float = 100.0
    sensitivity: float = 7.5
    max_hits: int = 1000
    gpu_enabled: bool = True
    gpu_device: int | None = None
    threads: int = 8
    temp_dir: str | None = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class RunConfig:
    """Configuration for an MSA run.

    Attributes:
      config: MSA tool config.
      chain_poly_type: The chain type for which the tools will be run.
      crop_size: The maximum number of sequences to keep in the MSA. If None, all
        sequences are kept. Note that the query is included in the MSA, so it
        doesn't make sense to set this to less than 2.
    """

    config: JackhmmerConfig | NhmmerConfig | MmseqsConfig
    chain_poly_type: str
    crop_size: int | None

    def __post_init__(self):
        if self.crop_size is not None and self.crop_size < 2:
            raise ValueError(f"crop_size must be None or >= 2: {self.crop_size}")

        _validate_chain_poly_type(self.chain_poly_type)


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class HmmsearchConfig:
    """Configuration for a hmmsearch."""

    hmmsearch_binary_path: str
    hmmbuild_binary_path: str

    e_value: float
    inc_e: float
    dom_e: float
    incdom_e: float
    alphabet: str = "amino"
    filter_f1: float | None = None
    filter_f2: float | None = None
    filter_f3: float | None = None
    filter_max: bool = False


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class TemplateToolConfig:
    """Configuration for a template tool.

    Supports either hmmsearch or MMseqs2 for template search. Exactly one of
    hmmsearch_config or mmseqs_config must be provided.
    """

    database_path: str
    chain_poly_type: str
    hmmsearch_config: HmmsearchConfig | None = None
    mmseqs_config: MmseqsTemplateConfig | None = None
    max_a3m_query_sequences: int | None = 300

    def __post_init__(self):
        _validate_chain_poly_type(self.chain_poly_type)
        if self.hmmsearch_config is None and self.mmseqs_config is None:
            raise ValueError(
                "Either hmmsearch_config or mmseqs_config must be provided"
            )
        if self.hmmsearch_config is not None and self.mmseqs_config is not None:
            raise ValueError(
                "Only one of hmmsearch_config or mmseqs_config can be provided"
            )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class TemplateFilterConfig:
    """Configuration for a template filter.

    Attributes:
        max_subsequence_ratio: If set, excludes hits which are an exact
            subsequence of the query sequence, and longer than this ratio.
        min_align_ratio: If set, excludes hits where the number of residues
            aligned to the query is less than this proportion of the template
            length. This is essentially a coverage filter.
        min_hit_length: If set, excludes hits with fewer residues than this.
        min_sequence_identity: If set, excludes hits with sequence identity
            below this threshold (0-1). Only applies to MMseqs2 template search.
        deduplicate_sequences: Whether to deduplicate by sequence.
        max_hits: Maximum number of hits to keep after filtering.
        max_template_date: Maximum release date for templates.
    """

    max_subsequence_ratio: float | None
    min_align_ratio: float | None
    min_hit_length: int | None
    min_sequence_identity: float | None = None
    deduplicate_sequences: bool
    max_hits: int | None
    max_template_date: datetime.date

    @classmethod
    def no_op_filter(cls) -> Self:
        """Returns a config for filter that keeps everything."""
        return cls(
            max_subsequence_ratio=None,
            min_align_ratio=None,
            min_hit_length=None,
            min_sequence_identity=None,
            deduplicate_sequences=False,
            max_hits=None,
            max_template_date=datetime.date(3000, 1, 1),  # Very far in the future.
        )

    @classmethod
    def strict_filter(
        cls,
        max_template_date: datetime.date,
        e_value: float = 1e-2,
        min_sequence_identity: float = 0.15,
        min_coverage: float = 0.3,
    ) -> Self:
        """Returns a strict filter for sequence-based template search.

        Used when Foldseek is the primary template source and sequence-based
        search is only a backup.

        Args:
            max_template_date: Maximum release date for templates.
            e_value: E-value threshold (applied at search level, not here).
            min_sequence_identity: Minimum sequence identity (0-1).
            min_coverage: Minimum alignment coverage (0-1).

        Returns:
            A strict TemplateFilterConfig.
        """
        del e_value  # Unused, applied at search level
        return cls(
            max_subsequence_ratio=0.95,
            min_align_ratio=min_coverage,
            min_hit_length=10,
            min_sequence_identity=min_sequence_identity,
            deduplicate_sequences=True,
            max_hits=4,
            max_template_date=max_template_date,
        )


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class TemplatesConfig:
    """Configuration for the template search pipeline."""

    template_tool_config: TemplateToolConfig
    filter_config: TemplateFilterConfig


# ============================================================================
# Foldseek Mode Configuration
# ============================================================================


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ESMFoldConfig:
    """Configuration for ESMFold structure prediction.

    Attributes:
        model_name: The ESMFold model to use (currently only "esmfold_v1").
        device: Device to run inference on ("cuda", "cpu", or None for auto).
        chunk_size: Optional chunk size for memory optimization with long seqs.
        min_plddt: Minimum mean pLDDT score to proceed with Foldseek search.
            If predicted structure has lower confidence, skip Foldseek.
    """

    model_name: str = "esmfold_v1"
    device: str | None = None
    chunk_size: int | None = None
    min_plddt: float = 50.0


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class FoldseekConfig:
    """Configuration for Foldseek structural search.

    Attributes:
        binary_path: Path to the Foldseek binary.
        database_path: Path to the AFDB Foldseek database.
        e_value: E-value threshold for hits.
        max_hits: Maximum number of hits to return from search.
        alignment_type: Alignment type (0=3Di, 1=TM, 2=3Di+AA).
        threads: Number of CPU threads to use.
        min_lddt: Minimum LDDT score for hit acceptance (0-1).
        gpu_enabled: Whether to use GPU acceleration (--gpu 1 flag).
        gpu_device: Specific GPU device to use (via CUDA_VISIBLE_DEVICES). If
            None, uses all available GPUs.
    """

    binary_path: str
    database_path: str
    e_value: float = 1e-3
    max_hits: int = 100
    alignment_type: int = 2
    threads: int = 8
    min_lddt: float = 0.5
    gpu_enabled: bool = True
    gpu_device: int | None = None


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class FoldseekFilterConfig:
    """Configuration for filtering Foldseek hits.

    Foldseek uses STRUCTURAL similarity (LDDT), not sequence identity.
    This is the key difference from sequence-based template search.

    Attributes:
        max_hits: Maximum number of templates to keep after filtering.
        min_sequence_identity: Minimum sequence identity (0-1). Default 0.0
            means no sequence identity filter - rely on structural LDDT only.
        min_lddt: Minimum LDDT structural similarity score (0-1). This is
            the primary filter for Foldseek - measures local structural distance.
        min_coverage: Minimum query coverage (0-1).
        deduplicate_by_uniprot: Whether to keep only one hit per UniProt ID.
    """

    max_hits: int = 4
    min_sequence_identity: float = 0.0  # No seq_id filter - use LDDT instead
    min_lddt: float = 0.5  # Primary structural similarity filter
    min_coverage: float = 0.3
    deduplicate_by_uniprot: bool = True


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class FoldseekTemplatesConfig:
    """Full configuration for Foldseek template search mode.

    Attributes:
        esmfold_config: Configuration for ESMFold structure prediction.
        foldseek_config: Configuration for Foldseek structural search.
        filter_config: Configuration for filtering Foldseek hits.
        afdb_cache_dir: Optional directory for caching downloaded AFDB structures.
        mode: Internal mode for template merging:
            - "supplement": PDB first, Foldseek fills remaining (foldseek_backup)
            - "foldseek_priority": Foldseek first, PDB fills remaining
            - "replace": Only Foldseek templates (legacy, fallback to PDB if none)
    """

    esmfold_config: ESMFoldConfig
    foldseek_config: FoldseekConfig
    filter_config: FoldseekFilterConfig
    afdb_cache_dir: str | None = None
    mode: str = "supplement"  # "supplement", "foldseek_priority", or "replace"
