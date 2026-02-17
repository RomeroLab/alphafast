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

"""Functions for running the MSA and template tools for the AlphaFold model."""

from concurrent import futures
import dataclasses
import datetime
import functools
import logging
import os
import time
from collections.abc import Sequence

from alphafold3.common import folding_input
from alphafold3.constants import mmcif_names
from alphafold3.data import msa
from alphafold3.data import msa_config
from alphafold3.data import structure_stores
from alphafold3.data import templates as templates_lib
from alphafold3.data.tools import mmseqs_batch


def _default_mmseqs_threads() -> int:
    return len(os.sched_getaffinity(0))


# Cache to avoid re-running template search for the same sequence in homomers.
@functools.cache
def _get_protein_templates(
    sequence: str,
    input_msa_a3m: str,
    run_template_search: bool,
    templates_config: msa_config.TemplatesConfig,
    pdb_database_path: str,
    precomputed_templates_a3m: str | None = None,
) -> templates_lib.Templates:
    """Searches for templates for a single protein chain."""
    if precomputed_templates_a3m:
        # Use pre-computed template A3M (skip template search)
        templates_start_time = time.time()
        logging.info("Loading pre-computed templates for sequence %s", sequence)
        protein_templates = templates_lib.Templates.from_hmmsearch_a3m(
            query_sequence=sequence,
            a3m=precomputed_templates_a3m,
            max_template_date=templates_config.filter_config.max_template_date,
            structure_store=structure_stores.StructureStore(pdb_database_path),
            filter_config=templates_config.filter_config,
        )
        logging.info(
            "Loaded %d pre-computed templates in %.2f seconds for sequence %s",
            protein_templates.num_hits,
            time.time() - templates_start_time,
            sequence,
        )
    elif run_template_search:
        templates_start_time = time.time()
        logging.info("Getting protein templates for sequence %s", sequence)
        protein_templates = templates_lib.Templates.from_seq_and_a3m(
            query_sequence=sequence,
            msa_a3m=input_msa_a3m,
            max_template_date=templates_config.filter_config.max_template_date,
            database_path=templates_config.template_tool_config.database_path,
            hmmsearch_config=templates_config.template_tool_config.hmmsearch_config,
            mmseqs_config=templates_config.template_tool_config.mmseqs_config,
            max_a3m_query_sequences=None,
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            structure_store=structure_stores.StructureStore(pdb_database_path),
            filter_config=templates_config.filter_config,
        )
        logging.info(
            "Getting %d protein templates took %.2f seconds for sequence %s",
            protein_templates.num_hits,
            time.time() - templates_start_time,
            sequence,
        )
    else:
        logging.info("Skipping template search for sequence %s", sequence)
        protein_templates = templates_lib.Templates(
            query_sequence=sequence,
            hits=[],
            max_template_date=templates_config.filter_config.max_template_date,
            structure_store=structure_stores.StructureStore(pdb_database_path),
        )
    return protein_templates


# Cache to avoid re-running the MSA tools for the same sequence in homomers.
@functools.cache
def _get_protein_msa_and_templates(
    sequence: str,
    run_template_search: bool,
    uniref90_msa_config: msa_config.RunConfig,
    mgnify_msa_config: msa_config.RunConfig,
    small_bfd_msa_config: msa_config.RunConfig,
    uniprot_msa_config: msa_config.RunConfig,
    templates_config: msa_config.TemplatesConfig,
    pdb_database_path: str,
    run_msa_sequential: bool = False,
    temp_dir: str | None = None,
) -> tuple[msa.Msa, msa.Msa, templates_lib.Templates]:
    """Processes a single protein chain.

    Args:
        sequence: The protein sequence to process.
        run_template_search: Whether to run template search.
        uniref90_msa_config: Config for UniRef90 MSA search.
        mgnify_msa_config: Config for MGnify MSA search.
        small_bfd_msa_config: Config for Small BFD MSA search.
        uniprot_msa_config: Config for UniProt MSA search.
        templates_config: Config for template search.
        pdb_database_path: Path to PDB database.
        run_msa_sequential: If True, run MSA searches sequentially instead of
            in parallel. Useful when GPU memory is limited.
        temp_dir: Directory for temporary files. If None, uses system default.
            Set to fast local storage on HPC clusters for better performance.

    Returns:
        Tuple of (unpaired_msa, paired_msa, templates).
    """
    logging.info("Getting protein MSAs for sequence %s", sequence)
    msa_start_time = time.time()

    if run_msa_sequential:
        # Run MSA searches sequentially to avoid GPU memory exhaustion.
        # Useful when MMseqs2-GPU is sharing GPU with JAX.
        logging.info("Running MSA searches sequentially")

        # Check if we're using MMseqs2 (supports pipelined execution)
        is_mmseqs = isinstance(uniref90_msa_config.config, msa_config.MmseqsConfig)

        if is_mmseqs:
            # Pipelined execution with shared query DB:
            # 1. Create query DB once (instead of 4 times)
            # 2. GPU searches run sequentially for each target DB
            # 3. CPU post-processing (result2msa, unpackdb) runs in parallel
            #    with subsequent GPU searches
            import shutil
            import tempfile
            from alphafold3.data.tools import mmseqs as mmseqs_module

            logging.info(
                "Using pipelined MMseqs2 execution with shared query DB "
                "(GPU sequential, CPU parallel)"
            )

            # Create shared query DB once
            # Use temp_dir if provided (for HPC with fast local storage)
            query_db_dir = tempfile.mkdtemp(prefix="mmseqs_shared_", dir=temp_dir)
            try:
                query_db, _ = mmseqs_module.create_query_db(
                    binary_path=uniref90_msa_config.config.binary_path,
                    sequences={"query": sequence},
                    output_dir=query_db_dir,
                )
                logging.info("Created shared query DB at %s", query_db)

                with futures.ThreadPoolExecutor() as postprocess_executor:
                    # Search all 4 databases using shared query DB
                    # GPU searches run sequentially, post-processing runs async
                    msa_futures = msa.get_msa_shared_db_pipelined(
                        target_sequence=sequence,
                        query_db=query_db,
                        run_configs=[
                            uniref90_msa_config,
                            mgnify_msa_config,
                            small_bfd_msa_config,
                            uniprot_msa_config,
                        ],
                        chain_poly_type=mmcif_names.PROTEIN_CHAIN,
                        executor=postprocess_executor,
                    )

                    # Wait for all post-processing to complete
                    uniref90_msa = msa_futures[0].result()
                    mgnify_msa = msa_futures[1].result()
                    small_bfd_msa = msa_futures[2].result()
                    uniprot_msa = msa_futures[3].result()

            finally:
                # Clean up shared query DB
                shutil.rmtree(query_db_dir, ignore_errors=True)
    else:
        # Run various MSA tools in parallel. Use a ThreadPoolExecutor because
        # they're not blocked by the GIL, as they're sub-shelled out.
        with futures.ThreadPoolExecutor(max_workers=4) as executor:
            uniref90_msa_future = executor.submit(
                msa.get_msa,
                target_sequence=sequence,
                run_config=uniref90_msa_config,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            )
            mgnify_msa_future = executor.submit(
                msa.get_msa,
                target_sequence=sequence,
                run_config=mgnify_msa_config,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            )
            small_bfd_msa_future = executor.submit(
                msa.get_msa,
                target_sequence=sequence,
                run_config=small_bfd_msa_config,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            )
            uniprot_msa_future = executor.submit(
                msa.get_msa,
                target_sequence=sequence,
                run_config=uniprot_msa_config,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            )
        uniref90_msa = uniref90_msa_future.result()
        mgnify_msa = mgnify_msa_future.result()
        small_bfd_msa = small_bfd_msa_future.result()
        uniprot_msa = uniprot_msa_future.result()
    logging.info(
        "Getting protein MSAs took %.2f seconds for sequence %s",
        time.time() - msa_start_time,
        sequence,
    )

    logging.info("Deduplicating MSAs for sequence %s", sequence)
    msa_dedupe_start_time = time.time()
    with futures.ThreadPoolExecutor() as executor:
        unpaired_protein_msa_future = executor.submit(
            msa.Msa.from_multiple_msas,
            msas=[uniref90_msa, small_bfd_msa, mgnify_msa],
            deduplicate=True,
        )
        paired_protein_msa_future = executor.submit(
            msa.Msa.from_multiple_msas, msas=[uniprot_msa], deduplicate=False
        )
    unpaired_protein_msa = unpaired_protein_msa_future.result()
    paired_protein_msa = paired_protein_msa_future.result()
    logging.info(
        "Deduplicating MSAs took %.2f seconds for sequence %s, found %d unpaired"
        " sequences, %d paired sequences",
        time.time() - msa_dedupe_start_time,
        sequence,
        unpaired_protein_msa.depth,
        paired_protein_msa.depth,
    )

    protein_templates = _get_protein_templates(
        sequence=sequence,
        input_msa_a3m=unpaired_protein_msa.to_a3m(),
        run_template_search=run_template_search,
        templates_config=templates_config,
        pdb_database_path=pdb_database_path,
    )

    return unpaired_protein_msa, paired_protein_msa, protein_templates


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DataPipelineConfig:
    """The configuration for the data pipeline.

    Attributes:
      pdb_database_path: PDB database directory with mmCIF files path, used for
        template search.
      max_template_date: The latest date of templates to use.
      mmseqs_binary_path: Path to MMseqs2 binary. If None, auto-detected at
        $HOME/.local/bin/mmseqs or via PATH.
      mmseqs_db_dir: Directory containing MMseqs2 padded databases. Required for
        GPU-accelerated protein MSA search.
      use_mmseqs_gpu: Whether to use GPU acceleration for MMseqs2 searches.
      gpu_device: Specific GPU device to use for MMseqs2 and Foldseek (via
        CUDA_VISIBLE_DEVICES). If None, uses all available GPUs.
      mmseqs_sensitivity: MMseqs2 sensitivity parameter (-s flag). Range 1-7.5,
        higher values find more remote homologs but are slower.
      mmseqs_n_threads: Number of CPU threads for MMseqs2 non-GPU operations.
      mmseqs_sequential: Whether to run MMseqs2 database searches sequentially
        instead of in parallel. Use this if encountering GPU OOM errors. When
        using the separate run_data_pipeline.py script, parallel execution is
        recommended since MMseqs2 gets full GPU access.
    """

    # Template search databases.
    pdb_database_path: str

    max_template_date: datetime.date

    # MMseqs2 configuration (for GPU-accelerated protein MSA search).
    mmseqs_binary_path: str | None = None
    mmseqs_db_dir: str | None = None
    use_mmseqs_gpu: bool = True
    gpu_device: int | None = None  # Used for both MMseqs2 and Foldseek
    mmseqs_sensitivity: float = 7.5
    mmseqs_n_threads: int = dataclasses.field(default_factory=_default_mmseqs_threads)
    mmseqs_sequential: bool = False

    # Temporary directory for intermediate files (for HPC with slow network storage).
    # When set, MMseqs2/Foldseek temp files are written to this fast local storage
    # instead of the system default. Can provide 10-13x speedup on clusters like DCC.
    temp_dir: str | None = None

    # Pre-computed MSA support (for inference-only mode).
    # When provided, MSA search is skipped and this file is used directly.
    precomputed_msa_path: str | None = None

    # Pre-computed template A3M support (for inference-only mode).
    # When provided, template search is skipped and this file is used directly.
    precomputed_templates_a3m_path: str | None = None

    # MMseqs2 template search parameters (separate from MSA search).
    # These are the thresholds for template search in ALL modes.
    # More selective than original hmmsearch (e-value=100) but reasonable.
    # Note: NO sequence identity filter - rely on e-value and coverage only.
    template_e_value: float = 1e-3  # Much stricter than MSA e-value
    template_min_coverage: float = 0.40  # 40% - good alignment coverage

    # Template mode configuration.
    # - "default": Only use MMseqs2-GPU with PDB (no Foldseek)
    # - "foldseek_backup": First try MMseqs2-GPU, then use Foldseek to fill remaining slots
    # - "foldseek_first": First try Foldseek, then backup with MMseqs2-GPU
    template_mode: str = "default"

    # Foldseek configuration (for structural template search from AFDB).
    foldseek_binary_path: str | None = None
    foldseek_database_path: str | None = None
    foldseek_max_templates: int = 4
    foldseek_min_lddt: float = 0.5
    foldseek_min_plddt: float = 50.0
    foldseek_e_value: float = 1e-3
    foldseek_threads: int = 8
    foldseek_gpu_enabled: bool = True
    esmfold_device: str | None = None
    esmfold_chunk_size: int | None = None
    afdb_cache_dir: str | None = None


class DataPipeline:
    """Runs the alignment tools and assembles the input features."""

    def __init__(self, data_pipeline_config: DataPipelineConfig):
        """Initializes the data pipeline with default configurations."""
        # Determine whether to use MMseqs2-GPU for protein MSA searches
        mmseqs_config = self._setup_mmseqs_config(data_pipeline_config)

        if mmseqs_config is None:
            raise ValueError(
                "MMseqs2-GPU is required for AlphaFast. Ensure mmseqs binary is "
                "installed and --mmseqs_db_dir points to padded databases."
            )

        mmseqs_binary, mmseqs_db_dir, template_db_available = mmseqs_config
        logging.info("Using MMseqs2-GPU for protein MSA searches")
        logging.info("MMseqs2 binary: %s", mmseqs_binary)
        logging.info("MMseqs2 database directory: %s", mmseqs_db_dir)
        if template_db_available:
            logging.info("Using MMseqs2-GPU for template search")

        self._uniref90_msa_config = msa_config.RunConfig(
            config=msa_config.MmseqsConfig(
                binary_path=mmseqs_binary,
                database_config=msa_config.DatabaseConfig(
                    name="uniref90",
                    path=f"{mmseqs_db_dir}/uniref90_padded",
                ),
                e_value=1e-4,
                sensitivity=data_pipeline_config.mmseqs_sensitivity,
                max_sequences=10_000,
                gpu_enabled=data_pipeline_config.use_mmseqs_gpu,
                gpu_device=data_pipeline_config.gpu_device,
                threads=data_pipeline_config.mmseqs_n_threads,
            ),
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            crop_size=None,
        )
        self._mgnify_msa_config = msa_config.RunConfig(
            config=msa_config.MmseqsConfig(
                binary_path=mmseqs_binary,
                database_config=msa_config.DatabaseConfig(
                    name="mgnify",
                    path=f"{mmseqs_db_dir}/mgnify_padded",
                ),
                e_value=1e-4,
                sensitivity=data_pipeline_config.mmseqs_sensitivity,
                max_sequences=5_000,
                gpu_enabled=data_pipeline_config.use_mmseqs_gpu,
                gpu_device=data_pipeline_config.gpu_device,
                threads=data_pipeline_config.mmseqs_n_threads,
            ),
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            crop_size=None,
        )
        self._small_bfd_msa_config = msa_config.RunConfig(
            config=msa_config.MmseqsConfig(
                binary_path=mmseqs_binary,
                database_config=msa_config.DatabaseConfig(
                    name="small_bfd",
                    path=f"{mmseqs_db_dir}/small_bfd_padded",
                ),
                e_value=1e-4,
                sensitivity=data_pipeline_config.mmseqs_sensitivity,
                max_sequences=5_000,
                gpu_enabled=data_pipeline_config.use_mmseqs_gpu,
                gpu_device=data_pipeline_config.gpu_device,
                threads=data_pipeline_config.mmseqs_n_threads,
            ),
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            crop_size=None,
        )
        self._uniprot_msa_config = msa_config.RunConfig(
            config=msa_config.MmseqsConfig(
                binary_path=mmseqs_binary,
                database_config=msa_config.DatabaseConfig(
                    name="uniprot",
                    path=f"{mmseqs_db_dir}/uniprot_padded",
                ),
                e_value=1e-4,
                sensitivity=data_pipeline_config.mmseqs_sensitivity,
                max_sequences=50_000,
                gpu_enabled=data_pipeline_config.use_mmseqs_gpu,
                gpu_device=data_pipeline_config.gpu_device,
                threads=data_pipeline_config.mmseqs_n_threads,
            ),
            chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            crop_size=None,
        )

        # Set up templates config (MMseqs2 if available, otherwise hmmsearch)
        # Template search uses SAME thresholds for ALL modes:
        # - e-value: 1e-3 (much stricter than old hmmsearch e-value=100)
        # - coverage: 40% (min_align_ratio)
        # - NO sequence identity filter (rely on e-value and coverage)
        template_filter_config = msa_config.TemplateFilterConfig(
            max_subsequence_ratio=0.95,
            min_align_ratio=data_pipeline_config.template_min_coverage,
            min_hit_length=10,
            min_sequence_identity=None,  # No sequence identity filter
            deduplicate_sequences=True,
            max_hits=4,
            max_template_date=data_pipeline_config.max_template_date,
        )
        template_e_value = data_pipeline_config.template_e_value
        logging.info(
            "Template search: e-value=%.2e, coverage≥%.0f%%, no seq_id filter",
            template_e_value,
            data_pipeline_config.template_min_coverage * 100,
        )

        if template_db_available:
            # Use MMseqs2 for template search
            self._templates_config = msa_config.TemplatesConfig(
                template_tool_config=msa_config.TemplateToolConfig(
                    database_path=f"{mmseqs_db_dir}/pdb_seqres_padded",
                    chain_poly_type=mmcif_names.PROTEIN_CHAIN,
                    mmseqs_config=msa_config.MmseqsTemplateConfig(
                        binary_path=mmseqs_binary,
                        database_config=msa_config.DatabaseConfig(
                            name="pdb_seqres",
                            path=f"{mmseqs_db_dir}/pdb_seqres_padded",
                        ),
                        e_value=template_e_value,
                        sensitivity=data_pipeline_config.mmseqs_sensitivity,
                        max_hits=1000,
                        gpu_enabled=data_pipeline_config.use_mmseqs_gpu,
                        gpu_device=data_pipeline_config.gpu_device,
                        threads=data_pipeline_config.mmseqs_n_threads,
                    ),
                ),
                filter_config=template_filter_config,
            )
        else:
            raise ValueError(
                "pdb_seqres_padded not found in %s. "
                "Run setup_databases.sh to create it." % mmseqs_db_dir
            )
        self._pdb_database_path = data_pipeline_config.pdb_database_path
        self._run_msa_sequential = data_pipeline_config.mmseqs_sequential
        self._temp_dir = data_pipeline_config.temp_dir

        # Store template_mode for reference
        self._template_mode = data_pipeline_config.template_mode

        # Store pre-computed MSA path (for inference-only mode)
        self._precomputed_msa_path = data_pipeline_config.precomputed_msa_path
        if self._precomputed_msa_path:
            logging.info("Using pre-computed MSA from: %s", self._precomputed_msa_path)

        # Store pre-computed template A3M path (for inference-only mode)
        self._precomputed_templates_a3m_path = (
            data_pipeline_config.precomputed_templates_a3m_path
        )
        if self._precomputed_templates_a3m_path:
            logging.info(
                "Using pre-computed templates from: %s",
                self._precomputed_templates_a3m_path,
            )

        # Set up Foldseek configuration (optional)
        self._foldseek_config = self._setup_foldseek_config(data_pipeline_config)

    def _setup_foldseek_config(
        self, data_pipeline_config: DataPipelineConfig
    ) -> msa_config.FoldseekTemplatesConfig | None:
        """Checks if Foldseek should be used and validates configuration.

        Args:
            data_pipeline_config: The data pipeline configuration.

        Returns:
            FoldseekTemplatesConfig if Foldseek should be used, None otherwise.
        """
        # Foldseek is used in "foldseek_backup" and "foldseek_first" modes
        if data_pipeline_config.template_mode == "default":
            return None

        # Check for Foldseek binary
        foldseek_binary = data_pipeline_config.foldseek_binary_path
        if foldseek_binary is None:
            # Try to find Foldseek binary
            from alphafold3.data.tools import foldseek as foldseek_tool

            foldseek_binary = foldseek_tool.find_foldseek_binary()

        if foldseek_binary is None:
            logging.warning(
                "Template mode '%s' requires Foldseek but binary not found. "
                "Install Foldseek or provide --foldseek_binary_path. "
                "Falling back to 'default' mode.",
                data_pipeline_config.template_mode,
            )
            return None

        # Check for Foldseek database
        foldseek_db = data_pipeline_config.foldseek_database_path
        if foldseek_db is None or not os.path.exists(foldseek_db):
            logging.warning(
                "Template mode '%s' requires Foldseek but database not found at %s. "
                "Run setup_foldseek_afdb.sh to download the AFDB database. "
                "Falling back to 'default' mode.",
                data_pipeline_config.template_mode,
                foldseek_db,
            )
            return None

        # Validate template_mode
        if data_pipeline_config.template_mode not in (
            "default",
            "foldseek_backup",
            "foldseek_first",
        ):
            raise ValueError(
                f"Invalid template_mode: {data_pipeline_config.template_mode}. "
                "Must be 'default', 'foldseek_backup', or 'foldseek_first'."
            )

        logging.info("Template mode: %s", data_pipeline_config.template_mode)
        logging.info("Foldseek binary: %s", foldseek_binary)
        logging.info("Foldseek database: %s", foldseek_db)
        if data_pipeline_config.template_mode in ("foldseek_backup", "foldseek_first"):
            logging.info(
                "Template search thresholds: e-value=%.2e, coverage=%.0f%%",
                data_pipeline_config.template_e_value,
                data_pipeline_config.template_min_coverage * 100,
            )

        # Determine the internal mode for _merge_templates
        # "foldseek_backup" -> Foldseek supplements PDB (PDB first)
        # "foldseek_first" -> PDB supplements Foldseek (Foldseek first)
        internal_mode = (
            "supplement"
            if data_pipeline_config.template_mode == "foldseek_backup"
            else "foldseek_priority"
        )

        return msa_config.FoldseekTemplatesConfig(
            esmfold_config=msa_config.ESMFoldConfig(
                model_name="esmfold_v1",
                device=data_pipeline_config.esmfold_device,
                chunk_size=data_pipeline_config.esmfold_chunk_size,
                min_plddt=data_pipeline_config.foldseek_min_plddt,
            ),
            foldseek_config=msa_config.FoldseekConfig(
                binary_path=foldseek_binary,
                database_path=foldseek_db,
                e_value=data_pipeline_config.foldseek_e_value,
                max_hits=100,  # Search for many, filter down later
                alignment_type=2,  # 3Di+AA
                threads=data_pipeline_config.foldseek_threads,
                min_lddt=data_pipeline_config.foldseek_min_lddt,
                gpu_enabled=data_pipeline_config.foldseek_gpu_enabled,
                gpu_device=data_pipeline_config.gpu_device,
            ),
            filter_config=msa_config.FoldseekFilterConfig(
                max_hits=data_pipeline_config.foldseek_max_templates,
                min_sequence_identity=0.0,
                min_lddt=data_pipeline_config.foldseek_min_lddt,
                min_coverage=0.3,
                deduplicate_by_uniprot=True,
            ),
            afdb_cache_dir=data_pipeline_config.afdb_cache_dir,
            mode=internal_mode,
        )

    def _setup_mmseqs_config(
        self, data_pipeline_config: DataPipelineConfig
    ) -> tuple[str, str, bool] | None:
        """Checks if MMseqs2-GPU should be used and validates configuration.

        Args:
            data_pipeline_config: The data pipeline configuration.

        Returns:
            Tuple of (mmseqs_binary_path, mmseqs_db_dir, template_db_available)
            if MMseqs2 should be used for MSA, None otherwise.
            template_db_available indicates if pdb_seqres_padded exists for
            template search.
        """
        import os

        from alphafold3.data.tools import mmseqs

        # Check if MMseqs2 binary path is provided or can be auto-detected
        mmseqs_binary = data_pipeline_config.mmseqs_binary_path
        if mmseqs_binary is None:
            mmseqs_binary = mmseqs.find_mmseqs_binary()

        if mmseqs_binary is None:
            logging.error(
                "MMseqs2 binary not found. Install MMseqs2-GPU:\n"
                "  wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz\n"
                "  tar xzf mmseqs-linux-gpu.tar.gz\n"
                "  cp mmseqs/bin/mmseqs $HOME/.local/bin/"
            )
            return None

        # Check if MMseqs2 database directory is provided
        mmseqs_db_dir = data_pipeline_config.mmseqs_db_dir
        if mmseqs_db_dir is None:
            logging.error(
                "MMseqs2 found at %s but --mmseqs_db_dir not specified. "
                "Provide the path to padded databases created by setup_databases.sh.",
                mmseqs_binary,
            )
            return None

        # Verify all required padded databases exist for MSA
        required_dbs = [
            ("uniref90_padded", "UniRef90"),
            ("mgnify_padded", "MGnify"),
            ("small_bfd_padded", "Small BFD"),
            ("uniprot_padded", "UniProt"),
        ]

        missing_dbs = []
        for db_name, display_name in required_dbs:
            db_path = os.path.join(mmseqs_db_dir, db_name)
            if not os.path.exists(f"{db_path}.dbtype"):
                missing_dbs.append((db_name, display_name))

        if missing_dbs:
            missing_list = ", ".join(name for _, name in missing_dbs)
            logging.error(
                "MMseqs2 padded databases not found: %s. "
                "Run setup_databases.sh to create them.",
                missing_list,
            )
            return None

        # Check if template database (pdb_seqres_padded) exists (optional)
        template_db_path = os.path.join(mmseqs_db_dir, "pdb_seqres_padded")
        template_db_available = os.path.exists(f"{template_db_path}.dbtype")
        if not template_db_available:
            logging.info(
                "pdb_seqres_padded not found in %s. "
                "Template search will use hmmsearch. "
                "Run convert_databases_to_mmseqs.sh to create pdb_seqres_padded "
                "for GPU-accelerated template search.",
                mmseqs_db_dir,
            )

        return (mmseqs_binary, mmseqs_db_dir, template_db_available)

    def _get_foldseek_templates(
        self, sequence: str
    ) -> Sequence[folding_input.Template]:
        """Gets structural templates using ESMFold + Foldseek.

        Args:
            sequence: The protein sequence to find templates for.

        Returns:
            List of Template objects from Foldseek search.
        """
        if self._foldseek_config is None:
            return []

        try:
            from alphafold3.data import foldseek_templates

            return foldseek_templates.get_foldseek_templates(
                sequence=sequence,
                foldseek_config=self._foldseek_config,
            )
        except Exception as e:
            logging.warning(
                "Foldseek template search failed: %s. Continuing without Foldseek templates.",
                e,
            )
            return []

    def _merge_templates(
        self,
        pdb_templates: list[folding_input.Template],
        foldseek_templates: Sequence[folding_input.Template],
        max_total_templates: int = 4,
    ) -> list[folding_input.Template]:
        """Merges PDB and Foldseek templates based on configuration.

        Template merging modes:
        - "supplement" (foldseek_backup): PDB templates first, Foldseek fills remaining
        - "foldseek_priority" (foldseek_first): Foldseek templates first, PDB fills remaining

        Args:
            pdb_templates: Templates from PDB search (sequence-based).
            foldseek_templates: Templates from Foldseek search (structure-based).
            max_total_templates: Maximum number of templates to return.

        Returns:
            Merged list of templates.
        """
        if self._foldseek_config is None:
            return pdb_templates[:max_total_templates]

        mode = self._foldseek_config.mode

        if mode == "supplement":
            # foldseek_backup: PDB first, Foldseek fills remaining slots
            combined = list(pdb_templates)

            # Add Foldseek templates that don't exceed the limit
            foldseek_to_add = max_total_templates - len(combined)
            if foldseek_to_add > 0 and foldseek_templates:
                combined.extend(foldseek_templates[:foldseek_to_add])
                logging.info(
                    "Template merge (foldseek_backup): %d PDB + %d Foldseek = %d total",
                    len(pdb_templates),
                    min(foldseek_to_add, len(foldseek_templates)),
                    len(combined[:max_total_templates]),
                )
            else:
                logging.info(
                    "Template merge (foldseek_backup): %d PDB templates (no Foldseek added)",
                    len(combined[:max_total_templates]),
                )

            return combined[:max_total_templates]

        elif mode == "foldseek_priority":
            # foldseek_first: Foldseek templates first, PDB fills remaining slots
            combined = list(foldseek_templates) if foldseek_templates else []

            # Add PDB templates that don't exceed the limit
            pdb_to_add = max_total_templates - len(combined)
            if pdb_to_add > 0 and pdb_templates:
                combined.extend(pdb_templates[:pdb_to_add])
                logging.info(
                    "Template merge (foldseek_first): %d Foldseek + %d PDB = %d total",
                    len(foldseek_templates) if foldseek_templates else 0,
                    min(pdb_to_add, len(pdb_templates)),
                    len(combined[:max_total_templates]),
                )
            else:
                logging.info(
                    "Template merge (foldseek_first): %d Foldseek templates (no PDB added)",
                    len(combined[:max_total_templates]),
                )

            return combined[:max_total_templates]

        elif mode == "replace":
            # Legacy mode: Use only Foldseek templates (fallback to PDB)
            if foldseek_templates:
                logging.info(
                    "Using %d Foldseek templates (replace mode)",
                    len(foldseek_templates),
                )
                return list(foldseek_templates[:max_total_templates])
            else:
                logging.info(
                    "No Foldseek templates found, falling back to %d PDB templates",
                    len(pdb_templates),
                )
                return pdb_templates[:max_total_templates]

        else:
            logging.warning("Unknown foldseek_mode: %s", mode)
            return pdb_templates[:max_total_templates]

    def process_protein_chain(
        self, chain: folding_input.ProteinChain
    ) -> folding_input.ProteinChain:
        """Processes a single protein chain."""
        # Check if we should use pre-computed MSA (inference-only mode)
        if self._precomputed_msa_path and chain.unpaired_msa is None:
            logging.info(
                "Loading pre-computed MSA for chain %s from %s",
                chain.id,
                self._precomputed_msa_path,
            )
            try:
                with open(self._precomputed_msa_path, "r") as f:
                    precomputed_msa = f.read()
                # Set the MSA on the chain - this will cause the pipeline to skip
                # MSA search and only do template search
                # Note: ProteinChain is not a dataclass, so we create a new instance
                chain = folding_input.ProteinChain(
                    id=chain.id,
                    sequence=chain.sequence,
                    ptms=chain.ptms,
                    description=chain.description,
                    unpaired_msa=precomputed_msa,
                    # Use empty paired MSA since we only have unpaired from benchmark
                    paired_msa="",
                    templates=chain.templates,
                )
                logging.info(
                    "Loaded pre-computed MSA with %d characters", len(precomputed_msa)
                )
            except Exception as e:
                logging.warning(
                    "Failed to load pre-computed MSA: %s. Falling back to search.", e
                )

        has_unpaired_msa = chain.unpaired_msa is not None
        has_paired_msa = chain.paired_msa is not None
        has_templates = chain.templates is not None

        if not has_unpaired_msa and not has_paired_msa and not chain.templates:
            # MSA None - search. Templates either [] - don't search, or None - search.
            unpaired_msa, paired_msa, template_hits = _get_protein_msa_and_templates(
                sequence=chain.sequence,
                run_template_search=not has_templates,  # Skip template search if [].
                uniref90_msa_config=self._uniref90_msa_config,
                mgnify_msa_config=self._mgnify_msa_config,
                small_bfd_msa_config=self._small_bfd_msa_config,
                uniprot_msa_config=self._uniprot_msa_config,
                templates_config=self._templates_config,
                pdb_database_path=self._pdb_database_path,
                run_msa_sequential=self._run_msa_sequential,
                temp_dir=self._temp_dir,
            )
            unpaired_msa = unpaired_msa.to_a3m()
            paired_msa = paired_msa.to_a3m()
            pdb_templates = [
                folding_input.Template(
                    mmcif=struc.to_mmcif(),
                    query_to_template_map=hit.query_to_hit_mapping,
                )
                for hit, struc in template_hits.get_hits_with_structures()
            ]

            # Get Foldseek templates and merge with PDB templates
            foldseek_tmpls = self._get_foldseek_templates(chain.sequence)
            templates = self._merge_templates(pdb_templates, foldseek_tmpls)

        elif has_unpaired_msa and has_paired_msa and not has_templates:
            # Has MSA, but doesn't have templates. Search for templates only.
            empty_msa = msa.Msa.from_empty(
                query_sequence=chain.sequence,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            ).to_a3m()
            unpaired_msa = chain.unpaired_msa or empty_msa
            paired_msa = chain.paired_msa or empty_msa

            # Check for pre-computed template A3M
            precomputed_templates_a3m = None
            if self._precomputed_templates_a3m_path:
                try:
                    with open(self._precomputed_templates_a3m_path, "r") as f:
                        precomputed_templates_a3m = f.read()
                    logging.info(
                        "Loaded pre-computed templates A3M for chain %s (%d chars)",
                        chain.id,
                        len(precomputed_templates_a3m),
                    )
                except Exception as e:
                    logging.warning(
                        "Failed to load pre-computed templates: %s. Falling back to search.",
                        e,
                    )

            template_hits = _get_protein_templates(
                sequence=chain.sequence,
                input_msa_a3m=unpaired_msa,
                run_template_search=True,
                precomputed_templates_a3m=precomputed_templates_a3m,
                templates_config=self._templates_config,
                pdb_database_path=self._pdb_database_path,
            )
            pdb_templates = [
                folding_input.Template(
                    mmcif=struc.to_mmcif(),
                    query_to_template_map=hit.query_to_hit_mapping,
                )
                for hit, struc in template_hits.get_hits_with_structures()
            ]

            # Get Foldseek templates and merge with PDB templates
            foldseek_tmpls = self._get_foldseek_templates(chain.sequence)
            templates = self._merge_templates(pdb_templates, foldseek_tmpls)

        else:
            # Has MSA and templates, don't search for anything.
            if not has_unpaired_msa or not has_paired_msa or not has_templates:
                raise ValueError(
                    f"Protein chain {chain.id} has unpaired MSA, paired MSA, or"
                    " templates set only partially. If you want to run the pipeline"
                    " with custom MSA/templates, you need to set all of them. You can"
                    " set MSA to empty string and templates to empty list to signify"
                    " that they should not be used and searched for."
                )
            logging.info(
                "Skipping MSA and template search for protein chain %s because it "
                "already has MSAs and templates.",
                chain.id,
            )
            if not chain.unpaired_msa:
                logging.info("Using empty unpaired MSA for protein chain %s", chain.id)
            if not chain.paired_msa:
                logging.info("Using empty paired MSA for protein chain %s", chain.id)
            if not chain.templates:
                logging.info("Using no templates for protein chain %s", chain.id)
            empty_msa = msa.Msa.from_empty(
                query_sequence=chain.sequence,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
            ).to_a3m()
            unpaired_msa = chain.unpaired_msa or empty_msa
            paired_msa = chain.paired_msa or empty_msa
            templates = chain.templates

        return folding_input.ProteinChain(
            id=chain.id,
            sequence=chain.sequence,
            ptms=chain.ptms,
            unpaired_msa=unpaired_msa,
            paired_msa=paired_msa,
            templates=templates,
        )

    def process_rna_chain(
        self, chain: folding_input.RnaChain
    ) -> folding_input.RnaChain:
        """Processes a single RNA chain.

        Note: RNA MSA search (via Nhmmer) is not supported in AlphaFast.
        RNA chains will use an empty MSA or any pre-provided MSA.
        """
        if chain.unpaired_msa is not None:
            logging.info(
                "Using provided MSA for RNA chain %s.", chain.id,
            )
            empty_msa = msa.Msa.from_empty(
                query_sequence=chain.sequence, chain_poly_type=mmcif_names.RNA_CHAIN
            ).to_a3m()
            unpaired_msa = chain.unpaired_msa or empty_msa
        else:
            logging.warning(
                "RNA MSA search is not supported in AlphaFast (requires HMMER). "
                "Using empty MSA for RNA chain %s.", chain.id,
            )
            unpaired_msa = msa.Msa.from_empty(
                query_sequence=chain.sequence, chain_poly_type=mmcif_names.RNA_CHAIN
            ).to_a3m()
        return folding_input.RnaChain(
            id=chain.id,
            sequence=chain.sequence,
            modifications=chain.modifications,
            unpaired_msa=unpaired_msa,
        )

    def process(self, fold_input: folding_input.Input) -> folding_input.Input:
        """Runs MSA and template tools and returns a new Input with the results."""
        processed_chains = []
        for chain in fold_input.chains:
            logging.info("Running data pipeline for chain %s...", chain.id)
            process_chain_start_time = time.time()
            match chain:
                case folding_input.ProteinChain():
                    processed_chains.append(self.process_protein_chain(chain))
                case folding_input.RnaChain():
                    processed_chains.append(self.process_rna_chain(chain))
                case _:
                    processed_chains.append(chain)
            logging.info(
                "Running data pipeline for chain %s took %.2f seconds",
                chain.id,
                time.time() - process_chain_start_time,
            )

        return dataclasses.replace(fold_input, chains=processed_chains)

    def process_batch(
        self, fold_inputs: Sequence[folding_input.Input]
    ) -> Sequence[folding_input.Input]:
        """Process multiple fold inputs with batched MSA search.

        This method is more efficient than processing individually because:
        1. Single createdb call for ALL protein sequences
        2. GPU processes all sequences in parallel
        3. Amortizes GPU kernel launch overhead

        Args:
            fold_inputs: Sequence of fold inputs to process together.

        Returns:
            Sequence of processed fold inputs with MSA and templates.
        """
        if not fold_inputs:
            return []

        # Check if using MMseqs2 (required for batch mode)
        is_mmseqs = isinstance(
            self._uniref90_msa_config.config, msa_config.MmseqsConfig
        )
        if not is_mmseqs:
            logging.warning(
                "Batch mode requires MMseqs2. Falling back to sequential processing."
            )
            return [self.process(fi) for fi in fold_inputs]

        batch_start_time = time.time()
        logging.info("Starting batch processing for %d fold inputs", len(fold_inputs))

        # Step 1: Collect all unique protein sequences that need MSA search
        # Use sequence content as key to handle identical sequences (homomers)
        all_sequences: dict[str, str] = {}  # seq_id -> sequence
        seq_to_locations: dict[
            str, list[tuple[int, str]]
        ] = {}  # seq -> [(fold_idx, chain_id)]

        for fold_idx, fold_input in enumerate(fold_inputs):
            for chain in fold_input.chains:
                if isinstance(chain, folding_input.ProteinChain):
                    # Only search if MSA is not already provided
                    if chain.unpaired_msa is None and chain.paired_msa is None:
                        seq = chain.sequence
                        if seq not in seq_to_locations:
                            # Create unique ID for this sequence
                            seq_id = f"seq_{len(all_sequences)}"
                            all_sequences[seq_id] = seq
                            seq_to_locations[seq] = []
                        seq_to_locations[seq].append((fold_idx, chain.id))

        if not all_sequences:
            logging.info("No protein sequences need MSA search, skipping batch")
            return [self.process(fi) for fi in fold_inputs]

        logging.info(
            "Collected %d unique protein sequences from %d fold inputs",
            len(all_sequences),
            len(fold_inputs),
        )

        # Step 2: Run batch MSA search using MmseqsMultiDBBatch
        mmseqs_cfg = self._uniref90_msa_config.config
        assert isinstance(mmseqs_cfg, msa_config.MmseqsConfig)

        # Get database paths from configs
        db_paths = {
            "uniref90": self._uniref90_msa_config.config.database_config.path,
            "mgnify": self._mgnify_msa_config.config.database_config.path,
            "small_bfd": self._small_bfd_msa_config.config.database_config.path,
            "uniprot": self._uniprot_msa_config.config.database_config.path,
        }

        # Get max sequences per database from configs
        max_seqs = {
            "uniref90": self._uniref90_msa_config.config.max_sequences,
            "mgnify": self._mgnify_msa_config.config.max_sequences,
            "small_bfd": self._small_bfd_msa_config.config.max_sequences,
            "uniprot": self._uniprot_msa_config.config.max_sequences,
        }

        batch_searcher = mmseqs_batch.MmseqsMultiDBBatch(
            binary_path=mmseqs_cfg.binary_path,
            database_paths=db_paths,
            e_value=mmseqs_cfg.e_value,
            sensitivity=mmseqs_cfg.sensitivity,
            max_sequences_per_db=max_seqs,
            gpu_enabled=mmseqs_cfg.gpu_enabled,
            gpu_device=mmseqs_cfg.gpu_device,
            threads=mmseqs_cfg.threads,
        )

        logging.info("Running batch MSA search across all databases...")
        msa_results = batch_searcher.search_all_databases_pipelined(all_sequences)
        logging.info(
            "Batch MSA search completed in %.2f seconds",
            time.time() - batch_start_time,
        )

        # Step 3: Build a mapping from sequence to MSA results
        # seq_id -> (uniref90_msa, mgnify_msa, small_bfd_msa, uniprot_msa)
        seq_id_to_msa: dict[str, tuple[msa.Msa, msa.Msa, msa.Msa, msa.Msa]] = {}

        for seq_id, sequence in all_sequences.items():
            # Get results for each database
            uniref90_result = msa_results.get("uniref90")
            mgnify_result = msa_results.get("mgnify")
            small_bfd_result = msa_results.get("small_bfd")
            uniprot_result = msa_results.get("uniprot")

            # Parse A3M to Msa objects
            uniref90_a3m = (
                uniref90_result.results[seq_id].a3m if uniref90_result else ""
            )
            mgnify_a3m = mgnify_result.results[seq_id].a3m if mgnify_result else ""
            small_bfd_a3m = (
                small_bfd_result.results[seq_id].a3m if small_bfd_result else ""
            )
            uniprot_a3m = uniprot_result.results[seq_id].a3m if uniprot_result else ""

            uniref90_msa = msa.Msa.from_a3m(
                query_sequence=sequence,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
                a3m=uniref90_a3m,
            )
            mgnify_msa = msa.Msa.from_a3m(
                query_sequence=sequence,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
                a3m=mgnify_a3m,
            )
            small_bfd_msa = msa.Msa.from_a3m(
                query_sequence=sequence,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
                a3m=small_bfd_a3m,
            )
            uniprot_msa = msa.Msa.from_a3m(
                query_sequence=sequence,
                chain_poly_type=mmcif_names.PROTEIN_CHAIN,
                a3m=uniprot_a3m,
            )

            seq_id_to_msa[seq_id] = (
                uniref90_msa,
                mgnify_msa,
                small_bfd_msa,
                uniprot_msa,
            )

        # Step 4: Create mapping from sequence content to seq_id
        seq_content_to_id = {seq: seq_id for seq_id, seq in all_sequences.items()}

        # Step 5: Process each fold input, distributing MSA results
        processed_fold_inputs = []

        for fold_idx, fold_input in enumerate(fold_inputs):
            processed_chains = []

            for chain in fold_input.chains:
                process_chain_start_time = time.time()
                logging.info("Processing chain %s from fold input %d...", chain.id, fold_idx)

                if isinstance(chain, folding_input.ProteinChain):
                    if chain.unpaired_msa is None and chain.paired_msa is None:
                        # Get MSAs from batch results
                        seq = chain.sequence
                        seq_id = seq_content_to_id.get(seq)

                        if seq_id and seq_id in seq_id_to_msa:
                            uniref90_msa, mgnify_msa, small_bfd_msa, uniprot_msa = (
                                seq_id_to_msa[seq_id]
                            )

                            # Deduplicate MSAs (same as sequential processing)
                            unpaired_protein_msa = msa.Msa.from_multiple_msas(
                                msas=[uniref90_msa, small_bfd_msa, mgnify_msa],
                                deduplicate=True,
                            )
                            paired_protein_msa = msa.Msa.from_multiple_msas(
                                msas=[uniprot_msa], deduplicate=False
                            )

                            # Run template search (depends on MSA, so still per-chain)
                            has_templates = chain.templates is not None
                            template_hits = _get_protein_templates(
                                sequence=chain.sequence,
                                input_msa_a3m=unpaired_protein_msa.to_a3m(),
                                run_template_search=not has_templates,
                                templates_config=self._templates_config,
                                pdb_database_path=self._pdb_database_path,
                            )

                            pdb_templates = [
                                folding_input.Template(
                                    mmcif=struc.to_mmcif(),
                                    query_to_template_map=hit.query_to_hit_mapping,
                                )
                                for hit, struc in template_hits.get_hits_with_structures()
                            ]

                            # Get Foldseek templates and merge
                            foldseek_tmpls = self._get_foldseek_templates(
                                chain.sequence
                            )
                            templates = self._merge_templates(
                                pdb_templates, foldseek_tmpls
                            )

                            processed_chain = folding_input.ProteinChain(
                                id=chain.id,
                                sequence=chain.sequence,
                                ptms=chain.ptms,
                                unpaired_msa=unpaired_protein_msa.to_a3m(),
                                paired_msa=paired_protein_msa.to_a3m(),
                                templates=templates,
                            )
                        else:
                            # Fallback to sequential if somehow missing
                            processed_chain = self.process_protein_chain(chain)
                    else:
                        # Chain already has MSA, process normally
                        processed_chain = self.process_protein_chain(chain)

                    processed_chains.append(processed_chain)

                elif isinstance(chain, folding_input.RnaChain):
                    # RNA chains use Nhmmer, process sequentially
                    processed_chains.append(self.process_rna_chain(chain))
                else:
                    # Other chain types pass through
                    processed_chains.append(chain)

                logging.info(
                    "Processing chain %s took %.2f seconds",
                    chain.id,
                    time.time() - process_chain_start_time,
                )

            processed_fold_inputs.append(
                dataclasses.replace(fold_input, chains=processed_chains)
            )

        total_time = time.time() - batch_start_time
        logging.info(
            "Batch processing completed: %d fold inputs, %d sequences in %.2f seconds",
            len(fold_inputs),
            len(all_sequences),
            total_time,
        )

        return processed_fold_inputs
