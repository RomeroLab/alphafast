# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
#
# Modifications Copyright 2026 Romero Lab, Duke University

"""AlphaFold 3 structure prediction script.

AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

To request access to the AlphaFold 3 model parameters, follow the process set
out at https://github.com/google-deepmind/alphafold3. You may only use these
if received directly from Google. Use is subject to terms of use available at
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
"""

from collections.abc import Sequence
import datetime
import json
import os
import pathlib
import string
import textwrap
import time
import typing

from absl import app
from absl import flags
from alphafold3.common import folding_input
from alphafold3.common import resources
from alphafold3.data import pipeline
from alphafold3.data.tools import shards
from alphafold3.model.inference import make_model_config
from alphafold3.model.inference import ModelRunner
from alphafold3.model.inference import process_fold_input
from alphafold3.model.inference import write_fold_input_json
import jax
import tokamax


_HOME_DIR = pathlib.Path(os.environ.get("HOME", "/root"))
_DEFAULT_MODEL_DIR = _HOME_DIR / "models"
_DEFAULT_DB_DIR = _HOME_DIR / "public_databases"


# Input and output paths.
_JSON_PATH = flags.DEFINE_string(
    "json_path",
    None,
    "Path to the input JSON file.",
)
_INPUT_DIR = flags.DEFINE_string(
    "input_dir",
    None,
    "Path to the directory containing input JSON files.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    None,
    "Path to a directory where the results will be saved.",
)
MODEL_DIR = flags.DEFINE_string(
    "model_dir",
    _DEFAULT_MODEL_DIR.as_posix(),
    "Path to the model to use for inference.",
)

# Control which stages to run.
_RUN_DATA_PIPELINE = flags.DEFINE_bool(
    "run_data_pipeline",
    True,
    "Whether to run the data pipeline on the fold inputs.",
)
_RUN_INFERENCE = flags.DEFINE_bool(
    "run_inference",
    True,
    "Whether to run inference on the fold inputs.",
)

# Database paths.
DB_DIR = flags.DEFINE_multi_string(
    "db_dir",
    (_DEFAULT_DB_DIR.as_posix(),),
    "Path to the directory containing the databases. Can be specified multiple"
    " times to search multiple directories in order.",
)

_PDB_DATABASE_PATH = flags.DEFINE_string(
    "pdb_database_path",
    "${DB_DIR}/mmcif_files",
    "PDB database directory with mmCIF files path, used for template search.",
)

# MMseqs2-GPU configuration (optional, for faster protein MSA searches).
_MMSEQS_BINARY_PATH = flags.DEFINE_string(
    "mmseqs_binary_path",
    None,
    "Path to the MMseqs2 binary. If not specified, auto-detected at "
    "$HOME/.local/bin/mmseqs or via PATH. MMseqs2-GPU provides 10-30x faster "
    "protein MSA searches compared to Jackhmmer.",
)
_MMSEQS_DB_DIR = flags.DEFINE_string(
    "mmseqs_db_dir",
    None,
    "Directory containing MMseqs2 padded databases. Required for MMseqs2-GPU "
    "mode. Expected databases: uniref90_padded, mgnify_padded, small_bfd_padded, "
    "uniprot_padded. Create them using convert_databases_to_mmseqs.sh.",
)
_USE_MMSEQS_GPU = flags.DEFINE_bool(
    "use_mmseqs_gpu",
    True,
    "Whether to use GPU acceleration for MMseqs2 searches. Requires padded "
    "databases created with `mmseqs makepaddedseqdb`. Set to False to use "
    "MMseqs2 on CPU only.",
)
_MMSEQS_SENSITIVITY = flags.DEFINE_float(
    "mmseqs_sensitivity",
    7.5,
    "MMseqs2 sensitivity (-s flag). Range 1-7.5, higher values find more "
    "remote homologs but are slower. 7.5 is recommended for AlphaFold.",
    lower_bound=1.0,
    upper_bound=7.5,
)
_MMSEQS_N_THREADS = flags.DEFINE_integer(
    "mmseqs_n_threads",
    len(os.sched_getaffinity(0)),
    "Number of CPU threads for MMseqs2 non-GPU operations.",
    lower_bound=1,
)
_MMSEQS_SEQUENTIAL = flags.DEFINE_bool(
    "mmseqs_sequential",
    False,
    "Run MMseqs2 database searches sequentially instead of in parallel. "
    "Use this if you encounter GPU out-of-memory errors when running the data "
    "pipeline together with inference. When using run_data_pipeline.py separately, "
    "parallel execution (the default) is recommended.",
)

# Batch processing configuration.
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    None,
    "Number of fold inputs to process together in a single batch. When set, "
    "all protein sequences from up to batch_size fold inputs are collected "
    "into a single MMseqs2 queryDB for GPU-accelerated batch search. This is "
    "much more efficient than sequential processing. If not set, processes "
    "each fold input sequentially (current default behavior).",
    lower_bound=1,
)

# Pre-computed MSA support (for inference-only mode).
_PRECOMPUTED_MSA_PATH = flags.DEFINE_string(
    "precomputed_msa_path",
    None,
    "Path to a pre-computed MSA A3M file. When provided, MSA search is skipped "
    "and this file is used directly. Useful for running inference on pre-computed "
    "MSAs from the benchmark suite.",
)

# Pre-computed template A3M support (for inference-only mode).
_PRECOMPUTED_TEMPLATES_A3M_PATH = flags.DEFINE_string(
    "precomputed_templates_a3m_path",
    None,
    "Path to a pre-computed template A3M file. When provided, template search is "
    "skipped and this file is used directly. Useful for running inference with "
    "pre-computed templates from the benchmark suite.",
)

# Template mode configuration.
_TEMPLATE_MODE = flags.DEFINE_enum(
    "template_mode",
    "default",
    ["default", "foldseek_backup", "foldseek_first"],
    "Template search strategy: "
    "'default' uses only MMseqs2-GPU/hmmsearch with PDB (no Foldseek); "
    "'foldseek_backup' uses PDB search first, then fills remaining slots with Foldseek/AFDB; "
    "'foldseek_first' uses Foldseek/AFDB first, then fills remaining slots with PDB search.",
)

# Template search thresholds (used in ALL template modes).
# These are separate from MSA search parameters - template search is more selective.
# Note: NO sequence identity filter - we rely on e-value and coverage only.
_TEMPLATE_E_VALUE = flags.DEFINE_float(
    "template_e_value",
    1e-3,
    "E-value threshold for template search. "
    "Much stricter than original hmmsearch (e-value=100).",
    lower_bound=0.0,
)
_TEMPLATE_MIN_COVERAGE = flags.DEFINE_float(
    "template_min_coverage",
    0.40,
    "Minimum alignment coverage (0-1) for template search. Default 0.40 (40%).",
    lower_bound=0.0,
    upper_bound=1.0,
)

# Foldseek configuration (for structural template search from AFDB).
_FOLDSEEK_BINARY_PATH = flags.DEFINE_string(
    "foldseek_binary_path",
    None,
    "Path to the Foldseek binary. If not specified, auto-detected at "
    "$HOME/.local/bin/foldseek or via PATH.",
)
_FOLDSEEK_DATABASE_PATH = flags.DEFINE_string(
    "foldseek_database_path",
    None,
    "Path to the AFDB Foldseek database. Create using setup_foldseek_afdb.sh. "
    "Required when --template_mode is foldseek_backup or foldseek_first.",
)
_FOLDSEEK_MAX_TEMPLATES = flags.DEFINE_integer(
    "foldseek_max_templates",
    4,
    "Maximum number of Foldseek templates to use.",
    lower_bound=1,
)
_FOLDSEEK_MIN_LDDT = flags.DEFINE_float(
    "foldseek_min_lddt",
    0.5,
    "Minimum LDDT score (0-1) for Foldseek template hits.",
    lower_bound=0.0,
    upper_bound=1.0,
)
_FOLDSEEK_MIN_PLDDT = flags.DEFINE_float(
    "foldseek_min_plddt",
    50.0,
    "Minimum mean pLDDT score (0-100) from ESMFold prediction to proceed "
    "with Foldseek search. If ESMFold confidence is below this threshold, "
    "Foldseek search is skipped.",
    lower_bound=0.0,
    upper_bound=100.0,
)
_FOLDSEEK_E_VALUE = flags.DEFINE_float(
    "foldseek_e_value",
    1e-3,
    "E-value threshold for Foldseek structural search.",
    lower_bound=0.0,
)
_FOLDSEEK_THREADS = flags.DEFINE_integer(
    "foldseek_threads",
    8,
    "Number of CPU threads for Foldseek search.",
    lower_bound=1,
)
_FOLDSEEK_GPU = flags.DEFINE_bool(
    "foldseek_gpu",
    True,
    "Whether to use GPU acceleration for Foldseek structural search. "
    "Requires Foldseek built with GPU support.",
)
_ESMFOLD_DEVICE = flags.DEFINE_string(
    "esmfold_device",
    None,
    "Device for ESMFold inference ('cuda', 'cpu', or None for auto). "
    "ESMFold requires ~40GB VRAM for full model. Use --esmfold_chunk_size "
    "for memory-constrained systems.",
)
_ESMFOLD_CHUNK_SIZE = flags.DEFINE_integer(
    "esmfold_chunk_size",
    None,
    "Optional chunk size for ESMFold memory optimization with long sequences. "
    "Reduces memory usage but may affect prediction quality.",
    lower_bound=1,
)
_AFDB_CACHE_DIR = flags.DEFINE_string(
    "afdb_cache_dir",
    None,
    "Optional directory for caching downloaded AFDB mmCIF structures. "
    "If not set, structures are downloaded on-demand without caching.",
)

# Data pipeline configuration.
_RESOLVE_MSA_OVERLAPS = flags.DEFINE_bool(
    "resolve_msa_overlaps",
    True,
    "Whether to deduplicate unpaired MSA against paired MSA. The default"
    " behaviour matches the method described in the AlphaFold 3 paper. Set this"
    " to false if providing custom paired MSA using the unpaired MSA field to"
    " keep it exactly as is as deduplication against the paired MSA could break"
    " the manually crafted pairing between MSA sequences.",
)
_MAX_TEMPLATE_DATE = flags.DEFINE_string(
    "max_template_date",
    "2021-09-30",  # By default, use the date from the AlphaFold 3 paper.
    "Maximum template release date to consider. Format: YYYY-MM-DD. All"
    " templates released after this date will be ignored. Controls also whether"
    " to allow use of model coordinates for a chemical component from the CCD"
    " if RDKit conformer generation fails and the component does not have ideal"
    " coordinates set. Only for components that have been released before this"
    " date the model coordinates can be used as a fallback.",
)
_CONFORMER_MAX_ITERATIONS = flags.DEFINE_integer(
    "conformer_max_iterations",
    None,  # Default to RDKit default parameters value.
    "Optional override for maximum number of iterations to run for RDKit "
    "conformer search.",
    lower_bound=0,
)

# JAX inference performance tuning.
_JAX_COMPILATION_CACHE_DIR = flags.DEFINE_string(
    "jax_compilation_cache_dir",
    None,
    "Path to a directory for the JAX compilation cache.",
)
_GPU_DEVICE = flags.DEFINE_integer(
    "gpu_device",
    0,
    "Optional override for the GPU device to use for inference, uses zero-based"
    " indexing. Defaults to the 0th GPU on the system. Useful on multi-GPU"
    " systems to pin each run to a specific GPU. Note that if GPUs are already"
    " pre-filtered by the environment (e.g. by using CUDA_VISIBLE_DEVICES),"
    " this flag refers to the GPU index after the filtering has been done.",
)
_BUCKETS = flags.DEFINE_list(
    "buckets",
    # pyformat: disable
    [
        "256",
        "512",
        "768",
        "1024",
        "1280",
        "1536",
        "2048",
        "2560",
        "3072",
        "3584",
        "4096",
        "4608",
        "5120",
    ],
    # pyformat: enable
    "Strictly increasing order of token sizes for which to cache compilations."
    " For any input with more tokens than the largest bucket size, a new bucket"
    " is created for exactly that number of tokens.",
)
_FLASH_ATTENTION_IMPLEMENTATION = flags.DEFINE_enum(
    "flash_attention_implementation",
    default="triton",
    enum_values=["triton", "cudnn", "xla"],
    help=(
        "Flash attention implementation to use. 'triton' and 'cudnn' uses a"
        " Triton and cuDNN flash attention implementation, respectively. The"
        " Triton kernel is fastest and has been tested more thoroughly. The"
        " Triton and cuDNN kernels require Ampere GPUs or later. 'xla' uses an"
        " XLA attention implementation (no flash attention) and is portable"
        " across GPU devices."
    ),
)
_NUM_RECYCLES = flags.DEFINE_integer(
    "num_recycles",
    10,
    "Number of recycles to use during inference.",
    lower_bound=1,
)
_NUM_DIFFUSION_SAMPLES = flags.DEFINE_integer(
    "num_diffusion_samples",
    5,
    "Number of diffusion samples to generate.",
    lower_bound=1,
)
_NUM_SEEDS = flags.DEFINE_integer(
    "num_seeds",
    None,
    "Number of seeds to use for inference. If set, only a single seed must be"
    " provided in the input JSON. AlphaFold 3 will then generate random seeds"
    " in sequence, starting from the single seed specified in the input JSON."
    " The full input JSON produced by AlphaFold 3 will include the generated"
    " random seeds. If not set, AlphaFold 3 will use the seeds as provided in"
    " the input JSON.",
    lower_bound=1,
)

# Output controls.
_SAVE_EMBEDDINGS = flags.DEFINE_bool(
    "save_embeddings",
    False,
    "Whether to save the final trunk single and pair embeddings in the output."
    " Note that the embeddings are large float16 arrays: num_tokens * 384"
    " + num_tokens * num_tokens * 128.",
)
_SAVE_DISTOGRAM = flags.DEFINE_bool(
    "save_distogram",
    False,
    "Whether to save the final distogram in the output. Note that the distogram"
    " is a large float16 array: num_tokens * num_tokens * 64.",
)
_FORCE_OUTPUT_DIR = flags.DEFINE_bool(
    "force_output_dir",
    False,
    "Whether to force the output directory to be used even if it already exists"
    " and is non-empty. Useful to set this to True to run the data pipeline and"
    " the inference separately, but use the same output directory.",
)
_WRITE_TIMING_JSON = flags.DEFINE_bool(
    "write_timing_json",
    False,
    "Whether to write per-input inference timing to a JSONL file at"
    " {output_dir}/inference_timing.jsonl. Each line contains the input name,"
    " inference time in seconds, and status.",
)


def replace_db_dir(path_with_db_dir: str, db_dirs: Sequence[str]) -> str:
    """Replaces the DB_DIR placeholder in a path with the given DB_DIR."""
    template = string.Template(path_with_db_dir)
    if "DB_DIR" in template.get_identifiers():
        for db_dir in db_dirs:
            path = template.substitute(DB_DIR=db_dir)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(
            f"{path_with_db_dir} with ${{DB_DIR}} not found in any of {db_dirs}."
        )
    if (
        sharded_paths := shards.get_sharded_paths(path_with_db_dir)
    ) is not None:
        db_exists = all(os.path.exists(p) for p in sharded_paths)
    else:
        db_exists = os.path.exists(path_with_db_dir)
    if not db_exists:
        raise FileNotFoundError(f"{path_with_db_dir} does not exist.")
    return path_with_db_dir


def main(_):
    if _JAX_COMPILATION_CACHE_DIR.value is not None:
        jax.config.update(
            "jax_compilation_cache_dir", _JAX_COMPILATION_CACHE_DIR.value
        )

    if _JSON_PATH.value is None == _INPUT_DIR.value is None:
        raise ValueError(
            "Exactly one of --json_path or --input_dir must be specified."
        )

    if not _RUN_INFERENCE.value and not _RUN_DATA_PIPELINE.value:
        raise ValueError(
            "At least one of --run_inference or --run_data_pipeline must be"
            " set to true."
        )

    if _INPUT_DIR.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_dir(
            pathlib.Path(_INPUT_DIR.value)
        )
    elif _JSON_PATH.value is not None:
        fold_inputs = folding_input.load_fold_inputs_from_path(
            pathlib.Path(_JSON_PATH.value)
        )
    else:
        raise AssertionError(
            "Exactly one of --json_path or --input_dir must be specified."
        )

    # Make sure we can create the output directory before running anything.
    try:
        os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
    except OSError as e:
        print(f"Failed to create output directory {_OUTPUT_DIR.value}: {e}")
        raise

    if _RUN_INFERENCE.value:
        # Fail early on incompatible devices, but only if we're running inference.
        gpu_devices = jax.local_devices(backend="gpu")
        if gpu_devices:
            compute_capability = float(
                gpu_devices[_GPU_DEVICE.value].compute_capability
            )
            if compute_capability < 6.0:
                raise ValueError(
                    "AlphaFold 3 requires at least GPU compute capability 6.0 (see"
                    " https://developer.nvidia.com/cuda-gpus)."
                )
            elif 7.0 <= compute_capability < 8.0:
                xla_flags = os.environ.get("XLA_FLAGS")
                required_flag = (
                    "--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
                )
                if not xla_flags or required_flag not in xla_flags:
                    raise ValueError(
                        "For devices with GPU compute capability 7.x (see"
                        " https://developer.nvidia.com/cuda-gpus) the ENV XLA_FLAGS must"
                        f' include "{required_flag}".'
                    )
                if _FLASH_ATTENTION_IMPLEMENTATION.value != "xla":
                    raise ValueError(
                        "For devices with GPU compute capability 7.x (see"
                        " https://developer.nvidia.com/cuda-gpus) the"
                        ' --flash_attention_implementation must be set to "xla".'
                    )

    notice = textwrap.wrap(
        "Running AlphaFold 3. Please note that standard AlphaFold 3 model"
        " parameters are only available under terms of use provided at"
        " https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md."
        " If you do not agree to these terms and are using AlphaFold 3 derived"
        " model parameters, cancel execution of AlphaFold 3 inference with"
        " CTRL-C, and do not use the model parameters.",
        break_long_words=False,
        break_on_hyphens=False,
        width=80,
    )
    print("\n" + "\n".join(notice) + "\n")

    max_template_date = datetime.date.fromisoformat(_MAX_TEMPLATE_DATE.value)
    if _RUN_DATA_PIPELINE.value:
        expand_path = lambda x: replace_db_dir(x, DB_DIR.value)

        data_pipeline_config = pipeline.DataPipelineConfig(
            pdb_database_path=expand_path(_PDB_DATABASE_PATH.value),
            max_template_date=max_template_date,
            # MMseqs2-GPU configuration (for MSA search)
            mmseqs_binary_path=_MMSEQS_BINARY_PATH.value,
            mmseqs_db_dir=_MMSEQS_DB_DIR.value,
            use_mmseqs_gpu=_USE_MMSEQS_GPU.value,
            gpu_device=_GPU_DEVICE.value,
            mmseqs_sensitivity=_MMSEQS_SENSITIVITY.value,
            mmseqs_n_threads=_MMSEQS_N_THREADS.value,
            # Pre-computed MSA path (for inference-only mode)
            precomputed_msa_path=_PRECOMPUTED_MSA_PATH.value,
            # Pre-computed templates A3M path (for inference-only mode)
            precomputed_templates_a3m_path=_PRECOMPUTED_TEMPLATES_A3M_PATH.value,
            mmseqs_sequential=_MMSEQS_SEQUENTIAL.value,
            # Template search thresholds
            template_e_value=_TEMPLATE_E_VALUE.value,
            template_min_coverage=_TEMPLATE_MIN_COVERAGE.value,
            # Template mode
            template_mode=_TEMPLATE_MODE.value,
            # Foldseek configuration
            foldseek_binary_path=_FOLDSEEK_BINARY_PATH.value,
            foldseek_database_path=_FOLDSEEK_DATABASE_PATH.value,
            foldseek_max_templates=_FOLDSEEK_MAX_TEMPLATES.value,
            foldseek_min_lddt=_FOLDSEEK_MIN_LDDT.value,
            foldseek_min_plddt=_FOLDSEEK_MIN_PLDDT.value,
            foldseek_e_value=_FOLDSEEK_E_VALUE.value,
            foldseek_threads=_FOLDSEEK_THREADS.value,
            foldseek_gpu_enabled=_FOLDSEEK_GPU.value,
            esmfold_device=_ESMFOLD_DEVICE.value,
            esmfold_chunk_size=_ESMFOLD_CHUNK_SIZE.value,
            afdb_cache_dir=_AFDB_CACHE_DIR.value,
        )
    else:
        data_pipeline_config = None

    # ==========================================================================
    # PHASE 1: Run data pipeline for ALL inputs BEFORE loading the AF3 model.
    # This prevents OOM errors when using GPU-accelerated tools (MMseqs2-GPU,
    # Foldseek-GPU, ESMFold) during the data pipeline, as the AF3 model
    # consumes ~95% of GPU memory.
    # ==========================================================================
    processed_fold_inputs = []  # List of (processed_fold_input, output_dir)

    if _RUN_DATA_PIPELINE.value:
        print("\n" + "=" * 70)
        print("PHASE 1: Running data pipeline for all inputs")
        print("(AF3 model will be loaded after data pipeline completes)")
        print("=" * 70 + "\n")

        # Expand seeds for all fold inputs first
        expanded_fold_inputs = []
        for fold_input in fold_inputs:
            if _NUM_SEEDS.value is not None:
                print(
                    f"Expanding fold job {fold_input.name} to {_NUM_SEEDS.value} seeds"
                )
                fold_input = fold_input.with_multiple_seeds(_NUM_SEEDS.value)
            expanded_fold_inputs.append(fold_input)

        # Check if batch mode is enabled
        if _BATCH_SIZE.value is not None and len(expanded_fold_inputs) > 1:
            # Batch mode: process multiple fold inputs together
            batch_size = _BATCH_SIZE.value
            print(
                f"\nBATCH MODE: Processing {len(expanded_fold_inputs)} fold inputs "
                f"in batches of {batch_size}\n"
            )

            data_pipeline = pipeline.DataPipeline(data_pipeline_config)

            # Process in batches
            for batch_start in range(0, len(expanded_fold_inputs), batch_size):
                batch_end = min(
                    batch_start + batch_size, len(expanded_fold_inputs)
                )
                batch = expanded_fold_inputs[batch_start:batch_end]

                print(
                    f"\n--- Processing batch {batch_start // batch_size + 1}: "
                    f"fold inputs {batch_start + 1} to {batch_end} ---\n"
                )

                # Run batch processing
                processed_inputs = data_pipeline.process_batch(batch)

                # Store each processed input with its output directory
                for processed_input in processed_inputs:
                    output_dir = os.path.join(
                        _OUTPUT_DIR.value, processed_input.sanitised_name()
                    )
                    # Write the processed data JSON
                    write_fold_input_json(processed_input, output_dir)
                    processed_fold_inputs.append((processed_input, output_dir))
                    print(
                        f"Fold job {processed_input.name} data pipeline done.\n"
                    )
        else:
            # Sequential mode: process each fold input individually
            if _BATCH_SIZE.value is None and len(expanded_fold_inputs) > 1:
                print(
                    "\nSequential mode: processing fold inputs one at a time"
                )
                print("(Use --batch_size=N to enable batch processing)\n")

            for fold_input in expanded_fold_inputs:
                output_dir = os.path.join(
                    _OUTPUT_DIR.value, fold_input.sanitised_name()
                )

                # Run data pipeline only (no model loaded yet)
                processed_input = process_fold_input(
                    fold_input=fold_input,
                    data_pipeline_config=data_pipeline_config,
                    model_runner=None,  # Don't load model during data pipeline
                    output_dir=output_dir,
                    buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
                    ref_max_modified_date=max_template_date,
                    conformer_max_iterations=_CONFORMER_MAX_ITERATIONS.value,
                    resolve_msa_overlaps=_RESOLVE_MSA_OVERLAPS.value,
                    force_output_dir=_FORCE_OUTPUT_DIR.value,
                )
                processed_fold_inputs.append((processed_input, output_dir))

        print(
            f"\nData pipeline completed for {len(processed_fold_inputs)} inputs.\n"
        )
    else:
        # If not running data pipeline, just collect the inputs for inference
        for fold_input in fold_inputs:
            if _NUM_SEEDS.value is not None:
                print(
                    f"Expanding fold job {fold_input.name} to {_NUM_SEEDS.value} seeds"
                )
                fold_input = fold_input.with_multiple_seeds(_NUM_SEEDS.value)
            # When using --json_path with inference-only mode, use output_dir directly
            # since user explicitly specifies where to write output.
            # When using --input_dir, create subdirectories for each input.
            if _JSON_PATH.value is not None:
                output_dir = _OUTPUT_DIR.value
            else:
                output_dir = os.path.join(
                    _OUTPUT_DIR.value, fold_input.sanitised_name()
                )
            processed_fold_inputs.append((fold_input, output_dir))

    # ==========================================================================
    # PHASE 2: Load AF3 model and run inference on all processed inputs.
    # Model is loaded AFTER data pipeline completes to avoid OOM.
    # ==========================================================================
    if _RUN_INFERENCE.value:
        print("\n" + "=" * 70)
        print("PHASE 2: Loading AF3 model and running inference")
        print("=" * 70 + "\n")

        devices = jax.local_devices(backend="gpu")
        print(
            f"Found local devices: {devices}, using device {_GPU_DEVICE.value}:"
            f" {devices[_GPU_DEVICE.value]}"
        )

        print("Building model from scratch...")
        model_runner = ModelRunner(
            config=make_model_config(
                flash_attention_implementation=typing.cast(
                    tokamax.DotProductAttentionImplementation,
                    _FLASH_ATTENTION_IMPLEMENTATION.value,
                ),
                num_diffusion_samples=_NUM_DIFFUSION_SAMPLES.value,
                num_recycles=_NUM_RECYCLES.value,
                return_embeddings=_SAVE_EMBEDDINGS.value,
                return_distogram=_SAVE_DISTOGRAM.value,
            ),
            device=devices[_GPU_DEVICE.value],
            model_dir=pathlib.Path(MODEL_DIR.value),
        )
        # Load model parameters
        print("Loading model parameters...")
        _ = model_runner.model_params
        print("Model loaded successfully.\n")

        # Optional timing JSONL file
        timing_path = None
        if _WRITE_TIMING_JSON.value:
            timing_path = os.path.join(
                _OUTPUT_DIR.value, "inference_timing.jsonl"
            )

        # Run inference on all processed inputs
        for processed_input, output_dir in processed_fold_inputs:
            t0 = time.time()
            status = "success"
            try:
                process_fold_input(
                    fold_input=processed_input,
                    data_pipeline_config=None,  # Data pipeline already done
                    model_runner=model_runner,
                    output_dir=output_dir,
                    buckets=tuple(int(bucket) for bucket in _BUCKETS.value),
                    ref_max_modified_date=max_template_date,
                    conformer_max_iterations=_CONFORMER_MAX_ITERATIONS.value,
                    resolve_msa_overlaps=_RESOLVE_MSA_OVERLAPS.value,
                    force_output_dir=True,  # Reuse same output dir from data pipeline
                )
            except Exception as e:
                status = "failed"
                print(f"ERROR: Fold job {processed_input.name} failed: {e}")
            elapsed = time.time() - t0

            if timing_path is not None:
                record = {
                    "name": processed_input.name,
                    "inference_seconds": round(elapsed, 3),
                    "status": status,
                }
                with open(timing_path, "at") as f:
                    f.write(json.dumps(record) + "\n")

    print(f"Done running {len(processed_fold_inputs)} fold jobs.")


if __name__ == "__main__":
    flags.mark_flags_as_required(["output_dir"])
    app.run(main)
