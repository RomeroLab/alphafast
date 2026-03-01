# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Standalone data pipeline script for AlphaFold 3.

This script runs the MSA and template search pipeline WITHOUT importing JAX.
This allows MMseqs2-GPU to use the full GPU memory without competing with
JAX's pre-allocated memory pool.

Usage:
    # Stage 1: Run data pipeline (MSA + template search)
    python run_data_pipeline.py \
        --json_path=/path/to/input.json \
        --output_dir=/path/to/output \
        --mmseqs_db_dir=/path/to/mmseqs/databases

    # Stage 2: Run inference on the output (in run_alphafold.py)
    python run_alphafold.py \
        --json_path=/path/to/output/job_name/job_name_data.json \
        --norun_data_pipeline \
        --output_dir=/path/to/output
"""

from collections.abc import Sequence
import datetime
import json
import os
import pathlib
import string
import time

from absl import app
from absl import flags

# IMPORTANT: Only import modules that do NOT depend on JAX
from alphafold3.common import folding_input
from alphafold3.data import pipeline
from alphafold3.data.tools import shards


_HOME_DIR = pathlib.Path(os.environ.get("HOME", "/root"))
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

# MMseqs2-GPU configuration.
_MMSEQS_BINARY_PATH = flags.DEFINE_string(
    "mmseqs_binary_path",
    None,
    "Path to the MMseqs2 binary. If not specified, auto-detected at "
    "$HOME/.local/bin/mmseqs or via PATH.",
)
_MMSEQS_DB_DIR = flags.DEFINE_string(
    "mmseqs_db_dir",
    None,
    "Directory containing MMseqs2 padded databases. Required for MMseqs2-GPU "
    "mode. Expected databases: uniref90_padded, mgnify_padded, small_bfd_padded, "
    "uniprot_padded.",
)
_USE_MMSEQS_GPU = flags.DEFINE_bool(
    "use_mmseqs_gpu",
    True,
    "Whether to use GPU acceleration for MMseqs2 searches.",
)
_GPU_DEVICE = flags.DEFINE_integer(
    "gpu_device",
    None,
    "GPU device to use for MMseqs2 and Foldseek searches (via CUDA_VISIBLE_DEVICES). "
    "If None, uses all available GPUs.",
)
_MMSEQS_SENSITIVITY = flags.DEFINE_float(
    "mmseqs_sensitivity",
    7.5,
    "MMseqs2 sensitivity (-s flag). Range 1-7.5.",
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
    "Use this if you encounter GPU out-of-memory errors with parallel searches.",
)

# Temporary directory configuration (for HPC with slow network storage).
_TEMP_DIR = flags.DEFINE_string(
    "temp_dir",
    None,
    "Directory for temporary files during MSA search. On HPC clusters with "
    "slow network-attached storage (e.g., GPFS, Lustre), set this to fast "
    "local storage (e.g., /scratch on DCC). This can provide 10-13x speedup "
    "for MMseqs2 searches. Typical space usage: 1-5 GB per batch. If not set, "
    "uses the system default temp directory.",
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

# Optional queue directory for producer/consumer inference workflows.
_QUEUE_DIR = flags.DEFINE_string(
    "queue_dir",
    None,
    "Optional directory for queue tokens. When set, a token is written for each "
    "completed MSA JSON, and a producer_done marker is created after the pipeline "
    "finishes. Tokens are written to <queue_dir>/ready.",
)

# Template mode configuration.
_TEMPLATE_MODE = flags.DEFINE_enum(
    "template_mode",
    "default",
    ["default", "foldseek_backup", "foldseek_first"],
    "Template search strategy: "
    "'default' uses only MMseqs2-GPU with PDB (no Foldseek); "
    "'foldseek_backup' uses PDB search first, then fills remaining slots with Foldseek/AFDB; "
    "'foldseek_first' uses Foldseek/AFDB first, then fills remaining slots with PDB search.",
)

# Template search thresholds (used in ALL template modes).
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
    "Path to the AFDB Foldseek database. Required when --template_mode is foldseek_backup or foldseek_first.",
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
    "ESMFold requires ~40GB VRAM for full model.",
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
_MAX_TEMPLATE_DATE = flags.DEFINE_string(
    "max_template_date",
    "2021-09-30",
    "Maximum template release date to consider. Format: YYYY-MM-DD.",
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


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> str:
    """Writes the input JSON to the output directory.

    Returns:
        The path to the written JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{fold_input.sanitised_name()}_data.json")
    print(f"Writing model input JSON to {path}")
    with open(path, "wt") as f:
        f.write(fold_input.to_json())
    return path


def _ensure_queue_dirs(queue_dir: str) -> dict[str, str]:
    """Ensures queue directories exist and returns their paths."""
    ready_dir = os.path.join(queue_dir, "ready")
    in_progress_dir = os.path.join(queue_dir, "in_progress")
    done_dir = os.path.join(queue_dir, "done")
    failed_dir = os.path.join(queue_dir, "failed")
    for path in (ready_dir, in_progress_dir, done_dir, failed_dir):
        os.makedirs(path, exist_ok=True)
    return {
        "ready": ready_dir,
        "in_progress": in_progress_dir,
        "done": done_dir,
        "failed": failed_dir,
    }


def _write_queue_token(
    *, queue_dir: str, fold_input: folding_input.Input, data_json_path: str
) -> None:
    """Writes a ready token for a processed fold input."""
    ready_dir = os.path.join(queue_dir, "ready")
    os.makedirs(ready_dir, exist_ok=True)
    name = fold_input.sanitised_name()
    token = {
        "name": name,
        "data_json_path": data_json_path,
    }
    tmp_path = os.path.join(ready_dir, f".{name}.json.tmp")
    token_path = os.path.join(ready_dir, f"{name}.json")
    with open(tmp_path, "wt") as f:
        json.dump(token, f)
    os.replace(tmp_path, token_path)


def _write_producer_done(queue_dir: str, total_inputs: int) -> None:
    """Writes a marker indicating the producer has finished."""
    marker = {
        "status": "done",
        "total_inputs": total_inputs,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    marker_path = os.path.join(queue_dir, "producer_done")
    tmp_path = f"{marker_path}.tmp"
    with open(tmp_path, "wt") as f:
        json.dump(marker, f)
    os.replace(tmp_path, marker_path)


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig,
    output_dir: os.PathLike[str] | str,
) -> str:
    """Runs data pipeline on a single fold input.

    Args:
        fold_input: Fold input to process.
        data_pipeline_config: Data pipeline config to use.
        output_dir: Output directory to write to.

    Returns:
        Path to the output JSON file.

    Raises:
        ValueError: If the fold input has no chains.
    """
    print(f"\nProcessing fold job {fold_input.name}...")

    if not fold_input.chains:
        raise ValueError("Fold input has no chains.")

    output_subdir = os.path.join(output_dir, fold_input.sanitised_name())
    print(f"Output will be written to {output_subdir}")

    print("Running data pipeline...")
    data_pipeline = pipeline.DataPipeline(data_pipeline_config)
    processed_input = data_pipeline.process(fold_input)

    output_path = write_fold_input_json(processed_input, output_subdir)
    print(f"Fold job {fold_input.name} done.\n")

    return output_path


def main(_):
    if _JSON_PATH.value is None and _INPUT_DIR.value is None:
        raise ValueError(
            "One of --json_path or --input_dir must be specified."
        )
    if _JSON_PATH.value is not None and _INPUT_DIR.value is not None:
        raise ValueError(
            "Only one of --json_path or --input_dir can be specified."
        )
    if _OUTPUT_DIR.value is None:
        raise ValueError("--output_dir must be specified.")

    # Load fold inputs
    if _INPUT_DIR.value is not None:
        fold_inputs = list(
            folding_input.load_fold_inputs_from_dir(
                pathlib.Path(_INPUT_DIR.value)
            )
        )
    else:
        fold_inputs = list(
            folding_input.load_fold_inputs_from_path(
                pathlib.Path(_JSON_PATH.value)
            )
        )

    print(f"Loaded {len(fold_inputs)} fold input(s)")

    # Create output directory
    try:
        os.makedirs(_OUTPUT_DIR.value, exist_ok=True)
    except OSError as e:
        print(f"Failed to create output directory {_OUTPUT_DIR.value}: {e}")
        raise

    if _QUEUE_DIR.value is not None:
        _ensure_queue_dirs(_QUEUE_DIR.value)

    # Build data pipeline config
    def expand_path(x):
        return replace_db_dir(x, DB_DIR.value)

    max_template_date = datetime.date.fromisoformat(_MAX_TEMPLATE_DATE.value)

    data_pipeline_config = pipeline.DataPipelineConfig(
        pdb_database_path=expand_path(_PDB_DATABASE_PATH.value),
        max_template_date=max_template_date,
        # MMseqs2-GPU configuration
        mmseqs_binary_path=_MMSEQS_BINARY_PATH.value,
        mmseqs_db_dir=_MMSEQS_DB_DIR.value,
        use_mmseqs_gpu=_USE_MMSEQS_GPU.value,
        gpu_device=_GPU_DEVICE.value,
        mmseqs_sensitivity=_MMSEQS_SENSITIVITY.value,
        mmseqs_n_threads=_MMSEQS_N_THREADS.value,
        mmseqs_sequential=_MMSEQS_SEQUENTIAL.value,
        temp_dir=_TEMP_DIR.value,
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

    # Process fold inputs - either in batch mode or sequentially
    output_paths = []
    data_pipeline = pipeline.DataPipeline(data_pipeline_config)

    pipeline_start_time = time.time()
    mode = (
        "batch"
        if (_BATCH_SIZE.value is not None and len(fold_inputs) > 1)
        else "sequential"
    )

    if _BATCH_SIZE.value is not None and len(fold_inputs) > 1:
        # Batch mode: process multiple fold inputs together
        batch_size = _BATCH_SIZE.value
        print(f"\n{'=' * 60}")
        print(
            f"BATCH MODE: Processing {len(fold_inputs)} fold inputs in batches of {batch_size}"
        )
        print(f"{'=' * 60}\n")

        # Process in batches
        for batch_start in range(0, len(fold_inputs), batch_size):
            batch_end = min(batch_start + batch_size, len(fold_inputs))
            batch = fold_inputs[batch_start:batch_end]

            print(
                f"\n--- Processing batch {batch_start // batch_size + 1}: "
                f"fold inputs {batch_start + 1} to {batch_end} ---\n"
            )

            processed_inputs = data_pipeline.process_batch(batch)

            # Write output for each processed input
            for fold_input in processed_inputs:
                output_subdir = os.path.join(
                    _OUTPUT_DIR.value, fold_input.sanitised_name()
                )
                output_path = write_fold_input_json(fold_input, output_subdir)
                output_paths.append(output_path)
                if _QUEUE_DIR.value is not None:
                    _write_queue_token(
                        queue_dir=_QUEUE_DIR.value,
                        fold_input=fold_input,
                        data_json_path=output_path,
                    )
                print(f"Fold job {fold_input.name} done.\n")
    else:
        # Sequential mode: process each fold input individually (default)
        if _BATCH_SIZE.value is None:
            print("\nSequential mode: processing fold inputs one at a time")
            print("(Use --batch_size=N to enable batch processing)\n")

        for fold_input in fold_inputs:
            output_path = process_fold_input(
                fold_input=fold_input,
                data_pipeline_config=data_pipeline_config,
                output_dir=_OUTPUT_DIR.value,
            )
            output_paths.append(output_path)
            if _QUEUE_DIR.value is not None:
                _write_queue_token(
                    queue_dir=_QUEUE_DIR.value,
                    fold_input=fold_input,
                    data_json_path=output_path,
                )

    pipeline_elapsed = time.time() - pipeline_start_time

    if _QUEUE_DIR.value is not None:
        _write_producer_done(_QUEUE_DIR.value, total_inputs=len(fold_inputs))

    # Count unique sequences across all fold inputs
    unique_sequences = set()
    total_chains = 0
    for fold_input in fold_inputs:
        for chain in fold_input.chains:
            if hasattr(chain, "sequence"):
                total_chains += 1
                unique_sequences.add(chain.sequence)
    num_unique_sequences = len(unique_sequences)

    # Calculate per-sequence times
    per_input_seconds = (
        pipeline_elapsed / len(fold_inputs) if fold_inputs else 0
    )
    per_unique_seq_seconds = (
        pipeline_elapsed / num_unique_sequences
        if num_unique_sequences > 0
        else 0
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Data pipeline complete!")
    print(f"\nTotal time: {pipeline_elapsed:.2f} seconds")
    print(f"Mode: {mode}")
    if _BATCH_SIZE.value:
        print(f"Batch size: {_BATCH_SIZE.value}")
    print(f"\nInputs processed: {len(fold_inputs)}")
    print(f"Total chains: {total_chains}")
    print(f"Unique sequences: {num_unique_sequences}")
    print(f"\nPer-input time: {per_input_seconds:.2f} seconds")
    print(f"Per-unique-seq time: {per_unique_seq_seconds:.2f} seconds")
    print("\nOutput JSON files:")
    for path in output_paths:
        print(f"  {path}")
    print("\nTo run inference, use:")
    print("  python run_alphafold.py \\")
    print("      --json_path=<output_json> \\")
    print("      --norun_data_pipeline \\")
    print(f"      --output_dir={_OUTPUT_DIR.value}")
    print("=" * 60)


if __name__ == "__main__":
    flags.mark_flags_as_required(["output_dir"])
    app.run(main)
