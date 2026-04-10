#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Universal AlphaFast run script.
#
# Supports Docker and Singularity backends with automatic detection.
# Handles single-GPU and phase-separated multi-GPU modes.
#
# Usage:
#   # Single GPU (device 0, the default)
#   ./scripts/run_alphafast.sh \
#       --input_dir /path/to/inputs \
#       --output_dir /path/to/outputs \
#       --db_dir /path/to/databases \
#       --weights_dir /path/to/weights
#
#   # Single GPU on a specific device
#   ./scripts/run_alphafast.sh ... --gpu_devices 2
#
#   # Multi-GPU (2 GPUs)
#   ./scripts/run_alphafast.sh ... --gpu_devices 0,1
#
#   # Multi-GPU on specific devices
#   ./scripts/run_alphafast.sh ... --gpu_devices 6,7

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
INPUT_DIR=""
OUTPUT_DIR=""
DB_DIR=""
WEIGHTS_DIR=""
NUM_GPUS=""
CONTAINER="romerolabduke/alphafast:latest"
BATCH_SIZE=""
GPU_DEVICES=""
BACKEND=""
RNA_MMSEQS_DB_DIR=""
USE_NHMMER=""

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 --input_dir DIR --output_dir DIR --db_dir DIR --weights_dir DIR [OPTIONS]"
    echo ""
    echo "Required:"
    echo "  --input_dir DIR       Directory containing input JSON files"
    echo "  --output_dir DIR      Output directory for results"
    echo "  --db_dir DIR          Database directory (from setup_databases.sh)"
    echo "  --weights_dir DIR     Directory containing af3.bin.zst"
    echo ""
    echo "Optional:"
    echo "  --gpu_devices IDS     Comma-separated GPU IDs (default: 0)."
    echo "                        Single device = single-GPU mode."
    echo "                        Multiple devices = multi-GPU mode."
    echo "                        Example: --gpu_devices 0,1 for 2-GPU parallel."
    echo "  --num_gpus N          Deprecated: use --gpu_devices instead."
    echo "                        If both given, --gpu_devices takes precedence."
    echo "  --container IMAGE     Container image or .sif path"
    echo "                        (default: romerolabduke/alphafast:latest)"
    echo "  --batch_size N        MSA batch size (default: auto = number of inputs)"
    echo "  --backend TYPE        Force 'docker' or 'singularity' (default: auto-detect)"
    echo "  --rna_mmseqs_db_dir DIR  Override RNA MMseqs2 database directory."
    echo "                        By default, AlphaFast auto-detects <db_dir>/mmseqs_rna"
    echo "                        and uses MMseqs2 nucleotide search for RNA MSA."
    echo "  --use_nhmmer          Force nhmmer for RNA MSA search instead of RNA MMseqs2."
    echo "                        Requires RNA FASTA fallback files, e.g. created with"
    echo "                        setup_databases.sh --include-nhmmer."
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --input_dir)    INPUT_DIR="$2"; shift 2 ;;
        --output_dir)   OUTPUT_DIR="$2"; shift 2 ;;
        --db_dir)       DB_DIR="$2"; shift 2 ;;
        --weights_dir)  WEIGHTS_DIR="$2"; shift 2 ;;
        --num_gpus)     NUM_GPUS="$2"; shift 2 ;;
        --container)    CONTAINER="$2"; shift 2 ;;
        --batch_size)   BATCH_SIZE="$2"; shift 2 ;;
        --gpu_devices)  GPU_DEVICES="$2"; shift 2 ;;
        --backend)      BACKEND="$2"; shift 2 ;;
        --rna_mmseqs_db_dir) RNA_MMSEQS_DB_DIR="$2"; shift 2 ;;
        --use_nhmmer)   USE_NHMMER="true"; shift ;;
        --help|-h)      usage ;;
        *)              echo "Unknown argument: $1"; usage ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_DIR" ] || [ -z "$DB_DIR" ] || [ -z "$WEIGHTS_DIR" ]; then
    echo "ERROR: --input_dir, --output_dir, --db_dir, and --weights_dir are required."
    echo ""
    usage
fi

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

if [ ! -d "$DB_DIR" ]; then
    echo "ERROR: Database directory not found: $DB_DIR"
    exit 1
fi

if [ ! -d "$WEIGHTS_DIR" ]; then
    echo "ERROR: Weights directory not found: $WEIGHTS_DIR"
    exit 1
fi

# ---------------------------------------------------------------------------
# Auto-detect backend
# ---------------------------------------------------------------------------
if [ -z "$BACKEND" ]; then
    if [[ "$CONTAINER" == *.sif ]]; then
        BACKEND="singularity"
    else
        BACKEND="docker"
    fi
fi

if [ "$BACKEND" != "docker" ] && [ "$BACKEND" != "singularity" ]; then
    echo "ERROR: --backend must be 'docker' or 'singularity', got: $BACKEND"
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve paths and defaults
# ---------------------------------------------------------------------------
INPUT_DIR="$(cd "$INPUT_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
DB_DIR="$(cd "$DB_DIR" && pwd)"
WEIGHTS_DIR="$(cd "$WEIGHTS_DIR" && pwd)"
MMSEQS_DB_DIR="${DB_DIR}/mmseqs"

# Auto batch size: count input JSON files
if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=$(find "$INPUT_DIR" -maxdepth 1 -name "*.json" -type f | wc -l | tr -d ' ')
    if [ "$BATCH_SIZE" -eq 0 ]; then
        echo "ERROR: No .json files found in $INPUT_DIR"
        exit 1
    fi
fi

# Resolve GPU devices and count.
# --gpu_devices is the primary way to specify GPUs. --num_gpus is deprecated
# but still accepted for backwards compat (generates 0,1,...,N-1).
if [ -z "$GPU_DEVICES" ]; then
    if [ -n "$NUM_GPUS" ] && [ "$NUM_GPUS" -gt 1 ]; then
        GPU_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
    else
        GPU_DEVICES="0"
    fi
fi

# Derive NUM_GPUS from the device list (authoritative source)
IFS=',' read -r -a _GPU_ARRAY <<< "$GPU_DEVICES"
NUM_GPUS=${#_GPU_ARRAY[@]}

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "AlphaFast Run"
echo "=========================================="
echo "Backend:    $BACKEND"
echo "Container:  $CONTAINER"
echo "Input dir:  $INPUT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "DB dir:     $DB_DIR"
echo "MMseqs dir: $MMSEQS_DB_DIR"
echo "Weights:    $WEIGHTS_DIR"
echo "GPUs:       $NUM_GPUS (devices: $GPU_DEVICES)"
echo "Batch size: $BATCH_SIZE"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# Helper: run a command inside the container
# ---------------------------------------------------------------------------
run_container() {
    local gpu_spec="$1"
    shift

    # Optional RNA MMseqs2 database mount
    local rna_mount_docker=""
    local rna_mount_singularity=""
    if [ -n "$RNA_MMSEQS_DB_DIR" ]; then
        rna_mount_docker="-v ${RNA_MMSEQS_DB_DIR}:/data/rna_mmseqs_databases:ro"
        rna_mount_singularity="--bind ${RNA_MMSEQS_DB_DIR}:/data/rna_mmseqs_databases:ro"
    fi

    if [ "$BACKEND" = "docker" ]; then
        docker run --rm \
            --user "$(id -u):$(id -g)" \
            --gpus "device=${gpu_spec}" \
            -v "${DB_DIR}:/data/public_databases" \
            -v "${MMSEQS_DB_DIR}:/data/mmseqs_databases" \
            -v "${WEIGHTS_DIR}:/data/models" \
            -v "${INPUT_DIR}:/data/af_input" \
            -v "${OUTPUT_DIR}:/data/af_output" \
            $rna_mount_docker \
            "$CONTAINER" \
            "$@"
    elif [ "$BACKEND" = "singularity" ]; then
        SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpu_spec" \
        singularity exec --nv \
            --bind "${DB_DIR}:/data/public_databases" \
            --bind "${MMSEQS_DB_DIR}:/data/mmseqs_databases" \
            --bind "${WEIGHTS_DIR}:/data/models" \
            --bind "${INPUT_DIR}:/data/af_input" \
            --bind "${OUTPUT_DIR}:/data/af_output" \
            $rna_mount_singularity \
            "$CONTAINER" \
            "$@"
    fi
}

# ---------------------------------------------------------------------------
# Single-GPU mode: two-stage pipeline
# ---------------------------------------------------------------------------
if [ "$NUM_GPUS" -eq 1 ]; then
    GPU_ID="${GPU_DEVICES%%,*}"  # Take first GPU

    PIPELINE_LOG="${LOG_DIR}/pipeline_${TIMESTAMP}.log"
    INFERENCE_LOG="${LOG_DIR}/inference_${TIMESTAMP}.log"

    # Stage 1: Data pipeline (MSA search)
    echo "=== Stage 1: Data Pipeline (MSA search) ==="
    echo "GPU: $GPU_ID"
    echo "Log: $PIPELINE_LOG"
    echo ""

    # Build RNA search flags:
    # 1. If --use_nhmmer is set, force nhmmer mode
    # 2. If --rna_mmseqs_db_dir is explicitly set, use that
    # 3. Auto-detect <db_dir>/mmseqs_rna/ for MMseqs2 nucleotide search
    # 4. Fall back to nhmmer if no MMseqs2 RNA databases found
    RNA_FLAGS=""
    if [ -n "$USE_NHMMER" ]; then
        RNA_FLAGS="--use_nhmmer --nhmmer_binary_path=/hmmer/bin/nhmmer --hmmalign_binary_path=/hmmer/bin/hmmalign --hmmbuild_binary_path=/hmmer/bin/hmmbuild"
    elif [ -n "$RNA_MMSEQS_DB_DIR" ]; then
        RNA_FLAGS="--rna_mmseqs_db_dir=/data/rna_mmseqs_databases"
    elif [ -d "${DB_DIR}/mmseqs_rna" ] && [ -f "${DB_DIR}/mmseqs_rna/rfam.dbtype" ]; then
        echo "Auto-detected RNA MMseqs2 databases at ${DB_DIR}/mmseqs_rna"
        RNA_FLAGS="--rna_mmseqs_db_dir=/data/public_databases/mmseqs_rna"
    else
        RNA_FLAGS="--nhmmer_binary_path=/hmmer/bin/nhmmer --hmmalign_binary_path=/hmmer/bin/hmmalign --hmmbuild_binary_path=/hmmer/bin/hmmbuild"
    fi

    run_container "$GPU_ID" \
        python /app/alphafold/run_data_pipeline.py \
        --input_dir=/data/af_input \
        --output_dir=/data/af_output \
        --db_dir=/data/public_databases \
        --mmseqs_db_dir=/data/mmseqs_databases \
        --use_mmseqs_gpu \
        --batch_size="$BATCH_SIZE" \
        $RNA_FLAGS \
        2>&1 | tee "$PIPELINE_LOG"

    # Stage 2: Inference (loop over data JSONs)
    echo ""
    echo "=== Stage 2: Inference ==="
    echo "Log: $INFERENCE_LOG"
    echo ""

    run_container "$GPU_ID" \
        python /app/alphafold/run_alphafold.py \
        --input_dir=/data/af_output \
        --model_dir=/data/models \
        --norun_data_pipeline \
        --output_dir=/data/af_output \
        --force_output_dir \
        2>&1 | tee "$INFERENCE_LOG"

# ---------------------------------------------------------------------------
# Multi-GPU mode: phase-separated parallel
# ---------------------------------------------------------------------------
else
    echo "=== Multi-GPU: Phase-Separated Parallel ==="
    echo ""

    MSA_OUTPUT_DIR="${OUTPUT_DIR}/msa_output"
    mkdir -p "$MSA_OUTPUT_DIR"

    # Determine RNA MMseqs2 DB path for multi-GPU container.
    # run_multigpu.sh reads RNA_MMSEQS_DB_DIR from environment.
    MULTIGPU_RNA_DB_DIR=""
    if [ -n "$USE_NHMMER" ]; then
        : # No RNA MMseqs2 DB — nhmmer will be used by run_data_pipeline.py via --use_nhmmer
    elif [ -n "$RNA_MMSEQS_DB_DIR" ]; then
        MULTIGPU_RNA_DB_DIR="/data/rna_mmseqs_databases"
    elif [ -d "${DB_DIR}/mmseqs_rna" ] && [ -f "${DB_DIR}/mmseqs_rna/rfam.dbtype" ]; then
        MULTIGPU_RNA_DB_DIR="/data/public_databases/mmseqs_rna"
    fi

    # Optional RNA MMseqs2 mount
    RNA_DOCKER_MOUNT=""
    RNA_SINGULARITY_BIND=""
    if [ -n "$RNA_MMSEQS_DB_DIR" ]; then
        RNA_DOCKER_MOUNT="-v ${RNA_MMSEQS_DB_DIR}:/data/rna_mmseqs_databases:ro"
        RNA_SINGULARITY_BIND="--bind ${RNA_MMSEQS_DB_DIR}:/data/rna_mmseqs_databases:ro"
    fi

    # Inside the container, CUDA_VISIBLE_DEVICES remaps physical device IDs
    # (e.g. 6,7) to logical indices (0,1). Pass logical indices to
    # run_multigpu.sh so map_visible_gpu() works correctly.
    LOGICAL_GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))

    if [ "$BACKEND" = "docker" ]; then
        docker run --rm \
            --user "$(id -u):$(id -g)" \
            --gpus all \
            -e CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
            -e RNA_MMSEQS_DB_DIR="${MULTIGPU_RNA_DB_DIR}" \
            -e USE_NHMMER="${USE_NHMMER}" \
            -v "${DB_DIR}:/data/public_databases" \
            -v "${MMSEQS_DB_DIR}:/data/mmseqs_databases" \
            -v "${WEIGHTS_DIR}:/data/models" \
            -v "${INPUT_DIR}:/data/af_input" \
            -v "${MSA_OUTPUT_DIR}:/data/af_msa_output" \
            -v "${OUTPUT_DIR}:/data/af_output" \
            $RNA_DOCKER_MOUNT \
            "$CONTAINER" \
            bash -lc "cd /app/alphafold && ./scripts/run_multigpu.sh \
                /data/af_input /data/af_msa_output /data/af_output \
                $NUM_GPUS $BATCH_SIZE $LOGICAL_GPU_LIST"
    elif [ "$BACKEND" = "singularity" ]; then
        SINGULARITYENV_CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
        SINGULARITYENV_RNA_MMSEQS_DB_DIR="${MULTIGPU_RNA_DB_DIR}" \
        SINGULARITYENV_USE_NHMMER="${USE_NHMMER}" \
        singularity exec --nv \
            --bind "${DB_DIR}:/data/public_databases" \
            --bind "${MMSEQS_DB_DIR}:/data/mmseqs_databases" \
            --bind "${WEIGHTS_DIR}:/data/models" \
            --bind "${INPUT_DIR}:/data/af_input" \
            --bind "${MSA_OUTPUT_DIR}:/data/af_msa_output" \
            --bind "${OUTPUT_DIR}:/data/af_output" \
            $RNA_SINGULARITY_BIND \
            "$CONTAINER" \
            bash -lc "cd /app/alphafold && ./scripts/run_multigpu.sh \
                /data/af_input /data/af_msa_output /data/af_output \
                $NUM_GPUS $BATCH_SIZE $LOGICAL_GPU_LIST"
    fi
fi

echo ""
echo "=========================================="
echo "AlphaFast Run Complete"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=========================================="
