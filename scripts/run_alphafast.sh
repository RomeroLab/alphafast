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
# Handles single-GPU (two-stage) and multi-GPU (producer-consumer) modes.
#
# Usage:
#   ./scripts/run_alphafast.sh \
#       --input_dir /path/to/inputs \
#       --output_dir /path/to/outputs \
#       --db_dir /path/to/databases \
#       --weights_dir /path/to/weights \
#       [--num_gpus 1] \
#       [--container romerolabduke/alphafast:latest] \
#       [--batch_size auto] \
#       [--gpu_devices 0,1,...] \
#       [--backend docker|singularity]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------
INPUT_DIR=""
OUTPUT_DIR=""
DB_DIR=""
WEIGHTS_DIR=""
NUM_GPUS=1
CONTAINER="romerolabduke/alphafast:latest"
BATCH_SIZE=""
GPU_DEVICES=""
BACKEND=""

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
    echo "  --num_gpus N          Number of GPUs (default: 1)"
    echo "  --container IMAGE     Container image or .sif path"
    echo "                        (default: romerolabduke/alphafast:latest)"
    echo "  --batch_size N        MSA batch size (default: auto = number of inputs)"
    echo "  --gpu_devices IDS     Comma-separated GPU IDs (default: 0 for single,"
    echo "                        0,1,...,N-1 for multi)"
    echo "  --backend TYPE        Force 'docker' or 'singularity' (default: auto-detect)"
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

# Default GPU devices
if [ -z "$GPU_DEVICES" ]; then
    if [ "$NUM_GPUS" -eq 1 ]; then
        GPU_DEVICES="0"
    else
        GPU_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
    fi
fi

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

    if [ "$BACKEND" = "docker" ]; then
        docker run --rm \
            --gpus "device=${gpu_spec}" \
            -v "${DB_DIR}:/root/public_databases" \
            -v "${MMSEQS_DB_DIR}:/root/mmseqs_databases" \
            -v "${WEIGHTS_DIR}:/root/models" \
            -v "${INPUT_DIR}:/root/af_input" \
            -v "${OUTPUT_DIR}:/root/af_output" \
            "$CONTAINER" \
            "$@"
    elif [ "$BACKEND" = "singularity" ]; then
        SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpu_spec" \
        singularity exec --nv \
            --bind "${DB_DIR}:/root/public_databases" \
            --bind "${MMSEQS_DB_DIR}:/root/mmseqs_databases" \
            --bind "${WEIGHTS_DIR}:/root/models" \
            --bind "${INPUT_DIR}:/root/af_input" \
            --bind "${OUTPUT_DIR}:/root/af_output" \
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

    run_container "$GPU_ID" \
        python /app/alphafold/run_data_pipeline.py \
        --input_dir=/root/af_input \
        --output_dir=/root/af_output \
        --db_dir=/root/public_databases \
        --mmseqs_db_dir=/root/mmseqs_databases \
        --use_mmseqs_gpu \
        --batch_size="$BATCH_SIZE" \
        2>&1 | tee "$PIPELINE_LOG"

    # Stage 2: Inference (loop over data JSONs)
    echo ""
    echo "=== Stage 2: Inference ==="
    echo "Log: $INFERENCE_LOG"
    echo ""

    DATA_FILES=$(find "$OUTPUT_DIR" -mindepth 2 -maxdepth 2 -name "*_data.json" -type f | sort)
    TOTAL_FILES=$(echo "$DATA_FILES" | grep -c . || echo 0)

    if [ "$TOTAL_FILES" -eq 0 ]; then
        echo "ERROR: No *_data.json files found in $OUTPUT_DIR"
        exit 1
    fi

    CURRENT=0
    for data_json in $DATA_FILES; do
        CURRENT=$((CURRENT + 1))
        PROTEIN_DIR=$(dirname "$data_json")
        PROTEIN_NAME=$(basename "$PROTEIN_DIR")

        echo "[$CURRENT/$TOTAL_FILES] Inference: ${PROTEIN_NAME}" | tee -a "$INFERENCE_LOG"

        run_container "$GPU_ID" \
            python /app/alphafold/run_alphafold.py \
            --json_path="/root/af_output/${PROTEIN_NAME}/${PROTEIN_NAME}_data.json" \
            --norun_data_pipeline \
            --output_dir="/root/af_output/${PROTEIN_NAME}" \
            --force_output_dir \
            2>&1 | tee -a "$INFERENCE_LOG"
    done

# ---------------------------------------------------------------------------
# Multi-GPU mode: phase-separated parallel
# ---------------------------------------------------------------------------
else
    echo "=== Multi-GPU: Phase-Separated Parallel ==="
    echo ""

    MSA_OUTPUT_DIR="${OUTPUT_DIR}/msa_output"
    mkdir -p "$MSA_OUTPUT_DIR"

    if [ "$BACKEND" = "docker" ]; then
        docker run --rm \
            --gpus all \
            -e CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
            -v "${DB_DIR}:/root/public_databases" \
            -v "${MMSEQS_DB_DIR}:/root/mmseqs_databases" \
            -v "${WEIGHTS_DIR}:/root/models" \
            -v "${INPUT_DIR}:/root/af_input" \
            -v "${MSA_OUTPUT_DIR}:/root/af_msa_output" \
            -v "${OUTPUT_DIR}:/root/af_output" \
            "$CONTAINER" \
            bash -lc "cd /app/alphafold && ./scripts/run_multigpu.sh \
                /root/af_input /root/af_msa_output /root/af_output \
                $NUM_GPUS $BATCH_SIZE $GPU_DEVICES"
    elif [ "$BACKEND" = "singularity" ]; then
        SINGULARITYENV_CUDA_VISIBLE_DEVICES="${GPU_DEVICES}" \
        singularity exec --nv \
            --bind "${DB_DIR}:/root/public_databases" \
            --bind "${MMSEQS_DB_DIR}:/root/mmseqs_databases" \
            --bind "${WEIGHTS_DIR}:/root/models" \
            --bind "${INPUT_DIR}:/root/af_input" \
            --bind "${MSA_OUTPUT_DIR}:/root/af_msa_output" \
            --bind "${OUTPUT_DIR}:/root/af_output" \
            "$CONTAINER" \
            bash -lc "cd /app/alphafold && ./scripts/run_multigpu.sh \
                /root/af_input /root/af_msa_output /root/af_output \
                $NUM_GPUS $BATCH_SIZE $GPU_DEVICES"
    fi
fi

echo ""
echo "=========================================="
echo "AlphaFast Run Complete"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "End time: $(date)"
echo "=========================================="
