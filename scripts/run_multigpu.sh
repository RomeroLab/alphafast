#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Phase-separated parallel multi-GPU pipeline.
#
# All N GPUs run MSA in parallel (Phase 1), then all N GPUs run fold in
# parallel (Phase 2). This provides clean timing separation for benchmarking
# while achieving near-perfect linear scaling.
#
# Usage (inside container):
#   scripts/run_multigpu.sh <input_dir> <msa_output_dir> <af_output_dir> \
#       <num_gpus> [batch_size] [gpu_list] [mmseqs_threads]
#
# Arguments:
#   input_dir       - Directory containing input JSON files
#   msa_output_dir  - Output directory for MSA JSONs
#   af_output_dir   - Output directory for inference outputs
#   num_gpus        - Number of GPUs to use
#   batch_size      - Batch size per GPU for MSA (default: 512)
#   gpu_list        - Comma-separated GPU indices (default: 0,1,...,N-1)
#   mmseqs_threads  - CPU threads per GPU (default: total_cores / num_gpus)

set -euo pipefail

usage() {
  echo "Usage: $0 <input_dir> <msa_output_dir> <af_output_dir> <num_gpus> [batch_size] [gpu_list] [mmseqs_threads]"
  echo "  input_dir:       directory containing input JSON files"
  echo "  msa_output_dir:  output directory for MSA JSONs"
  echo "  af_output_dir:   output directory for inference outputs"
  echo "  num_gpus:        number of GPUs to use"
  echo "  batch_size:      batch size per GPU for MSA (default: partition size)"
  echo "  gpu_list:        comma-separated GPU indices (default: 0,1,...,N-1)"
  echo "  mmseqs_threads:  CPU threads per GPU (default: total_cores / num_gpus)"
}

if [ "$#" -lt 4 ]; then
  usage
  exit 1
fi

INPUT_DIR="$1"
MSA_OUTPUT_DIR="$2"
AF_OUTPUT_DIR="$3"
NUM_GPUS="$4"
BATCH_SIZE="${5:-}"
GPU_LIST="${6:-}"
MMSEQS_THREADS="${7:-}"

export DB_DIR="${DB_DIR:-/data/public_databases}"
export MMSEQS_DB_DIR="${MMSEQS_DB_DIR:-/data/mmseqs_databases}"
export MODEL_DIR="${MODEL_DIR:-/data/models}"
export LOG_DIR="${LOG_DIR:-logs}"
RUN_DATA_PIPELINE_PATH="${RUN_DATA_PIPELINE_PATH:-/app/alphafold/run_data_pipeline.py}"
RUN_ALPHAFOLD_PATH="${RUN_ALPHAFOLD_PATH:-/app/alphafold/run_alphafold.py}"

# Default GPU list: 0,1,...,N-1
if [ -z "$GPU_LIST" ]; then
  GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
fi

# Default threads: total cores / num_gpus
TOTAL_CORES=$(nproc 2>/dev/null || echo 128)
if [ -z "$MMSEQS_THREADS" ]; then
  MMSEQS_THREADS=$((TOTAL_CORES / NUM_GPUS))
fi

mkdir -p "$MSA_OUTPUT_DIR" "$AF_OUTPUT_DIR" "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── GPU mapping ───────────────────────────────────────────────────────────
map_visible_gpu() {
  local requested_index="$1"
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a _VISIBLE_LIST <<< "${CUDA_VISIBLE_DEVICES}"
    if [ "$requested_index" -ge "${#_VISIBLE_LIST[@]}" ]; then
      echo "ERROR: Requested GPU index ${requested_index} but only ${#_VISIBLE_LIST[@]} visible GPUs: ${CUDA_VISIBLE_DEVICES}" >&2
      exit 1
    fi
    echo "${_VISIBLE_LIST[$requested_index]}"
  else
    echo "${requested_index}"
  fi
}

# ── Partition inputs ──────────────────────────────────────────────────────
INPUT_FILES=($(ls "$INPUT_DIR"/*.json 2>/dev/null | grep -v '/index\.json$' | sort))
TOTAL_INPUTS=${#INPUT_FILES[@]}

if [ "$TOTAL_INPUTS" -eq 0 ]; then
  echo "ERROR: No .json files found in $INPUT_DIR"
  exit 1
fi

echo "=========================================="
echo "Multi-GPU Parallel Pipeline"
echo "Input dir: ${INPUT_DIR} (${TOTAL_INPUTS} files)"
echo "MSA output dir: ${MSA_OUTPUT_DIR}"
echo "AF output dir: ${AF_OUTPUT_DIR}"
echo "Num GPUs: ${NUM_GPUS}"
echo "GPU list: ${GPU_LIST}"
echo "Threads per GPU: ${MMSEQS_THREADS}"
echo "CUDA_VISIBLE_DEVICES (initial): ${CUDA_VISIBLE_DEVICES:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Visible GPUs inside container:"
  nvidia-smi -L || true
fi
echo "Start time: $(date)"
echo "=========================================="

START_TIME=$(date +%s)

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"

# Create per-GPU partition directories with symlinks (round-robin)
for ((i=0; i<NUM_GPUS; i++)); do
  mkdir -p "${MSA_OUTPUT_DIR}/partition_${i}"
  mkdir -p "${INPUT_DIR}/.partition_${i}"
done

for ((j=0; j<TOTAL_INPUTS; j++)); do
  GPU_IDX=$((j % NUM_GPUS))
  BASENAME=$(basename "${INPUT_FILES[$j]}")
  ln -sf "$(realpath "${INPUT_FILES[$j]}")" "${INPUT_DIR}/.partition_${GPU_IDX}/${BASENAME}"
done

# Report partition sizes
for ((i=0; i<NUM_GPUS; i++)); do
  PART_COUNT=$(ls "${INPUT_DIR}/.partition_${i}"/*.json 2>/dev/null | wc -l | tr -d ' ')
  echo "  GPU ${GPU_ARRAY[$i]}: ${PART_COUNT} inputs"
done

# ══════════════════════════════════════════════════════════════════════════
# Phase 1: Parallel MSA
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Phase 1: Parallel MSA (${NUM_GPUS} GPUs) ==="
MSA_START=$(date +%s)

MSA_PIDS=()
MSA_LOGS=()
for ((i=0; i<NUM_GPUS; i++)); do
  GPU_IDX="${GPU_ARRAY[$i]}"
  VISIBLE_GPU=$(map_visible_gpu "${GPU_IDX}")
  PARTITION_DIR="${INPUT_DIR}/.partition_${i}"
  MSA_OUT="${MSA_OUTPUT_DIR}/partition_${i}"
  MSA_LOG="${LOG_DIR}/msa_gpu${GPU_IDX}_${TIMESTAMP}.log"

  PART_COUNT=$(ls "${PARTITION_DIR}"/*.json 2>/dev/null | wc -l | tr -d ' ')
  GPU_BATCH_SIZE="${BATCH_SIZE:-512}"

  echo "  Starting MSA on GPU ${GPU_IDX} (${PART_COUNT} inputs, batch_size=${GPU_BATCH_SIZE})"

  CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}" \
    python "$RUN_DATA_PIPELINE_PATH" \
    --input_dir="$PARTITION_DIR" \
    --output_dir="$MSA_OUT" \
    --db_dir="$DB_DIR" \
    --mmseqs_db_dir="$MMSEQS_DB_DIR" \
    --use_mmseqs_gpu \
    --mmseqs_n_threads="$MMSEQS_THREADS" \
    --batch_size="$GPU_BATCH_SIZE" \
    ${TEMP_DIR:+--temp_dir="$TEMP_DIR"} \
    > "$MSA_LOG" 2>&1 &
  MSA_PIDS+=("$!")
  MSA_LOGS+=("$MSA_LOG")
done

# Wait for all MSA processes
set +e
MSA_FAILURES=0
MSA_EXIT_CODES=()
for ((i=0; i<NUM_GPUS; i++)); do
  wait "${MSA_PIDS[$i]}"
  STATUS=$?
  MSA_EXIT_CODES+=("$STATUS")
  if [ "$STATUS" -ne 0 ]; then
    MSA_FAILURES=$((MSA_FAILURES + 1))
    echo "  WARNING: MSA on GPU ${GPU_ARRAY[$i]} failed with exit code ${STATUS}"
    echo "  Log: ${MSA_LOGS[$i]}"
  fi
done
set -e

MSA_END=$(date +%s)
MSA_ELAPSED=$((MSA_END - MSA_START))
echo ""
echo "Phase 1 complete: ${MSA_ELAPSED} seconds (${MSA_FAILURES} failures)"

# Clean up partition symlink directories
for ((i=0; i<NUM_GPUS; i++)); do
  rm -rf "${INPUT_DIR}/.partition_${i}"
done

# ── Collect MSA outputs ───────────────────────────────────────────────────
# Find all *_data.json files across all partitions
ALL_DATA_FILES=()
for ((i=0; i<NUM_GPUS; i++)); do
  MSA_OUT="${MSA_OUTPUT_DIR}/partition_${i}"
  while IFS= read -r -d '' f; do
    ALL_DATA_FILES+=("$f")
  done < <(find "$MSA_OUT" -mindepth 2 -maxdepth 2 -name "*_data.json" -type f -print0 2>/dev/null)
done

MSA_SUCCESS_COUNT=${#ALL_DATA_FILES[@]}
MSA_FAILED_COUNT=$((TOTAL_INPUTS - MSA_SUCCESS_COUNT))

echo ""
echo "MSA results: ${MSA_SUCCESS_COUNT} succeeded, ${MSA_FAILED_COUNT} failed out of ${TOTAL_INPUTS}"

if [ "$MSA_SUCCESS_COUNT" -eq 0 ]; then
  echo "ERROR: No MSA outputs produced. Check logs:"
  for log in "${MSA_LOGS[@]}"; do
    echo "  $log"
  done
  exit 1
fi

# ══════════════════════════════════════════════════════════════════════════
# Phase 2: Parallel Fold
# ══════════════════════════════════════════════════════════════════════════
echo ""
echo "=== Phase 2: Parallel Fold (${NUM_GPUS} GPUs) ==="

# Create flat symlink directories for fold — re-partition found data JSONs
for ((i=0; i<NUM_GPUS; i++)); do
  mkdir -p "${MSA_OUTPUT_DIR}/.fold_partition_${i}"
done

# Sort for deterministic assignment
IFS=$'\n' SORTED_DATA_FILES=($(printf '%s\n' "${ALL_DATA_FILES[@]}" | sort)); unset IFS

for ((j=0; j<${#SORTED_DATA_FILES[@]}; j++)); do
  GPU_IDX=$((j % NUM_GPUS))
  DATA_JSON="${SORTED_DATA_FILES[$j]}"
  BASENAME=$(basename "$DATA_JSON")
  ln -sf "$(realpath "$DATA_JSON")" "${MSA_OUTPUT_DIR}/.fold_partition_${GPU_IDX}/${BASENAME}"
done

# Report fold partition sizes
for ((i=0; i<NUM_GPUS; i++)); do
  FOLD_COUNT=$(ls "${MSA_OUTPUT_DIR}/.fold_partition_${i}"/*.json 2>/dev/null | wc -l | tr -d ' ')
  echo "  GPU ${GPU_ARRAY[$i]}: ${FOLD_COUNT} proteins"
done

FOLD_START=$(date +%s)

JAX_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-}"

FOLD_PIDS=()
FOLD_LOGS=()
for ((i=0; i<NUM_GPUS; i++)); do
  GPU_IDX="${GPU_ARRAY[$i]}"
  VISIBLE_GPU=$(map_visible_gpu "${GPU_IDX}")
  FOLD_PARTITION="${MSA_OUTPUT_DIR}/.fold_partition_${i}"
  FOLD_LOG="${LOG_DIR}/fold_gpu${GPU_IDX}_${TIMESTAMP}.log"

  echo "  Starting fold on GPU ${GPU_IDX}"

  CUDA_VISIBLE_DEVICES="${VISIBLE_GPU}" \
    python "$RUN_ALPHAFOLD_PATH" \
    --input_dir="$FOLD_PARTITION" \
    --norun_data_pipeline \
    --output_dir="$AF_OUTPUT_DIR" \
    --gpu_device=0 \
    --force_output_dir \
    --write_timing_json \
    ${JAX_CACHE_DIR:+--jax_compilation_cache_dir="$JAX_CACHE_DIR"} \
    > "$FOLD_LOG" 2>&1 &
  FOLD_PIDS+=("$!")
  FOLD_LOGS+=("$FOLD_LOG")
done

# Wait for all fold processes
set +e
FOLD_FAILURES=0
FOLD_EXIT_CODES=()
for ((i=0; i<NUM_GPUS; i++)); do
  wait "${FOLD_PIDS[$i]}"
  STATUS=$?
  FOLD_EXIT_CODES+=("$STATUS")
  if [ "$STATUS" -ne 0 ]; then
    FOLD_FAILURES=$((FOLD_FAILURES + 1))
    echo "  WARNING: Fold on GPU ${GPU_ARRAY[$i]} failed with exit code ${STATUS}"
    echo "  Log: ${FOLD_LOGS[$i]}"
  fi
done
set -e

FOLD_END=$(date +%s)
FOLD_ELAPSED=$((FOLD_END - FOLD_START))
echo ""
echo "Phase 2 complete: ${FOLD_ELAPSED} seconds (${FOLD_FAILURES} failures)"

# Clean up fold partition symlink directories
for ((i=0; i<NUM_GPUS; i++)); do
  rm -rf "${MSA_OUTPUT_DIR}/.fold_partition_${i}"
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

# ══════════════════════════════════════════════════════════════════════════
# Merge timing
# ══════════════════════════════════════════════════════════════════════════
TIMING_FILE="${AF_OUTPUT_DIR}/multigpu_timing.json"

# Count successful fold outputs
FOLD_SUCCESS_COUNT=$(find "$AF_OUTPUT_DIR" -maxdepth 2 -name "*_model.cif" -type f 2>/dev/null | wc -l | tr -d ' ')

# Write lightweight summary JSON
cat > "$TIMING_FILE" <<EOF
{
  "num_gpus": ${NUM_GPUS},
  "total_inputs": ${TOTAL_INPUTS},
  "successful_msa": ${MSA_SUCCESS_COUNT},
  "successful_fold": ${FOLD_SUCCESS_COUNT},
  "msa_wall_seconds": ${MSA_ELAPSED},
  "fold_wall_seconds": ${FOLD_ELAPSED},
  "total_wall_seconds": ${TOTAL_SECONDS}
}
EOF
echo "Timing JSON written to ${TIMING_FILE}"

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Multi-GPU parallel pipeline complete"
echo "Total wall time: ${TOTAL_SECONDS} seconds"
echo "  MSA phase:  ${MSA_ELAPSED}s"
echo "  Fold phase: ${FOLD_ELAPSED}s"
echo "MSA: ${MSA_SUCCESS_COUNT} succeeded, ${MSA_FAILED_COUNT} failed"
echo "Timing JSON: ${TIMING_FILE}"
echo "MSA logs: ${MSA_LOGS[*]}"
echo "Fold logs: ${FOLD_LOGS[*]}"
echo "End time: $(date)"
echo "=========================================="

if [ "$MSA_FAILURES" -ne 0 ] || [ "$FOLD_FAILURES" -ne 0 ]; then
  exit 1
fi
