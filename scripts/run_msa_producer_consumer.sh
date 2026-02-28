#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

set -euo pipefail

usage() {
  echo "Usage: $0 <input_dir> <msa_output_dir> <af_output_dir> <queue_dir> [batch_size] [producer_gpu] [consumer_gpus]"
  echo "  input_dir: directory containing input JSON files"
  echo "  msa_output_dir: output directory for MSA JSONs"
  echo "  af_output_dir: output directory for inference outputs"
  echo "  queue_dir: directory for queue tokens"
  echo "  batch_size: batch size for MSA producer (default: 32)"
  echo "  producer_gpu: GPU index for MSA producer (default: 0)"
  echo "  consumer_gpus: comma-separated GPU indices (default: 1,2,3,4)"
}

if [ "$#" -lt 4 ]; then
  usage
  exit 1
fi

INPUT_DIR="$1"
MSA_OUTPUT_DIR="$2"
AF_OUTPUT_DIR="$3"
QUEUE_DIR="$4"
BATCH_SIZE="${5:-32}"
PRODUCER_GPU="${6:-0}"
CONSUMER_GPUS="${7:-1,2,3,4}"

export DB_DIR="${DB_DIR:-/data/public_databases}"
export MMSEQS_DB_DIR="${MMSEQS_DB_DIR:-/data/mmseqs_databases}"
export MODEL_DIR="${MODEL_DIR:-/data/models}"
export LOG_DIR="${LOG_DIR:-logs}"
POLL_INTERVAL="${POLL_INTERVAL:-2.0}"
IDLE_GRACE_SECONDS="${IDLE_GRACE_SECONDS:-10.0}"
RUN_DATA_PIPELINE_PATH="${RUN_DATA_PIPELINE_PATH:-/app/alphafold/run_data_pipeline.py}"
RUN_ALPHAFOLD_PATH="${RUN_ALPHAFOLD_PATH:-/app/alphafold/run_alphafold.py}"
QUEUE_WORKER_PATH="${QUEUE_WORKER_PATH:-/app/alphafold/scripts/queue_worker.py}"

mkdir -p "$MSA_OUTPUT_DIR" "$AF_OUTPUT_DIR" "$QUEUE_DIR" "$LOG_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PRODUCER_LOG="${LOG_DIR}/producer_${TIMESTAMP}.log"

echo "=========================================="
echo "Producer/consumer run"
echo "Input dir: ${INPUT_DIR}"
echo "MSA output dir: ${MSA_OUTPUT_DIR}"
echo "AF output dir: ${AF_OUTPUT_DIR}"
echo "Queue dir: ${QUEUE_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Producer GPU: ${PRODUCER_GPU}"
echo "Consumer GPUs: ${CONSUMER_GPUS}"
echo "CUDA_VISIBLE_DEVICES (initial): ${CUDA_VISIBLE_DEVICES:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "Visible GPUs inside container:"
  nvidia-smi -L || true
fi
echo "Start time: $(date)"
echo "=========================================="

START_TIME=$(date +%s)

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

PRODUCER_VISIBLE_GPU=$(map_visible_gpu "${PRODUCER_GPU}")

CUDA_VISIBLE_DEVICES="${PRODUCER_VISIBLE_GPU}" \
  python "$RUN_DATA_PIPELINE_PATH" \
  --input_dir="$INPUT_DIR" \
  --output_dir="$MSA_OUTPUT_DIR" \
  --db_dir="$DB_DIR" \
  --mmseqs_db_dir="$MMSEQS_DB_DIR" \
  --use_mmseqs_gpu \
  --mmseqs_n_threads=$(nproc) \
  --gpu_device=0 \
  --batch_size="$BATCH_SIZE" \
  --queue_dir="$QUEUE_DIR" \
  ${TEMP_DIR:+--temp_dir="$TEMP_DIR"} \
  > "$PRODUCER_LOG" 2>&1 &
PRODUCER_PID=$!

IFS=',' read -r -a GPU_LIST <<< "$CONSUMER_GPUS"
WORKER_PIDS=()
WORKER_LOGS=()

JAX_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-}"

for GPU_ID in "${GPU_LIST[@]}"; do
  WORKER_ID="gpu${GPU_ID}"
  WORKER_LOG="${LOG_DIR}/consumer_${WORKER_ID}_${TIMESTAMP}.log"
  WORKER_VISIBLE_GPU=$(map_visible_gpu "${GPU_ID}")
  CUDA_VISIBLE_DEVICES="${WORKER_VISIBLE_GPU}" \
    python "$QUEUE_WORKER_PATH" \
    --queue_dir="$QUEUE_DIR" \
    --output_dir="$AF_OUTPUT_DIR" \
    --worker_id="$WORKER_ID" \
    --model_dir="$MODEL_DIR" \
    --gpu_device=0 \
    --poll_interval="$POLL_INTERVAL" \
    --idle_grace_seconds="$IDLE_GRACE_SECONDS" \
    --force_output_dir \
    ${JAX_CACHE_DIR:+--jax_compilation_cache_dir="$JAX_CACHE_DIR"} \
    > "$WORKER_LOG" 2>&1 &
  WORKER_PIDS+=("$!")
  WORKER_LOGS+=("$WORKER_LOG")
done

set +e
wait "$PRODUCER_PID"
PRODUCER_STATUS=$?
set -e

if [ "$PRODUCER_STATUS" -ne 0 ]; then
  echo "Producer failed with exit code ${PRODUCER_STATUS}" | tee -a "$PRODUCER_LOG"
  for PID in "${WORKER_PIDS[@]}"; do
    kill "$PID" 2>/dev/null || true
  done
  exit "$PRODUCER_STATUS"
fi

WORKER_FAILURES=0
for PID in "${WORKER_PIDS[@]}"; do
  set +e
  wait "$PID"
  STATUS=$?
  set -e
  if [ "$STATUS" -ne 0 ]; then
    WORKER_FAILURES=$((WORKER_FAILURES + 1))
  fi
done

END_TIME=$(date +%s)
TOTAL_SECONDS=$((END_TIME - START_TIME))

DONE_COUNT=$(find "${QUEUE_DIR}/done" -name "*.json" 2>/dev/null | wc -l)
FAILED_COUNT=$(find "${QUEUE_DIR}/failed" -name "*.json" 2>/dev/null | wc -l)
TOTAL_COUNT=$((DONE_COUNT + FAILED_COUNT))
PER_SEQUENCE_SECONDS=0
if [ "$DONE_COUNT" -gt 0 ]; then
  PER_SEQUENCE_SECONDS=$((TOTAL_SECONDS / DONE_COUNT))
fi

SUMMARY_PATH="${QUEUE_DIR}/summary_overall.json"
export QUEUE_DIR SUMMARY_PATH TOTAL_SECONDS DONE_COUNT FAILED_COUNT TOTAL_COUNT
export PER_SEQUENCE_SECONDS TIMESTAMP
python - <<'PY'
import json
import os

queue_dir = os.environ["QUEUE_DIR"]
summary_path = os.environ["SUMMARY_PATH"]
data = {
    "total_wall_seconds": int(os.environ["TOTAL_SECONDS"]),
    "completed": int(os.environ["DONE_COUNT"]),
    "failed": int(os.environ["FAILED_COUNT"]),
    "total_count": int(os.environ["TOTAL_COUNT"]),
    "per_sequence_wall_seconds": int(os.environ["PER_SEQUENCE_SECONDS"]),
    "timestamp": os.environ["TIMESTAMP"],
}
producer_done = os.path.join(queue_dir, "producer_done")
if os.path.exists(producer_done):
    with open(producer_done, "rt") as f:
        data["producer_done"] = json.load(f)
with open(summary_path, "wt") as f:
    json.dump(data, f, indent=2)
PY

echo "=========================================="
echo "Producer/consumer run complete"
echo "Total wall time: ${TOTAL_SECONDS} seconds"
echo "Completed: ${DONE_COUNT}, Failed: ${FAILED_COUNT}"
echo "Per-sequence wall time: ${PER_SEQUENCE_SECONDS} seconds"
echo "Queue summary: ${SUMMARY_PATH}"
echo "Producer log: ${PRODUCER_LOG}"
echo "Worker logs: ${WORKER_LOGS[*]}"
echo "End time: $(date)"
echo "=========================================="

if [ "$WORKER_FAILURES" -ne 0 ]; then
  exit 1
fi
