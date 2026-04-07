#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Generate a mixed-type benchmark test set (5 categories).
#
# Categories: protein-monomer, protein-ligand, protein-protein, protein-rna, protein-dna
#
# This script runs inside a Docker container with access to the
# alphafold3 module (required for mmCIF parsing).
#
# Usage:
#   # Using Docker (recommended)
#   ./scripts/benchmarks/generate_mixed_test_set.sh \
#       --mmcif_dir /path/to/databases/mmcif_files \
#       --output_dir benchmarks/benchmark_set_mixed_40 \
#       --container romerolabduke/alphafast:latest
#
#   # Direct (requires alphafold3 module in Python path)
#   ./scripts/benchmarks/generate_mixed_test_set.sh \
#       --mmcif_dir /path/to/databases/mmcif_files \
#       --output_dir benchmarks/benchmark_set_mixed_40

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
MMCIF_DIR=""
OUTPUT_DIR="${REPO_DIR}/benchmarks/benchmark_set_mixed_40"
CONTAINER_IMAGE=""
SAMPLES=8
TOTAL_SAMPLES=""
SEED=42
NUM_WORKERS=8

usage() {
    echo "Usage: $0 --mmcif_dir DIR [OPTIONS]"
    echo ""
    echo "Generate a mixed-type benchmark test set (5 categories)."
    echo ""
    echo "Required:"
    echo "  --mmcif_dir DIR       Path to PDB mmCIF files directory"
    echo ""
    echo "Optional:"
    echo "  --output_dir DIR      Output directory (default: benchmarks/benchmark_set_mixed_40)"
    echo "  --container IMAGE     Run inside Docker container (recommended)"
    echo "  --samples N           Samples per category (default: 8)"
    echo "  --total_samples N     Total samples distributed across 5 categories"
    echo "                        (mutually exclusive with --samples)"
    echo "  --seed N              Random seed (default: 42)"
    echo "  --num_workers N       Parallel workers for scanning (default: 8)"
    exit 1
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --mmcif_dir)      MMCIF_DIR="$2"; shift 2 ;;
        --output_dir)     OUTPUT_DIR="$2"; shift 2 ;;
        --container)      CONTAINER_IMAGE="$2"; shift 2 ;;
        --samples)        SAMPLES="$2"; shift 2 ;;
        --total_samples)  TOTAL_SAMPLES="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --num_workers)    NUM_WORKERS="$2"; shift 2 ;;
        --help|-h)        usage ;;
        *)                echo "Unknown argument: $1"; usage ;;
    esac
done

if [ -z "$MMCIF_DIR" ]; then
    echo "ERROR: --mmcif_dir is required."
    usage
fi

# Build sample count args
SAMPLE_ARGS=""
if [ -n "$TOTAL_SAMPLES" ]; then
    SAMPLE_ARGS="--total_samples=$TOTAL_SAMPLES"
    DISPLAY_SAMPLES="${TOTAL_SAMPLES} total"
else
    SAMPLE_ARGS="--samples_per_category=$SAMPLES"
    DISPLAY_SAMPLES="${SAMPLES} per category ($(( SAMPLES * 5 )) total)"
fi

echo "=========================================="
echo "Mixed-Type Benchmark Test Set Generator"
echo "=========================================="
echo "mmCIF dir:  $MMCIF_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Samples:    $DISPLAY_SAMPLES"
echo "Seed:       $SEED"
echo "Workers:    $NUM_WORKERS"
echo "=========================================="
echo ""

mkdir -p "$OUTPUT_DIR"

COMMON_ARGS="--cutoff_date=2021-09-30 --max_resolution=3.0 --min_seq_length=50 --max_seq_length=500 --max_total_residues=1500 --num_workers=$NUM_WORKERS --seed=$SEED"

if [ -n "$CONTAINER_IMAGE" ]; then
    docker run --rm \
        -v "${MMCIF_DIR}:/data/mmcif_files:ro" \
        -v "${OUTPUT_DIR}:/data/output" \
        -v "${REPO_DIR}/benchmarks:/app/alphafold/benchmarks" \
        "$CONTAINER_IMAGE" \
        python /app/alphafold/benchmarks/create_mixed_benchmark.py \
            --mmcif_dir /data/mmcif_files \
            --output_dir /data/output \
            $SAMPLE_ARGS \
            $COMMON_ARGS
else
    python "${REPO_DIR}/benchmarks/create_mixed_benchmark.py" \
        --mmcif_dir "$MMCIF_DIR" \
        --output_dir "$OUTPUT_DIR" \
        $SAMPLE_ARGS \
        $COMMON_ARGS
fi

# Count results
TOTAL_COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "*_input.json" 2>/dev/null | wc -l | tr -d ' ')

echo ""
echo "=========================================="
echo "Benchmark test set generated!"
echo "Output: $OUTPUT_DIR"
echo "Total:  ${TOTAL_COUNT} files"
echo ""
echo "Category breakdown:"
for cat in protein_monomer protein_ligand protein_protein protein_rna protein_dna; do
    COUNT=$(find "$OUTPUT_DIR" -maxdepth 1 -name "${cat}_*_input.json" 2>/dev/null | wc -l | tr -d ' ')
    echo "  ${cat}: ${COUNT}"
done
echo ""
echo "To run predictions:"
echo "  ./scripts/run_alphafast.sh \\"
echo "      --input_dir $OUTPUT_DIR \\"
echo "      --output_dir ./output \\"
echo "      --db_dir /path/to/databases \\"
echo "      --weights_dir /path/to/weights"
echo "=========================================="
