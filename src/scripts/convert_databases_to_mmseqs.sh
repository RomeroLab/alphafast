#!/bin/bash
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Convert AlphaFold 3 protein databases to MMseqs2 GPU-compatible format
#
# This script converts the FASTA databases used by AlphaFold 3 (for Jackhmmer)
# to MMseqs2 padded databases suitable for GPU-accelerated searches.
#
# Usage:
#   ./convert_databases_to_mmseqs.sh [SOURCE_DIR] [TARGET_DIR]
#
# Arguments:
#   SOURCE_DIR: Directory containing original FASTA databases
#               Default: /opt/alphafold3_data/databases
#   TARGET_DIR: Directory where MMseqs2 databases will be created
#               Default: $SOURCE_DIR/mmseqs
#
# Requirements:
#   - MMseqs2 installed and in PATH (https://github.com/soedinglab/MMseqs2)
#   - ~540 GB free disk space for padded databases
#   - Several hours for conversion (depending on disk speed)
#
# The script will create the following databases:
#   - uniref90_padded    (from uniref90_2022_05.fa)
#   - mgnify_padded      (from mgy_clusters_2022_05.fa)
#   - small_bfd_padded   (from bfd-first_non_consensus_sequences.fasta)
#   - uniprot_padded     (from uniprot_all_2021_04.fa)
#   - pdb_seqres_padded  (from pdb_seqres_2022_09_28.fasta) - for template search

set -euo pipefail

# Default paths
SOURCE_DIR="${1:-/opt/alphafold3_data/databases}"
TARGET_DIR="${2:-$SOURCE_DIR/mmseqs}"

# Check if mmseqs is available
if ! command -v mmseqs &> /dev/null; then
    echo "ERROR: MMseqs2 not found in PATH."
    echo "Install MMseqs2 with GPU support:"
    echo "  wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz"
    echo "  tar xzf mmseqs-linux-gpu.tar.gz"
    echo "  sudo cp mmseqs/bin/mmseqs /usr/local/bin/"
    echo "  # Or: cp mmseqs/bin/mmseqs \$HOME/.local/bin/"
    exit 1
fi

# Print MMseqs2 version
echo "Using MMseqs2 version:"
mmseqs version
echo ""

# Database mapping: target_name -> source_fasta
declare -A DATABASES=(
    ["uniref90"]="uniref90_2022_05.fa"
    ["mgnify"]="mgy_clusters_2022_05.fa"
    ["small_bfd"]="bfd-first_non_consensus_sequences.fasta"
    ["uniprot"]="uniprot_all_2021_04.fa"
    ["pdb_seqres"]="pdb_seqres_2022_09_28.fasta"
)

# Create target directory
echo "Creating target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Track overall progress
total_dbs=${#DATABASES[@]}
current_db=0

echo ""
echo "=========================================="
echo "Database Conversion Summary"
echo "=========================================="
echo "Source directory: $SOURCE_DIR"
echo "Target directory: $TARGET_DIR"
echo "Databases to convert: $total_dbs"
echo "=========================================="
echo ""

# Process each database
for db_name in "${!DATABASES[@]}"; do
    current_db=$((current_db + 1))
    source_fasta="$SOURCE_DIR/${DATABASES[$db_name]}"
    target_base="$TARGET_DIR/${db_name}"
    target_padded="${target_base}_padded"
    
    echo "=========================================="
    echo "[$current_db/$total_dbs] Processing: $db_name"
    echo "=========================================="
    echo "Source: $source_fasta"
    echo "Target: $target_padded"
    echo ""
    
    # Check if source FASTA exists
    if [[ ! -f "$source_fasta" ]]; then
        echo "WARNING: Source FASTA not found: $source_fasta"
        echo "Skipping this database."
        echo ""
        continue
    fi
    
    # Check if padded database already exists
    if [[ -f "${target_padded}.dbtype" ]]; then
        echo "SKIP: Padded database already exists at ${target_padded}"
        echo ""
        continue
    fi
    
    # Check if intermediate database exists
    if [[ -f "${target_base}.dbtype" ]]; then
        echo "Found existing intermediate database at ${target_base}"
        echo "Skipping createdb step..."
    else
        # Step 1: Create standard MMseqs2 database from FASTA
        echo "Step 1/2: Creating MMseqs2 database from FASTA..."
        echo "Command: mmseqs createdb $source_fasta $target_base"
        time mmseqs createdb "$source_fasta" "$target_base"
        echo ""
    fi
    
    # Step 2: Create padded database for GPU acceleration
    echo "Step 2/2: Creating padded database for GPU..."
    echo "Command: mmseqs makepaddedseqdb $target_base $target_padded"
    time mmseqs makepaddedseqdb "$target_base" "$target_padded"
    echo ""
    
    # Verify the padded database was created
    if [[ -f "${target_padded}.dbtype" ]]; then
        echo "SUCCESS: Created padded database at ${target_padded}"
        
        # Show file sizes
        echo "Database files:"
        ls -lh "${target_padded}"* 2>/dev/null || true
    else
        echo "ERROR: Failed to create padded database at ${target_padded}"
    fi
    
    echo ""
done

echo "=========================================="
echo "Database Conversion Complete!"
echo "=========================================="
echo "Padded databases created in: $TARGET_DIR"
echo ""
echo "To use with AlphaFold 3, run:"
echo "  python run_alphafold.py \\"
echo "      --mmseqs_db_dir=$TARGET_DIR \\"
echo "      --json_path=your_input.json \\"
echo "      --output_dir=/path/to/output"
echo "=========================================="
