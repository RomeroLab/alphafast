#!/bin/bash
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Set up Foldseek and download AlphaFold Database for structural template search
#
# This script:
# 1. Downloads and installs Foldseek (if not already installed)
# 2. Downloads the AlphaFold Database (AFDB) in Foldseek format
#
# Usage:
#   ./setup_foldseek_afdb.sh [TARGET_DIR] [DATABASE_SIZE]
#
# Arguments:
#   TARGET_DIR:     Directory where Foldseek and AFDB will be set up
#                   Default: $HOME/foldseek_afdb
#   DATABASE_SIZE:  Size of AFDB to download (swissport, uniprot50, or full)
#                   Default: uniprot50 (recommended, ~100GB)
#
# Database Options:
#   - swissprot:  SwissProt subset only (~15GB, fastest setup, limited coverage)
#   - uniprot50:  UniProt50 clustered (~100GB, recommended balance)
#   - full:       Full AFDB (~2.5TB, maximum coverage, large storage needed)
#
# Requirements:
#   - wget or curl for downloading
#   - ~100GB free disk space (for uniprot50), more for full AFDB
#   - Internet connection for database download

set -euo pipefail

# Default configuration
TARGET_DIR="${1:-$HOME/foldseek_afdb}"
DATABASE_SIZE="${2:-uniprot50}"

# Foldseek version and download URLs
FOLDSEEK_VERSION="9-427df8a"
FOLDSEEK_LINUX_URL="https://mmseqs.com/foldseek/foldseek-linux-avx2.tar.gz"
FOLDSEEK_LINUX_GPU_URL="https://mmseqs.com/foldseek/foldseek-linux-gpu.tar.gz"

# Database names for foldseek databases command
declare -A AFDB_DATABASES=(
    ["swissprot"]="Alphafold/Swiss-Prot"
    ["uniprot50"]="Alphafold/UniProt50"
    ["full"]="Alphafold/UniProt"
)

# Estimated sizes
declare -A AFDB_SIZES=(
    ["swissprot"]="~15GB"
    ["uniprot50"]="~100GB"
    ["full"]="~2.5TB"
)

echo "=========================================="
echo "Foldseek + AFDB Setup Script"
echo "=========================================="
echo "Target directory: $TARGET_DIR"
echo "Database size: $DATABASE_SIZE (${AFDB_SIZES[$DATABASE_SIZE]:-unknown})"
echo "=========================================="
echo ""

# Validate database size option
if [[ ! ${AFDB_DATABASES[$DATABASE_SIZE]+_} ]]; then
    echo "ERROR: Invalid DATABASE_SIZE: $DATABASE_SIZE"
    echo "Valid options: swissprot, uniprot50, full"
    exit 1
fi

# Create target directory
echo "Creating target directory: $TARGET_DIR"
mkdir -p "$TARGET_DIR"

# Check if Foldseek is already installed
FOLDSEEK_BIN="$TARGET_DIR/foldseek/bin/foldseek"
SYSTEM_FOLDSEEK=$(command -v foldseek 2>/dev/null || true)

if [[ -x "$FOLDSEEK_BIN" ]]; then
    echo "Found Foldseek at: $FOLDSEEK_BIN"
    FOLDSEEK="$FOLDSEEK_BIN"
elif [[ -n "$SYSTEM_FOLDSEEK" ]]; then
    echo "Found system Foldseek at: $SYSTEM_FOLDSEEK"
    FOLDSEEK="$SYSTEM_FOLDSEEK"
else
    echo "=========================================="
    echo "Step 1: Installing Foldseek"
    echo "=========================================="

    # Detect if we have a GPU
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU detected. Downloading GPU-enabled Foldseek..."
        DOWNLOAD_URL="$FOLDSEEK_LINUX_GPU_URL"
    else
        echo "No GPU detected. Downloading CPU-only Foldseek..."
        DOWNLOAD_URL="$FOLDSEEK_LINUX_URL"
    fi

    cd "$TARGET_DIR"
    echo "Downloading from: $DOWNLOAD_URL"

    if command -v wget &> /dev/null; then
        wget -q --show-progress "$DOWNLOAD_URL" -O foldseek.tar.gz
    elif command -v curl &> /dev/null; then
        curl -L --progress-bar "$DOWNLOAD_URL" -o foldseek.tar.gz
    else
        echo "ERROR: Neither wget nor curl found. Please install one of them."
        exit 1
    fi

    echo "Extracting Foldseek..."
    tar xzf foldseek.tar.gz
    rm foldseek.tar.gz

    FOLDSEEK="$TARGET_DIR/foldseek/bin/foldseek"

    if [[ -x "$FOLDSEEK" ]]; then
        echo "Foldseek installed successfully at: $FOLDSEEK"
    else
        echo "ERROR: Foldseek installation failed."
        exit 1
    fi

    echo ""
fi

# Print Foldseek version
echo "Using Foldseek:"
"$FOLDSEEK" version
echo ""

# Download AFDB database
echo "=========================================="
echo "Step 2: Downloading AlphaFold Database"
echo "=========================================="
echo "Database: ${AFDB_DATABASES[$DATABASE_SIZE]}"
echo "Estimated size: ${AFDB_SIZES[$DATABASE_SIZE]}"
echo ""

AFDB_PATH="$TARGET_DIR/afdb_$DATABASE_SIZE"
TMP_DIR="$TARGET_DIR/tmp"

# Check if database already exists
if [[ -f "${AFDB_PATH}.dbtype" ]] || [[ -f "${AFDB_PATH}_ss.dbtype" ]]; then
    echo "SKIP: AFDB database already exists at: $AFDB_PATH"
    echo ""
else
    echo "This may take a while depending on your internet connection..."
    echo "Database will be downloaded to: $AFDB_PATH"
    echo ""

    mkdir -p "$TMP_DIR"

    # Use foldseek databases command to download
    echo "Command: foldseek databases ${AFDB_DATABASES[$DATABASE_SIZE]} $AFDB_PATH $TMP_DIR"
    "$FOLDSEEK" databases "${AFDB_DATABASES[$DATABASE_SIZE]}" "$AFDB_PATH" "$TMP_DIR"

    echo ""
    echo "Database downloaded successfully!"
fi

# Verify database
echo "=========================================="
echo "Verifying Database"
echo "=========================================="

# List database files
echo "Database files:"
ls -lh "${AFDB_PATH}"* 2>/dev/null | head -20 || echo "Database files not found at expected location."
echo ""

# Clean up tmp directory
if [[ -d "$TMP_DIR" ]]; then
    echo "Cleaning up temporary files..."
    rm -rf "$TMP_DIR"
fi

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Foldseek binary: $FOLDSEEK"
echo "AFDB database: $AFDB_PATH"
echo ""
echo "To use with AlphaFold 3, add the following flags:"
echo ""
echo "  python run_alphafold.py \\"
echo "      --use_foldseek_templates \\"
echo "      --foldseek_binary_path=$FOLDSEEK \\"
echo "      --foldseek_database_path=$AFDB_PATH \\"
echo "      --json_path=your_input.json \\"
echo "      --output_dir=/path/to/output"
echo ""
echo "Optional: Add Foldseek to your PATH:"
echo "  export PATH=\"$TARGET_DIR/foldseek/bin:\$PATH\""
echo ""
echo "For 'replace' mode (use only Foldseek templates):"
echo "  --foldseek_mode=replace"
echo ""
echo "=========================================="
