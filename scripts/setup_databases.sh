#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Download AlphaFold 3 databases and convert protein DBs to MMseqs2 GPU format.
#
# This is the single setup script for AlphaFast. It downloads all required
# databases from Google Cloud Storage: protein FASTA files are converted to
# MMseqs2 padded databases for GPU-accelerated MSA search, and RNA FASTA files
# are downloaded as-is for nhmmer-based RNA MSA search (HMMER suite).
#
# Usage:
#   ./scripts/setup_databases.sh <target_dir> [--keep-fasta]
#
# Arguments:
#   target_dir:    Directory where databases will be downloaded and converted
#   --keep-fasta:  Keep raw FASTA files after MMseqs2 conversion (default: keep)
#                  Use --no-keep-fasta to remove them after conversion
#
# Requirements:
#   - wget, zstd, tar in PATH
#   - mmseqs (GPU version) in PATH
#   - ~800 GB free disk space (250 GB download + 540 GB MMseqs2 padded)
#   - (optional) nhmmer, hmmalign, hmmbuild for RNA MSA search
#     Install: conda install -c bioconda hmmer
#
# Output directory structure:
#   <target_dir>/
#     mmcif_files/              # PDB structures for template retrieval
#     uniref90_2022_05.fa       # Raw protein FASTA (if kept)
#     mgy_clusters_2022_05.fa
#     ...
#     rnacentral_active_seq_id_90_cov_80_linclust.fasta   # RNA (nhmmer)
#     rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta      # RNA (nhmmer)
#     nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta  # RNA (nhmmer)
#     mmseqs/
#       uniref90_padded*        # MMseqs2 GPU-ready databases (protein)
#       mgnify_padded*
#       small_bfd_padded*
#       uniprot_padded*
#       pdb_seqres_padded*
#     mmseqs_rna/
#       rfam*                   # MMseqs2 nucleotide databases (optional)
#       rnacentral*
#       nt_rna*

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 <target_dir> [--keep-fasta | --no-keep-fasta]"
    echo ""
    echo "Downloads AlphaFold 3 databases and converts protein DBs to MMseqs2 GPU format."
    echo ""
    echo "Arguments:"
    echo "  target_dir       Directory where databases will be stored"
    echo "  --keep-fasta     Keep raw FASTA files after conversion (default)"
    echo "  --no-keep-fasta  Remove raw FASTA files after conversion"
    exit 1
}

if [ "$#" -lt 1 ]; then
    usage
fi

TARGET_DIR="$1"
shift

KEEP_FASTA=true
while [ "$#" -gt 0 ]; do
    case "$1" in
        --keep-fasta)    KEEP_FASTA=true; shift ;;
        --no-keep-fasta) KEEP_FASTA=false; shift ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------
MISSING=0
for cmd in wget tar zstd mmseqs; do
    if ! command -v "$cmd" &> /dev/null; then
        echo "ERROR: $cmd is not installed or not in PATH."
        MISSING=1
    fi
done
if [ "$MISSING" -ne 0 ]; then
    echo ""
    echo "Install missing dependencies before running this script."
    echo "For mmseqs with GPU support:"
    echo "  wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz"
    echo "  tar xzf mmseqs-linux-gpu.tar.gz"
    echo "  sudo cp mmseqs/bin/mmseqs /usr/local/bin/"
    exit 1
fi

echo "Using MMseqs2 version: $(mmseqs version)"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
readonly SOURCE="https://storage.googleapis.com/alphafold-databases/v3.0"
MMSEQS_DIR="${TARGET_DIR}/mmseqs"

mkdir -p "$TARGET_DIR" "$MMSEQS_DIR"

echo "=========================================="
echo "AlphaFast Database Setup"
echo "=========================================="
echo "Target directory: $TARGET_DIR"
echo "MMseqs2 directory: $MMSEQS_DIR"
echo "Keep FASTA files: $KEEP_FASTA"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# ---------------------------------------------------------------------------
# Step 1: Download databases
# ---------------------------------------------------------------------------
echo "=== Step 1: Download databases ==="
echo ""

# PDB mmCIF structures (for template retrieval)
MMCIF_DIR="${TARGET_DIR}/mmcif_files"
if [ -d "$MMCIF_DIR" ] && [ "$(ls -A "$MMCIF_DIR" 2>/dev/null)" ]; then
    echo "SKIP: mmcif_files already exists at $MMCIF_DIR"
else
    echo "Downloading PDB mmCIF structures..."
    wget --progress=bar:force:noscroll -O - \
        "${SOURCE}/pdb_2022_09_28_mmcif_files.tar.zst" | \
        tar --no-same-owner --no-same-permissions \
        --use-compress-program=zstd -xf - --directory="$TARGET_DIR"
    echo "Done: mmCIF structures"
fi
echo ""

# Protein FASTA databases (for MSA search)
declare -A PROTEIN_FASTAS=(
    ["uniref90_2022_05.fa"]="UniRef90"
    ["mgy_clusters_2022_05.fa"]="MGnify"
    ["bfd-first_non_consensus_sequences.fasta"]="Small BFD"
    ["uniprot_all_2021_04.fa"]="UniProt"
    ["pdb_seqres_2022_09_28.fasta"]="PDB SeqRes"
)

for fasta_file in "${!PROTEIN_FASTAS[@]}"; do
    db_label="${PROTEIN_FASTAS[$fasta_file]}"
    target_path="${TARGET_DIR}/${fasta_file}"

    if [ -f "$target_path" ]; then
        echo "SKIP: $db_label already exists at $target_path"
    else
        echo "Downloading $db_label ($fasta_file)..."
        wget --progress=bar:force:noscroll -O - \
            "${SOURCE}/${fasta_file}.zst" | \
            zstd --decompress > "$target_path"
        echo "Done: $db_label"
    fi
done
echo ""

# RNA databases (for RNA MSA search via nhmmer - not converted to MMseqs2).
# RNA MSA search requires HMMER suite (nhmmer, hmmalign, hmmbuild).
# Install: conda install -c bioconda hmmer
RNA_FASTAS=(
    "rnacentral_active_seq_id_90_cov_80_linclust.fasta"
    "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"
    "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"
)

for fasta_file in "${RNA_FASTAS[@]}"; do
    target_path="${TARGET_DIR}/${fasta_file}"
    if [ -f "$target_path" ]; then
        echo "SKIP: RNA database already exists at $target_path"
    else
        echo "Downloading RNA database ($fasta_file)..."
        wget --progress=bar:force:noscroll -O - \
            "${SOURCE}/${fasta_file}.zst" | \
            zstd --decompress > "$target_path"
        echo "Done: $fasta_file"
    fi
done
echo ""

# ---------------------------------------------------------------------------
# Step 1b: Build MMseqs2 databases from RNA FASTA (for nucleotide search)
# ---------------------------------------------------------------------------
# These are optional — only needed if using --rna_mmseqs_db_dir instead of
# nhmmer. MMseqs2 nucleotide search (--search-type 3, CPU-only) is faster
# than nhmmer but does not support profile-based iterative search.
# DNA chains do not require MSA search (AlphaFold 3 leaves them blank).
echo "=== Step 1b: Build MMseqs2 RNA databases (optional) ==="
echo ""

RNA_MMSEQS_DIR="${TARGET_DIR}/mmseqs_rna"
mkdir -p "$RNA_MMSEQS_DIR"

declare -A RNA_MMSEQS_DATABASES=(
    ["rfam"]="rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"
    ["rnacentral"]="rnacentral_active_seq_id_90_cov_80_linclust.fasta"
    ["nt_rna"]="nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"
)

for db_name in "${!RNA_MMSEQS_DATABASES[@]}"; do
    source_fasta="${TARGET_DIR}/${RNA_MMSEQS_DATABASES[$db_name]}"
    target_db="${RNA_MMSEQS_DIR}/${db_name}"

    if [ -f "${target_db}.dbtype" ]; then
        echo "SKIP: MMseqs2 RNA database ${db_name} already exists"
        continue
    fi

    if [ ! -f "$source_fasta" ]; then
        echo "SKIP: Source FASTA not found: $source_fasta"
        continue
    fi

    echo "Creating MMseqs2 nucleotide database: $db_name"
    time mmseqs createdb "$source_fasta" "$target_db"

    # Build k-mer index for fast search. Without this, MMseqs2 rebuilds
    # the index on every query, which is very slow for large databases.
    # Use --split 4 for databases >10GB to avoid OOM (~372GB peak RAM unsplit).
    if [ ! -f "${target_db}.idx" ]; then
        echo "Creating search index for $db_name..."
        idx_tmp=$(mktemp -d)
        source_size_gb=$(du -g "${TARGET_DIR}/${RNA_MMSEQS_DATABASES[$db_name]}" 2>/dev/null | cut -f1)
        split_flag=""
        if [ "${source_size_gb:-0}" -gt 10 ]; then
            split_flag="--split 4"
            echo "  Using --split 4 for large database (${source_size_gb}G)"
        fi
        time mmseqs createindex "$target_db" "$idx_tmp" --search-type 3 $split_flag
        rm -rf "$idx_tmp"
    fi
    echo "Done: $db_name"
done
echo ""

# ---------------------------------------------------------------------------
# Step 2: Convert protein FASTA to MMseqs2 padded format
# ---------------------------------------------------------------------------
echo "=== Step 2: Convert to MMseqs2 GPU format ==="
echo ""

# Database mapping: mmseqs_name -> source_fasta_filename
declare -A MMSEQS_DATABASES=(
    ["uniref90"]="uniref90_2022_05.fa"
    ["mgnify"]="mgy_clusters_2022_05.fa"
    ["small_bfd"]="bfd-first_non_consensus_sequences.fasta"
    ["uniprot"]="uniprot_all_2021_04.fa"
    ["pdb_seqres"]="pdb_seqres_2022_09_28.fasta"
)

total_dbs=${#MMSEQS_DATABASES[@]}
current_db=0

for db_name in "${!MMSEQS_DATABASES[@]}"; do
    current_db=$((current_db + 1))
    source_fasta="${TARGET_DIR}/${MMSEQS_DATABASES[$db_name]}"
    target_base="${MMSEQS_DIR}/${db_name}"
    target_padded="${target_base}_padded"

    echo "[$current_db/$total_dbs] Converting: $db_name"

    # Check if padded database already exists
    if [ -f "${target_padded}.dbtype" ]; then
        echo "  SKIP: Padded database already exists"
        echo ""
        continue
    fi

    if [ ! -f "$source_fasta" ]; then
        echo "  WARNING: Source FASTA not found: $source_fasta"
        echo "  Skipping."
        echo ""
        continue
    fi

    # Step 1: createdb
    if [ -f "${target_base}.dbtype" ]; then
        echo "  Found intermediate database, skipping createdb..."
    else
        echo "  Creating MMseqs2 database..."
        time mmseqs createdb "$source_fasta" "$target_base"
    fi

    # Step 2: makepaddedseqdb
    echo "  Creating padded database for GPU..."
    time mmseqs makepaddedseqdb "$target_base" "$target_padded"

    # Clean up intermediate (non-padded) database files
    echo "  Cleaning up intermediate database files..."
    rm -f "${target_base}" "${target_base}".dbtype "${target_base}".index \
          "${target_base}".lookup "${target_base}".source \
          "${target_base}_h" "${target_base}_h".dbtype "${target_base}_h".index

    if [ -f "${target_padded}.dbtype" ]; then
        echo "  SUCCESS: Created ${target_padded}"
    else
        echo "  ERROR: Failed to create padded database"
    fi
    echo ""
done

# ---------------------------------------------------------------------------
# Step 3: Optionally clean up raw FASTA files
# ---------------------------------------------------------------------------
if [ "$KEEP_FASTA" = false ]; then
    echo "=== Step 3: Removing raw FASTA files ==="
    for fasta_file in "${!PROTEIN_FASTAS[@]}"; do
        target_path="${TARGET_DIR}/${fasta_file}"
        if [ -f "$target_path" ]; then
            echo "Removing: $target_path"
            rm -f "$target_path"
        fi
    done
    echo ""
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "End time: $(date)"
echo ""
echo "Database directory:  $TARGET_DIR"
echo "MMseqs2 directory:   $MMSEQS_DIR"
echo "mmCIF directory:     ${TARGET_DIR}/mmcif_files"
echo ""
echo "RNA databases (for nhmmer):"
for rna_f in "${RNA_FASTAS[@]}"; do
    if [ -f "${TARGET_DIR}/${rna_f}" ]; then
        echo "  OK: ${rna_f}"
    else
        echo "  MISSING: ${rna_f}"
    fi
done
echo ""
echo "RNA MMseqs2 databases (for --rna_mmseqs_db_dir, optional):"
for rna_db in rfam rnacentral nt_rna; do
    if [ -f "${RNA_MMSEQS_DIR}/${rna_db}.dbtype" ]; then
        echo "  OK: ${rna_db}"
    else
        echo "  NOT BUILT: ${rna_db}"
    fi
done
echo ""
echo "Use these paths with run_alphafast.sh:"
echo "  ./scripts/run_alphafast.sh \\"
echo "      --db_dir $TARGET_DIR \\"
echo "      --weights_dir /path/to/weights \\"
echo "      --input_dir /path/to/inputs \\"
echo "      --output_dir /path/to/outputs"
echo "=========================================="
