#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Download AlphaFold 3 databases for AlphaFast.
#
# Default mode downloads protein and RNA databases from HuggingFace.
# Use --protein-only or --rna-only to narrow the scope.
# Use --from-source to download raw databases from Google Cloud Storage and
# build MMseqs databases locally.
#
# Usage:
#   ./scripts/setup_databases.sh <target_dir> [OPTIONS]
#
# Arguments:
#   target_dir:    Directory where databases will be stored
#
# Requirements (default / HuggingFace mode):
#   - hf CLI (HuggingFace): curl -LsSf https://hf.co/cli/install.sh | bash -s
#   - zstd, tar in PATH (for mmCIF extraction)
#   - ~800 GB free disk space
#
# Requirements (--from-source mode):
#   - wget, zstd, tar in PATH
#   - mmseqs (GPU version) in PATH
#   - ~800 GB free disk space (250 GB download + 540 GB MMseqs2 padded)
#
# Output directory structure:
#   <target_dir>/
#     mmcif_files/              # PDB structures for template retrieval
#     mmseqs/
#       uniref90_padded*        # MMseqs2 GPU-ready databases (protein)
#       mgnify_padded*
#       small_bfd_padded*
#       uniprot_padded*
#       pdb_seqres_padded*
#     mmseqs_rna/
#       rfam*                   # MMseqs2 nucleotide databases (default RNA search)
#       rnacentral*
#       nt_rna*
#     rnacentral_active_seq_id_90_cov_80_linclust.fasta   # RNA FASTA (optional nhmmer fallback)
#     rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta      # RNA FASTA (optional nhmmer fallback)
#     nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta  # RNA FASTA (optional nhmmer fallback)

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
  local exit_code="${1:-1}"
  echo "Usage: $0 <target_dir> [OPTIONS]"
  echo ""
  echo "Downloads AlphaFold 3 databases for AlphaFast."
  echo ""
  echo "Default mode downloads protein and RNA databases from HuggingFace."
  echo "Use --protein-only or --rna-only to narrow the scope."
  echo "Use --from-source to download raw databases from Google Cloud Storage"
  echo "and build MMseqs databases locally."
  echo ""
  echo "Arguments:"
  echo "  target_dir           Directory where databases will be stored"
  echo ""
  echo "Options:"
  echo "  -h, --help           Show this help message and exit."
  echo "  --protein-only       Download/build only protein databases and mmCIF structures."
  echo "  --rna-only           Download/build only RNA databases."
  echo "  --include-nhmmer     Download/build RNA FASTA fallback files."
  echo "  --from-source        Build from FASTA files (Google Cloud Storage)."
  echo "                       Requires wget, zstd, tar, mmseqs in PATH."
  echo "  --keep-fasta         Keep raw FASTA files after conversion (default, --from-source only)"
  echo "  --no-keep-fasta      Remove raw FASTA files after conversion (--from-source only)"
  exit "$exit_code"
}

if [ "$#" -eq 0 ]; then
  usage
fi

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  usage 0
fi

TARGET_DIR="$1"
shift

KEEP_FASTA=true
FROM_PREBUILT=true # HuggingFace is the default
INSTALL_MODE="all" # "all", "protein-only", or "rna-only"
INCLUDE_NHMMER=false
DOWNLOAD_RNA_MMSEQS=true
DOWNLOAD_RNA_FASTA=false

while [ "$#" -gt 0 ]; do
  case "$1" in
  -h|--help)
    usage 0
    ;;
  --keep-fasta)
    KEEP_FASTA=true
    shift
    ;;
  --no-keep-fasta)
    KEEP_FASTA=false
    shift
    ;;
  --from-prebuilt)
    FROM_PREBUILT=true
    shift
    ;;
  --from-source)
    FROM_PREBUILT=false
    shift
    ;;
  --protein-only)
    INSTALL_MODE="protein-only"
    shift
    ;;
  --rna-only)
    INSTALL_MODE="rna-only"
    shift
    ;;
  --include-nhmmer)
    INCLUDE_NHMMER=true
    shift
    ;;
  *)
    echo "Unknown argument: $1"
    usage
    ;;
  esac
done

if $FROM_PREBUILT && [ "$KEEP_FASTA" = false ]; then
  echo "ERROR: --no-keep-fasta can only be used with --from-source."
  usage
fi

case "$INSTALL_MODE" in
all)
  DOWNLOAD_PROTEIN=true
  DOWNLOAD_RNA_MMSEQS=true
  DOWNLOAD_RNA_FASTA="$INCLUDE_NHMMER"
  ;;
protein-only)
  DOWNLOAD_PROTEIN=true
  DOWNLOAD_RNA_MMSEQS=false
  DOWNLOAD_RNA_FASTA=false
  ;;
rna-only)
  DOWNLOAD_PROTEIN=false
  DOWNLOAD_RNA_MMSEQS=true
  DOWNLOAD_RNA_FASTA="$INCLUDE_NHMMER"
  ;;
esac

# ---------------------------------------------------------------------------
# Check prerequisites
# ---------------------------------------------------------------------------
MISSING=0
if $FROM_PREBUILT; then
  for cmd in hf tar zstd; do
    if ! command -v "$cmd" &>/dev/null; then
      if [ "$cmd" = "hf" ]; then
        echo "ERROR: $cmd is not installed. Install with: curl -LsSf https://hf.co/cli/install.sh | bash -s"
      else
        echo "ERROR: $cmd is not installed or not in PATH."
      fi
      MISSING=1
    fi
  done
else
  for cmd in wget tar zstd mmseqs; do
    if ! command -v "$cmd" &>/dev/null; then
      echo "ERROR: $cmd is not installed or not in PATH."
      MISSING=1
    fi
  done
fi
if [ "$MISSING" -ne 0 ]; then
  echo ""
  echo "Install missing dependencies before running this script."
  if $FROM_PREBUILT; then
    echo "For HuggingFace CLI:"
    echo "  curl -LsSf https://hf.co/cli/install.sh | bash -s"
  else
    echo "For mmseqs with GPU support:"
    echo "  wget https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz"
    echo "  tar xzf mmseqs-linux-gpu.tar.gz"
    echo "  sudo cp mmseqs/bin/mmseqs /usr/local/bin/"
  fi
  exit 1
fi

if ! $FROM_PREBUILT; then
  echo "Using MMseqs2 version: $(mmseqs version)"
fi

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
readonly SOURCE="https://storage.googleapis.com/alphafold-databases/v3.0"
readonly HF_REPO="RomeroLab-Duke/af3-mmseqs-db"
MMSEQS_DIR="${TARGET_DIR}/mmseqs"
RNA_MMSEQS_DIR="${TARGET_DIR}/mmseqs_rna"
RNA_FASTAS=(
  "rnacentral_active_seq_id_90_cov_80_linclust.fasta"
  "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"
  "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"
)

mkdir -p "$TARGET_DIR"
if [ "$DOWNLOAD_PROTEIN" = true ]; then
  mkdir -p "$MMSEQS_DIR"
fi
if [ "$DOWNLOAD_RNA_MMSEQS" = true ]; then
  mkdir -p "$RNA_MMSEQS_DIR"
fi

reassemble_part_files() {
  local target_dir="$1"
  shopt -s nullglob
  for part_prefix in "${target_dir}"/*.part00; do
    if [ -f "$part_prefix" ]; then
      local base="${part_prefix%.part00}"
      echo "Reassembling $(basename "$base")..."
      cat "${base}.part"* > "$base"
      rm -f "${base}.part"*
    fi
  done
  shopt -u nullglob
}

decompress_zst_files() {
  local target_dir="$1"
  shopt -s nullglob
  for compressed in "${target_dir}"/*.zst; do
    if [ -f "$compressed" ]; then
      local output="${compressed%.zst}"
      if [ -f "$output" ]; then
        echo "SKIP: $(basename "$output") already decompressed"
        rm -f "$compressed"
        continue
      fi
      echo "Decompressing $(basename "$compressed")..."
      zstd --decompress --force --rm -o "$output" "$compressed"
    fi
  done
  shopt -u nullglob
}

echo "=========================================="
echo "AlphaFast Database Setup"
echo "=========================================="
echo "Target directory: $TARGET_DIR"
echo "MMseqs2 directory: $MMSEQS_DIR"
echo "Mode:             $(if $FROM_PREBUILT; then echo 'pre-built (HuggingFace)'; else echo 'download raw + build locally'; fi)"
echo "Install mode:     $INSTALL_MODE"
echo "RNA MMseqs2:      $DOWNLOAD_RNA_MMSEQS"
echo "RNA FASTA:        $DOWNLOAD_RNA_FASTA"
echo "Keep FASTA files: $KEEP_FASTA"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# ===========================================================================
# Pre-built mode: download from HuggingFace
# ===========================================================================
if $FROM_PREBUILT; then
  echo "=== Downloading pre-built databases from HuggingFace ==="
  echo "Repository: $HF_REPO"
  echo ""
  if [ "$DOWNLOAD_PROTEIN" = true ]; then
    # Download mmCIF structures
    MMCIF_DIR="${TARGET_DIR}/mmcif_files"
    if [ -d "$MMCIF_DIR" ] && [ "$(ls -A "$MMCIF_DIR" 2>/dev/null)" ]; then
      echo "SKIP: mmcif_files already exists"
    else
      echo "Downloading mmCIF structures..."
      hf download "$HF_REPO" --repo-type dataset --include "mmcif_files.tar.zst.*" --local-dir "$TARGET_DIR"
      # Reassemble and extract
      cat "${TARGET_DIR}/mmcif_files.tar.zst.part"* | tar --use-compress-program=zstd -xf - --directory="$TARGET_DIR"
      rm -f "${TARGET_DIR}/mmcif_files.tar.zst.part"*
      echo "Done: mmCIF structures"
    fi
    echo ""
  else
    echo "SKIP: Protein databases (--rna-only mode)"
    echo ""
  fi

  # Download protein MMseqs2 padded databases
  if [ "$DOWNLOAD_PROTEIN" = true ]; then
    if [ -f "${MMSEQS_DIR}/uniref90_padded.dbtype" ]; then
      echo "SKIP: Protein MMseqs2 databases already exist"
    else
      echo "Downloading protein MMseqs2 padded databases..."
      hf download "$HF_REPO" --repo-type dataset \
        --include "mmseqs/*.zst" --include "mmseqs/*.zst.*" \
        --local-dir "$TARGET_DIR"
      # Reassemble any split .partNN files and decompress .zst payloads.
      reassemble_part_files "$MMSEQS_DIR"
      decompress_zst_files "$MMSEQS_DIR"
      echo "Done: Protein MMseqs2 databases"
    fi
    echo ""
  fi

  if [ "$DOWNLOAD_RNA_MMSEQS" = true ]; then
    RNA_MMSEQS_DIR="${TARGET_DIR}/mmseqs_rna"
    if [ -f "${RNA_MMSEQS_DIR}/rfam.dbtype" ]; then
      echo "SKIP: RNA MMseqs2 databases already exist"
    else
      # Download everything including pre-built indices (default)
      echo "Downloading RNA MMseqs2 nucleotide databases (with pre-built indices)..."
      hf download "$HF_REPO" --repo-type dataset \
        --include "mmseqs_rna/*.zst" --include "mmseqs_rna/*.zst.*" \
        --local-dir "$TARGET_DIR"
      reassemble_part_files "$RNA_MMSEQS_DIR"
      decompress_zst_files "$RNA_MMSEQS_DIR"
      echo "Done: RNA MMseqs2 databases"
    fi
    echo ""

  else
    echo "SKIP: RNA MMseqs2 databases (--protein-only mode)"
    echo ""
  fi

  if [ "$DOWNLOAD_RNA_FASTA" = true ]; then
    # Download RNA FASTA databases (for nhmmer fallback)
    echo "Downloading RNA FASTA databases (for nhmmer fallback)..."
    hf download "$HF_REPO" --repo-type dataset --include "*.fasta" --include "*.fasta.*" --local-dir "$TARGET_DIR"
    # Reassemble any split files
    for part_prefix in "${TARGET_DIR}"/*.fasta.part00; do
      if [ -f "$part_prefix" ]; then
        base="${part_prefix%.part00}"
        echo "Reassembling $(basename "$base")..."
        cat "${base}.part"* >"$base"
        rm -f "${base}.part"*
      fi
    done
    echo "Done: RNA FASTA databases"
    echo ""
  elif [ "$INSTALL_MODE" = "rna-only" ] || [ "$INSTALL_MODE" = "all" ]; then
    echo "SKIP: RNA FASTA databases (use --include-nhmmer to download them)"
    echo ""
  else
    echo "SKIP: RNA databases (--protein-only mode)"
    echo ""
  fi

  echo "=== Pre-built download complete ==="
  echo ""

# ===========================================================================
# Build mode: download FASTA and convert
# ===========================================================================
else

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
      "${SOURCE}/pdb_2022_09_28_mmcif_files.tar.zst" |
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

  if [ "$DOWNLOAD_PROTEIN" = true ]; then
    for fasta_file in "${!PROTEIN_FASTAS[@]}"; do
      db_label="${PROTEIN_FASTAS[$fasta_file]}"
      target_path="${TARGET_DIR}/${fasta_file}"

      if [ -f "$target_path" ]; then
        echo "SKIP: $db_label already exists at $target_path"
      else
        echo "Downloading $db_label ($fasta_file)..."
        wget --progress=bar:force:noscroll -O - \
          "${SOURCE}/${fasta_file}.zst" |
          zstd --decompress >"$target_path"
        echo "Done: $db_label"
      fi
    done
  else
    echo "SKIP: Protein FASTA downloads (--rna-only mode)"
  fi
  echo ""

  # RNA databases (for RNA MSA search via nhmmer).
  RNA_FASTAS=(
    "rnacentral_active_seq_id_90_cov_80_linclust.fasta"
    "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta"
    "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta"
  )

  if [ "$DOWNLOAD_RNA_MMSEQS" = true ] || [ "$DOWNLOAD_RNA_FASTA" = true ]; then
    for fasta_file in "${RNA_FASTAS[@]}"; do
      target_path="${TARGET_DIR}/${fasta_file}"
      if [ -f "$target_path" ]; then
        echo "SKIP: RNA database already exists at $target_path"
      else
        echo "Downloading RNA database ($fasta_file)..."
        wget --progress=bar:force:noscroll -O - \
          "${SOURCE}/${fasta_file}.zst" |
          zstd --decompress >"$target_path"
        echo "Done: $fasta_file"
      fi
    done
  elif [ "$INSTALL_MODE" = "rna-only" ] || [ "$INSTALL_MODE" = "all" ]; then
    echo "SKIP: RNA downloads (RNA mode disabled)"
  else
    echo "SKIP: RNA FASTA downloads (--protein-only mode)"
  fi
  echo ""

  # ---------------------------------------------------------------------------
  # Step 1b: Build MMseqs2 databases from RNA FASTA (for nucleotide search)
  # ---------------------------------------------------------------------------
  if [ "$DOWNLOAD_RNA_MMSEQS" = true ]; then
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

      # Build k-mer index for fast search.
      if [ ! -f "${target_db}.idx" ]; then
        echo "Creating search index for $db_name..."
        idx_tmp=$(mktemp -d)
        source_size_gb=$(du -B1G "${TARGET_DIR}/${RNA_MMSEQS_DATABASES[$db_name]}" 2>/dev/null | cut -f1)
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
  else
    echo "SKIP: RNA MMseqs2 database build (--protein-only mode)"
    echo ""
    RNA_MMSEQS_DIR="${TARGET_DIR}/mmseqs_rna"
  fi

  # ---------------------------------------------------------------------------
  # Step 2: Convert protein FASTA to MMseqs2 padded format
  # ---------------------------------------------------------------------------
  if [ "$DOWNLOAD_PROTEIN" = true ]; then
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
  else
    echo "=== Step 2: Convert to MMseqs2 GPU format ==="
    echo "SKIP: Protein MMseqs2 conversion (--rna-only mode)"
    echo ""
  fi

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
    if [ "$DOWNLOAD_RNA_FASTA" = false ]; then
      for fasta_file in "${RNA_FASTAS[@]}"; do
        target_path="${TARGET_DIR}/${fasta_file}"
        if [ -f "$target_path" ]; then
          echo "Removing: $target_path"
          rm -f "$target_path"
        fi
      done
    fi
    echo ""
  fi

fi # end of: if $FROM_PREBUILT; then ... else (build mode)

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
echo "RNA MMseqs2 databases (default RNA search):"
if [ "$DOWNLOAD_RNA_MMSEQS" = true ]; then
  for rna_db in rfam rnacentral nt_rna; do
    if [ -f "${RNA_MMSEQS_DIR}/${rna_db}.dbtype" ]; then
      echo "  OK: ${rna_db}"
    else
      echo "  NOT BUILT: ${rna_db}"
    fi
  done
else
  echo "  SKIPPED: --protein-only mode"
fi
echo ""
echo "RNA FASTA databases (nhmmer fallback, --use_nhmmer):"
if [ "$DOWNLOAD_RNA_FASTA" = true ]; then
  for rna_f in "${RNA_FASTAS[@]}"; do
    if [ -f "${TARGET_DIR}/${rna_f}" ]; then
      echo "  OK: ${rna_f}"
    else
      echo "  MISSING: ${rna_f}"
    fi
  done
elif [ "$INSTALL_MODE" = "rna-only" ] || [ "$INSTALL_MODE" = "all" ]; then
  echo "  SKIPPED: use --include-nhmmer to download them"
else
  echo "  SKIPPED: --protein-only mode"
fi
echo ""
echo "Use these paths with run_alphafast.sh:"
echo "  ./scripts/run_alphafast.sh \\"
echo "      --db_dir $TARGET_DIR \\"
echo "      --weights_dir /path/to/weights \\"
echo "      --input_dir /path/to/inputs \\"
echo "      --output_dir /path/to/outputs"
echo "=========================================="
