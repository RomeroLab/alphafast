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
# databases from Google Cloud Storage and converts the protein FASTA files to
# MMseqs2 padded databases for GPU-accelerated MSA search.
#
# Usage:
#   ./scripts/setup_databases.sh <target_dir> [--keep-fasta]
#   ./scripts/setup_databases.sh <target_dir> --from-prebuilt
#
# Arguments:
#   target_dir:      Directory where databases will be downloaded and converted
#   --keep-fasta:    Keep raw FASTA files after MMseqs2 conversion (default: keep)
#                    Use --no-keep-fasta to remove them after conversion
#   --from-prebuilt: Download pre-built MMseqs2 GPU databases from HuggingFace
#                    (~569 GB, no conversion needed, requires pip: huggingface_hub)
#
# Requirements (default mode):
#   - wget, zstd, tar in PATH
#   - mmseqs (GPU version) in PATH
#   - ~800 GB free disk space (250 GB download + 540 GB MMseqs2 padded)
#
# Requirements (--from-prebuilt mode):
#   - python3 with huggingface_hub installed (pip install huggingface_hub)
#   - zstd, tar in PATH
#   - ~569 GB free disk space
#
# Output directory structure:
#   <target_dir>/
#     mmcif_files/              # PDB structures for template retrieval
#     uniref90_2022_05.fa       # Raw FASTA (if kept)
#     mgy_clusters_2022_05.fa
#     ...
#     mmseqs/
#       uniref90_padded*        # MMseqs2 GPU-ready databases
#       mgnify_padded*
#       small_bfd_padded*
#       uniprot_padded*
#       pdb_seqres_padded*

set -euo pipefail

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
usage() {
    echo "Usage: $0 <target_dir> [--keep-fasta | --no-keep-fasta]"
    echo "       $0 <target_dir> --from-prebuilt"
    echo ""
    echo "Downloads AlphaFold 3 databases and converts protein DBs to MMseqs2 GPU format."
    echo ""
    echo "Arguments:"
    echo "  target_dir       Directory where databases will be stored"
    echo "  --keep-fasta     Keep raw FASTA files after conversion (default)"
    echo "  --no-keep-fasta  Remove raw FASTA files after conversion"
    echo "  --from-prebuilt  Download pre-built databases from HuggingFace (no mmseqs needed)"
    exit 1
}

if [ "$#" -lt 1 ]; then
    usage
fi

TARGET_DIR="$1"
shift

KEEP_FASTA=true
FROM_PREBUILT=false
while [ "$#" -gt 0 ]; do
    case "$1" in
        --keep-fasta)    KEEP_FASTA=true; shift ;;
        --no-keep-fasta) KEEP_FASTA=false; shift ;;
        --from-prebuilt) FROM_PREBUILT=true; shift ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

# ---------------------------------------------------------------------------
# Pre-built download from HuggingFace (--from-prebuilt)
# ---------------------------------------------------------------------------
if [ "$FROM_PREBUILT" = true ]; then
    # Check prerequisites for HF download
    MISSING=0
    for cmd in python3 tar zstd; do
        if ! command -v "$cmd" &> /dev/null; then
            echo "ERROR: $cmd is not installed or not in PATH."
            MISSING=1
        fi
    done
    if [ "$MISSING" -ne 0 ]; then
        echo ""
        echo "Install missing dependencies before running this script."
        echo "For python3: install huggingface_hub with: pip install huggingface_hub"
        exit 1
    fi

    # Check huggingface_hub is importable
    if ! python3 -c "import huggingface_hub" 2>/dev/null; then
        echo "ERROR: huggingface_hub Python package is not installed."
        echo "Install it with: pip install huggingface_hub"
        exit 1
    fi

    HF_REPO="RomeroLab-Duke/af3-mmseqs-db"
    MMSEQS_DIR="${TARGET_DIR}/mmseqs"
    mkdir -p "$TARGET_DIR" "$MMSEQS_DIR"

    echo "=========================================="
    echo "AlphaFast Database Setup (from HuggingFace)"
    echo "=========================================="
    echo "Repository: ${HF_REPO}"
    echo "Target directory: $TARGET_DIR"
    echo "Start time: $(date)"
    echo "=========================================="
    echo ""

    # Download all files from HuggingFace to target directory
    echo "=== Step 1: Download pre-built databases from HuggingFace ==="
    echo ""

    python3 - "$TARGET_DIR" "$HF_REPO" <<'PYEOF'
import sys
import os
from pathlib import Path
from collections import defaultdict

from huggingface_hub import HfApi, hf_hub_download

target_dir = Path(sys.argv[1])
repo_id = sys.argv[2]

api = HfApi()
repo_files = api.list_repo_files(repo_id, repo_type="dataset")

# Ignore .gitattributes and other metadata
repo_files = [f for f in repo_files if not f.startswith(".")]
print(f"Found {len(repo_files)} files in {repo_id}")
print()

# Separate regular files from .part* split files
part_files = sorted(f for f in repo_files if ".part" in f)
regular_files = sorted(f for f in repo_files if ".part" not in f)

# Download regular files
for repo_file in regular_files:
    local_path = target_dir / repo_file

    if local_path.exists() and local_path.stat().st_size > 0:
        print(f"  SKIP: {repo_file} (exists)")
        continue

    # For mmcif tar files, check if already extracted
    if "mmcif" in repo_file and repo_file.endswith((".tar", ".tar.zst")):
        mmcif_dir = target_dir / "mmcif_files"
        if mmcif_dir.exists() and any(mmcif_dir.iterdir()):
            print(f"  SKIP: {repo_file} (mmcif_files/ already extracted)")
            continue

    print(f"  Downloading: {repo_file}...")
    local_path.parent.mkdir(parents=True, exist_ok=True)
    hf_hub_download(
        repo_id=repo_id,
        filename=repo_file,
        repo_type="dataset",
        local_dir=str(target_dir),
    )
    size_gb = local_path.stat().st_size / (1024**3) if local_path.exists() else 0
    print(f"  DONE: {repo_file} ({size_gb:.1f} GB)")

# Download and reassemble split .part* files
if part_files:
    part_groups = defaultdict(list)
    for pf in part_files:
        base = pf.rsplit(".part", 1)[0]
        part_groups[base].append(pf)

    for base_name, parts in sorted(part_groups.items()):
        reassembled_path = target_dir / base_name

        if reassembled_path.exists() and reassembled_path.stat().st_size > 0:
            print(f"  SKIP: {base_name} (already reassembled)")
            continue

        # For mmcif tar, check if already extracted
        if "mmcif" in base_name and base_name.endswith((".tar", ".tar.zst")):
            mmcif_dir = target_dir / "mmcif_files"
            if mmcif_dir.exists() and any(mmcif_dir.iterdir()):
                print(f"  SKIP: {base_name} (mmcif_files/ already extracted)")
                continue

        print(f"  Downloading {len(parts)} parts for {base_name}...")
        part_paths = []
        for part_file in sorted(parts):
            print(f"    Downloading: {part_file}...")
            hf_hub_download(
                repo_id=repo_id,
                filename=part_file,
                repo_type="dataset",
                local_dir=str(target_dir),
            )
            part_paths.append(target_dir / part_file)
            print(f"    DONE: {part_file}")

        # Reassemble
        reassembled_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"  Reassembling {base_name} from {len(part_paths)} parts...")
        with open(reassembled_path, "wb") as out_f:
            for p in sorted(part_paths):
                with open(p, "rb") as in_f:
                    while True:
                        chunk = in_f.read(64 * 1024 * 1024)  # 64 MB chunks
                        if not chunk:
                            break
                        out_f.write(chunk)

        size_gb = reassembled_path.stat().st_size / (1024**3)
        print(f"  Reassembled: {base_name} ({size_gb:.1f} GB)")

        # Clean up parts
        for p in part_paths:
            p.unlink()
        print(f"  Cleaned up {len(part_paths)} part files")

print()
print("HuggingFace download complete.")
PYEOF

    echo ""

    # Extract mmcif tar archive if present
    echo "=== Step 2: Extract mmcif archive ==="
    MMCIF_DIR="${TARGET_DIR}/mmcif_files"
    for tar_name in "mmcif_files.tar.zst" "mmcif_files.tar"; do
        TAR_PATH="${TARGET_DIR}/${tar_name}"
        if [ -f "$TAR_PATH" ]; then
            if [ -d "$MMCIF_DIR" ] && [ "$(ls -A "$MMCIF_DIR" 2>/dev/null)" ]; then
                echo "SKIP: mmcif_files/ already extracted"
            else
                echo "Extracting ${tar_name}..."
                if [[ "$tar_name" == *.tar.zst ]]; then
                    tar --use-compress-program=zstd -xf "$TAR_PATH" -C "$TARGET_DIR"
                else
                    tar -xf "$TAR_PATH" -C "$TARGET_DIR"
                fi
                echo "Done. Removing archive..."
                rm -f "$TAR_PATH"
            fi
            break
        fi
    done
    echo ""

    # Summary
    echo "=========================================="
    echo "Setup Complete! (from HuggingFace pre-built)"
    echo "=========================================="
    echo "End time: $(date)"
    echo ""
    echo "Database directory:  $TARGET_DIR"
    echo "MMseqs2 directory:   ${TARGET_DIR}/mmseqs"
    echo "mmCIF directory:     ${TARGET_DIR}/mmcif_files"
    echo ""
    echo "Use these paths with run_alphafast.sh:"
    echo "  ./scripts/run_alphafast.sh \\"
    echo "      --db_dir $TARGET_DIR \\"
    echo "      --weights_dir /path/to/weights \\"
    echo "      --input_dir /path/to/inputs \\"
    echo "      --output_dir /path/to/outputs"
    echo "=========================================="

    exit 0
fi

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

# RNA databases (for RNA MSA search - not converted to MMseqs2)
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
echo "Use these paths with run_alphafast.sh:"
echo "  ./scripts/run_alphafast.sh \\"
echo "      --db_dir $TARGET_DIR \\"
echo "      --weights_dir /path/to/weights \\"
echo "      --input_dir /path/to/inputs \\"
echo "      --output_dir /path/to/outputs"
echo "=========================================="
