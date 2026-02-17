# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Shared configuration for Modal deployment."""

# Volume names - these are created in the user's Modal account
DATABASE_VOLUME_NAME = "af3-databases"
WEIGHTS_VOLUME_NAME = "af3-weights"

# Mount paths inside the container
DATABASE_MOUNT_PATH = "/databases"
WEIGHTS_MOUNT_PATH = "/weights"
MMSEQS_DB_PATH = "/databases/mmseqs"

# GPU options mapping CLI flag to Modal GPU spec
GPU_OPTIONS = {
    "a100": "A100-80GB",
    "a100-40gb": "A100-40GB",
    "h100": "H100",
    "h200": "H200",
    "l40s": "L40S",
}

# Workspace volume for producer-consumer batch pipeline
WORKSPACE_VOLUME_NAME = "af3-workspace"
WORKSPACE_MOUNT_PATH = "/workspace"

# Producer-consumer GPU assignments
PRODUCER_GPU = "L40S"
CONSUMER_GPU = "A100-80GB"

# Default timeouts (in seconds)
PREDICTION_TIMEOUT = 3600 * 8  # 8 hours
PRODUCER_TIMEOUT = 3600 * 4   # 4 hours for batch MSA
CONSUMER_TIMEOUT = 3600 * 2   # 2 hours per protein inference

# Hot-path data transfer (Queue + Dict)
DATA_DICT_NAME = "af3-data-dict"
DATA_QUEUE_NAME = "af3-data-queue"

# Warm consumer configuration
NUM_CONSUMERS = 4
WARM_CONSUMER_TIMEOUT = 3600 * 6  # 6 hours (one container processes many proteins)
DATABASE_SETUP_TIMEOUT = 3600 * 24  # 24 hours
WEIGHTS_UPLOAD_TIMEOUT = 1800  # 30 minutes

# Database source URL
DATABASE_SOURCE_URL = "https://storage.googleapis.com/alphafold-databases/v3.0"

# Database files to download
DATABASE_FILES = {
    # PDB mmCIF files
    "pdb_mmcif": "pdb_2022_09_28_mmcif_files.tar.zst",
    # Protein sequence databases
    "mgnify": "mgy_clusters_2022_05.fa.zst",
    "small_bfd": "bfd-first_non_consensus_sequences.fasta.zst",
    "uniref90": "uniref90_2022_05.fa.zst",
    "uniprot": "uniprot_all_2021_04.fa.zst",
    "pdb_seqres": "pdb_seqres_2022_09_28.fasta.zst",
    # RNA databases
    "rna_central": "rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst",
    "nt_rna": "nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst",
    "rfam": "rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst",
}

# Databases that need MMseqs2 conversion for GPU acceleration
MMSEQS_CONVERTIBLE_DBS = {
    "uniref90": "uniref90_2022_05.fa",
    "mgnify": "mgy_clusters_2022_05.fa",
    "small_bfd": "bfd-first_non_consensus_sequences.fasta",
    "uniprot": "uniprot_all_2021_04.fa",
    "pdb_seqres": "pdb_seqres_2022_09_28.fasta",
}
