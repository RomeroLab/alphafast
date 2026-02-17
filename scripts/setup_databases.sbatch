#!/bin/bash
# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/
#
#SBATCH --job-name=alphafast-setup
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --partition=common
#SBATCH --output=logs/setup_databases_%j.out
#SBATCH --error=logs/setup_databases_%j.err
#
# SLURM wrapper for setup_databases.sh
#
# Usage:
#   sbatch scripts/setup_databases.sbatch /path/to/databases
#
# All arguments are forwarded to setup_databases.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p logs

exec "$SCRIPT_DIR/setup_databases.sh" "$@"
