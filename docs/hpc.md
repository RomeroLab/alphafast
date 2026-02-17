# HPC Computing Guide

This guide covers running AlphaFast on HPC clusters using Singularity/Apptainer and SLURM.

## Prerequisites

- NVIDIA GPU with CUDA compute capability >= 6.0
- CUDA driver >= 560.28.03 (required for CUDA 12.6 runtime)
- Singularity >= 3.7 or Apptainer >= 1.0
- SLURM workload manager

## Pulling the Container

Convert the Docker image to a Singularity SIF file:

```bash
singularity pull alphafast.sif docker://romerolabduke/alphafast:latest
```

Set `SINGULARITY_CACHEDIR` to scratch storage to avoid filling your home directory:

```bash
export SINGULARITY_CACHEDIR=/scratch/$USER/.singularity
mkdir -p $SINGULARITY_CACHEDIR
singularity pull alphafast.sif docker://romerolabduke/alphafast:latest
```

The resulting `alphafast.sif` file is approximately 10 GB.

## Filesystem Considerations

HPC filesystems vary in performance characteristics. The data pipeline writes many small temporary files during MMseqs2 searches, which can be very slow on network-attached storage.

| Filesystem | Temp Files | Databases | Recommendations |
|------------|-----------|-----------|-----------------|
| Lustre | Slow (metadata-heavy) | Good (large sequential reads) | Use `--temp_dir` on local storage |
| GPFS | Slow (metadata-heavy) | Good | Use `--temp_dir` on local storage |
| NFS | Slow | Adequate | Use `--temp_dir` on local storage |
| Local SSD / /scratch | Fast | Fast | Preferred for temp files |

**Recommendation**: Always set `--temp_dir` to fast local storage (e.g., `/scratch` or `/tmp`) when databases reside on network-attached storage. This can provide a 10-13x speedup for MMseqs2 searches. Typical temporary space usage is 1-5 GB per batch.

### Bind Mounts

Singularity requires explicit bind mounts for directories outside the container. The `run_alphafast.sh` script handles this automatically for `--db_dir`, `--weights_dir`, `--input_dir`, and `--output_dir`. If you need to pass additional paths (e.g., `--temp_dir`), you may need to add bind mounts manually or ensure the paths are under an already-bound parent directory.

## CUDA Driver Compatibility

AlphaFast uses CUDA 12.6 inside the container. The host system must have a compatible NVIDIA driver:

| CUDA Version | Minimum Driver |
|-------------|---------------|
| CUDA 12.6 | >= 560.28.03 |

Check your driver version:

```bash
nvidia-smi | head -3
```

If your driver is too old, contact your HPC system administrator.

## Single-GPU SLURM Job

```bash
#!/bin/bash
#SBATCH --job-name=alphafast
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=alphafast_%j.out
#SBATCH --error=alphafast_%j.err

# Set paths
INPUT_DIR=/path/to/inputs
OUTPUT_DIR=/path/to/outputs
DB_DIR=/path/to/databases
WEIGHTS_DIR=/path/to/weights
CONTAINER=/path/to/alphafast.sif

# Use local scratch for temporary files (recommended)
export SINGULARITY_CACHEDIR=/scratch/$USER/.singularity

./scripts/run_alphafast.sh \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --db_dir $DB_DIR \
    --weights_dir $WEIGHTS_DIR \
    --container $CONTAINER
```

## Multi-GPU SLURM Job

Multi-GPU mode uses a **phase-separated parallel** architecture: all N GPUs run MSA search in parallel (Phase 1), then all N GPUs run inference in parallel (Phase 2).

```bash
#!/bin/bash
#SBATCH --job-name=alphafast-multi
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --output=alphafast_multi_%j.out
#SBATCH --error=alphafast_multi_%j.err

# Set paths
INPUT_DIR=/path/to/inputs
OUTPUT_DIR=/path/to/outputs
DB_DIR=/path/to/databases
WEIGHTS_DIR=/path/to/weights
CONTAINER=/path/to/alphafast.sif

# Use local scratch for temporary files (recommended)
export SINGULARITY_CACHEDIR=/scratch/$USER/.singularity

./scripts/run_alphafast.sh \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --db_dir $DB_DIR \
    --weights_dir $WEIGHTS_DIR \
    --container $CONTAINER \
    --num_gpus 4
```

With 4 GPUs, all 4 GPUs run MSA search simultaneously in Phase 1. After all MSA jobs complete, the results are re-distributed and all 4 GPUs run inference simultaneously in Phase 2.

## Resource Sizing

| Workload | GPUs | CPUs | Memory | Wall Time (est.) |
|----------|------|------|--------|-----------------|
| 1 protein, single-GPU | 1 | 8 | 64 GB | 30-60 min |
| 32 proteins, single-GPU | 1 | 8 | 64 GB | 4-8 hours |
| 32 proteins, 4 GPUs | 4 | 32 | 256 GB | 1-3 hours |
| 512 proteins, 4 GPUs | 4 | 32 | 256 GB | 12-24 hours |

Memory requirements depend on protein size. For sequences longer than 2000 residues, increase to 128 GB (single-GPU) or 512 GB (multi-GPU).

## Troubleshooting

**Container fails to start with GPU errors**: Verify the CUDA driver version is >= 560.28.03. Run `nvidia-smi` to check.

**Slow MSA searches**: Set `--temp_dir` to local scratch storage. Network filesystems add significant overhead for the many small I/O operations in MMseqs2.

**Out of memory during inference**: Reduce `XLA_CLIENT_MEM_FRACTION` (default 0.95) or use `--flash_attention_implementation xla` for lower memory usage on older GPUs.

**Singularity permission errors**: Ensure `SINGULARITY_CACHEDIR` is writable and on a filesystem that supports file locking.
