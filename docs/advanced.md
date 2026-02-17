# Advanced Configuration

## Template Modes

AlphaFast supports three template search strategies, controlled by `--template_mode`:

| Mode | Description |
|------|-------------|
| `default` | Uses MMseqs2-GPU to search against the PDB sequence database only. No Foldseek. This is the default and fastest option. |
| `foldseek_backup` | Searches PDB first with MMseqs2-GPU, then fills remaining template slots with Foldseek structural search against the AlphaFold Database (AFDB). |
| `foldseek_first` | Searches AFDB with Foldseek first, then fills remaining template slots with PDB search via MMseqs2-GPU. |

The `foldseek_backup` and `foldseek_first` modes require:

- A Foldseek AFDB database (`--foldseek_database_path`)
- ESMFold for generating initial structure predictions that Foldseek uses as queries

## Memory Settings

Two environment variables control GPU memory allocation. Both are pre-configured in the Docker image.

**`XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"`**

Required to avoid XLA compilation slowdowns. Set in the container by default. For GPUs with compute capability 7.x (e.g., V100), use instead:

```bash
export XLA_FLAGS="--xla_disable_hlo_passes=custom-kernel-fusion-rewriter"
```

**`XLA_CLIENT_MEM_FRACTION=0.95`**

Controls the fraction of GPU memory JAX pre-allocates. The default of 0.95 is optimal when using the two-stage pipeline (data pipeline runs separately from inference). If running both stages in a single `run_alphafold.py` invocation, you may need to reduce this to avoid OOM errors during the data pipeline stage.

## Multi-GPU Phase-Separated Parallel

When `--num_gpus > 1`, AlphaFast uses a **phase-separated parallel** architecture orchestrated by `scripts/run_multigpu.sh`:

1. **Partition** — Inputs are distributed round-robin across all N GPUs.
2. **Phase 1: Parallel MSA** — All N GPUs run `run_data_pipeline.py` simultaneously, each processing its partition with batched MMseqs2-GPU search.
3. **Barrier** — Wait for all MSA jobs to complete.
4. **Re-partition** — MSA output files (`*_data.json`) are re-distributed across GPUs for balanced inference.
5. **Phase 2: Parallel Fold** — All N GPUs run `run_alphafold.py --norun_data_pipeline` simultaneously, each loading the model once and processing its assigned inputs.

This achieves near-perfect linear scaling because every GPU is 100% utilized in each phase.

### Producer-Consumer Mode (Modal)

The Modal deployment uses an alternative **producer-consumer** architecture via `scripts/run_msa_producer_consumer.sh`, where a single producer GPU runs MSA search and consumer GPUs run inference concurrently. See [Modal documentation](../docs/modal.md) for details.

## Batch Size Tuning

The `--batch_size` flag controls how many input JSON files are processed together in a single MMseqs2 GPU search. All protein sequences from a batch are collected into a single MMseqs2 query database, which is significantly more efficient than sequential processing.

When using `run_alphafast.sh`, the batch size defaults to the number of input JSON files in `--input_dir`. For manual runs:

```bash
python run_data_pipeline.py \
    --input_dir=/root/af_input \
    --output_dir=/root/af_output \
    --batch_size=32 \
    ...
```

Larger batch sizes improve GPU utilization but increase peak memory usage. If you encounter GPU OOM during MSA search, reduce the batch size.

## Temporary Directory

On HPC clusters with slow network-attached storage (Lustre, GPFS, NFS), set `--temp_dir` to fast local storage for a major speedup:

```bash
python run_data_pipeline.py \
    --temp_dir=/scratch/$USER/alphafast_tmp \
    ...
```

Typical space usage: 1-5 GB per batch. The directory is cleaned up automatically after each run.

## GPU Device Selection

The `--gpu_device` flag selects which GPU to use (zero-indexed). This is useful on multi-GPU systems when you want to pin a run to a specific device:

```bash
# Use the second GPU for inference
python run_alphafold.py \
    --gpu_device=1 \
    ...
```

Note: If `CUDA_VISIBLE_DEVICES` is set, `--gpu_device` refers to the index within the visible devices, not the physical GPU index.

## run_alphafold.py Flag Reference

### Input/Output

| Flag | Default | Description |
|------|---------|-------------|
| `--json_path` | -- | Path to a single input JSON file |
| `--input_dir` | -- | Directory containing input JSON files (alternative to `--json_path`) |
| `--output_dir` | (required) | Output directory for results |
| `--model_dir` | `~/models` | Directory containing `af3.bin.zst` model weights |
| `--force_output_dir` | `false` | Allow writing to non-empty output directories |

### Pipeline Control

| Flag | Default | Description |
|------|---------|-------------|
| `--run_data_pipeline` | `true` | Run MSA/template search |
| `--run_inference` | `true` | Run structure prediction |
| `--norun_data_pipeline` | -- | Skip data pipeline (use pre-computed MSA data) |

### MSA Search

| Flag | Default | Description |
|------|---------|-------------|
| `--mmseqs_sensitivity` | `7.5` | MMseqs2 sensitivity (1.0-7.5) |
| `--use_mmseqs_gpu` | `true` | Use GPU-accelerated MMseqs2 |
| `--mmseqs_n_threads` | all CPUs | CPU threads for MMseqs2 non-GPU operations |
| `--mmseqs_sequential` | `false` | Run database searches sequentially (lower memory) |
| `--batch_size` | -- | Batch multiple inputs into one MMseqs2 search |

### Template Search

| Flag | Default | Description |
|------|---------|-------------|
| `--template_mode` | `default` | Template strategy: `default`, `foldseek_backup`, `foldseek_first` |
| `--template_e_value` | `1e-3` | E-value threshold for template hits |
| `--template_min_coverage` | `0.40` | Minimum alignment coverage (0-1) |
| `--max_template_date` | `2021-09-30` | Maximum template release date (YYYY-MM-DD) |

### Foldseek (for `foldseek_backup` / `foldseek_first` modes)

| Flag | Default | Description |
|------|---------|-------------|
| `--foldseek_database_path` | -- | Path to AFDB Foldseek database |
| `--foldseek_max_templates` | `4` | Maximum Foldseek templates |
| `--foldseek_min_lddt` | `0.5` | Minimum LDDT score for hits (0-1) |
| `--foldseek_min_plddt` | `50.0` | Minimum ESMFold pLDDT to proceed (0-100) |
| `--foldseek_e_value` | `1e-3` | E-value threshold for Foldseek |
| `--foldseek_threads` | `8` | CPU threads for Foldseek |
| `--foldseek_gpu` | `true` | Use GPU-accelerated Foldseek |

### Inference Tuning

| Flag | Default | Description |
|------|---------|-------------|
| `--gpu_device` | `0` | GPU device index for inference |
| `--flash_attention_implementation` | `triton` | Flash attention backend: `triton`, `cudnn`, or `xla` |
| `--num_recycles` | `10` | Number of recycling iterations |
| `--num_diffusion_samples` | `5` | Number of diffusion samples to generate |
| `--num_seeds` | -- | Generate N seeds from a single input seed |
| `--buckets` | `256,512,...,5120` | Token bucket sizes for compilation caching |
| `--jax_compilation_cache_dir` | -- | Directory for JAX compilation cache |

### Output Options

| Flag | Default | Description |
|------|---------|-------------|
| `--save_embeddings` | `false` | Save trunk single/pair embeddings |
| `--save_distogram` | `false` | Save predicted distogram |
| `--write_timing_json` | `false` | Write per-input inference timing to JSONL |
