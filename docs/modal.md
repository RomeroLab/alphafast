# Modal Serverless Deployment

All serverless scripts live in the `modal/` directory and run on [Modal](https://modal.com) with pay-per-second billing.

There are **two ways** to get started. We strongly recommend Option A — it's free, fast, and identical in MSA quality. Option B exists for legacy/advanced use cases but offers no practical benefit.

| | Option A: Free MSA Server (Recommended) | Option B: Self-Hosted Databases |
|---|---|---|
| **Setup time** | ~5 minutes | 6-7 hours |
| **Storage cost** | None | ~$120/month (~1.2 TB) |
| **One-time DB cost** | None | ~$30 (download + convert) |
| **MSA speed** | Same (GPU-accelerated MMseqs2) | Same (GPU-accelerated MMseqs2) |
| **MSA quality** | Identical | Identical |
| **Best for** | Everyone | Not recommended |

## Step 1: Install Modal

```bash
pip install modal
modal token new
```

## Step 2: Upload Model Weights (one-time)

```bash
modal run modal/upload_weights.py --file /path/to/af3.bin
modal run modal/upload_weights.py --status         # Check weights status
```

## Option A: Use Our Free MSA Server (Recommended)

The fastest way to get started. Our public MSA server handles all sequence database searches for you — no need to download or store the ~1.2 TB of databases yourself. Just upload your model weights and start predicting.

```bash
# Single protein
modal run modal/af3_predict.py --input protein.json \
    --msa-server https://romero-lab--alphafold3-msa-server-msa.modal.run

# Batch directory
modal run modal/af3_predict.py --input-dir ./proteins/ --output ./results/ \
    --msa-server https://romero-lab--alphafold3-msa-server-msa.modal.run

# Large batch split across multiple GPUs (each GPU: batch MSA then fold one-by-one)
modal run modal/af3_predict.py --input-dir ./proteins/ --mode multi-gpu --num-gpus 4 \
    --msa-server https://romero-lab--alphafold3-msa-server-msa.modal.run

# Large batch with warm producer-consumer pipeline
modal run modal/af3_predict.py --input-dir ./proteins/ --mode producer-consumer \
    --msa-server https://romero-lab--alphafold3-msa-server-msa.modal.run
```

That's it — no database setup, no storage costs. The `--msa-server` flag routes all MSA/template searches to our hosted endpoint, and inference runs on your own Modal GPUs.

> **Note:** The sequence databases (~1.2 TB) are hosted exclusively on our Modal infrastructure and are not available for local download. All MSA/template searches must go through the `--msa-server` endpoint.

## Option B: Self-Hosted Databases (Not Recommended)

> **Why not?** You will spend 6-7 hours downloading and converting ~530 GB of databases, pay ~$30 for the compute, and then ~$120/month to store them on a Modal volume. The MSA results are identical to Option A. There is no speed advantage — both options run the same GPU-accelerated MMseqs2 pipeline on the same databases. Unless you have a very specific reason (e.g., air-gapped environment, custom database modifications), use Option A instead.

### 1. Prepare Databases (one-time, ~6-7 hours, ~$30)

Downloads ~530GB from Google Cloud and converts to MMseqs2 GPU format.

```bash
modal run modal/prepare_databases.py              # Full setup (download + convert)
modal run modal/prepare_databases.py --status      # Check database status
modal run modal/prepare_databases.py --convert-only # Only MMseqs2 conversion
```

### 2. Run Predictions (without --msa-server)

```bash
# Single protein
modal run modal/af3_predict.py --input protein.json

# Batch directory
modal run modal/af3_predict.py --input-dir ./proteins/ --output ./results/

# Single-GPU mode
modal run modal/af3_predict.py --input-dir ./proteins/ --single-gpu

# Multi-GPU mode (split inputs across 4 GPUs)
modal run modal/af3_predict.py --input-dir ./proteins/ --mode multi-gpu --num-gpus 4

# Warm producer-consumer pipeline
modal run modal/af3_predict.py --input-dir ./proteins/ --mode producer-consumer
```

## CLI Reference

All prediction modes use a single unified script (`af3_predict.py`):

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | - | Path to single input JSON file |
| `--input-dir` | - | Path to directory of input JSON files |
| `--output` | `./af3_output` | Local output directory |
| `--msa-server` | - | URL of remote MSA server (skips local DB for MSA) |
| `--mode` | `single` | Pipeline mode: `single`, `multi-gpu`, or `producer-consumer` |
| `--gpu` | `a100` | Inference GPU type (`l40s`, `h100`, `h200`, `a100`) |
| `--num-gpus` | - | Number of GPUs for multi-gpu mode (splits inputs across GPUs) |
| `--producer-gpu` | `l40s` | Producer GPU type for P-C mode |
| `--num-consumers` | `4` | Number of warm consumer containers (P-C mode) |
| `--parallel` | `4` | Max concurrent GPU invocations (single mode batch) |
| `--batch-size` | `32` | Sequences per MMseqs2 batch |
| `--skip-msa` | `false` | Skip MSA (input must have pre-computed data) |
| `--msa-only` | `false` | Only run MSA, skip inference |
| `--single-gpu` | `false` | Batch MSA + sequential inference on one GPU |
| `--keep-workspace` | `false` | Don't clean up workspace volume after P-C completion |
| `--check` | `false` | Check setup status and exit |

## Examples

```bash
# L40S single mode, batch size 128
modal run modal/af3_predict.py --input-dir ./proteins/ --gpu l40s --batch-size 128 \
    --msa-server https://romero-lab--alphafold3-msa-server-msa.modal.run

# H200 single mode, batch size 128
modal run modal/af3_predict.py --input-dir ./proteins/ --gpu h200 --batch-size 128 \
    --msa-server https://romero-lab--alphafold3-msa-server-msa.modal.run

# Multi-GPU with L40S: split across 4 GPUs, batch MSA in groups of 128
modal run modal/af3_predict.py --input-dir ./proteins/ --gpu l40s --batch-size 128 \
    --mode multi-gpu --num-gpus 4 \
    --msa-server https://romero-lab--alphafold3-msa-server-msa.modal.run

# Multi-GPU with H200s: split across 8 GPUs
modal run modal/af3_predict.py --input-dir ./proteins/ --gpu h200 --batch-size 128 \
    --mode multi-gpu --num-gpus 8 \
    --msa-server https://romero-lab--alphafold3-msa-server-msa.modal.run

# L40S producer-consumer, 4 consumers, batch size 128
modal run modal/af3_predict.py --input-dir ./proteins/ --gpu l40s --batch-size 128 \
    --mode producer-consumer --num-consumers 4 \
    --msa-server https://romero-lab--alphafold3-msa-server-msa.modal.run

# H200 producer-consumer, 4 consumers, batch size 128
modal run modal/af3_predict.py --input-dir ./proteins/ --gpu h200 --batch-size 128 \
    --mode producer-consumer --num-consumers 4 \
    --msa-server https://romero-lab--alphafold3-msa-server-msa.modal.run
```

## Cost Estimates

| Item | Option A (MSA Server) | Option B (Self-Hosted) |
|------|---|---|
| Database setup | Free | ~$30 (one-time, 6-7 hours) |
| Database storage | Free | ~$120/month (~1.2 TB) |
| Per prediction (varies) | $2-15 | $2-15 |
