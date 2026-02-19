<p align="center"><img src=".github/logo.png" height="96" /></p>

# AlphaFast

Ultra-high-throughput inference with [AlphaFold 3](https://github.com/google-deepmind/alphafold3). Replaces Jackhmmer with MMseqs2-GPU for **over 68x** speedup in homology search and **over 22x** speedup in end-to-end inference on a single H200 GPU.

AlphaFast has **multi-GPU** capabilities capable of reaching throughput of **8s per input** on 4 H200 GPUs, **4.5s per input** on 8 H200 GPUs, and even higher throughput on larger systems, scaling approximately linearly with number of devices.

For minimal setup or those without significant computational resources, see our [Modal Setup](#modal-setup) section for serverless inference at a cost of **$0.035** and time of **28s** per input.

Check out our bioarxiv preprint [here]([https://www.biorxiv.org/content/10.64898/2026.02.17.706409v1.article-info](https://www.biorxiv.org/content/10.64898/2026.02.17.706409v1))!

> **Disclaimer**: AlphaFast requires AlphaFold 3 model weights, which are subject to
> [Google DeepMind's Terms of Use](WEIGHTS_TERMS_OF_USE.md).
> You must apply for and receive weights directly from Google. This is not an officially supported Google product.
>
> **Note**: DNA/RNA MSA search is not currently supported, but we are rapidly working on this.
>
## Quick Start

### Step 1: Acquire Model Weights

Request access to AlphaFold 3 model parameters via
[this form](https://forms.gle/svvpY4u2jsHEwWYS6) from Google. Approval typically takes 2-5
business days. You will receive a file of compressed weights named `af3.bin.zst`.

### Step 2: Choose Your Compute Environment

| Environment | Requirements | Jump to |
|-------------|-------------|---------|
| Local Server | Docker, Sudo Access | [Docker Setup](#docker-setup) |
| HPC Cluster | Singularity, SLURM | [HPC Setup](#hpc-setup) |
| Serverless | Modal Billing Account | [Modal Setup](#modal-setup) |

---

## Docker Setup

### Step 3: Download and Convert Databases

Downloads and converts protein sequence databases to MMseqs2 GPU format.

> **Important:** Point `path/to/databases` to a fast data drive (NVMe recommended). You will need a minimum of **800 GB** free disk space (250 GB download + 540 GB MMseqs2 padded databases).
>
> **Prerequisite:** The `mmseqs` binary (GPU version), `wget`, `zstd`, and `tar` must be installed and in your `PATH` before running this script. See [docs/building.md](docs/building.md) for MMseqs2 installation instructions.

```bash
./scripts/setup_databases.sh /path/to/databases
```

### Step 4: Pull Container

> **Optional:** To build the container from source instead, see [docs/building.md](docs/building.md).

```bash
docker pull romerolabduke/alphafast:latest
```

### Step 5: Place Weights
>
> **Note:** There are several ways to move the weights to your server such as direct download from the link provided by the DeepMind team, SSH transfer via utilities like [rsync](https://linux.die.net/man/1/rsync) or [scp](https://linux.die.net/man/1/scp).

```bash
cp /path/to/downloaded/af3.bin.zst /path/to/weights/
```

### Step 6: Create Input

Create a directory of input `.json` files. See [docs/input_format.md](docs/input_format.md) for the full format reference. Minimal example:

```json
{
  "name": "2PV7",
  "sequences": [
    {
      "protein": {
        "id": ["A", "B"],
        "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGG..."
      }
    }
  ],
  "modelSeeds": [1,2,3],
  "dialect": "alphafold3",
  "version": 3
}
```

### Step 7: Run Inference

> **Note:** Performance gains from AlphaFast scale from: input batch size, GPU compute capability/VRAM, and number of GPUs.

**Single GPU:**

```bash
./scripts/run_alphafast.sh \
    --input_dir /path/to/inputs \
    --output_dir /path/to/outputs \
    --db_dir /path/to/databases \
    --weights_dir /path/to/weights
```

**Multi-GPU:**

```bash
./scripts/run_alphafast.sh \
    --input_dir /path/to/inputs \
    --output_dir /path/to/outputs \
    --db_dir /path/to/databases \
    --weights_dir /path/to/weights \
    --num_gpus 4 \
    --gpu_devices 0,1,2,3
```

### How Multi-GPU Mode Works

When `--num_gpus` > 1, AlphaFast runs a **phase-separated parallel pipeline**:

1. **Partition** — Inputs are distributed round-robin across GPUs. Identical protein sequences are deduplicated within each partition.
2. **Phase 1: Parallel MSA** — All N GPUs run batched MMseqs2-GPU search simultaneously.
3. **Phase 2: Parallel Fold** — AlphaFast waits until MSAs are complete then data files are re-distributed and all N GPUs run inference simultaneously.

At large batch sizes, every GPU is 100% utilized in each phase, achieving near-linear scaling.

---

## HPC Setup

### Step 3: Install Databases
>
> **Important:** Point `path/to/databases` to a high speed volume with fast network transfer. You will need a minimum of **800 GB** free disk space on this partition (250 GB download + 540 GB MMseqs2 padded databases). AlphaFast will spend roughly ~1 hour to copy the databases to a local NVMe volume (often called `/scratch` on HPC systems). If this is not available, then make sure the databases are on the fastest I/O partition possible.
> **Note:** You may need to edit the SLURM directives to match your university's specific HPC formatting.

```bash
# Submit as SLURM job (CPU node, no GPU needed)
sbatch scripts/setup_databases.sbatch /path/to/databases

# Or run directly in an interactive session:
./scripts/setup_databases.sh /path/to/databases
```

### Step 4: Pull Container
>
> **Important:** Most university HPC systems contain apptainer or singularity rather than Docker for permission management. Depending on your HPC setup, your home directory may very small; therefore, you should ensure your apptainer or singularity cache directory is set to an appropriately sized and speed volume. For more information, see [docs/hpc.md](docs/hpc.md) for specific guidance.

```bash
singularity pull alphafast.sif docker://romerolabduke/alphafast:latest
```

### Step 5: Place Weights
>
> **Note:** There are several ways to move the weights to your university HPC system such as direct download from the link provided by the DeepMind team, SSH transfer via utilities like [rsync](https://linux.die.net/man/1/rsync) or [scp](https://linux.die.net/man/1/scp). Most university systems will have a data transfer node with services like [Globus](https://www.globus.org/) that may be useful.

```bash
rsync -avP /local/path/af3.bin.zst user@hpc:/path/to/weights/
```

### Step 6: Create Input

Create a directory of input `.json` files. See [docs/input_format.md](docs/input_format.md) for the full format reference. Minimal example:

```json
{
  "name": "2PV7",
  "sequences": [
    {
      "protein": {
        "id": ["A", "B"],
        "sequence": "GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGG..."
      }
    }
  ],
  "modelSeeds": [1,2,3],
  "dialect": "alphafold3",
  "version": 1
}
```

### Step 7: Run
>
> **Note:** Performance gains from AlphaFast scale from: input batch size, GPU compute capability/VRAM, and number of GPUs. On HPC systems specifically, AlphaFast will attempt to transfer almost all code, the container, databases to the local `/scratch` directory of a compute node. This transfer can take up to 1-2 hours depending on network speed; therefore, AlphaFast is optimally used for very large input batch sizes.
> Furthermore, if you note significant slowdowns, you should ensure the cache directory for package managers like uv and other system packages is not set to a slow filesystem on your cluster setup. If all else fails, our [Modal Setup](#modal-setup) section should be used instead for your needs.

```bash
./scripts/run_alphafast.sh \
    --input_dir /path/to/inputs \
    --output_dir /path/to/outputs \
    --db_dir /path/to/databases \
    --weights_dir /path/to/weights \
    --container /path/to/alphafast.sif \
    --num_gpus 4
```

---

## Modal Setup

Modal provides serverless GPU inference with pay-per-second billing. A free hosted MSA server eliminates the need to download databases.

```bash
pip install modal && modal token new
modal run modal/upload_weights.py --file /path/to/af3.bin.zst

# Run predictions using the free MSA server
modal run modal/af3_predict.py --input protein.json \
    --msa-server https://romero-lab--alphafold3-msa-server-msa.modal.run
```

See [docs/modal.md](docs/modal.md) for the full CLI reference, batch processing, multi-GPU modes, and cost estimates.

---

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--input_dir` | (required) | Directory containing input JSON files |
| `--output_dir` | (required) | Output directory for results |
| `--db_dir` | (required) | Database directory (from `setup_databases.sh`) |
| `--weights_dir` | (required) | Directory containing `af3.bin.zst` |
| `--num_gpus` | `1` | Number of GPUs |
| `--container` | `romerolabduke/alphafast:latest` | Docker image or `.sif` path |
| `--batch_size` | auto (count of inputs) | MSA batch size |
| `--gpu_devices` | `0` (single) or `0,1,...` (multi) | Comma-separated GPU device IDs |
| `--backend` | auto-detect | Force `docker` or `singularity` |

For advanced flags, see [docs/advanced.md](docs/advanced.md).

## Citing This Work

If you use AlphaFast in your research, please cite both our work and AlphaFold 3:

### AlphaFast Citation

```bibtex
@article {Perry2026.02.17.706409,
	author = {Perry, Benjamin C and Kim, Jeonghyeon and Romero, Philip A},
	title = {AlphaFast: High-throughput AlphaFold 3 via GPU-accelerated MSA construction},
	year = {2026},
	doi = {10.64898/2026.02.17.706409},
	publisher = {Cold Spring Harbor Laboratory},
	abstract = {AlphaFold 3 (AF3) enables accurate biomolecular modeling but is limited by slow, CPU-bound multiple sequence alignment (MSA) generation. We introduce AlphaFast, a drop-in framework that integrates GPU-accelerated MMseqs2 sequence search to remove this bottleneck. AlphaFast achieves a 68.5x speedup in MSA construction and a 22.8x reduction in end-to-end runtime on a single GPU, and delivers predictions in 8 seconds per input on four GPUs while maintaining indistinguishable structural accuracy. A serverless deployment enables structure prediction for as little as $0.035 per input. Code is available at https://github.com/RomeroLab/alphafast.},
	URL = {https://www.biorxiv.org/content/early/2026/02/18/2026.02.17.706409},
	journal = {bioRxiv}
}

```

### AlphaFold 3 Citation

```bibtex
@article{Abramson2024,
  author  = {Abramson, Josh and Adler, Jonas and Dunger, Jack and Evans, Richard and Green, Tim and Pritzel, Alexander and Ronneberger, Olaf and Willmore, Lindsay and Ballard, Andrew J. and Bambrick, Joshua and Bodenstein, Sebastian W. and Evans, David A. and Hung, Chia-Chun and O’Neill, Michael and Reiman, David and Tunyasuvunakool, Kathryn and Wu, Zachary and Žemgulytė, Akvilė and Arvaniti, Eirini and Beattie, Charles and Bertolli, Ottavia and Bridgland, Alex and Cherepanov, Alexey and Congreve, Miles and Cowen-Rivers, Alexander I. and Cowie, Andrew and Figurnov, Michael and Fuchs, Fabian B. and Gladman, Hannah and Jain, Rishub and Khan, Yousuf A. and Low, Caroline M. R. and Perlin, Kuba and Potapenko, Anna and Savy, Pascal and Singh, Sukhdeep and Stecula, Adrian and Thillaisundaram, Ashok and Tong, Catherine and Yakneen, Sergei and Zhong, Ellen D. and Zielinski, Michal and Žídek, Augustin and Bapst, Victor and Kohli, Pushmeet and Jaderberg, Max and Hassabis, Demis and Jumper, John M.},
  journal = {Nature},
  title   = {Accurate structure prediction of biomolecular interactions with AlphaFold 3},
  year    = {2024},
  volume  = {630},
  number  = {8016},
  pages   = {493--500},
  doi     = {10.1038/s41586-024-07487-w}
}
```

## License

Source code is licensed under [CC-BY-NC-SA 4.0](LICENSE). Model parameters are
subject to the [AlphaFold 3 Model Parameters Terms of Use](WEIGHTS_TERMS_OF_USE.md).
Output is subject to the [Output Terms of Use](OUTPUT_TERMS_OF_USE.md).

This is not an officially supported Google product.
