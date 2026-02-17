# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""
AlphaFold3 inference on Modal with MMseqs2-GPU acceleration.

Unified prediction script supporting single-GPU and warm producer-consumer modes.

Usage:
    # Single protein
    modal run modal/af3_predict.py --input protein.json --output ./results/

    # Batch on single GPU
    modal run modal/af3_predict.py --input-dir ./proteins/ --single-gpu

    # Batch split across multiple GPUs (each GPU: batch MSA → fold one-by-one)
    modal run modal/af3_predict.py --input-dir ./proteins/ --mode multi-gpu --num-gpus 4

    # Batch with warm producer-consumer pipeline
    modal run modal/af3_predict.py --input-dir ./proteins/ --mode producer-consumer

    # Custom GPUs
    modal run modal/af3_predict.py --input-dir ./proteins/ --gpu h100 --producer-gpu l40s --mode producer-consumer

    # Using remote MSA server
    modal run modal/af3_predict.py --input-dir ./proteins/ --msa-server https://your-app.modal.run/msa

Prerequisites:
    1. Databases prepared: modal run modal/prepare_databases.py
    2. Weights uploaded: modal run modal/upload_weights.py --file weights.tar.zst
"""

import json
import math
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import modal

from config import (
    DATA_DICT_NAME,
    DATA_QUEUE_NAME,
    DATABASE_MOUNT_PATH,
    DATABASE_VOLUME_NAME,
    CONSUMER_GPU,
    GPU_OPTIONS,
    MMSEQS_DB_PATH,
    NUM_CONSUMERS,
    PREDICTION_TIMEOUT,
    PRODUCER_GPU,
    PRODUCER_TIMEOUT,
    WARM_CONSUMER_TIMEOUT,
    WEIGHTS_MOUNT_PATH,
    WEIGHTS_VOLUME_NAME,
    WORKSPACE_MOUNT_PATH,
    WORKSPACE_VOLUME_NAME,
)


# ── GPU resolution from CLI flags ─────────────────────────────────────


def _resolve_gpu_from_argv(flag_name, default):
    """Parse --flag_name VALUE from sys.argv at module level (before decorators).

    Returns the Modal GPU spec string (e.g. "H100", "A100-80GB").
    """
    for i, arg in enumerate(sys.argv):
        if arg == f"--{flag_name}" and i + 1 < len(sys.argv):
            return GPU_OPTIONS.get(sys.argv[i + 1].lower(), sys.argv[i + 1].upper())
    return default


_CONSUMER_GPU = _resolve_gpu_from_argv("gpu", CONSUMER_GPU)
_PRODUCER_GPU = _resolve_gpu_from_argv("producer-gpu", PRODUCER_GPU)


# ── Modal app and resources ───────────────────────────────────────────


app = modal.App("alphafold3-inference")

db_volume = modal.Volume.from_name(DATABASE_VOLUME_NAME, create_if_missing=True)
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME_NAME, create_if_missing=True)
workspace_volume = modal.Volume.from_name(WORKSPACE_VOLUME_NAME, create_if_missing=True)

data_dict = modal.Dict.from_name(DATA_DICT_NAME, create_if_missing=True)
data_queue = modal.Queue.from_name(DATA_QUEUE_NAME, create_if_missing=True)

image = (
    modal.Image.from_dockerfile(
        path="docker/Dockerfile",
        context_dir=".",
    )
    .add_local_python_source("config")
    .add_local_python_source("utils")
)

# XLA/JAX env vars scoped to inference subprocess only.
# Setting these on the image would cause JAX to preallocate GPU memory
# during the data pipeline stage, starving MMseqs2-GPU of CUDA memory.
_XLA_INFERENCE_ENV = {
    "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    "XLA_CLIENT_MEM_FRACTION": "0.95",
}


from utils.batch_utils import split_into_chunks  # noqa: E402


# ── Pure helper functions (testable without Modal) ────────────────────


def flatten_data_jsons(output_dir, flat_dir):
    """Create a flat directory with symlinks to *_data.json files from subdirs.

    AF3's --input_dir expects *_data.json at the top level, but the data
    pipeline writes them into per-protein subdirectories.  This bridges the gap.
    """
    flat_dir = Path(flat_dir)
    flat_dir.mkdir(parents=True, exist_ok=True)

    for data_json in Path(output_dir).rglob("*_data.json"):
        dst = flat_dir / data_json.name
        try:
            os.symlink(data_json, dst)
        except OSError:
            shutil.copy2(data_json, dst)


def create_flat_input_dir(
    protein_names: list[str],
    msa_output_dir: str,
    flat_dir: str,
) -> list[str]:
    """Create a flat directory with symlinks to *_data.json files.

    Args:
        protein_names: List of protein names to include.
        msa_output_dir: Directory containing per-protein MSA output subdirs.
        flat_dir: Path to create the flat directory at.

    Returns:
        List of protein names that were successfully linked/copied.
    """
    flat_path = Path(flat_dir)
    flat_path.mkdir(parents=True, exist_ok=True)

    created = []
    msa_path = Path(msa_output_dir)

    for name in protein_names:
        protein_dir = msa_path / name
        data_jsons = list(protein_dir.glob("*_data.json")) if protein_dir.exists() else []

        if not data_jsons:
            print(f"Warning: no data JSON found for {name}, skipping")
            continue

        src = data_jsons[0]
        dst = flat_path / src.name

        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)

        created.append(name)

    return created


def write_batch_to_dict(
    protein_names: list[str],
    msa_output_dir: str,
    data_dict: dict,
) -> list[str]:
    """Read *_data.json files and store contents in Dict.

    Args:
        protein_names: List of protein names to read.
        msa_output_dir: Directory containing per-protein MSA output subdirs.
        data_dict: modal.Dict or plain dict for testing.

    Returns:
        List of protein names successfully written to dict.
    """
    written = []
    msa_path = Path(msa_output_dir)

    for name in protein_names:
        protein_dir = msa_path / name
        data_jsons = list(protein_dir.glob("*_data.json")) if protein_dir.exists() else []

        if not data_jsons:
            print(f"Warning: no data JSON found for {name}, skipping")
            continue

        content = data_jsons[0].read_text()
        data_dict[name] = content
        written.append(name)

    return written


def create_flat_input_dir_from_dict(
    protein_names: list[str],
    data_dict: dict,
    flat_dir: str,
) -> tuple[list[str], list[str]]:
    """Read data from Dict, write to local temp files.

    Args:
        protein_names: List of protein names to read from dict.
        data_dict: modal.Dict or plain dict for testing.
        flat_dir: Path to create the flat directory at.

    Returns:
        Tuple of (created_names, fallback_needed_names).
    """
    flat_path = Path(flat_dir)
    flat_path.mkdir(parents=True, exist_ok=True)

    created = []
    fallback_needed = []

    for name in protein_names:
        try:
            content = data_dict.pop(name)
        except KeyError:
            fallback_needed.append(name)
            continue

        dst = flat_path / f"{name}_data.json"
        dst.write_text(content)
        created.append(name)

    return created, fallback_needed


def collect_chunk_results(
    protein_names: list[str],
    output_dir: str,
) -> list[dict]:
    """Collect per-protein results from a chunk inference output directory.

    Args:
        protein_names: Names of proteins that were in this chunk.
        output_dir: Directory where inference wrote per-protein subdirs.

    Returns:
        List of per-protein result dicts with status, name, files.
    """
    results = []
    output_path = Path(output_dir)

    for name in protein_names:
        protein_dir = output_path / name

        if not protein_dir.exists() or not protein_dir.is_dir():
            results.append({
                "name": name,
                "status": "error",
                "error": f"No output directory found for {name}",
                "files": {},
            })
            continue

        output_files = {}
        for fpath in protein_dir.rglob("*"):
            if fpath.is_file():
                rel = str(fpath.relative_to(protein_dir))
                output_files[rel] = fpath.stat().st_size

        if output_files:
            results.append({
                "name": name,
                "status": "success",
                "files": output_files,
            })
        else:
            results.append({
                "name": name,
                "status": "error",
                "error": f"Output directory exists but is empty for {name}",
                "files": {},
            })

    return results


def _call_msa_server(inputs: list[dict], server_url: str, batch_size: int = 32) -> dict:
    """Call a remote MSA server via HTTP, batching inputs to avoid timeouts.

    Args:
        inputs: List of AF3 input JSON dicts.
        server_url: URL of the MSA server endpoint.
        batch_size: Batch size for MSA processing.

    Returns:
        Dict mapping protein names to enriched JSON strings.
    """
    import urllib.request
    import urllib.error

    all_data = {}

    # Send in chunks to avoid server timeout / OOM on large batches
    chunk_size = batch_size
    for chunk_start in range(0, len(inputs), chunk_size):
        chunk = inputs[chunk_start:chunk_start + chunk_size]
        chunk_end = min(chunk_start + chunk_size, len(inputs))
        print(f"  MSA server: sending batch {chunk_start + 1}-{chunk_end}/{len(inputs)}")

        payload = json.dumps({
            "inputs": chunk,
            "batch_size": batch_size,
        }).encode("utf-8")

        req = urllib.request.Request(
            server_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=3600 * 24) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            print(f"  MSA server error for batch {chunk_start + 1}-{chunk_end}: {e}")
            continue

        if result.get("status") == "error":
            print(f"  MSA server returned error: {result.get('error', 'Unknown')}")
            continue

        chunk_data = result.get("data", {})
        print(f"  MSA server: got {len(chunk_data)} results")
        all_data.update(chunk_data)

    return all_data


# ── Modal remote functions ────────────────────────────────────────────


@app.function(
    image=image,
    gpu=_CONSUMER_GPU,
    volumes={
        DATABASE_MOUNT_PATH: db_volume,
        WEIGHTS_MOUNT_PATH: weights_volume,
    },
    timeout=86400,  # 24 hours (Modal maximum)
    memory=65536,
)
def predict_structure(
    input_json: dict | list[dict],
    run_msa: bool = True,
    num_seeds: int = 1,
    batch_size: int = 32,
) -> dict | list[dict]:
    """
    Run AlphaFold3 prediction on one or more inputs.

    Args:
        input_json: AlphaFold3 input specification (dict or list of dicts)
        run_msa: Whether to run MSA search (set False if input has pre-computed MSA)
        num_seeds: Number of model seeds for sampling
        batch_size: Number of sequences to batch together for MMseqs2-GPU

    Returns:
        Dictionary with prediction results including structure and confidence scores
    """
    import subprocess
    import tempfile
    import time as _time
    from utils.log_utils import format_subprocess_log

    total_start = _time.time()
    timing = {"msa_seconds": 0.0, "inference_seconds": 0.0, "total_seconds": 0.0}
    msa_log_content = None
    inference_log_content = None

    inputs = input_json if isinstance(input_json, list) else [input_json]
    is_batch = len(inputs) > 1

    work_dir = Path(tempfile.mkdtemp())
    input_dir = work_dir / "inputs"
    input_dir.mkdir()
    output_dir = work_dir / "output"
    output_dir.mkdir()
    inference_output_dir = work_dir / "inference_output"
    inference_output_dir.mkdir()

    input_paths = []
    for i, inp in enumerate(inputs):
        input_path = input_dir / f"input_{i}.json"
        clean_inp = {k: v for k, v in inp.items() if not k.startswith("_")}
        with open(input_path, "w") as f:
            json.dump(clean_inp, f)
        input_paths.append(input_path)

    job_names = [inp.get("name", f"prediction_{i}") for i, inp in enumerate(inputs)]
    print(f"Starting prediction for: {', '.join(job_names[:5])}{'...' if len(job_names) > 5 else ''}")
    if is_batch:
        print(f"Batch mode: {len(inputs)} inputs, batch_size={batch_size}")

    try:
        if run_msa:
            msa_start = _time.time()
            print("Stage 1: Running MSA and template search...")
            if is_batch:
                cmd = [
                    "python", "run_data_pipeline.py",
                    f"--input_dir={input_dir}",
                    f"--output_dir={output_dir}",
                    f"--db_dir={DATABASE_MOUNT_PATH}",
                    f"--mmseqs_db_dir={MMSEQS_DB_PATH}",
                    f"--batch_size={batch_size}",
                    "--use_mmseqs_gpu",
                ]
            else:
                cmd = [
                    "python", "run_data_pipeline.py",
                    f"--json_path={input_paths[0]}",
                    f"--output_dir={output_dir}",
                    f"--db_dir={DATABASE_MOUNT_PATH}",
                    f"--mmseqs_db_dir={MMSEQS_DB_PATH}",
                    "--use_mmseqs_gpu",
                ]
            print(f"Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd="/app/alphafold",
            )

            msa_log_content = format_subprocess_log(
                cmd=cmd, returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr, stage="msa",
            )

            if result.returncode != 0:
                print(f"MSA stderr: {result.stderr}")
                raise RuntimeError(f"Data pipeline failed: {result.stderr[-2000:]}")

            timing["msa_seconds"] = round(_time.time() - msa_start, 2)
            print(f"MSA search complete ({timing['msa_seconds']}s)")

            data_json_files = list(output_dir.rglob("*_data.json"))
            print(f"Found {len(data_json_files)} data JSON files")

        inference_start = _time.time()
        print("Stage 2: Running structure inference...")
        if is_batch:
            if run_msa:
                flat_input_dir = work_dir / "flat_input"
                flatten_data_jsons(output_dir, flat_input_dir)
                data_dir = flat_input_dir
            else:
                data_dir = input_dir
            cmd = [
                "python", "run_alphafold.py",
                f"--input_dir={data_dir}",
                f"--model_dir={WEIGHTS_MOUNT_PATH}",
                f"--output_dir={inference_output_dir}",
                f"--db_dir={DATABASE_MOUNT_PATH}",
                "--norun_data_pipeline",
            ]
        else:
            if run_msa:
                data_json_files = list(output_dir.rglob("*_data.json"))
                inference_input = data_json_files[0] if data_json_files else input_paths[0]
            else:
                inference_input = input_paths[0]
            cmd = [
                "python", "run_alphafold.py",
                f"--json_path={inference_input}",
                f"--model_dir={WEIGHTS_MOUNT_PATH}",
                f"--output_dir={inference_output_dir}",
                f"--db_dir={DATABASE_MOUNT_PATH}",
                "--norun_data_pipeline",
            ]
        print(f"Command: {' '.join(cmd)}")

        inference_env = {**os.environ, **_XLA_INFERENCE_ENV}
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd="/app/alphafold",
            env=inference_env,
        )

        inference_log_content = format_subprocess_log(
            cmd=cmd, returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr, stage="inference",
        )

        if result.stdout:
            print(f"Inference stdout (last 2000 chars):\n{result.stdout[-2000:]}")
        if result.stderr:
            print(f"Inference stderr (last 2000 chars):\n{result.stderr[-2000:]}")

        if result.returncode != 0:
            raise RuntimeError(f"Inference failed (exit code {result.returncode})")

        timing["inference_seconds"] = round(_time.time() - inference_start, 2)
        timing["total_seconds"] = round(_time.time() - total_start, 2)
        print(f"Inference complete ({timing['inference_seconds']}s)")
        print(f"Total compute time: {timing['total_seconds']}s")

        gpu_cost_per_hour = 3.73
        estimated_cost = round(gpu_cost_per_hour * timing["total_seconds"] / 3600, 4)
        print(f"Estimated cost: ${estimated_cost:.4f}")

        all_inference_files = [f for f in inference_output_dir.rglob("*") if f.is_file()]
        print(f"Inference produced {len(all_inference_files)} files:")
        for f in all_inference_files:
            print(f"  {f.relative_to(inference_output_dir)} ({f.stat().st_size} bytes)")

        all_results = []
        for job_name in job_names:
            result_dict = {
                "name": job_name,
                "status": "success",
                "timing": timing,
                "estimated_cost_usd": estimated_cost,
                "files": {},
                "log": {"msa": msa_log_content, "inference": inference_log_content},
            }

            job_output_dir = inference_output_dir / job_name
            if not job_output_dir.exists():
                job_output_dir = inference_output_dir

            for fpath in job_output_dir.rglob("*"):
                if not fpath.is_file():
                    continue
                rel_name = str(fpath.relative_to(job_output_dir))
                is_root_level = fpath.parent == job_output_dir
                suffix = fpath.suffix.lower()

                if is_root_level and suffix == ".cif":
                    with open(fpath) as f:
                        result_dict["structure"] = f.read()
                    result_dict["structure_filename"] = fpath.name
                elif is_root_level and suffix == ".json" and "confidence" in fpath.name:
                    with open(fpath) as f:
                        result_dict["confidence"] = json.load(f)
                elif is_root_level and suffix == ".json" and "summary" in fpath.name:
                    with open(fpath) as f:
                        result_dict["summary"] = json.load(f)
                elif suffix == ".json":
                    with open(fpath) as f:
                        result_dict["files"][rel_name] = json.load(f)
                elif suffix == ".cif":
                    with open(fpath) as f:
                        result_dict["files"][rel_name] = f.read()
                else:
                    size = fpath.stat().st_size
                    if size < 10_000_000:
                        try:
                            with open(fpath) as f:
                                result_dict["files"][rel_name] = f.read()
                        except UnicodeDecodeError:
                            result_dict["files"][rel_name] = f"<binary, {size} bytes>"

            if "structure" not in result_dict:
                print(f"WARNING: No CIF file found for {job_name}")

            all_results.append(result_dict)

        return all_results if is_batch else all_results[0]

    except Exception as e:
        log = {"msa": msa_log_content, "inference": inference_log_content}
        if is_batch:
            return [
                {
                    "name": name,
                    "status": "error",
                    "error": str(e),
                    "timing": {"total_seconds": round(_time.time() - total_start, 2)},
                    "log": log,
                }
                for name in job_names
            ]
        return {
            "name": job_names[0] if job_names else "unknown",
            "status": "error",
            "error": str(e),
            "timing": {"total_seconds": round(_time.time() - total_start, 2)},
            "log": log,
        }

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.function(
    image=image,
    gpu=_CONSUMER_GPU,
    volumes={
        DATABASE_MOUNT_PATH: db_volume,
        WEIGHTS_MOUNT_PATH: weights_volume,
    },
    timeout=3600 * 2,
    memory=32768,
)
def run_msa_only(input_json: dict) -> dict:
    """Run only the MSA/data pipeline stage."""
    import subprocess
    import tempfile

    work_dir = Path(tempfile.mkdtemp())
    input_path = work_dir / "input.json"
    output_dir = work_dir / "output"
    output_dir.mkdir()

    with open(input_path, "w") as f:
        json.dump(input_json, f)

    job_name = input_json.get("name", "msa_job")
    print(f"Running MSA for: {job_name}")

    try:
        cmd = [
            "python", "run_data_pipeline.py",
            f"--json_path={input_path}",
            f"--output_dir={output_dir}",
            f"--db_dir={DATABASE_MOUNT_PATH}",
            f"--mmseqs_db_dir={MMSEQS_DB_PATH}",
            "--use_mmseqs_gpu",
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd="/app/alphafold",
        )

        if result.returncode != 0:
            raise RuntimeError(f"MSA failed: {result.stderr[-2000:]}")

        data_json_files = list(output_dir.rglob("*_data.json"))
        if not data_json_files:
            raise RuntimeError("No data JSON output found")

        with open(data_json_files[0]) as f:
            enriched_data = json.load(f)

        return {"name": job_name, "status": "success", "data": enriched_data}

    except Exception as e:
        return {"name": job_name, "status": "error", "error": str(e)}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ── Warm producer-consumer functions ──────────────────────────────────


@app.function(
    image=image,
    gpu=_PRODUCER_GPU,
    volumes={
        DATABASE_MOUNT_PATH: db_volume,
        WORKSPACE_MOUNT_PATH: workspace_volume,
    },
    timeout=PRODUCER_TIMEOUT,
    memory=32768,
)
def run_producer(
    job_id: str,
    batch_size: int = 32,
    protein_names: list[str] | None = None,
    msa_server_url: str | None = None,
) -> dict:
    """Run batch MSA/data pipeline on producer GPU.

    Reads input JSONs from workspace volume, runs run_data_pipeline.py,
    and writes enriched *_data.json files back to the workspace volume.
    """
    import subprocess
    import tempfile
    from utils.log_utils import save_subprocess_log
    from utils.batch_utils import filter_input_jsons, build_workspace_dirs

    start = time.time()
    dirs = build_workspace_dirs(WORKSPACE_MOUNT_PATH, job_id)

    input_dir = Path(dirs["inputs"])
    msa_output_dir = Path(dirs["msa_outputs"])
    msa_output_dir.mkdir(parents=True, exist_ok=True)

    input_jsons = sorted(input_dir.glob("*.json"))
    input_jsons = filter_input_jsons(input_jsons, protein_names)
    if not input_jsons:
        return {
            "status": "error",
            "error": "No input JSON files found in workspace",
            "data_json_names": [],
            "timing": {"total_seconds": round(time.time() - start, 2)},
            "failures": [],
        }

    print(f"Producer: processing {len(input_jsons)} inputs with batch_size={batch_size}")

    work_dir = Path(tempfile.mkdtemp())
    dp_output = work_dir / "dp_output"
    dp_output.mkdir()

    if protein_names is not None:
        filtered_input_dir = work_dir / "filtered_inputs"
        filtered_input_dir.mkdir()
        for src in input_jsons:
            shutil.copy2(src, filtered_input_dir / src.name)
        effective_input_dir = filtered_input_dir
    else:
        effective_input_dir = input_dir

    try:
        if msa_server_url:
            # Remote MSA via HTTP
            inputs_data = []
            input_sources = []
            for src in input_jsons:
                with open(src) as f:
                    inputs_data.append(json.load(f))
                input_sources.append(src)
            enriched = _call_msa_server(inputs_data, msa_server_url, batch_size)
            data_json_names = []
            failures = []
            for inp, inp_src in zip(inputs_data, input_sources):
                name = inp.get("name", Path(inp_src).stem)
                if name in enriched:
                    dest_dir = msa_output_dir / name
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = dest_dir / f"{name}_data.json"
                    if isinstance(enriched[name], str):
                        dest_file.write_text(enriched[name])
                    else:
                        dest_file.write_text(json.dumps(enriched[name]))
                    data_json_names.append(name)
                else:
                    failures.append(name)
        else:
            cmd = [
                "python", "run_data_pipeline.py",
                f"--input_dir={effective_input_dir}",
                f"--output_dir={dp_output}",
                f"--db_dir={DATABASE_MOUNT_PATH}",
                f"--mmseqs_db_dir={MMSEQS_DB_PATH}",
                f"--batch_size={batch_size}",
                "--use_mmseqs_gpu",
            ]
            print(f"Command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd="/app/alphafold",
            )

            save_subprocess_log(
                log_dir=str(msa_output_dir), name="producer",
                stage="msa", cmd=cmd, returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr,
            )

            if result.returncode != 0:
                print(f"Producer stderr: {result.stderr[-2000:]}")
                return {
                    "status": "error",
                    "error": f"Data pipeline failed: {result.stderr[-2000:]}",
                    "data_json_names": [],
                    "timing": {"total_seconds": round(time.time() - start, 2)},
                    "failures": [],
                }

            data_json_files = list(dp_output.rglob("*_data.json"))
            data_json_names = []
            failures = []

            for djf in data_json_files:
                protein_name = djf.parent.name if djf.parent != dp_output else djf.stem.replace("_data", "")
                dest_dir = msa_output_dir / protein_name
                dest_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(djf, dest_dir / djf.name)
                data_json_names.append(protein_name)

            input_names = {p.stem for p in input_jsons}
            produced_names = set(data_json_names)
            for missing in input_names - produced_names:
                failures.append(missing)

        # HOT PATH: Write to Dict + signal Queue
        written = write_batch_to_dict(data_json_names, str(dp_output if not msa_server_url else msa_output_dir), data_dict)
        data_queue.put(written)

        # COLD PATH: Copy to volume for persistence
        workspace_volume.commit()

        elapsed = round(time.time() - start, 2)
        print(f"Producer complete: {len(data_json_names)} succeeded, {len(failures)} failed ({elapsed}s)")

        return {
            "status": "success" if data_json_names else "error",
            "data_json_names": data_json_names,
            "timing": {"total_seconds": elapsed},
            "failures": failures,
        }

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.cls(
    image=image,
    gpu=_CONSUMER_GPU,
    volumes={
        DATABASE_MOUNT_PATH: db_volume,
        WEIGHTS_MOUNT_PATH: weights_volume,
        WORKSPACE_MOUNT_PATH: workspace_volume,
    },
    timeout=WARM_CONSUMER_TIMEOUT,
    memory=65536,
    scaledown_window=600,
)
class InferenceWorker:
    """Warm consumer that loads model once and processes a chunk of proteins."""

    @modal.enter()
    def startup(self):
        """Runs once on container start — verify GPU/CUDA readiness."""
        import subprocess as sp

        result = sp.run(
            ["python", "-c", "import jax; print('GPU:', jax.devices())"],
            capture_output=True,
            text=True,
        )
        print(f"InferenceWorker ready: {result.stdout.strip()}")

    @modal.method()
    def warmup(self) -> str:
        """No-op method to trigger container creation and @modal.enter()."""
        return "warm"

    @modal.method()
    def process_chunk(self, job_id: str, protein_names: list[str]) -> list[dict]:
        """Process a chunk of proteins via --input_dir.

        Creates a flat temp dir with symlinks to data JSONs, then runs
        run_alphafold.py --input_dir on the whole chunk. Model is loaded
        once for the entire chunk.
        """
        import subprocess
        import tempfile
        from utils.log_utils import save_subprocess_log
        from utils.batch_utils import build_workspace_dirs

        start = time.time()
        dirs = build_workspace_dirs(WORKSPACE_MOUNT_PATH, job_id)
        msa_output_dir = dirs["msa_outputs"]

        work_dir = Path(tempfile.mkdtemp())
        flat_input_dir = work_dir / "flat_input"
        inference_output = work_dir / "inference_output"
        inference_output.mkdir()

        # HOT PATH: Read data from Dict
        created, fallback_needed = create_flat_input_dir_from_dict(
            protein_names=protein_names,
            data_dict=data_dict,
            flat_dir=str(flat_input_dir),
        )

        # COLD PATH: Fallback to Volume
        if fallback_needed:
            print(f"  Dict miss for {len(fallback_needed)} proteins, falling back to Volume")
            workspace_volume.reload()
            fallback_created = create_flat_input_dir(
                protein_names=fallback_needed,
                msa_output_dir=msa_output_dir,
                flat_dir=str(flat_input_dir),
            )
            created.extend(fallback_created)

        if not created:
            return [{
                "name": name,
                "status": "error",
                "error": "No data JSON found after MSA",
                "timing": {"total_seconds": round(time.time() - start, 2)},
            } for name in protein_names]

        print(f"InferenceWorker: processing chunk of {len(created)} proteins")

        try:
            cmd = [
                "python", "run_alphafold.py",
                f"--input_dir={flat_input_dir}",
                f"--model_dir={WEIGHTS_MOUNT_PATH}",
                f"--output_dir={inference_output}",
                f"--db_dir={DATABASE_MOUNT_PATH}",
                "--norun_data_pipeline",
            ]
            print(f"Command: {' '.join(cmd)}")

            inference_env = {**os.environ, **_XLA_INFERENCE_ENV}
            result = subprocess.run(
                cmd, capture_output=True, text=True, cwd="/app/alphafold",
                env=inference_env,
            )

            if result.stdout:
                print(f"stdout (last 2000):\n{result.stdout[-2000:]}")
            if result.stderr:
                print(f"stderr (last 2000):\n{result.stderr[-2000:]}")

            results_dir = Path(dirs["results"])
            results_dir.mkdir(parents=True, exist_ok=True)
            save_subprocess_log(
                log_dir=str(results_dir), name="chunk",
                stage="inference", cmd=cmd, returncode=result.returncode,
                stdout=result.stdout, stderr=result.stderr,
            )

            results = collect_chunk_results(
                protein_names=created,
                output_dir=str(inference_output),
            )

            elapsed = round(time.time() - start, 2)
            for r in results:
                r["timing"] = {"total_seconds": elapsed}

            results_dir = Path(dirs["results"])
            results_dir.mkdir(parents=True, exist_ok=True)

            for r in results:
                if r["status"] == "success":
                    src_dir = inference_output / r["name"]
                    dst_dir = results_dir / r["name"]
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    for fpath in src_dir.rglob("*"):
                        if fpath.is_file():
                            rel = fpath.relative_to(src_dir)
                            dest = dst_dir / rel
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(fpath, dest)

            workspace_volume.commit()

            missing = set(protein_names) - set(created)
            for name in missing:
                results.append({
                    "name": name,
                    "status": "error",
                    "error": "No data JSON found after MSA",
                    "timing": {"total_seconds": elapsed},
                })

            print(f"InferenceWorker: chunk done ({elapsed}s)")
            return results

        except Exception as e:
            elapsed = round(time.time() - start, 2)
            return [{
                "name": name,
                "status": "error",
                "error": str(e),
                "timing": {"total_seconds": elapsed},
            } for name in protein_names]

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


# ── Setup check ───────────────────────────────────────────────────────


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={
        DATABASE_MOUNT_PATH: db_volume,
        WEIGHTS_MOUNT_PATH: weights_volume,
    },
    timeout=300,
)
def check_setup() -> dict:
    """Verify that databases and weights are properly configured."""
    status = {
        "databases": {"ready": False, "details": {}},
        "weights": {"ready": False, "details": {}},
    }

    db_path = Path(DATABASE_MOUNT_PATH)
    mmseqs_path = db_path / "mmseqs"

    # Raw FASTAs are deleted after MMseqs conversion to save volume space.
    # Only the MMseqs padded databases are required for GPU predictions.
    required_mmseqs = [
        "uniref90_padded.dbtype",
        "mgnify_padded.dbtype",
        "small_bfd_padded.dbtype",
    ]

    db_ok = True
    for db in required_mmseqs:
        exists = (mmseqs_path / db).exists()
        status["databases"]["details"][f"mmseqs/{db}"] = exists
        if not exists:
            db_ok = False

    status["databases"]["ready"] = db_ok

    weights_path = Path(WEIGHTS_MOUNT_PATH)
    model_files = list(weights_path.glob("*.bin")) + list(weights_path.glob("**/*.bin"))

    if model_files:
        status["weights"]["ready"] = True
        status["weights"]["details"]["files"] = [f.name for f in model_files[:5]]
    else:
        status["weights"]["details"]["files"] = []

    print("=" * 60)
    print("Setup Status Check")
    print("=" * 60)
    print()
    print(f"Databases: {'READY' if status['databases']['ready'] else 'NOT READY'}")
    for name, exists in status["databases"]["details"].items():
        print(f"  {'[OK]' if exists else '[MISSING]'} {name}")
    print()
    print(f"Weights: {'READY' if status['weights']['ready'] else 'NOT READY'}")
    if status["weights"]["details"]["files"]:
        for f in status["weights"]["details"]["files"]:
            print(f"  [OK] {f}")
    else:
        print("  [MISSING] No model files found")
    print()
    if status["databases"]["ready"] and status["weights"]["ready"]:
        print("Status: Ready to run predictions!")
    else:
        print("Status: Setup incomplete")
        if not status["databases"]["ready"]:
            print("  Run: modal run modal/prepare_databases.py")
        if not status["weights"]["ready"]:
            print("  Run: modal run modal/upload_weights.py --file <weights>")
    print("=" * 60)

    return status


# ── Local helpers ─────────────────────────────────────────────────────


def save_results_local(results: list[dict] | dict, output_dir: str):
    """Save prediction results to local filesystem."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if isinstance(results, dict):
        results = [results]

    for result in results:
        name = result.get("name", "prediction")
        job_dir = output_path / name
        job_dir.mkdir(exist_ok=True)

        if "structure" in result:
            filename = result.get("structure_filename", f"{name}_model.cif")
            with open(job_dir / filename, "w") as f:
                f.write(result["structure"])
            print(f"  Saved: {job_dir / filename}")
        else:
            print(f"  WARNING: No structure file for {name}")

        if "confidence" in result:
            with open(job_dir / f"{name}_confidence.json", "w") as f:
                json.dump(result["confidence"], f, indent=2)
            print(f"  Saved: {job_dir / f'{name}_confidence.json'}")

        if "summary" in result:
            with open(job_dir / f"{name}_summary.json", "w") as f:
                json.dump(result["summary"], f, indent=2)
            print(f"  Saved: {job_dir / f'{name}_summary.json'}")

        for rel_name, content in result.get("files", {}).items():
            file_path = job_dir / rel_name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(content, str):
                with open(file_path, "w") as f:
                    f.write(content)
            else:
                with open(file_path, "w") as f:
                    json.dump(content, f, indent=2)
            print(f"  Saved: {file_path}")

        log_data = result.get("log", {})
        if log_data:
            for stage, content in log_data.items():
                if content is not None:
                    log_file = job_dir / f"{name}_{stage}.log"
                    with open(log_file, "w") as f:
                        f.write(content)
                    print(f"  Saved: {log_file}")

        result_copy = {k: v for k, v in result.items() if k not in ("structure", "files", "log")}
        with open(job_dir / f"{name}_result.json", "w") as f:
            json.dump(result_copy, f, indent=2)


def _print_timing_summary(results: list[dict] | dict):
    """Print a timing and cost summary table."""
    if isinstance(results, dict):
        results = [results]

    print()
    print("-" * 60)
    print("Timing & Cost Summary")
    print("-" * 60)
    print(f"{'Name':<25} {'MSA':>8} {'Infer':>8} {'Total':>8} {'Cost':>8}")
    print(f"{'':.<25} {'(s)':>8} {'(s)':>8} {'(s)':>8} {'(USD)':>8}")
    print("-" * 60)

    total_cost = 0.0
    total_time = 0.0
    for r in results:
        name = r.get("name", "unknown")[:25]
        timing = r.get("timing", {})
        cost = r.get("estimated_cost_usd", 0.0)
        msa_s = timing.get("msa_seconds", 0.0)
        inf_s = timing.get("inference_seconds", 0.0)
        tot_s = timing.get("total_seconds", 0.0)
        total_cost += cost
        total_time += tot_s
        print(f"{name:<25} {msa_s:>8.1f} {inf_s:>8.1f} {tot_s:>8.1f} ${cost:>7.4f}")

    if len(results) > 1:
        print("-" * 60)
        print(f"{'TOTAL':<25} {'':>8} {'':>8} {total_time:>8.1f} ${total_cost:>7.4f}")

    print("-" * 60)


def _mount_to_volume_path(mount_path: str) -> str:
    """Convert a container mount path to a volume-relative path."""
    if mount_path.startswith(WORKSPACE_MOUNT_PATH):
        return mount_path[len(WORKSPACE_MOUNT_PATH):]
    return mount_path


def _upload_inputs_to_volume(inputs: list[dict], job_id: str):
    """Upload input JSONs to workspace volume via batch_upload."""
    import io
    from utils.batch_utils import build_workspace_dirs

    dirs = build_workspace_dirs(WORKSPACE_MOUNT_PATH, job_id)
    vol_input_dir = _mount_to_volume_path(dirs["inputs"])

    with workspace_volume.batch_upload() as batch:
        for inp in inputs:
            name = inp.get("name", f"protein_{inputs.index(inp)}")
            json_bytes = json.dumps(inp).encode("utf-8")
            remote_path = f"{vol_input_dir}/{name}.json"
            batch.put_file(io.BytesIO(json_bytes), remote_path)


def _download_results_from_volume(job_id: str, output_dir: str, protein_names: list[str]):
    """Download result files from workspace volume to local disk."""
    from utils.batch_utils import build_workspace_dirs

    dirs = build_workspace_dirs(WORKSPACE_MOUNT_PATH, job_id)
    vol_results_dir = _mount_to_volume_path(dirs["results"])
    output_path = Path(output_dir)

    for protein_name in protein_names:
        vol_protein_dir = f"{vol_results_dir}/{protein_name}"
        local_dir = output_path / protein_name
        local_dir.mkdir(parents=True, exist_ok=True)

        try:
            entries = workspace_volume.listdir(vol_protein_dir, recursive=True)
        except Exception:
            print(f"  Warning: no results found for {protein_name}")
            continue

        for entry in entries:
            if entry.type.name == "FILE":
                remote_path = entry.path
                vol_protein_prefix = vol_protein_dir.lstrip("/")
                if remote_path.startswith(vol_protein_prefix):
                    rel = remote_path[len(vol_protein_prefix):].lstrip("/")
                else:
                    rel = Path(remote_path).name
                dest = local_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                data = b"".join(workspace_volume.read_file(remote_path))
                dest.write_bytes(data)
                print(f"  Downloaded: {dest}")


def _run_multi_gpu_pipeline(
    inputs: list[dict],
    output: str,
    num_gpus: int,
    batch_size: int,
    skip_msa: bool,
    msa_server_url: str | None,
):
    """Run batch prediction split across multiple GPUs.

    Each GPU receives a subset of inputs, runs batch MSA with the given
    batch_size, then folds its chunk sequentially via --input_dir.
    """
    from utils.batch_utils import split_into_chunks

    print(f"GPU: {_CONSUMER_GPU}")
    print(f"Num GPUs: {num_gpus}")
    print(f"Batch size (MMseqs2): {batch_size}")
    print(f"Skip MSA: {skip_msa}")
    if msa_server_url:
        print(f"MSA server: {msa_server_url}")
    print()

    total_start = time.time()

    # If using remote MSA server, pre-compute MSA for all inputs first,
    # then distribute enriched inputs across GPUs with skip_msa=True.
    run_msa_locally = not skip_msa and not msa_server_url
    if msa_server_url and not skip_msa:
        print(f"Calling MSA server for {len(inputs)} inputs (batch_size={batch_size})...")
        msa_start = time.time()
        enriched = _call_msa_server(inputs, msa_server_url, batch_size=batch_size)
        msa_elapsed = time.time() - msa_start
        print(f"MSA server returned {len(enriched)}/{len(inputs)} results ({msa_elapsed:.1f}s)")

        enriched_inputs = []
        missing = []
        for inp in inputs:
            name = inp.get("name", "unknown")
            if name in enriched:
                data = enriched[name]
                enriched_inputs.append(json.loads(data) if isinstance(data, str) else data)
            else:
                missing.append(name)
        if missing:
            print(f"  WARNING: MSA server missing {len(missing)} proteins: {missing[:10]}")
        inputs = enriched_inputs
        run_msa_locally = False
        if not inputs:
            print("Error: MSA server returned no results")
            return

    chunks = split_into_chunks(inputs, num_gpus)
    non_empty_chunks = [c for c in chunks if c]

    print(f"Splitting {len(inputs)} inputs across {len(non_empty_chunks)} GPUs")
    for i, chunk in enumerate(non_empty_chunks):
        names = [inp.get("name", "?") for inp in chunk]
        print(f"  GPU {i}: {len(chunk)} proteins — {', '.join(names[:5])}{'...' if len(names) > 5 else ''}")
    print()

    # Spawn all GPU jobs in parallel
    futures = []
    for chunk in non_empty_chunks:
        future = predict_structure.spawn(
            chunk, run_msa=run_msa_locally, batch_size=batch_size
        )
        futures.append((chunk, future))

    # Gather results
    all_results = []
    for gpu_idx, (chunk, future) in enumerate(futures):
        chunk_names = [inp.get("name", "?") for inp in chunk]
        try:
            result = future.get()
            chunk_results = result if isinstance(result, list) else [result]
            all_results.extend(chunk_results)
            ok = sum(1 for r in chunk_results if r.get("status") == "success")
            fail = len(chunk_results) - ok
            print(f"  GPU {gpu_idx}: {ok} succeeded, {fail} failed")
        except Exception as e:
            for name in chunk_names:
                all_results.append({
                    "name": name,
                    "status": "error",
                    "error": str(e),
                    "timing": {"total_seconds": 0},
                })
            print(f"  GPU {gpu_idx}: error ({e})")

    total_seconds = time.time() - total_start
    successes = sum(1 for r in all_results if r.get("status") == "success")
    failures = len(all_results) - successes

    print()
    print(f"Saving {successes} successful results to {output}")
    save_results_local(all_results, output)
    _print_timing_summary(all_results)

    print()
    print(f"Total wall time: {total_seconds:.1f}s ({total_seconds/3600:.2f}h)")


def _run_producer_consumer_pipeline(
    inputs: list[dict],
    output: str,
    batch_size: int,
    num_consumers: int,
    skip_msa: bool,
    keep_workspace: bool,
    msa_server_url: str | None,
):
    """Run the warm producer-consumer pipeline."""
    from utils.batch_utils import (
        estimate_warm_producer_consumer_cost,
        generate_job_id,
        split_into_chunks,
        collect_consumer_results,
        print_batch_summary,
    )

    print(f"Producer GPU: {_PRODUCER_GPU}")
    print(f"Consumer GPU: {_CONSUMER_GPU}")
    print(f"Batch size: {batch_size}")
    print(f"Num consumers: {num_consumers}")
    print(f"Skip MSA: {skip_msa}")
    if msa_server_url:
        print(f"MSA server: {msa_server_url}")
    print()

    estimate = estimate_warm_producer_consumer_cost(
        num_proteins=len(inputs),
        num_consumers=num_consumers,
        batch_size=batch_size,
    )
    print(f"Estimated cost: ${estimate['total_cost_usd']:.2f}")
    print(f"Estimated wall time: {estimate['total_wall_hours']:.1f}h")
    print()

    job_id = generate_job_id()
    print(f"Job ID: {job_id}")

    total_start = time.time()

    data_dict.clear()

    worker = InferenceWorker()

    if not skip_msa:
        print("Uploading inputs to workspace volume...")
        _upload_inputs_to_volume(inputs, job_id)

        protein_names = [inp.get("name", f"protein_{i}") for i, inp in enumerate(inputs)]

        producer_batches = [
            protein_names[i:i + batch_size]
            for i in range(0, len(protein_names), batch_size)
        ]

        print()
        print("-" * 60)
        print(f"Pipelined: {len(producer_batches)} producer batches, "
              f"{num_consumers} warm consumers")
        print("-" * 60)

        warmup_futures = [worker.warmup.spawn() for _ in range(num_consumers)]

        chunk_futures = []
        data_json_names = []
        producer_seconds = 0.0
        producer_failures = []

        for batch_idx, batch_names in enumerate(producer_batches):
            print(f"\n  Producer batch {batch_idx + 1}/{len(producer_batches)}: "
                  f"{len(batch_names)} proteins")

            producer_result = run_producer.remote(
                job_id, batch_size, protein_names=batch_names,
                msa_server_url=msa_server_url,
            )
            producer_seconds += producer_result["timing"]["total_seconds"]

            if batch_idx == 0:
                print("  Waiting for consumer warmups...")
                for wf in warmup_futures:
                    wf.get()
                print(f"  All {num_consumers} consumers warm")

            if producer_result["status"] == "error":
                print(f"  Producer batch {batch_idx + 1} failed: "
                      f"{producer_result.get('error', 'Unknown')}")
                producer_failures.extend(
                    producer_result.get("failures", batch_names)
                )
                continue

            batch_data = producer_result["data_json_names"]
            data_json_names.extend(batch_data)

            if producer_result["failures"]:
                producer_failures.extend(producer_result["failures"])
                print(f"  Partial failures: {producer_result['failures']}")

            consumer_chunks = split_into_chunks(batch_data, num_consumers)
            for chunk in consumer_chunks:
                if chunk:
                    future = worker.process_chunk.spawn(job_id, chunk)
                    chunk_futures.append((chunk, future))

            print(f"  Dispatched {len(batch_data)} proteins to consumers")

        if producer_failures:
            print(f"\n  Total producer failures: {producer_failures}")

        if not data_json_names:
            print("  All producer batches failed — nothing to infer.")
            return

        print(f"\n  Producer total: {len(data_json_names)} proteins "
              f"({producer_seconds:.1f}s)")

    else:
        print("Skipping MSA — using pre-computed data")
        data_json_names = [inp.get("name", f"protein_{i}") for i, inp in enumerate(inputs)]
        producer_seconds = 0.0

        warmup_futures = [worker.warmup.spawn() for _ in range(num_consumers)]
        for wf in warmup_futures:
            wf.get()
        print(f"  All {num_consumers} consumers warm")

        chunk_futures = []
        consumer_chunks = split_into_chunks(data_json_names, num_consumers)
        for chunk in consumer_chunks:
            if chunk:
                future = worker.process_chunk.spawn(job_id, chunk)
                chunk_futures.append((chunk, future))

    print()
    print("-" * 60)
    print(f"Gathering results from {len(chunk_futures)} consumer chunks...")
    print("-" * 60)

    all_results = []
    for chunk_idx, (chunk, future) in enumerate(chunk_futures):
        try:
            chunk_results = future.get()
            all_results.extend(chunk_results)
            ok = sum(1 for r in chunk_results if r.get("status") == "success")
            fail = len(chunk_results) - ok
            print(f"  Chunk {chunk_idx}: {ok} succeeded, {fail} failed")
        except Exception as e:
            for name in chunk:
                all_results.append({
                    "name": name,
                    "status": "error",
                    "error": str(e),
                    "timing": {"total_seconds": 0},
                })
            print(f"  Chunk {chunk_idx}: error ({e})")

    summary = collect_consumer_results(all_results)
    total_seconds = time.time() - total_start

    print_batch_summary(summary, producer_seconds=producer_seconds)

    succeeded_names = [r["name"] for r in all_results if r.get("status") == "success"]
    if succeeded_names:
        print()
        print(f"Downloading {len(succeeded_names)} results to {output}...")
        _download_results_from_volume(job_id, output, succeeded_names)

    if not keep_workspace:
        print()
        print(f"Cleaning up workspace job {job_id}...")

    print()
    print(f"Total wall time: {total_seconds:.1f}s ({total_seconds/3600:.2f}h)")


# ── Local entrypoint ──────────────────────────────────────────────────


@app.local_entrypoint()
def main(
    input: str | None = None,
    input_dir: str | None = None,
    output: str = "./af3_output",
    parallel: int = 4,
    batch_size: int = 32,
    skip_msa: bool = False,
    check: bool = False,
    msa_only: bool = False,
    single_gpu: bool = False,
    mode: str = "single",
    gpu: str | None = None,
    producer_gpu: str | None = None,
    num_consumers: int = NUM_CONSUMERS,
    num_gpus: int | None = None,
    keep_workspace: bool = False,
    msa_server: str | None = None,
):
    """
    Run AlphaFold3 structure predictions on Modal.

    Args:
        input: Path to single input JSON file
        input_dir: Path to directory of input JSON files (batch mode)
        output: Output directory for results
        parallel: Number of parallel Modal function invocations for batch mode
        batch_size: Number of sequences to batch together for MMseqs2-GPU
        skip_msa: Skip MSA search (input must have pre-computed MSA data)
        check: Check setup status and exit
        msa_only: Only run MSA, don't run inference
        single_gpu: Process all proteins on one GPU (batch MSA + sequential inference)
        mode: Pipeline mode — 'single' (default), 'multi-gpu', or 'producer-consumer'
        gpu: Consumer/inference GPU type (e.g. h100, l40s, a100)
        producer_gpu: Producer GPU type for P-C mode (e.g. l40s)
        num_consumers: Number of warm consumer containers (P-C mode)
        num_gpus: Number of GPUs for multi-gpu mode (splits inputs across GPUs)
        keep_workspace: Don't clean up workspace volume after P-C completion
        msa_server: URL of remote MSA server (skip local DB for MSA)
    """
    print()
    print("=" * 60)
    print("AlphaFold3 Prediction on Modal")
    print("=" * 60)
    print()

    if check:
        check_setup.remote()
        return

    if input is None and input_dir is None:
        print("Error: Please specify --input or --input-dir")
        print()
        print("Usage:")
        print("  modal run modal/af3_predict.py --input protein.json")
        print("  modal run modal/af3_predict.py --input-dir ./proteins/")
        print("  modal run modal/af3_predict.py --input-dir ./proteins/ --mode producer-consumer")
        print()
        print("To check setup:")
        print("  modal run modal/af3_predict.py --check")
        return

    # Load inputs
    inputs = []
    if input:
        input_path = Path(input).expanduser().resolve()
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}")
            return
        with open(input_path) as f:
            inputs.append(json.load(f))
        print(f"Input: {input_path}")

    elif input_dir:
        input_path = Path(input_dir).expanduser().resolve()
        if not input_path.exists():
            print(f"Error: Input directory not found: {input_path}")
            return
        for json_file in sorted(input_path.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            if "sequences" not in data:
                print(f"Skipping non-input JSON: {json_file.name}")
                continue
            if "name" not in data:
                data["name"] = json_file.stem
            data["_source_file"] = json_file.name
            inputs.append(data)
        print(f"Input directory: {input_path}")
        print(f"Found {len(inputs)} JSON files")

    if not inputs:
        print("Error: No input files found")
        return

    print(f"Output: {output}")
    print(f"GPU: {_CONSUMER_GPU}")
    print(f"MSA: {'Skip' if skip_msa else 'Run'}")
    print(f"Batch size (MMseqs2): {batch_size}")
    if msa_server:
        print(f"MSA server: {msa_server}")

    # Dispatch to appropriate mode
    if mode == "multi-gpu":
        effective_gpus = num_gpus or parallel
        print(f"Mode: Multi-GPU ({effective_gpus} GPUs)")
        print()
        _run_multi_gpu_pipeline(
            inputs=inputs,
            output=output,
            num_gpus=effective_gpus,
            batch_size=batch_size,
            skip_msa=skip_msa,
            msa_server_url=msa_server,
        )
    elif mode == "producer-consumer":
        print(f"Mode: Producer-Consumer (warm pipeline)")
        print()
        _run_producer_consumer_pipeline(
            inputs=inputs,
            output=output,
            batch_size=batch_size,
            num_consumers=num_consumers,
            skip_msa=skip_msa,
            keep_workspace=keep_workspace,
            msa_server_url=msa_server,
        )
    else:
        # Single mode (default)
        if single_gpu:
            print("Mode: Single-GPU (batch MSA + sequential inference)")
        if msa_only:
            print("Mode: MSA only (no inference)")
        print()

        # If --msa-server is provided, call the remote MSA server first,
        # then run inference with pre-computed MSA data (skip local pipeline).
        run_msa_locally = not skip_msa and not msa_server
        if msa_server and not skip_msa and not msa_only:
            print(f"Calling MSA server for {len(inputs)} inputs (batch_size={batch_size})...")
            input_names = [inp.get("name", "unknown") for inp in inputs]
            print(f"  Input names (first 5): {input_names[:5]}")
            msa_start = time.time()
            enriched = _call_msa_server(inputs, msa_server, batch_size=batch_size)
            msa_elapsed = time.time() - msa_start
            print(f"MSA server returned {len(enriched)}/{len(inputs)} results ({msa_elapsed:.1f}s)")
            if enriched:
                returned_names = list(enriched.keys())
                print(f"  Returned names (first 5): {returned_names[:5]}")
            print()

            # Replace inputs with enriched data from MSA server
            enriched_inputs = []
            missing = []
            for inp in inputs:
                name = inp.get("name", "unknown")
                if name in enriched:
                    data = enriched[name]
                    if isinstance(data, str):
                        enriched_inputs.append(json.loads(data))
                    else:
                        enriched_inputs.append(data)
                else:
                    missing.append(name)
            if missing:
                print(f"  WARNING: MSA server missing {len(missing)} proteins")
                print(f"  First 10 missing: {missing[:10]}")
            inputs = enriched_inputs

            if not inputs:
                print("Error: MSA server returned no results")
                return

        if len(inputs) == 1:
            print(f"Running prediction for: {inputs[0].get('name', 'protein')}")
            print()

            if msa_only:
                result = run_msa_only.remote(inputs[0])
            else:
                result = predict_structure.remote(inputs[0], run_msa=run_msa_locally, batch_size=batch_size)

            if result.get("status") == "success":
                print("Prediction successful!")
                save_results_local(result, output)
                _print_timing_summary(result)
            else:
                print(f"Prediction failed: {result.get('error', 'Unknown error')}")
                if "timing" in result:
                    _print_timing_summary(result)

        else:
            if single_gpu:
                print(f"Running single-GPU batch ({len(inputs)} proteins, batch_size={batch_size})")
            else:
                print(f"Running batch prediction ({len(inputs)} proteins, {parallel} parallel)")
            print()

            if msa_only:
                futures = [run_msa_only.spawn(inp) for inp in inputs]
                results = [f.get() for f in futures]
            elif single_gpu:
                print("Sending all inputs to one GPU for batch processing...")
                batch_result = predict_structure.remote(
                    inputs, run_msa=run_msa_locally, batch_size=batch_size
                )
                results = batch_result if isinstance(batch_result, list) else [batch_result]
                for i, r in enumerate(results):
                    status = r.get("status", "unknown")
                    name = r.get("name", f"protein_{i}")
                    print(f"  [{i+1}/{len(results)}] {name}: {status}")
            else:
                futures = []
                for inp in inputs:
                    future = predict_structure.spawn(inp, run_msa=run_msa_locally, batch_size=batch_size)
                    futures.append(future)

                results = []
                for i, future in enumerate(futures):
                    try:
                        result = future.get()
                        results.append(result)
                        status = result.get("status", "unknown")
                        name = result.get("name", f"protein_{i}")
                        print(f"  [{i+1}/{len(inputs)}] {name}: {status}")
                    except Exception as e:
                        results.append({
                            "name": inputs[i].get("name", f"protein_{i}"),
                            "status": "error",
                            "error": str(e),
                        })
                        print(f"  [{i+1}/{len(inputs)}] Error: {e}")

            successes = sum(1 for r in results if r.get("status") == "success")
            print(f"\nSaving {successes} successful results to {output}")
            save_results_local(results, output)
            _print_timing_summary(results)

    print()
    print("=" * 60)
    print("Done!")
    print("=" * 60)
