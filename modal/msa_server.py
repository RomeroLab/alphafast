# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""
AlphaFold3 MSA Server on Modal.

Persistent web endpoint that runs the MSA/data pipeline with MMseqs2-GPU
acceleration. Deployed separately from the prediction script so that
multiple users or scripts can share a single database instance.

Deploy:
    modal deploy modal/msa_server.py

Usage:
    curl -X POST https://your-app.modal.run/msa \\
        -H "Content-Type: application/json" \\
        -d '{"inputs": [{"name": "2PV7", "sequences": [...]}], "token": "..."}'
"""

import json
import os
import shutil
from pathlib import Path

import modal

from config import (
    DATABASE_MOUNT_PATH,
    DATABASE_VOLUME_NAME,
    MMSEQS_DB_PATH,
)


app = modal.App("alphafold3-msa-server")

db_volume = modal.Volume.from_name(DATABASE_VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.from_dockerfile(
        path="docker/Dockerfile",
        context_dir=".",
    )
    .env({
        "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
    })
    .run_commands("uv pip install 'fastapi[standard]'")
    .add_local_python_source("config")
    .add_local_python_source("utils")
)


def _run_msa_pipeline(inputs: list[dict], batch_size: int = 32) -> dict:
    """Run the MSA/data pipeline on a list of AF3 inputs.

    Args:
        inputs: List of AF3 input JSON dicts.
        batch_size: Batch size for MMseqs2 processing.

    Returns:
        Dict mapping protein names to enriched JSON strings.
    """
    import subprocess
    import tempfile

    work_dir = Path(tempfile.mkdtemp())
    input_dir = work_dir / "inputs"
    input_dir.mkdir()
    output_dir = work_dir / "output"
    output_dir.mkdir()

    for i, inp in enumerate(inputs):
        name = inp.get("name", f"protein_{i}")
        clean_inp = {k: v for k, v in inp.items() if not k.startswith("_")}
        with open(input_dir / f"{name}.json", "w") as f:
            json.dump(clean_inp, f)

    try:
        if len(inputs) == 1:
            input_file = list(input_dir.glob("*.json"))[0]
            cmd = [
                "python", "run_data_pipeline.py",
                f"--json_path={input_file}",
                f"--output_dir={output_dir}",
                f"--db_dir={DATABASE_MOUNT_PATH}",
                f"--mmseqs_db_dir={MMSEQS_DB_PATH}",
                "--use_mmseqs_gpu",
            ]
        else:
            cmd = [
                "python", "run_data_pipeline.py",
                f"--input_dir={input_dir}",
                f"--output_dir={output_dir}",
                f"--db_dir={DATABASE_MOUNT_PATH}",
                f"--mmseqs_db_dir={MMSEQS_DB_PATH}",
                f"--batch_size={batch_size}",
                "--use_mmseqs_gpu",
            ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd="/app/alphafold",
        )

        if result.returncode != 0:
            raise RuntimeError(f"Data pipeline failed: {result.stderr[-2000:]}")

        data = {}
        for data_json in output_dir.rglob("*_data.json"):
            protein_name = data_json.parent.name if data_json.parent != output_dir else data_json.stem.replace("_data", "")
            data[protein_name] = data_json.read_text()

        return data

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.function(
    image=image,
    gpu="L40S",
    volumes={DATABASE_MOUNT_PATH: db_volume},
    timeout=3600 * 24,
    memory=65536,
)
@modal.fastapi_endpoint(method="POST")
def msa(item: dict) -> dict:
    """MSA web endpoint.

    Accepts:
        {
            "inputs": [<AF3 input JSON>, ...],
            "batch_size": 32,       # optional
        }

    Returns:
        {
            "status": "success",
            "data": {"protein_name": "<enriched JSON string>", ...}
        }
    """
    inputs = item.get("inputs", [])
    if not inputs:
        return {"status": "error", "error": "No inputs provided"}

    batch_size = item.get("batch_size", 32)

    try:
        data = _run_msa_pipeline(inputs, batch_size=batch_size)
        return {"status": "success", "data": data}
    except Exception as e:
        return {"status": "error", "error": str(e)}
