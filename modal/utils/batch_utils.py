# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Batch processing utilities for Modal deployment."""

from typing import TypeVar, Iterator
import json
import math
import time
import uuid
from pathlib import Path

T = TypeVar("T")


def chunk_inputs(items: list[T], chunk_size: int) -> Iterator[list[T]]:
    """Split a list into chunks of specified size."""
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def gather_results(futures: list, timeout: float | None = None) -> list:
    """Gather results from a list of futures, handling errors gracefully."""
    results = []
    errors = []

    for i, future in enumerate(futures):
        try:
            result = future.get(timeout=timeout) if timeout else future.get()
            results.append({"index": i, "status": "success", "result": result})
        except Exception as e:
            results.append({"index": i, "status": "error", "error": str(e)})
            errors.append((i, str(e)))

    if errors:
        print(f"Warning: {len(errors)} jobs failed:")
        for idx, err in errors[:5]:  # Show first 5 errors
            print(f"  Job {idx}: {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")

    return results


def load_inputs_from_directory(input_dir: str) -> list[dict]:
    """Load all JSON input files from a directory."""
    path = Path(input_dir)
    inputs = []

    for json_file in sorted(path.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
                data["_source_file"] = json_file.name
                inputs.append(data)
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse {json_file}: {e}")

    return inputs


def save_result(result: dict, output_dir: str, name: str | None = None):
    """Save prediction result to output directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Determine output name
    if name is None:
        name = result.get("name", "prediction")

    # Save main output files
    output_subdir = path / name
    output_subdir.mkdir(exist_ok=True)

    # Save structure if present
    if "structure" in result:
        cif_path = output_subdir / f"{name}_model.cif"
        with open(cif_path, "w") as f:
            f.write(result["structure"])

    # Save confidence scores if present
    if "confidence" in result:
        conf_path = output_subdir / f"{name}_confidence.json"
        with open(conf_path, "w") as f:
            json.dump(result["confidence"], f, indent=2)

    # Save full result as JSON
    result_path = output_subdir / f"{name}_result.json"
    # Remove large binary data before saving
    result_copy = {k: v for k, v in result.items() if k != "structure"}
    with open(result_path, "w") as f:
        json.dump(result_copy, f, indent=2)

    return str(output_subdir)


def estimate_producer_consumer_cost(
    num_proteins: int,
    batch_size: int = 32,
    parallel: int = 10,
    avg_msa_minutes_per_batch: float = 15.0,
    avg_inference_minutes: float = 20.0,
    producer_gpu_rate: float = 1.50,
    consumer_gpu_rate: float = 3.80,
) -> dict:
    """Estimate cost for the two-stage producer-consumer batch pipeline.

    Args:
        num_proteins: Number of proteins to process.
        batch_size: Proteins batched per MSA producer call.
        parallel: Max concurrent consumer GPUs.
        avg_msa_minutes_per_batch: Average minutes per MSA batch on producer GPU.
        avg_inference_minutes: Average minutes per protein on consumer GPU.
        producer_gpu_rate: Hourly rate for producer GPU (e.g. L40S).
        consumer_gpu_rate: Hourly rate for consumer GPU (e.g. A100-80GB).

    Returns:
        Dictionary with cost breakdown.
    """
    import math

    num_batches = math.ceil(num_proteins / batch_size) if num_proteins > 0 else 0
    producer_hours = (num_batches * avg_msa_minutes_per_batch) / 60
    producer_cost = producer_hours * producer_gpu_rate

    # Consumer wall-clock time depends on parallelism
    num_waves = math.ceil(num_proteins / parallel) if num_proteins > 0 else 0
    consumer_wall_hours = (num_waves * avg_inference_minutes) / 60
    # Total GPU-hours is per-protein, not per-wave
    consumer_gpu_hours = (num_proteins * avg_inference_minutes) / 60
    consumer_cost = consumer_gpu_hours * consumer_gpu_rate

    total_cost = producer_cost + consumer_cost

    # Pipelined wall time: first batch runs sequentially, then remaining
    # producer batches overlap with consumer inference.
    # wall = first_batch + max(remaining_producer, all_consumer_wall)
    first_batch_hours = avg_msa_minutes_per_batch / 60 if num_batches > 0 else 0
    remaining_producer_hours = max(0, (num_batches - 1) * avg_msa_minutes_per_batch) / 60
    total_wall_hours = first_batch_hours + max(remaining_producer_hours, consumer_wall_hours)

    return {
        "num_proteins": num_proteins,
        "batch_size": batch_size,
        "parallel": parallel,
        "producer_batches": num_batches,
        "producer_hours": round(producer_hours, 2),
        "producer_cost_usd": round(producer_cost, 2),
        "consumer_gpu_hours": round(consumer_gpu_hours, 2),
        "consumer_wall_hours": round(consumer_wall_hours, 2),
        "consumer_cost_usd": round(consumer_cost, 2),
        "total_cost_usd": round(total_cost, 2),
        "total_wall_hours": round(total_wall_hours, 2),
    }


def estimate_cost(
    num_proteins: int,
    avg_length: int = 300,
    gpu_type: str = "a100",
) -> dict:
    """Estimate cost for a batch of predictions."""
    # Rough estimates based on typical runtimes
    GPU_HOURLY_RATES = {
        "a100": 3.80,  # A100-80GB
        "a100-40gb": 2.78,
        "h100": 4.50,
        "l40s": 1.50,
    }

    # Estimate runtime based on protein length
    # These are rough estimates - actual times vary significantly
    if avg_length < 200:
        base_minutes = 30
    elif avg_length < 500:
        base_minutes = 60
    elif avg_length < 1000:
        base_minutes = 120
    else:
        base_minutes = 240

    hourly_rate = GPU_HOURLY_RATES.get(gpu_type, 3.80)
    total_hours = (base_minutes * num_proteins) / 60

    return {
        "num_proteins": num_proteins,
        "avg_length": avg_length,
        "gpu_type": gpu_type,
        "estimated_hours": round(total_hours, 2),
        "estimated_cost_usd": round(total_hours * hourly_rate, 2),
        "hourly_rate": hourly_rate,
        "note": "Estimates are rough - actual costs may vary significantly",
    }


def estimate_warm_producer_consumer_cost(
    num_proteins: int,
    num_consumers: int = 4,
    batch_size: int = 32,
    avg_msa_minutes_per_batch: float = 15.0,
    avg_inference_minutes_per_protein: float = 0.5,
    warmup_minutes: float = 3.0,
    producer_gpu_rate: float = 1.50,
    consumer_gpu_rate: float = 3.80,
) -> dict:
    """Estimate cost for the pipelined warm producer-consumer batch pipeline.

    Producer runs MSA in batches; after each batch, results are dispatched to
    warm consumers immediately. This overlaps MSA and inference:

        wall = max(first_batch, warmup) + max(remaining_producer, consumer_inference)

    Args:
        num_proteins: Number of proteins to process.
        num_consumers: Number of warm consumer containers.
        batch_size: Proteins batched per MSA producer call.
        avg_msa_minutes_per_batch: Average minutes per MSA batch on producer GPU.
        avg_inference_minutes_per_protein: Minutes per protein on a warm consumer.
        warmup_minutes: Minutes for each consumer to warm up (overlaps with first batch).
        producer_gpu_rate: Hourly rate for producer GPU.
        consumer_gpu_rate: Hourly rate for consumer GPU.

    Returns:
        Dictionary with cost breakdown.
    """
    if num_proteins == 0:
        return {
            "num_proteins": 0,
            "num_consumers": num_consumers,
            "batch_size": batch_size,
            "producer_batches": 0,
            "producer_hours": 0.0,
            "producer_cost_usd": 0.0,
            "consumer_gpu_hours": 0.0,
            "consumer_wall_hours": 0.0,
            "consumer_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "total_wall_hours": 0.0,
        }

    # Producer cost: batched MSA
    num_batches = math.ceil(num_proteins / batch_size)
    producer_total_minutes = num_batches * avg_msa_minutes_per_batch
    producer_hours = producer_total_minutes / 60
    producer_cost = producer_hours * producer_gpu_rate

    # Consumer inference: proteins split across N consumers
    effective_consumers = min(num_consumers, num_proteins)
    proteins_per_consumer = math.ceil(num_proteins / effective_consumers)
    consumer_inference_minutes = proteins_per_consumer * avg_inference_minutes_per_protein
    consumer_wall_hours = consumer_inference_minutes / 60

    # Consumer GPU-hours: each consumer runs for (warmup + inference) time
    per_consumer_hours = (warmup_minutes + consumer_inference_minutes) / 60
    consumer_gpu_hours = effective_consumers * per_consumer_hours
    consumer_cost = consumer_gpu_hours * consumer_gpu_rate

    total_cost = producer_cost + consumer_cost

    # Pipelined wall time:
    # - First MSA batch and warmup run in parallel
    # - Remaining MSA batches overlap with consumer inference
    # wall = max(first_batch, warmup) + max(remaining_producer, consumer_inference)
    first_batch_minutes = avg_msa_minutes_per_batch
    remaining_producer_minutes = max(0, (num_batches - 1)) * avg_msa_minutes_per_batch
    wall_minutes = (
        max(first_batch_minutes, warmup_minutes)
        + max(remaining_producer_minutes, consumer_inference_minutes)
    )
    wall_hours = wall_minutes / 60

    return {
        "num_proteins": num_proteins,
        "num_consumers": num_consumers,
        "batch_size": batch_size,
        "producer_batches": num_batches,
        "producer_hours": round(producer_hours, 2),
        "producer_cost_usd": round(producer_cost, 2),
        "consumer_gpu_hours": round(consumer_gpu_hours, 2),
        "consumer_wall_hours": round(consumer_wall_hours, 2),
        "consumer_cost_usd": round(consumer_cost, 2),
        "total_cost_usd": round(total_cost, 2),
        "total_wall_hours": round(wall_hours, 2),
    }


# ── Shared helpers (no Modal dependency) ──────────────────────────────


def split_into_chunks(items: list, num_chunks: int) -> list[list]:
    """Split items into num_chunks roughly equal chunks.

    Uses round-robin distribution so chunk sizes differ by at most 1.

    Args:
        items: List of items to split.
        num_chunks: Number of chunks to create.

    Returns:
        List of num_chunks lists (some may be empty if len(items) < num_chunks).
    """
    if num_chunks <= 0:
        return [list(items)]
    chunks = [[] for _ in range(num_chunks)]
    for i, item in enumerate(items):
        chunks[i % num_chunks].append(item)
    return chunks


def generate_job_id() -> str:
    """Generate a unique, timestamp-prefixed job ID."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{ts}_{short_uuid}"


def build_workspace_dirs(workspace_root: str, job_id: str) -> dict[str, str]:
    """Return workspace subdirectory paths for a given job.

    Args:
        workspace_root: Root mount path of the workspace volume.
        job_id: Unique identifier for this batch job.

    Returns:
        Dict with keys 'inputs', 'msa_outputs', 'results'.
    """
    base = f"{workspace_root}/{job_id}"
    return {
        "inputs": f"{base}/inputs",
        "msa_outputs": f"{base}/msa_outputs",
        "results": f"{base}/results",
    }


def filter_input_jsons(
    input_jsons: list[Path], protein_names: list[str] | None
) -> list[Path]:
    """Filter input JSON paths to only those whose stems match protein_names.

    Args:
        input_jsons: List of Path objects for input JSON files.
        protein_names: Names to keep (matched against file stem). If None,
            return all inputs unchanged.

    Returns:
        Filtered (or original) list of Paths.
    """
    if protein_names is None:
        return input_jsons
    name_set = set(protein_names)
    return [p for p in input_jsons if p.stem in name_set]


def collect_consumer_results(raw_results: list[dict]) -> dict:
    """Aggregate consumer results into a summary.

    Args:
        raw_results: List of per-protein result dicts from consumers.

    Returns:
        Summary dict with counts, successes, failures, and timing.
    """
    succeeded = []
    failures = []

    for r in raw_results:
        if r.get("status") == "success":
            succeeded.append(r)
        else:
            failures.append(r.get("name", "unknown"))

    return {
        "total": len(raw_results),
        "succeeded": len(succeeded),
        "failed": len(failures),
        "failures": failures,
        "results": raw_results,
    }


def print_batch_summary(summary: dict, producer_seconds: float = 0.0):
    """Print a human-readable summary of the batch run."""
    total = summary["total"]
    ok = summary["succeeded"]
    fail = summary["failed"]

    print()
    print("=" * 60)
    print("Batch Summary")
    print("=" * 60)
    print(f"  Total proteins:  {total}")
    print(f"  Succeeded:       {ok}")
    print(f"  Failed:          {fail}")
    if producer_seconds > 0:
        print(f"  Producer (MSA):  {producer_seconds:.1f}s")

    if summary["failures"]:
        print()
        print("  Failed proteins:")
        for name in summary["failures"]:
            print(f"    - {name}")

    successes = [r for r in summary["results"] if r.get("status") == "success"]
    if successes:
        times = [r.get("timing", {}).get("total_seconds", 0) for r in successes]
        if any(t > 0 for t in times):
            print()
            print("  Inference timing:")
            print(f"    Min:  {min(times):.1f}s")
            print(f"    Max:  {max(times):.1f}s")
            print(f"    Mean: {sum(times) / len(times):.1f}s")

    print("=" * 60)
