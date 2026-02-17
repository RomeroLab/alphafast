# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Queue worker for producer/consumer inference.

By default, loads the AF3 model once at startup and runs inference in-process
for each queued job. Use --legacy_subprocess to fall back to the old behavior
of spawning a new subprocess (and reloading the model) per job.
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
import pathlib
import subprocess
import sys
import time


def _timestamp() -> str:
    return datetime.datetime.now().isoformat()


def _queue_paths(queue_dir: str) -> dict[str, str]:
    return {
        "ready": os.path.join(queue_dir, "ready"),
        "in_progress": os.path.join(queue_dir, "in_progress"),
        "done": os.path.join(queue_dir, "done"),
        "failed": os.path.join(queue_dir, "failed"),
        "producer_done": os.path.join(queue_dir, "producer_done"),
        "summary": os.path.join(queue_dir, "summary.json"),
        "summary_lock": os.path.join(queue_dir, "summary.lock"),
    }


def _ensure_queue_dirs(queue_dir: str) -> None:
    paths = _queue_paths(queue_dir)
    for key in ("ready", "in_progress", "done", "failed"):
        os.makedirs(paths[key], exist_ok=True)


def _claim_token(queue_dir: str) -> str | None:
    paths = _queue_paths(queue_dir)
    try:
        ready_files = sorted(
            f
            for f in os.listdir(paths["ready"])
            if f.endswith(".json") and not f.startswith(".")
        )
    except FileNotFoundError:
        return None

    for filename in ready_files:
        ready_path = os.path.join(paths["ready"], filename)
        in_progress_path = os.path.join(paths["in_progress"], filename)
        try:
            os.replace(ready_path, in_progress_path)
            return in_progress_path
        except FileNotFoundError:
            continue
    return None


def _read_token(path: str) -> dict[str, str]:
    with open(path, "rt") as f:
        return json.load(f)


def _write_token(path: str, payload: dict[str, object]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "wt") as f:
        json.dump(payload, f)
    os.replace(tmp_path, path)


def _append_jsonl(path: str, payload: dict[str, object]) -> None:
    with open(path, "at") as f:
        f.write(json.dumps(payload) + "\n")


def _update_summary(queue_dir: str, record: dict[str, object]) -> None:
    paths = _queue_paths(queue_dir)
    os.makedirs(queue_dir, exist_ok=True)
    with open(paths["summary_lock"], "wt") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if os.path.exists(paths["summary"]):
            with open(paths["summary"], "rt") as f:
                summary = json.load(f)
        else:
            summary = {
                "completed": 0,
                "failed": 0,
                "total_inference_seconds": 0.0,
                "last_updated": None,
            }
        status = record.get("status")
        elapsed = float(record.get("elapsed_seconds", 0.0))
        if status == "success":
            summary["completed"] += 1
            summary["total_inference_seconds"] += elapsed
        else:
            summary["failed"] += 1
        if summary["completed"]:
            summary["per_protein_seconds"] = (
                summary["total_inference_seconds"] / summary["completed"]
            )
        summary["last_updated"] = _timestamp()
        _write_token(paths["summary"], summary)


def _run_inference_subprocess(
    *,
    run_alphafold_path: str,
    data_json_path: str,
    output_dir: str,
    force_output_dir: bool,
    model_dir: str | None,
) -> int:
    """Legacy: spawn a subprocess per job (reloads model each time)."""
    cmd = [
        sys.executable,
        run_alphafold_path,
        f"--json_path={data_json_path}",
        "--norun_data_pipeline",
        f"--output_dir={output_dir}",
    ]
    if model_dir:
        cmd.append(f"--model_dir={model_dir}")
    if force_output_dir:
        cmd.append("--force_output_dir")
    return subprocess.call(cmd)


def _load_model(
    model_dir: str,
    gpu_device: int,
    jax_compilation_cache_dir: str | None,
):
    """Load the AF3 model once. Returns (model_runner, folding_input_module)."""
    import jax
    import tokamax

    if jax_compilation_cache_dir is not None:
        jax.config.update(
            "jax_compilation_cache_dir", jax_compilation_cache_dir
        )

    from alphafold3.model.inference import make_model_config
    from alphafold3.model.inference import ModelRunner

    devices = jax.local_devices(backend="gpu")
    print(
        f"[{_timestamp()}] Found GPUs: {devices}, using device"
        f" {gpu_device}: {devices[gpu_device]}"
    )
    model_runner = ModelRunner(
        config=make_model_config(
            flash_attention_implementation=tokamax.DotProductAttentionImplementation.XLA,
        ),
        device=devices[gpu_device],
        model_dir=pathlib.Path(model_dir),
    )
    # Force model parameter loading now so JIT compile happens upfront.
    print(f"[{_timestamp()}] Loading model parameters...")
    _ = model_runner.model_params
    print(f"[{_timestamp()}] Model loaded successfully.")
    return model_runner


def _run_inference_inprocess(
    model_runner,
    data_json_path: str,
    output_dir: str,
    buckets: tuple[int, ...] | None,
) -> None:
    """Run inference in-process using an already-loaded model."""
    from alphafold3.common import folding_input
    from alphafold3.model.inference import process_fold_input

    fold_input = next(
        folding_input.load_fold_inputs_from_path(pathlib.Path(data_json_path))
    )
    process_fold_input(
        fold_input=fold_input,
        data_pipeline_config=None,
        model_runner=model_runner,
        output_dir=output_dir,
        buckets=buckets,
        force_output_dir=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consume MSA tokens and run AF3 inference."
    )
    parser.add_argument("--queue_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--worker_id", required=True)
    parser.add_argument("--poll_interval", type=float, default=2.0)
    parser.add_argument("--idle_grace_seconds", type=float, default=10.0)
    parser.add_argument("--force_output_dir", action="store_true")
    parser.add_argument("--model_dir", default=None)
    parser.add_argument(
        "--gpu_device", type=int, default=0,
        help="GPU device index for inference.",
    )
    parser.add_argument(
        "--jax_compilation_cache_dir", default=None,
        help="Directory for JAX compilation cache.",
    )
    parser.add_argument(
        "--buckets", nargs="*", type=int, default=None,
        help="Bucket sizes for padding. If omitted, auto-computed.",
    )
    parser.add_argument(
        "--legacy_subprocess", action="store_true",
        help="Use legacy subprocess mode (reloads model per job).",
    )
    # Legacy flag — only required in subprocess mode.
    parser.add_argument("--run_alphafold_path", default=None)
    args = parser.parse_args()

    if args.legacy_subprocess and not args.run_alphafold_path:
        parser.error("--run_alphafold_path is required with --legacy_subprocess")

    _ensure_queue_dirs(args.queue_dir)
    timings_path = os.path.join(
        args.queue_dir, f"timings_worker_{args.worker_id}.jsonl"
    )

    # Load model once at startup (unless legacy mode).
    model_runner = None
    if not args.legacy_subprocess:
        if not args.model_dir:
            parser.error("--model_dir is required for in-process inference")
        print(f"[{_timestamp()}] worker {args.worker_id}: loading model...")
        model_runner = _load_model(
            model_dir=args.model_dir,
            gpu_device=args.gpu_device,
            jax_compilation_cache_dir=args.jax_compilation_cache_dir,
        )
        print(f"[{_timestamp()}] worker {args.worker_id}: model ready")

    buckets = tuple(args.buckets) if args.buckets else None

    idle_start = None
    while True:
        token_path = _claim_token(args.queue_dir)
        if token_path is None:
            producer_done = os.path.exists(
                _queue_paths(args.queue_dir)["producer_done"]
            )
            ready_dir = _queue_paths(args.queue_dir)["ready"]
            ready_empty = True
            try:
                ready_empty = len(os.listdir(ready_dir)) == 0
            except FileNotFoundError:
                ready_empty = True

            if producer_done and ready_empty:
                if idle_start is None:
                    idle_start = time.time()
                elif time.time() - idle_start >= args.idle_grace_seconds:
                    print(f"[{_timestamp()}] worker {args.worker_id}: no work, exiting")
                    break
            else:
                idle_start = None
            time.sleep(args.poll_interval)
            continue

        idle_start = None
        token = _read_token(token_path)
        name = token.get("name")
        data_json_path = token.get("data_json_path")
        if not name or not data_json_path:
            error = "missing name or data_json_path in token"
            print(f"[{_timestamp()}] worker {args.worker_id}: {error}")
            failed_path = os.path.join(
                _queue_paths(args.queue_dir)["failed"], os.path.basename(token_path)
            )
            _write_token(
                failed_path,
                {
                    "status": "failed",
                    "error": error,
                    "worker_id": args.worker_id,
                    "timestamp": _timestamp(),
                },
            )
            continue

        output_dir = os.path.join(args.output_dir, name)
        start_time = time.time()
        print(
            f"[{_timestamp()}] worker {args.worker_id}: running {name} from"
            f" {data_json_path}"
        )
        exit_code = 1
        status = "failed"
        if os.path.exists(data_json_path):
            if args.legacy_subprocess:
                exit_code = _run_inference_subprocess(
                    run_alphafold_path=args.run_alphafold_path,
                    data_json_path=data_json_path,
                    output_dir=output_dir,
                    force_output_dir=args.force_output_dir,
                    model_dir=args.model_dir,
                )
                status = "success" if exit_code == 0 else "failed"
            else:
                try:
                    _run_inference_inprocess(
                        model_runner=model_runner,
                        data_json_path=data_json_path,
                        output_dir=output_dir,
                        buckets=buckets,
                    )
                    exit_code = 0
                    status = "success"
                except Exception as e:
                    print(
                        f"[{_timestamp()}] worker {args.worker_id}:"
                        f" inference failed for {name}: {e}"
                    )
                    exit_code = 1
                    status = "failed"
        else:
            status = "failed"
            exit_code = 2
            print(f"[{_timestamp()}] worker {args.worker_id}: missing {data_json_path}")

        elapsed = time.time() - start_time
        record = {
            "name": name,
            "status": status,
            "exit_code": exit_code,
            "elapsed_seconds": round(elapsed, 3),
            "worker_id": args.worker_id,
            "timestamp": _timestamp(),
        }
        _append_jsonl(timings_path, record)
        _update_summary(args.queue_dir, record)

        dest_dir = _queue_paths(args.queue_dir)[
            "done" if status == "success" else "failed"
        ]
        dest_path = os.path.join(dest_dir, os.path.basename(token_path))
        _write_token(dest_path, {**token, **record})
        try:
            os.remove(token_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
