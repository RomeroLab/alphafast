# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/
"""Subprocess logging helpers for AF3 Modal pipelines.

Stdlib-only — no Modal dependency.
"""

from datetime import datetime, timezone
from pathlib import Path


def format_subprocess_log(
    cmd: list[str],
    returncode: int,
    stdout: str | None,
    stderr: str | None,
    stage: str,
) -> str:
    """Format subprocess output into a structured log string.

    Args:
        cmd: Command that was run.
        returncode: Process exit code.
        stdout: Captured stdout (may be None).
        stderr: Captured stderr (may be None).
        stage: Pipeline stage name (e.g. "msa", "inference").

    Returns:
        Formatted log string.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    stdout = stdout or ""
    stderr = stderr or ""

    lines = [
        f"Stage: {stage}",
        f"Timestamp: {ts}",
        f"Command: {' '.join(cmd)}",
        f"Return code: {returncode}",
        "",
        "=== STDOUT ===",
        stdout,
        "",
        "=== STDERR ===",
        stderr,
    ]
    return "\n".join(lines)


def save_subprocess_log(
    log_dir: str,
    name: str,
    stage: str,
    cmd: list[str],
    returncode: int,
    stdout: str | None,
    stderr: str | None,
) -> str:
    """Write a subprocess log file to disk.

    Args:
        log_dir: Directory to write the log file.
        name: Protein/job name.
        stage: Pipeline stage name (e.g. "msa", "inference").
        cmd: Command that was run.
        returncode: Process exit code.
        stdout: Captured stdout.
        stderr: Captured stderr.

    Returns:
        Path to the written log file.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    content = format_subprocess_log(cmd, returncode, stdout, stderr, stage)
    filepath = log_path / f"{name}_{stage}.log"
    filepath.write_text(content)

    return str(filepath)
