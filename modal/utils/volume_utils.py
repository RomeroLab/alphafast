# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Volume management utilities for Modal deployment."""

from pathlib import Path


def check_volume_exists(volume_path: str) -> bool:
    """Check if a Modal volume has data."""
    path = Path(volume_path)
    if not path.exists():
        return False
    # Check if directory has any files
    return any(path.iterdir())


def get_volume_size(volume_path: str) -> int:
    """Get total size of files in a volume in bytes."""
    path = Path(volume_path)
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        if f.is_file():
            total += f.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """Format size in bytes to human readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def check_database_completeness(db_path: str) -> dict:
    """Check which databases are present and their sizes."""
    from config import DATABASE_FILES, MMSEQS_CONVERTIBLE_DBS

    path = Path(db_path)
    status = {}

    # Check raw databases (non-convertible ones that must persist)
    for name, filename in DATABASE_FILES.items():
        decompressed = filename.replace(".zst", "")
        if name == "pdb_mmcif":
            check_path = path / "mmcif_files"
            status[name] = {
                "exists": check_path.exists() and any(check_path.iterdir()),
                "path": str(check_path),
            }
        elif name not in MMSEQS_CONVERTIBLE_DBS:
            # Only check non-convertible FASTAs; protein FASTAs are deleted
            # after MMseqs conversion to save volume space.
            check_path = path / decompressed
            status[name] = {
                "exists": check_path.exists(),
                "path": str(check_path),
                "size": check_path.stat().st_size if check_path.exists() else 0,
            }

    # Check MMseqs2 databases (replace raw protein FASTAs)
    mmseqs_path = path / "mmseqs"
    for name in MMSEQS_CONVERTIBLE_DBS:
        padded_db = mmseqs_path / f"{name}_padded.dbtype"
        status[f"mmseqs_{name}"] = {
            "exists": padded_db.exists(),
            "path": str(mmseqs_path / f"{name}_padded"),
        }

    return status


def check_weights_completeness(weights_path: str) -> dict:
    """Check if model weights are present."""
    path = Path(weights_path)
    status = {
        "exists": False,
        "files": [],
    }

    if not path.exists():
        return status

    # Look for model files (af3.bin or similar)
    model_files = list(path.glob("*.bin")) + list(path.glob("*.npz"))
    if model_files:
        status["exists"] = True
        status["files"] = [f.name for f in model_files]

    return status
