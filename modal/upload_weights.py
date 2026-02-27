# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Upload AlphaFold3 model weights to Modal Volume.

This script uploads the model parameters to a Modal Volume in the user's
account. You must first request access to the weights from Google DeepMind.

Usage:
    modal run modal/upload_weights.py --file ~/Downloads/af3_params.tar.zst
    modal run modal/upload_weights.py --file ~/Downloads/af3.bin --no-extract

Prerequisites:
    1. Request weights from: https://forms.gle/svvpY4u2jsHEwWYS6
    2. Wait for Google DeepMind approval (2-3 business days)
    3. Download the weights file
"""

import modal
from pathlib import Path

# Configuration
WEIGHTS_VOLUME_NAME = "af3-weights"
WEIGHTS_MOUNT_PATH = "/weights"

app = modal.App("af3-weights-setup")

# Create or get the user's weights volume
weights_volume = modal.Volume.from_name(WEIGHTS_VOLUME_NAME, create_if_missing=True)

# Simple image for file operations
upload_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("zstd", "tar", "pigz")
)


@app.function(
    image=upload_image,
    volumes={WEIGHTS_MOUNT_PATH: weights_volume},
    timeout=1800,  # 30 minutes
)
def extract_weights_on_volume(filename: str):
    """Extract an already-uploaded archive on the volume."""
    import subprocess
    import os
    from pathlib import Path

    weights_path = Path(WEIGHTS_MOUNT_PATH)
    file_path = weights_path / filename

    if not file_path.exists():
        raise RuntimeError(f"File {file_path} not found on volume")

    if filename.endswith(".tar.zst"):
        print("Extracting tar.zst archive...")
        result = subprocess.run(
            ["tar", "--use-compress-program=zstd", "-xf", str(file_path), "-C", str(weights_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error extracting: {result.stderr}")
            raise RuntimeError(f"Failed to extract {filename}")
        os.remove(file_path)
        print("Extraction complete, archive removed")

    elif filename.endswith(".tar.gz") or filename.endswith(".tgz"):
        print("Extracting tar.gz archive...")
        result = subprocess.run(
            ["tar", "-xzf", str(file_path), "-C", str(weights_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error extracting: {result.stderr}")
            raise RuntimeError(f"Failed to extract {filename}")
        os.remove(file_path)
        print("Extraction complete, archive removed")

    elif filename.endswith(".tar"):
        print("Extracting tar archive...")
        result = subprocess.run(
            ["tar", "-xf", str(file_path), "-C", str(weights_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Error extracting: {result.stderr}")
            raise RuntimeError(f"Failed to extract {filename}")
        os.remove(file_path)
        print("Extraction complete, archive removed")

    else:
        print(f"No extraction needed for {filename}")

    # Commit the volume
    print("Committing volume...")
    weights_volume.commit()

    # List contents
    print("\nVolume contents:")
    for f in sorted(weights_path.rglob("*")):
        if f.is_file():
            rel_path = f.relative_to(weights_path)
            size_mb = f.stat().st_size / (1024**2)
            print(f"  {rel_path}: {size_mb:.2f} MB")

    print("\nWeights uploaded successfully!")


@app.function(
    image=upload_image,
    volumes={WEIGHTS_MOUNT_PATH: weights_volume},
    timeout=300,  # 5 minutes
)
def check_weights():
    """Check if model weights are present in the volume."""
    from pathlib import Path

    weights_path = Path(WEIGHTS_MOUNT_PATH)

    print("=" * 60)
    print("Weights Volume Status")
    print("=" * 60)
    print()

    if not weights_path.exists():
        print("Status: Volume is empty")
        print()
        print("To upload weights:")
        print("  modal run modal/upload_weights.py --file ~/Downloads/af3_params.tar.zst")
        return {"exists": False, "files": []}

    files = list(weights_path.rglob("*"))
    if not files:
        print("Status: Volume is empty")
        print()
        print("To upload weights:")
        print("  modal run modal/upload_weights.py --file ~/Downloads/af3_params.tar.zst")
        return {"exists": False, "files": []}

    print("Volume contents:")
    total_size = 0
    file_list = []
    for f in sorted(files):
        if f.is_file():
            rel_path = f.relative_to(weights_path)
            size_mb = f.stat().st_size / (1024**2)
            total_size += f.stat().st_size
            print(f"  {rel_path}: {size_mb:.2f} MB")
            file_list.append(str(rel_path))

    print()
    print(f"Total size: {total_size / (1024**3):.2f} GB")

    # Check for expected model files
    model_files = [f for f in file_list if f.endswith(".bin") or f.endswith(".npz")]
    if model_files:
        print()
        print("Status: Model weights found!")
        print("You can run predictions with:")
        print("  modal run modal/af3_predict.py --input protein.json")
    else:
        print()
        print("Status: No model files (.bin, .npz) found")
        print("The weights may not have been uploaded correctly.")

    return {"exists": bool(model_files), "files": file_list}


@app.function(
    image=upload_image,
    volumes={WEIGHTS_MOUNT_PATH: weights_volume},
    timeout=300,
)
def clear_weights():
    """Clear all weights from the volume (use with caution)."""
    import shutil
    from pathlib import Path

    weights_path = Path(WEIGHTS_MOUNT_PATH)

    if not weights_path.exists():
        print("Volume is already empty")
        return

    # Remove all contents
    for item in weights_path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    weights_volume.commit()
    print("Weights volume cleared")


@app.local_entrypoint()
def main(
    file: str | None = None,
    no_extract: bool = False,
    status: bool = False,
    clear: bool = False,
):
    """
    Upload AlphaFold3 model weights to your Modal Volume.

    Args:
        file: Path to the weights file (e.g., af3_params.tar.zst)
        no_extract: Don't extract archive files, keep as-is
        status: Check current weights status
        clear: Clear the weights volume (use with caution)
    """
    if status:
        check_weights.remote()
        return

    if clear:
        clear_weights.remote()
        return

    if file is None:
        print("Error: Please specify a weights file with --file")
        print()
        print("Usage:")
        print("  modal run modal/upload_weights.py --file ~/Downloads/af3_params.tar.zst")
        print()
        print("To check current status:")
        print("  modal run modal/upload_weights.py --status")
        return

    path = Path(file).expanduser().resolve()

    if not path.exists():
        print(f"Error: File not found: {path}")
        return

    file_size = path.stat().st_size
    print()
    print("=" * 60)
    print("AlphaFold3 Weights Upload")
    print("=" * 60)
    print()
    print(f"File: {path}")
    print(f"Size: {file_size / 1e9:.2f} GB")
    print(f"Extract: {'No' if no_extract else 'Yes'}")
    print()

    # Stream upload directly to volume (avoids loading entire file into memory)
    print(f"Streaming upload to Modal Volume '{WEIGHTS_VOLUME_NAME}'...")
    with weights_volume.batch_upload() as batch:
        batch.put_file(path, f"/{path.name}")
    print("Upload complete")

    # Extract on the remote side if needed
    if not no_extract:
        print("Extracting on remote...")
        extract_weights_on_volume.remote(filename=path.name)
    else:
        print("Skipping extraction (--no-extract)")

    print()
    print("=" * 60)
    print("Upload complete!")
    print("=" * 60)
    print()
    print("You can now run predictions:")
    print("  modal run modal/af3_predict.py --input protein.json")
