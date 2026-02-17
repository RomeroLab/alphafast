# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""
Prepare AlphaFold3 databases on Modal.

Downloads databases from Google Cloud Storage directly to Modal volume,
then converts to MMseqs2 GPU-padded format.

Usage:
    # Full setup (download + convert)
    modal run modal/prepare_databases.py

    # Check current status
    modal run modal/prepare_databases.py --status

    # Only run MMseqs2 conversion (if raw databases exist)
    modal run modal/prepare_databases.py --convert-only
"""

import modal
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database download source URL (Google Cloud Storage)
DATABASE_SOURCE_URL = "https://storage.googleapis.com/alphafold-databases/v3.0"

# Volume configuration
DATABASE_VOLUME_NAME = "af3-databases"
DATABASE_MOUNT_PATH = "/databases"

# Database files to download
# Format: (filename, download_type) where download_type is "tar" or "zstd"
DATABASE_FILES = [
    # PDB mmCIF files (large, downloads as tar)
    ("pdb_2022_09_28_mmcif_files.tar.zst", "tar"),
    # Protein sequence databases
    ("mgy_clusters_2022_05.fa.zst", "zstd"),
    ("bfd-first_non_consensus_sequences.fasta.zst", "zstd"),
    ("uniref90_2022_05.fa.zst", "zstd"),
    ("uniprot_all_2021_04.fa.zst", "zstd"),
    ("pdb_seqres_2022_09_28.fasta.zst", "zstd"),
    # RNA databases
    ("rnacentral_active_seq_id_90_cov_80_linclust.fasta.zst", "zstd"),
    ("nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta.zst", "zstd"),
    ("rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta.zst", "zstd"),
]

# MMseqs2 databases to create (target_name -> source_fasta)
MMSEQS_DATABASES = {
    "uniref90": "uniref90_2022_05.fa",
    "mgnify": "mgy_clusters_2022_05.fa",
    "small_bfd": "bfd-first_non_consensus_sequences.fasta",
    "uniprot": "uniprot_all_2021_04.fa",
    "pdb_seqres": "pdb_seqres_2022_09_28.fasta",
}

# =============================================================================
# MODAL APP SETUP
# =============================================================================

app = modal.App("af3-database-setup")

# Create or get the database volume
db_volume = modal.Volume.from_name(DATABASE_VOLUME_NAME, create_if_missing=True)

# Image for downloading (lightweight)
download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "zstd", "pigz", "tar", "curl")
)

# Image with MMseqs2 for database conversion
mmseqs_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("wget", "pigz", "tar", "zstd")
    .run_commands(
        "wget -q -O mmseqs.tar.gz https://mmseqs.com/latest/mmseqs-linux-gpu.tar.gz",
        "pigz -dc mmseqs.tar.gz | tar xf -",
        "cp mmseqs/bin/mmseqs /usr/local/bin/",
        "rm -rf mmseqs mmseqs.tar.gz",
    )
)


# =============================================================================
# DOWNLOAD DATABASES
# =============================================================================

@app.function(
    image=download_image,
    volumes={DATABASE_MOUNT_PATH: db_volume},
    timeout=3600 * 24,  # 24 hours
    cpu=8,  # More CPUs for parallel downloads
    memory=32768,  # 32GB RAM
)
def download_databases():
    """
    Download AlphaFold3 databases from Google Cloud Storage.

    Downloads in PARALLEL directly from Google servers to Modal volume.
    """
    import subprocess
    import concurrent.futures
    from pathlib import Path

    db_path = Path(DATABASE_MOUNT_PATH)
    db_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading AlphaFold3 Databases (PARALLEL)")
    print("=" * 60)
    print(f"Source: {DATABASE_SOURCE_URL}")
    print(f"Target: {db_path}")
    print(f"Files: {len(DATABASE_FILES)}")
    print("=" * 60)
    print()

    def download_file(args):
        """Download a single file."""
        filename, decompress_type = args
        output_name = filename.replace(".zst", "")

        # Check if already downloaded
        if decompress_type == "tar":
            check_path = db_path / "mmcif_files"
            if check_path.exists() and any(check_path.iterdir()):
                return f"SKIP: mmcif_files/ (exists)"
        else:
            check_path = db_path / output_name
            if check_path.exists() and check_path.stat().st_size > 0:
                return f"SKIP: {output_name} (exists)"

        url = f"{DATABASE_SOURCE_URL}/{filename}"

        if decompress_type == "tar":
            cmd = f"wget -q -O - '{url}' | tar --use-compress-program=zstd -xf - -C '{db_path}'"
        else:
            output_path = db_path / output_name
            cmd = f"wget -q -O - '{url}' | zstd -d > '{output_path}'"

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

        if result.returncode != 0:
            return f"ERROR: {filename} - {result.stderr}"

        return f"DONE: {output_name}"

    # Download all files in parallel
    print("Starting parallel downloads...")
    print()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_file, f): f for f in DATABASE_FILES}

        for future in concurrent.futures.as_completed(futures):
            filename = futures[future][0]
            try:
                result = future.result()
                print(f"  [{filename}] {result}")
            except Exception as e:
                print(f"  [{filename}] EXCEPTION: {e}")

    # Commit the volume
    print()
    print("Committing volume...")
    db_volume.commit()

    print()
    print("=" * 60)
    print("Download Complete!")
    print("=" * 60)


# =============================================================================
# CONVERT TO MMSEQS2
# =============================================================================

# Databases ordered by size (smallest first for quick wins)
# MGnify is processed last as it's the largest (~300M sequences)
MMSEQS_DB_ORDER = ["pdb_seqres", "small_bfd", "uniprot", "uniref90", "mgnify"]


@app.function(
    image=mmseqs_image,
    volumes={DATABASE_MOUNT_PATH: db_volume},
    timeout=3600 * 24,  # 24 hours
    cpu=8,
    memory=65536,  # 64GB RAM (MMseqs2 uses streaming, doesn't need full DB in memory)
    ephemeral_disk=1024 * 1024,  # 1TiB ephemeral disk for /tmp
)
def convert_to_mmseqs():
    """
    Convert FASTA databases to MMseqs2 GPU-padded format.
    """
    import subprocess
    import os
    import shutil
    import sys
    from pathlib import Path

    # Force unbuffered output for real-time logging
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    print("=" * 60, flush=True)
    print("Starting MMseqs2 conversion...", flush=True)
    print("=" * 60, flush=True)

    try:
        db_path = Path(DATABASE_MOUNT_PATH)
        mmseqs_path = db_path / "mmseqs"
        mmseqs_path.mkdir(exist_ok=True)

        # IMPORTANT: Use LOCAL /tmp for MMseqs temp files (much faster than network volume)
        tmp_path = Path("/tmp/mmseqs_tmp")
        tmp_path.mkdir(exist_ok=True)
        os.environ["TMPDIR"] = str(tmp_path)

        # Get number of CPUs available
        num_cpus = os.cpu_count() or 16

        print(f"Source: {db_path}", flush=True)
        print(f"Target: {mmseqs_path}", flush=True)
        print(f"Temp dir: {tmp_path} (local SSD)", flush=True)
        print(f"CPUs: {num_cpus}", flush=True)
        print(f"Databases: {len(MMSEQS_DB_ORDER)}", flush=True)
        print("=" * 60, flush=True)
        print(flush=True)

        # Print MMseqs2 version
        result = subprocess.run(["mmseqs", "version"], capture_output=True, text=True)
        print(f"MMseqs2 version: {result.stdout.strip()}", flush=True)
        print(flush=True)

        # Use local ephemeral disk for building MMseqs databases, then copy to volume.
        # Writing large databases directly to network volume causes "write error" failures.
        local_build_path = Path("/tmp/mmseqs_build")
        local_build_path.mkdir(exist_ok=True)

        # Free volume space: delete raw FASTAs and redundant base sequence
        # files for databases already converted
        print("Checking for reclaimable space from already-converted databases...", flush=True)
        vol_usage = shutil.disk_usage(DATABASE_MOUNT_PATH)
        print(f"  Volume space: {vol_usage.used / (1024**3):.1f} GB used, {vol_usage.free / (1024**3):.1f} GB free", flush=True)
        freed_total = 0
        for db_name, source_fasta in MMSEQS_DATABASES.items():
            # Only consider fully converted databases (all 3 critical files)
            padded_complete = all(
                (mmseqs_path / f"{db_name}_padded{ext}").exists()
                for ext in ("", ".index", ".dbtype")
            )
            if not padded_complete:
                continue
            # Delete raw FASTA
            fasta_path = db_path / source_fasta
            if fasta_path.exists():
                freed_gb = fasta_path.stat().st_size / (1024**3)
                fasta_path.unlink()
                freed_total += freed_gb
                print(f"  Deleted {source_fasta} ({freed_gb:.1f} GB)", flush=True)
            # Delete ALL base DB files (padded version has its own headers/lookup).
            import glob as glob_mod
            for match in glob_mod.glob(str(mmseqs_path / f"{db_name}*")):
                base_file = Path(match)
                if base_file.name.startswith(f"{db_name}_padded"):
                    continue  # keep padded files
                freed_gb = base_file.stat().st_size / (1024**3)
                base_file.unlink()
                freed_total += freed_gb
                print(f"  Deleted {base_file.name} ({freed_gb:.1f} GB)", flush=True)
        if freed_total > 0:
            print(f"  Total freed: {freed_total:.1f} GB", flush=True)
        db_volume.commit()
        vol_usage = shutil.disk_usage(DATABASE_MOUNT_PATH)
        print(f"  Volume space: {vol_usage.free / (1024**3):.1f} GB free", flush=True)
        print(flush=True)

        # Process databases in order (smallest first)
        for i, db_name in enumerate(MMSEQS_DB_ORDER, 1):
            if db_name not in MMSEQS_DATABASES:
                print(f"[{i}/{len(MMSEQS_DB_ORDER)}] {db_name} - not in config, skipping", flush=True)
                continue

            source_fasta = MMSEQS_DATABASES[db_name]
            source_path = db_path / source_fasta

            print(f"[{i}/{len(MMSEQS_DB_ORDER)}] {db_name}", flush=True)
            print(f"  Source: {source_fasta}", flush=True)

            # Check if padded database is fully converted on volume
            # Require ALL three critical files: data, index, and dbtype
            padded_complete = all(
                (mmseqs_path / f"{db_name}_padded{ext}").exists()
                for ext in ("", ".index", ".dbtype")
            )
            if padded_complete:
                print(f"  SKIP: Already converted", flush=True)
                continue

            # Clean up partial padded files from a previous failed run
            partial_freed = 0
            for f in sorted(mmseqs_path.glob(f"{db_name}*")):
                freed = f.stat().st_size / (1024**3)
                f.unlink()
                partial_freed += freed
                print(f"  Removed partial: {f.name} ({freed:.1f} GB)", flush=True)
            if partial_freed > 0:
                db_volume.commit()
                vol_usage = shutil.disk_usage(DATABASE_MOUNT_PATH)
                print(f"  Freed {partial_freed:.1f} GB from previous run, volume: {vol_usage.free / (1024**3):.1f} GB free", flush=True)

            # Re-download source FASTA if missing (e.g. deleted by a previous
            # failed conversion that ran out of space mid-copy)
            if not source_path.exists():
                compressed = source_fasta + ".zst"
                url = f"{DATABASE_SOURCE_URL}/{compressed}"
                print(f"  Source FASTA missing, re-downloading {compressed}...", flush=True)
                result = subprocess.run(
                    f"wget -q -O - '{url}' | zstd -d > '{source_path}'",
                    shell=True, capture_output=True, text=True,
                )
                if result.returncode != 0 or not source_path.exists():
                    print(f"  ERROR: Failed to download {compressed}: {result.stderr}", flush=True)
                    raise RuntimeError(f"Cannot download {compressed}")
                db_volume.commit()
                print(f"  Downloaded {source_fasta} ({source_path.stat().st_size / (1024**3):.1f} GB)", flush=True)

            # Show source file size
            source_size_gb = source_path.stat().st_size / (1024**3)
            print(f"  Size: {source_size_gb:.1f} GB", flush=True)

            # Build on local ephemeral disk (fast local SSD, no network write errors)
            local_base = local_build_path / db_name
            local_padded = local_build_path / f"{db_name}_padded"

            # Step 1: Create MMseqs2 database from FASTA
            tmp_usage = shutil.disk_usage("/tmp")
            print(f"  /tmp space: {tmp_usage.free / (1024**3):.1f} GB free", flush=True)

            print(f"  Step 1/2: createdb on local disk (threads={num_cpus})...", flush=True)

            result = subprocess.run(
                [
                    "mmseqs", "createdb",
                    str(source_path),
                    str(local_base),
                    "--threads", str(num_cpus),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"  ERROR stdout: {result.stdout}", flush=True)
                print(f"  ERROR stderr: {result.stderr}", flush=True)
                tmp_usage = shutil.disk_usage("/tmp")
                print(f"  /tmp space after error: {tmp_usage.free / (1024**3):.1f} GB free", flush=True)
                raise RuntimeError(f"Failed to create database {db_name}: {result.stderr}")
            print(f"  createdb complete", flush=True)

            # Step 2: Create padded database for GPU
            tmp_usage = shutil.disk_usage("/tmp")
            print(f"  /tmp space: {tmp_usage.free / (1024**3):.1f} GB free", flush=True)

            print(f"  Step 2/2: makepaddedseqdb on local disk (threads={num_cpus})...", flush=True)

            result = subprocess.run(
                [
                    "mmseqs", "makepaddedseqdb",
                    str(local_base),
                    str(local_padded),
                    "--threads", str(num_cpus),
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                print(f"  ERROR stdout: {result.stdout}", flush=True)
                print(f"  ERROR stderr: {result.stderr}", flush=True)
                tmp_usage = shutil.disk_usage("/tmp")
                print(f"  /tmp space after error: {tmp_usage.free / (1024**3):.1f} GB free", flush=True)
                raise RuntimeError(f"Failed to create padded database {db_name}: {result.stderr}")

            # Step 3: Free volume space BEFORE copying results
            # Delete the raw FASTA now — createdb already read it.
            # If the copy fails later, the re-download logic above will fetch it again.
            if source_path.exists():
                freed_gb = source_path.stat().st_size / (1024**3)
                source_path.unlink()
                db_volume.commit()
                print(f"  Deleted raw FASTA: {source_fasta} ({freed_gb:.1f} GB freed)", flush=True)

            # Step 4: Copy only padded DB files to volume
            # makepaddedseqdb creates {name}_padded* with its own headers/lookup,
            # so base DB files ({name}_h*, {name}.lookup, etc.) are redundant.
            print(f"  Copying padded DB files to volume...", flush=True)
            import glob as glob_mod
            padded_prefix = f"{db_name}_padded"
            for f in sorted(glob_mod.glob(str(local_build_path / f"{db_name}*"))):
                src = Path(f)
                if not src.name.startswith(padded_prefix):
                    print(f"    SKIP {src.name} ({src.stat().st_size / (1024**3):.1f} GB) - base DB", flush=True)
                    continue
                dst = mmseqs_path / src.name
                src_size = src.stat().st_size
                src_size_gb = src_size / (1024**3)
                if dst.exists() and dst.stat().st_size == src_size:
                    print(f"    EXISTS {src.name} ({src_size_gb:.1f} GB) - already on volume", flush=True)
                    continue
                # Remove partial destination file from a previous failed copy
                if dst.exists():
                    dst.unlink()
                    db_volume.commit()
                shutil.copy2(str(src), str(dst))
                print(f"    {src.name} ({src_size_gb:.1f} GB)", flush=True)
                # Commit after each large file to persist progress and free space
                if src_size_gb > 1.0:
                    db_volume.commit()

            print(f"  Done: {db_name}_padded", flush=True)

            # Commit volume after each database to save progress
            print(f"  Committing volume...", flush=True)
            db_volume.commit()

            # Clean up local build files to free ephemeral disk for next database
            for f in local_build_path.glob(f"{db_name}*"):
                try:
                    f.unlink()
                except Exception:
                    pass

            # Clean up temp files
            for tmp_file in tmp_path.glob("*"):
                try:
                    if tmp_file.is_file():
                        tmp_file.unlink()
                    elif tmp_file.is_dir():
                        shutil.rmtree(tmp_file)
                except Exception:
                    pass

        # Final cleanup
        print(flush=True)
        print("Cleaning up temp files...", flush=True)
        shutil.rmtree(tmp_path, ignore_errors=True)
        shutil.rmtree(local_build_path, ignore_errors=True)

        print(flush=True)
        print("=" * 60, flush=True)
        print("MMseqs2 Conversion Complete!", flush=True)
        print("=" * 60, flush=True)

    except Exception as e:
        import traceback
        print(f"ERROR: {e}", flush=True)
        print(traceback.format_exc(), flush=True)
        raise


# =============================================================================
# STATUS CHECK
# =============================================================================

@app.function(
    image=download_image,
    volumes={DATABASE_MOUNT_PATH: db_volume},
    timeout=300,
)
def check_status():
    """Check database preparation status."""
    from pathlib import Path

    db_path = Path(DATABASE_MOUNT_PATH)

    print("=" * 60)
    print("Database Status")
    print("=" * 60)
    print()

    # Check non-convertible databases (needed at prediction time)
    # Protein FASTAs are deleted after MMseqs conversion to save space.
    required_dbs = [
        ("mmcif_files", "directory"),
        ("rnacentral_active_seq_id_90_cov_80_linclust.fasta", "file"),
        ("nt_rna_2023_02_23_clust_seq_id_90_cov_80_rep_seq.fasta", "file"),
        ("rfam_14_9_clust_seq_id_90_cov_80_rep_seq.fasta", "file"),
    ]

    print("Required Databases:")
    dbs_complete = True
    for name, db_type in required_dbs:
        path = db_path / name
        if db_type == "directory":
            exists = path.exists() and path.is_dir() and any(path.iterdir())
        else:
            exists = path.exists() and path.stat().st_size > 0

        status = "+" if exists else "-"
        if not exists:
            dbs_complete = False

        if exists and db_type == "file":
            size_gb = path.stat().st_size / (1024**3)
            print(f"  [{status}] {name}: {size_gb:.1f} GB")
        elif exists and db_type == "directory":
            num_files = sum(1 for _ in path.rglob("*") if _.is_file())
            print(f"  [{status}] {name}/: {num_files} files")
        else:
            print(f"  [{status}] {name}")

    print()

    # Check MMseqs2 databases (protein FASTAs converted to GPU format)
    mmseqs_path = db_path / "mmseqs"

    print("MMseqs2 Databases:")
    mmseqs_complete = True
    for db_name in MMSEQS_DATABASES.keys():
        padded_db = mmseqs_path / f"{db_name}_padded.dbtype"
        exists = padded_db.exists()
        if not exists:
            mmseqs_complete = False
        status = "+" if exists else "-"
        print(f"  [{status}] {db_name}_padded")

    # Check for unconverted protein FASTAs (should be deleted after conversion)
    unconverted = []
    for db_name, fasta in MMSEQS_DATABASES.items():
        fasta_path = db_path / fasta
        padded_exists = (mmseqs_path / f"{db_name}_padded.dbtype").exists()
        if fasta_path.exists() and padded_exists:
            size_gb = fasta_path.stat().st_size / (1024**3)
            unconverted.append((fasta, size_gb))
    if unconverted:
        print()
        print("  Reclaimable space (raw FASTAs with existing MMseqs DBs):")
        for fasta, size_gb in unconverted:
            print(f"    {fasta}: {size_gb:.1f} GB")

    print()
    print("=" * 60)

    if dbs_complete and mmseqs_complete:
        print("STATUS: Ready for predictions!")
    elif dbs_complete and not mmseqs_complete:
        print("STATUS: Need MMseqs2 conversion")
        print("  Run: modal run modal/prepare_databases.py --convert-only")
    else:
        print("STATUS: Need to download databases")
        print("  Run: modal run modal/prepare_databases.py")

    print("=" * 60)

    return {"dbs_complete": dbs_complete, "mmseqs_complete": mmseqs_complete}


# =============================================================================
# MAIN ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main(
    status: bool = False,
    convert_only: bool = False,
):
    """
    Prepare AlphaFold3 databases on Modal.

    Downloads from Google Cloud Storage and converts to MMseqs2 format.

    Args:
        status: Check current database status
        convert_only: Only run MMseqs2 conversion (databases must exist)
    """
    print()
    print("=" * 60)
    print("AlphaFold3 Database Setup")
    print("=" * 60)
    print()

    if status:
        check_status.remote()
        return

    if convert_only:
        print("Mode: MMseqs2 conversion only")
        print()
        convert_to_mmseqs.remote()
        check_status.remote()
        return

    # Full setup: download + convert
    print("Mode: Full setup (download + convert)")
    print()
    print("This will:")
    print("  1. Download databases from Google Cloud (~530GB)")
    print("  2. Convert to MMseqs2 GPU format")
    print()

    print("Step 1/2: Downloading databases...")
    download_databases.remote()

    print()
    print("Step 2/2: Converting to MMseqs2...")
    convert_to_mmseqs.remote()

    print()
    check_status.remote()

    print()
    print("Next steps:")
    print("  1. Upload model weights:")
    print("     modal run modal/upload_weights.py --file /path/to/af3.bin")
    print()
    print("  2. Run predictions:")
    print("     modal run modal/af3_predict.py --input protein.json")
