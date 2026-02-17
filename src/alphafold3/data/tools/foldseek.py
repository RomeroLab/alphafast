# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Library to run Foldseek structural search from Python.

Foldseek is a fast structural alignment tool that can search protein structures
against large databases like AlphaFold Database (AFDB). It uses 3Di encoding
to represent protein structures efficiently.

Requirements:
  - Foldseek binary (https://github.com/steineggerlab/foldseek)
  - AFDB Foldseek database (created with `foldseek databases`)
"""

import dataclasses
import os
import pathlib
import tempfile
import time
from collections.abc import Sequence

from absl import logging
from alphafold3.data.tools import subprocess_utils


@dataclasses.dataclass(frozen=True, slots=True)
class FoldseekHit:
    """Single Foldseek search result.

    Attributes:
        target_id: Target structure identifier (e.g., "AF-Q9Y6K1-F1-model_v4").
        query_start: Start position in query (0-indexed).
        query_end: End position in query (0-indexed).
        target_start: Start position in target (0-indexed).
        target_end: End position in target (0-indexed).
        evalue: E-value of the alignment.
        bitscore: Bit score of the alignment.
        sequence_identity: Sequence identity (0-1).
        lddt: Local distance difference test score (0-1).
        aligned_length: Length of the alignment.
        query_aligned: Aligned query sequence.
        target_aligned: Aligned target sequence.
    """

    target_id: str
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    evalue: float
    bitscore: float
    sequence_identity: float
    lddt: float
    aligned_length: int
    query_aligned: str
    target_aligned: str


@dataclasses.dataclass(frozen=True, slots=True)
class FoldseekResult:
    """Result from Foldseek structural search.

    Attributes:
        query_pdb: The query structure in PDB format.
        hits: List of structural hits from the search.
    """

    query_pdb: str
    hits: Sequence[FoldseekHit]


class Foldseek:
    """Python wrapper for Foldseek structural search."""

    def __init__(
        self,
        *,
        binary_path: str,
        database_path: str,
        e_value: float = 1e-3,
        max_hits: int = 100,
        alignment_type: int = 2,
        threads: int = 32,
        min_lddt: float = 0.0,
        gpu_enabled: bool = False,  # CPU is faster for single queries (no preprocessing overhead)
        gpu_device: int | None = None,
        temp_dir: str | None = None,
    ):
        """Initializes the Foldseek wrapper.

        Args:
            binary_path: Path to the Foldseek binary.
            database_path: Path to the AFDB Foldseek database.
            e_value: E-value threshold for hits.
            max_hits: Maximum number of hits to return.
            alignment_type: Alignment type: 0=3Di, 1=TM, 2=3Di+AA (recommended).
            threads: Number of CPU threads to use.
            min_lddt: Minimum LDDT score for hits (0-1).
            gpu_enabled: Whether to use GPU acceleration (--gpu 1 flag).
            gpu_device: Specific GPU device to use (via CUDA_VISIBLE_DEVICES).
                If None, uses all available GPUs.
            temp_dir: Directory for temporary files. If None, uses system default.
                Set to fast local storage on HPC clusters for better performance.

        Raises:
            RuntimeError: If Foldseek binary not found.
            ValueError: If database doesn't exist.
        """
        self._binary_path = binary_path
        subprocess_utils.check_binary_exists(path=self._binary_path, name="Foldseek")

        self._database_path = database_path
        self._e_value = e_value
        self._max_hits = max_hits
        self._alignment_type = alignment_type
        self._threads = threads
        self._min_lddt = min_lddt
        self._gpu_enabled = gpu_enabled
        self._gpu_device = gpu_device
        self._temp_dir = temp_dir

        # Verify database exists
        if not os.path.exists(database_path):
            raise ValueError(
                f"Foldseek database not found at {database_path}. "
                "Run setup_foldseek_afdb.sh to download and set up the database."
            )

    def search(self, query_pdb: str) -> FoldseekResult:
        """Searches for structural matches against AFDB.

        Args:
            query_pdb: Query structure in PDB format.

        Returns:
            FoldseekResult containing the query and list of hits.
        """
        logging.info("Running Foldseek structural search...")
        search_start_time = time.time()

        # Use self._temp_dir for HPC clusters with fast local storage
        with tempfile.TemporaryDirectory(dir=self._temp_dir) as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)

            # Write query PDB to file
            query_file = tmp_path / "query.pdb"
            query_file.write_text(query_pdb)

            # Output file for results
            result_file = tmp_path / "result.m8"

            # Run Foldseek easy-search
            self._run_search(
                query_path=str(query_file),
                result_path=str(result_file),
                tmp_dir=str(tmp_path / "tmp"),
            )

            # Parse results
            if result_file.exists():
                hits = self._parse_results(result_file.read_text())
            else:
                logging.warning("No Foldseek results file created.")
                hits = []

        search_time = time.time() - search_start_time
        logging.info(
            "Foldseek search completed in %.2f seconds, found %d hits",
            search_time,
            len(hits),
        )

        return FoldseekResult(query_pdb=query_pdb, hits=hits)

    def _run_search(self, query_path: str, result_path: str, tmp_dir: str) -> None:
        """Runs Foldseek easy-search command."""
        os.makedirs(tmp_dir, exist_ok=True)

        # Build command with extended output format
        # Output columns: query,target,fident,alnlen,mismatch,gapopen,
        #                 qstart,qend,tstart,tend,evalue,bits,lddt,
        #                 qaln,taln
        cmd = [
            self._binary_path,
            "easy-search",
            query_path,
            self._database_path,
            result_path,
            tmp_dir,
            "--threads",
            str(self._threads),
            "-e",
            str(self._e_value),
            "--max-seqs",
            str(self._max_hits),
            "--alignment-type",
            str(self._alignment_type),
            "--format-output",
            "query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,lddt,qaln,taln",
            "--db-load-mode",
            "2",  # mmap mode - uses prebuilt index efficiently
        ]

        # Add GPU flag if enabled
        if self._gpu_enabled:
            cmd.extend(["--gpu", "1"])
            logging.info("Running Foldseek with GPU acceleration")

        # Set up environment for specific GPU device
        env = None
        if self._gpu_device is not None:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self._gpu_device)
            logging.info("Using GPU device %d for Foldseek", self._gpu_device)

        subprocess_utils.run(
            cmd=cmd,
            cmd_name="Foldseek search",
            log_stdout=False,
            log_stderr=True,
            log_on_process_error=True,
            env=env,
        )

    def _parse_results(self, result_text: str) -> list[FoldseekHit]:
        """Parses Foldseek tabular output.

        Args:
            result_text: Tab-separated output from Foldseek.

        Returns:
            List of FoldseekHit objects.
        """
        hits = []

        for line in result_text.strip().split("\n"):
            if not line:
                continue

            fields = line.split("\t")
            if len(fields) < 15:
                logging.warning("Skipping malformed Foldseek result line: %s", line)
                continue

            try:
                # Parse fields according to format-output specification
                # query,target,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,lddt,qaln,taln
                lddt = float(fields[12]) if fields[12] else 0.0

                # Apply LDDT filter
                if lddt < self._min_lddt:
                    continue

                hit = FoldseekHit(
                    target_id=fields[1],
                    query_start=int(fields[6]) - 1,  # Convert to 0-indexed
                    query_end=int(fields[7]) - 1,
                    target_start=int(fields[8]) - 1,
                    target_end=int(fields[9]) - 1,
                    evalue=float(fields[10]),
                    bitscore=float(fields[11]),
                    sequence_identity=float(fields[2]),
                    lddt=lddt,
                    aligned_length=int(fields[3]),
                    query_aligned=fields[13] if len(fields) > 13 else "",
                    target_aligned=fields[14] if len(fields) > 14 else "",
                )
                hits.append(hit)
            except (ValueError, IndexError) as e:
                logging.warning("Error parsing Foldseek hit: %s - %s", line, e)
                continue

        return hits


def find_foldseek_binary() -> str | None:
    """Finds the Foldseek binary.

    Returns:
        Path to the Foldseek binary if found, None otherwise.
    """
    import shutil

    # Check common locations
    home_local = os.path.expandvars("$HOME/.local/bin/foldseek")
    if os.path.isfile(home_local) and os.access(home_local, os.X_OK):
        return home_local

    # Fall back to PATH
    return shutil.which("foldseek")
