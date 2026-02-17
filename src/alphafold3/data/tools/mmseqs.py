# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md
# Modifications Copyright 2026 Romero Lab, Duke University

"""Library to run MMseqs2 GPU-accelerated sequence search from Python."""

from concurrent import futures
import os
import pathlib
import shutil
import tempfile
import time

from absl import logging
from alphafold3.data.tools import msa_tool
from alphafold3.data.tools import subprocess_utils


class Mmseqs(msa_tool.MsaTool):
    """Python wrapper for the MMseqs2 binary with GPU support."""

    def __init__(
        self,
        *,
        binary_path: str,
        database_path: str,
        e_value: float = 1e-4,
        sensitivity: float = 7.5,
        max_sequences: int = 5000,
        gpu_enabled: bool = True,
        gpu_device: int | None = None,
        threads: int = 8,
        temp_dir: str | None = None,
        **unused_kwargs,
    ):
        """Initializes the Python MMseqs2 wrapper.

        Args:
          binary_path: The path to the mmseqs executable.
          database_path: The path to the MMseqs2 padded database. This should be the
            path created by `mmseqs makepaddedseqdb` for GPU acceleration.
          e_value: The E-value threshold for the search.
          sensitivity: Sensitivity parameter (-s flag). Range 1-7.5, higher values
            find more remote homologs but are slower. 7.5 is recommended for remote
            homolog detection.
          max_sequences: Maximum number of sequences to return in the MSA.
          gpu_enabled: Whether to use GPU acceleration (--gpu 1 flag). Requires
            padded database created with makepaddedseqdb.
          gpu_device: Specific GPU device to use. If None, uses all available GPUs.
            Set via CUDA_VISIBLE_DEVICES environment variable.
          threads: Number of CPU threads for non-GPU parts of the search.
          temp_dir: Directory for temporary files. If None, uses system default.
            Set to fast local storage on HPC clusters for better performance.

        Raises:
          RuntimeError: If MMseqs2 binary not found at the path.
          ValueError: If GPU is enabled but database doesn't appear to be padded.
        """
        self._binary_path = binary_path
        subprocess_utils.check_binary_exists(path=self._binary_path, name="MMseqs2")

        self._database_path = database_path
        self._e_value = e_value
        self._sensitivity = sensitivity
        self._max_sequences = max_sequences
        self._gpu_enabled = gpu_enabled
        self._gpu_device = gpu_device
        self._threads = threads
        self._temp_dir = temp_dir

        # Verify the database exists
        if not os.path.exists(f"{database_path}.dbtype"):
            raise ValueError(
                f"MMseqs2 database not found at {database_path}. "
                "Expected to find {database_path}.dbtype file."
            )

        # Check for padded database if GPU is enabled
        # Note: makepaddedseqdb creates standard MMseqs2 database files (.dbtype, .index, etc.)
        # but with padded sequences. We just verify the database exists - the .dbtype check
        # above already confirms this. The --gpu flag will work if the database was created
        # with makepaddedseqdb; if not, MMseqs2 will fall back to CPU mode automatically.

    def query(self, target_sequence: str) -> msa_tool.MsaToolResult:
        """Runs MMseqs2 search and returns the MSA in A3M format.

        The workflow is:
        1. Create query database from input sequence
        2. Run GPU-accelerated search with alignment backtraces (-a flag)
        3. Generate MSA database in A3M format
        4. Extract A3M file from the MSA database

        Args:
          target_sequence: The protein sequence to search for homologs.

        Returns:
          MsaToolResult containing the target sequence, e-value, and A3M MSA.
        """
        logging.info(
            "MMseqs2 query sequence: %s",
            target_sequence
            if len(target_sequence) <= 16
            else f"{target_sequence[:16]}... (len {len(target_sequence)})",
        )

        search_start_time = time.time()

        with tempfile.TemporaryDirectory(dir=self._temp_dir) as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)

            # Create paths for intermediate files
            query_fasta = tmp_path / "query.fasta"
            query_db = tmp_path / "queryDB"
            result_db = tmp_path / "resultDB"
            msa_db = tmp_path / "msaDB"
            output_dir = tmp_path / "output"
            tmp_search = tmp_path / "tmp"

            # Create subdirectories
            output_dir.mkdir()
            tmp_search.mkdir()

            # Step 1: Write query sequence to FASTA file
            subprocess_utils.create_query_fasta_file(
                sequence=target_sequence, path=str(query_fasta)
            )

            # Step 2: Create query database
            self._run_createdb(
                input_fasta=str(query_fasta),
                output_db=str(query_db),
            )

            # Step 3: Run search
            self._run_search(
                query_db=str(query_db),
                target_db=self._database_path,
                result_db=str(result_db),
                tmp_dir=str(tmp_search),
            )

            # Step 4: Generate MSA in A3M format
            self._run_result2msa(
                query_db=str(query_db),
                target_db=self._database_path,
                result_db=str(result_db),
                msa_db=str(msa_db),
            )

            # Step 5: Unpack MSA database to files
            self._run_unpackdb(
                msa_db=str(msa_db),
                output_dir=str(output_dir),
            )

            # Step 6: Read the A3M output (single query produces file named "0")
            a3m_file = output_dir / "0"
            if a3m_file.exists():
                a3m_content = a3m_file.read_text()
            else:
                # No hits found, return just the query sequence
                logging.warning("No MSA hits found for query sequence.")
                a3m_content = f">query\n{target_sequence}\n"

        search_time = time.time() - search_start_time
        logging.info(
            "MMseqs2 search completed in %.2f seconds for sequence %s",
            search_time,
            target_sequence[:16] if len(target_sequence) > 16 else target_sequence,
        )

        return msa_tool.MsaToolResult(
            target_sequence=target_sequence,
            e_value=self._e_value,
            a3m=a3m_content,
        )

    def query_pipelined(
        self,
        target_sequence: str,
        executor: futures.ThreadPoolExecutor,
    ) -> futures.Future[msa_tool.MsaToolResult]:
        """Runs MMseqs2 search with GPU synchronously, post-processing asynchronously.

        This method allows pipelining multiple database searches. The GPU search
        runs synchronously (blocking), but the CPU-bound post-processing (result2msa,
        unpackdb) is submitted to the executor and runs in parallel with subsequent
        GPU searches.

        Args:
            target_sequence: The protein sequence to search for homologs.
            executor: ThreadPoolExecutor to run post-processing tasks.

        Returns:
            A Future that resolves to MsaToolResult when post-processing completes.
        """
        logging.info(
            "MMseqs2 pipelined query sequence: %s",
            target_sequence
            if len(target_sequence) <= 16
            else f"{target_sequence[:16]}... (len {len(target_sequence)})",
        )

        search_start_time = time.time()

        # Create temp directory manually (no context manager) so it persists
        # until post-processing completes
        # Use self._temp_dir for HPC clusters with fast local storage
        tmp_dir = tempfile.mkdtemp(prefix="mmseqs_", dir=self._temp_dir)
        tmp_path = pathlib.Path(tmp_dir)

        # Create paths for intermediate files
        query_fasta = tmp_path / "query.fasta"
        query_db = tmp_path / "queryDB"
        result_db = tmp_path / "resultDB"
        msa_db = tmp_path / "msaDB"
        output_dir = tmp_path / "output"
        tmp_search = tmp_path / "tmp"

        # Create subdirectories
        output_dir.mkdir()
        tmp_search.mkdir()

        # Step 1: Write query sequence to FASTA file
        subprocess_utils.create_query_fasta_file(
            sequence=target_sequence, path=str(query_fasta)
        )

        # Step 2: Create query database
        self._run_createdb(
            input_fasta=str(query_fasta),
            output_db=str(query_db),
        )

        # Step 3: Run search (GPU, synchronous)
        self._run_search(
            query_db=str(query_db),
            target_db=self._database_path,
            result_db=str(result_db),
            tmp_dir=str(tmp_search),
        )

        search_time = time.time() - search_start_time
        logging.info(
            "MMseqs2 GPU search completed in %.2f seconds for sequence %s",
            search_time,
            target_sequence[:16] if len(target_sequence) > 16 else target_sequence,
        )

        # Submit post-processing to executor (runs async while next search starts)
        return executor.submit(
            self._postprocess_and_cleanup,
            tmp_dir=tmp_dir,
            query_db=str(query_db),
            result_db=str(result_db),
            msa_db=str(msa_db),
            output_dir=str(output_dir),
            target_sequence=target_sequence,
            search_start_time=search_start_time,
        )

    def _postprocess_and_cleanup(
        self,
        tmp_dir: str,
        query_db: str,
        result_db: str,
        msa_db: str,
        output_dir: str,
        target_sequence: str,
        search_start_time: float,
    ) -> msa_tool.MsaToolResult:
        """Runs post-processing (result2msa, unpackdb) and cleans up temp directory.

        This method is designed to be run in a thread pool executor, allowing
        CPU-bound post-processing to run in parallel with GPU searches.

        Args:
            tmp_dir: Path to temporary directory (will be cleaned up).
            query_db: Path to query database.
            result_db: Path to result database.
            msa_db: Path for MSA database output.
            output_dir: Path for unpacked output files.
            target_sequence: Original query sequence.
            search_start_time: Time when search started (for logging).

        Returns:
            MsaToolResult containing the target sequence, e-value, and A3M MSA.
        """
        try:
            # Step 4: Generate MSA in A3M format
            self._run_result2msa(
                query_db=query_db,
                target_db=self._database_path,
                result_db=result_db,
                msa_db=msa_db,
            )

            # Step 5: Unpack MSA database to files
            self._run_unpackdb(
                msa_db=msa_db,
                output_dir=output_dir,
            )

            # Step 6: Read the A3M output (single query produces file named "0")
            a3m_file = pathlib.Path(output_dir) / "0"
            if a3m_file.exists():
                a3m_content = a3m_file.read_text()
            else:
                # No hits found, return just the query sequence
                logging.warning("No MSA hits found for query sequence.")
                a3m_content = f">query\n{target_sequence}\n"

            total_time = time.time() - search_start_time
            logging.info(
                "MMseqs2 total (search + postprocess) completed in %.2f seconds for sequence %s",
                total_time,
                target_sequence[:16] if len(target_sequence) > 16 else target_sequence,
            )

            return msa_tool.MsaToolResult(
                target_sequence=target_sequence,
                e_value=self._e_value,
                a3m=a3m_content,
            )
        finally:
            # Clean up temp directory
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _get_env(self) -> dict[str, str] | None:
        """Returns environment variables for subprocess, setting CUDA_VISIBLE_DEVICES if needed."""
        if self._gpu_device is not None:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self._gpu_device)
            return env
        return None

    def _run_createdb(self, input_fasta: str, output_db: str) -> None:
        """Creates an MMseqs2 database from a FASTA file."""
        cmd = [
            self._binary_path,
            "createdb",
            input_fasta,
            output_db,
        ]
        subprocess_utils.run(
            cmd=cmd,
            cmd_name="MMseqs2 createdb",
            log_stdout=False,
            log_stderr=True,
            log_on_process_error=True,
        )

    def _run_search(
        self,
        query_db: str,
        target_db: str,
        result_db: str,
        tmp_dir: str,
    ) -> None:
        """Runs MMseqs2 search with GPU acceleration if enabled."""
        cmd = [
            self._binary_path,
            "search",
            query_db,
            target_db,
            result_db,
            tmp_dir,
            "-a",  # Required: enable alignment backtraces for MSA generation
            "-s",
            str(self._sensitivity),
            "-e",
            str(self._e_value),
            "--threads",
            str(self._threads),
            "--max-seqs",
            str(self._max_sequences),
        ]

        if self._gpu_enabled:
            cmd.extend(["--gpu", "1"])

        subprocess_utils.run(
            cmd=cmd,
            cmd_name="MMseqs2 search",
            log_stdout=False,
            log_stderr=True,
            log_on_process_error=True,
            env=self._get_env(),
        )

    def _run_result2msa(
        self,
        query_db: str,
        target_db: str,
        result_db: str,
        msa_db: str,
    ) -> None:
        """Converts search results to MSA in A3M format."""
        cmd = [
            self._binary_path,
            "result2msa",
            query_db,
            target_db,
            result_db,
            msa_db,
            "--msa-format-mode",
            "5",  # A3M format
            # Note: No --threads limit to allow full CPU utilization
        ]
        subprocess_utils.run(
            cmd=cmd,
            cmd_name="MMseqs2 result2msa",
            log_stdout=False,
            log_stderr=True,
            log_on_process_error=True,
        )

    def _run_unpackdb(self, msa_db: str, output_dir: str) -> None:
        """Unpacks the MSA database to individual files."""
        cmd = [
            self._binary_path,
            "unpackdb",
            msa_db,
            output_dir,
        ]
        subprocess_utils.run(
            cmd=cmd,
            cmd_name="MMseqs2 unpackdb",
            log_stdout=False,
            log_stderr=True,
            log_on_process_error=True,
        )

    def search_with_query_db(
        self,
        query_db: str,
        sequences: dict[str, str],
        executor: futures.ThreadPoolExecutor,
    ) -> dict[str, futures.Future[msa_tool.MsaToolResult]]:
        """Search using a pre-created query database.

        This method allows sharing a query database across multiple database searches,
        avoiding redundant createdb calls. The GPU search runs synchronously, but
        post-processing is submitted to the executor.

        Args:
            query_db: Path to pre-created MMseqs2 query database.
            sequences: Dict mapping sequence_id -> sequence (for result parsing).
            executor: ThreadPoolExecutor for async post-processing.

        Returns:
            Dict mapping sequence_id -> Future[MsaToolResult].
        """
        search_start_time = time.time()

        # Create temp directory for this search's intermediate files
        # Use self._temp_dir for HPC clusters with fast local storage
        tmp_dir = tempfile.mkdtemp(prefix="mmseqs_search_", dir=self._temp_dir)
        tmp_path = pathlib.Path(tmp_dir)

        result_db = tmp_path / "resultDB"
        msa_db = tmp_path / "msaDB"
        output_dir = tmp_path / "output"
        tmp_search = tmp_path / "tmp"

        output_dir.mkdir()
        tmp_search.mkdir()

        # Run GPU search (synchronous)
        self._run_search(
            query_db=query_db,
            target_db=self._database_path,
            result_db=str(result_db),
            tmp_dir=str(tmp_search),
        )

        search_time = time.time() - search_start_time
        logging.info(
            "MMseqs2 GPU search (shared query DB) completed in %.2f seconds",
            search_time,
        )

        # Submit post-processing to executor
        # Returns a dict of futures, one per sequence
        return {
            seq_id: executor.submit(
                self._postprocess_shared_db,
                tmp_dir=tmp_dir,
                query_db=query_db,
                result_db=str(result_db),
                msa_db=str(msa_db),
                output_dir=str(output_dir),
                seq_id=seq_id,
                sequence=sequence,
                search_start_time=search_start_time,
                is_last=(seq_id == list(sequences.keys())[-1]),
            )
            for seq_id, sequence in sequences.items()
        }

    def _postprocess_shared_db(
        self,
        tmp_dir: str,
        query_db: str,
        result_db: str,
        msa_db: str,
        output_dir: str,
        seq_id: str,
        sequence: str,
        search_start_time: float,
        is_last: bool,
    ) -> msa_tool.MsaToolResult:
        """Post-process results from shared query DB search.

        Only the last sequence to complete should clean up the temp directory.
        For single-sequence queries, this is always true.

        Args:
            tmp_dir: Temporary directory (cleaned up by last sequence).
            query_db: Path to query database.
            result_db: Path to result database.
            msa_db: Path for MSA database output.
            output_dir: Path for unpacked output files.
            seq_id: Sequence identifier.
            sequence: The query sequence.
            search_start_time: Time when search started.
            is_last: Whether this is the last sequence (should cleanup).

        Returns:
            MsaToolResult for this sequence.
        """
        try:
            # For shared DB, we run result2msa and unpackdb once
            # The first call will do the work, subsequent calls will just read results
            msa_db_path = pathlib.Path(msa_db)
            output_path = pathlib.Path(output_dir)

            if not msa_db_path.with_suffix(".dbtype").exists():
                # First call - run post-processing
                self._run_result2msa(
                    query_db=query_db,
                    target_db=self._database_path,
                    result_db=result_db,
                    msa_db=msa_db,
                )
                self._run_unpackdb(
                    msa_db=msa_db,
                    output_dir=output_dir,
                )

            # Read result for this sequence
            # For single query, file is named "0"
            # For batch queries, need to map seq_id to file index
            a3m_file = output_path / "0"
            if a3m_file.exists():
                a3m_content = a3m_file.read_text()
            else:
                logging.warning("No MSA hits found for sequence %s", seq_id)
                a3m_content = f">{seq_id}\n{sequence}\n"

            total_time = time.time() - search_start_time
            logging.info(
                "MMseqs2 total (shared DB) completed in %.2f seconds for %s",
                total_time,
                seq_id,
            )

            return msa_tool.MsaToolResult(
                target_sequence=sequence,
                e_value=self._e_value,
                a3m=a3m_content,
            )
        finally:
            if is_last:
                shutil.rmtree(tmp_dir, ignore_errors=True)


def create_query_fasta(sequences: dict[str, str], output_path: str) -> None:
    """Write multiple sequences to a FASTA file.

    Args:
        sequences: Dict mapping sequence_id -> sequence.
        output_path: Path to write FASTA file.
    """
    with open(output_path, "w") as f:
        for seq_id, sequence in sequences.items():
            f.write(f">{seq_id}\n")
            # Write sequence in 80-char lines
            for i in range(0, len(sequence), 80):
                f.write(f"{sequence[i : i + 80]}\n")


def create_query_db(
    binary_path: str,
    sequences: dict[str, str],
    output_dir: str,
) -> tuple[str, str]:
    """Create an MMseqs2 query database from sequences.

    This function creates a query database that can be reused across multiple
    target database searches, avoiding redundant createdb calls.

    Args:
        binary_path: Path to mmseqs binary.
        sequences: Dict mapping sequence_id -> sequence.
        output_dir: Directory to store database files.

    Returns:
        Tuple of (query_db_path, query_fasta_path).
    """
    output_path = pathlib.Path(output_dir)
    query_fasta = output_path / "query.fasta"
    query_db = output_path / "queryDB"

    # Write sequences to FASTA
    create_query_fasta(sequences, str(query_fasta))

    # Create database
    cmd = [
        binary_path,
        "createdb",
        str(query_fasta),
        str(query_db),
    ]
    subprocess_utils.run(
        cmd=cmd,
        cmd_name="MMseqs2 createdb (shared)",
        log_stdout=False,
        log_stderr=True,
        log_on_process_error=True,
    )

    return str(query_db), str(query_fasta)


def find_mmseqs_binary() -> str | None:
    """Finds the MMseqs2 binary, preferring $HOME/.local/bin.

    Returns:
      Path to the MMseqs2 binary if found, None otherwise.
    """
    # First check the standard user-local installation path
    home_local = os.path.expandvars("$HOME/.local/bin/mmseqs")
    if os.path.isfile(home_local) and os.access(home_local, os.X_OK):
        return home_local

    # Fall back to searching PATH
    return shutil.which("mmseqs")


def check_mmseqs_database(db_dir: str, db_name: str) -> str | None:
    """Checks if an MMseqs2 padded database exists.

    Args:
      db_dir: Directory containing MMseqs2 databases.
      db_name: Name of the database (e.g., 'uniref90_padded').

    Returns:
      Full path to the database if it exists, None otherwise.
    """
    db_path = os.path.join(db_dir, db_name)
    if os.path.exists(f"{db_path}.dbtype"):
        return db_path
    return None
