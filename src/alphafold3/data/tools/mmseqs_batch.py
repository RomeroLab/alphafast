# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Batch MSA search using MMseqs2-GPU for multiple sequences.

This module provides batch processing capabilities for MMseqs2 searches,
allowing multiple query sequences to be processed in a single search command.
This is more efficient than individual searches because:
1. Single createdb call instead of N
2. GPU processes all queries in parallel
3. Amortizes GPU kernel launch overhead
"""

from concurrent import futures
import os
import pathlib
import shutil
import tempfile
import time
from typing import NamedTuple

from absl import logging
from alphafold3.data.tools import mmseqs
from alphafold3.data.tools import msa_tool
from alphafold3.data.tools import subprocess_utils


class BatchTimingStats(NamedTuple):
    """Timing statistics for a batch search."""

    batch_size: int
    num_sequences: int
    createdb_time: float
    search_time: float
    result2msa_time: float
    unpackdb_time: float
    total_time: float
    sequences_per_second: float


class BatchSearchResult(NamedTuple):
    """Results from a batch MSA search."""

    results: dict[str, msa_tool.MsaToolResult]  # seq_id -> result
    timing: BatchTimingStats


class MmseqsBatch:
    """Batch MSA search for multiple sequences using MMseqs2-GPU.

    This class processes multiple query sequences in a single MMseqs2 search,
    which is more efficient than searching each sequence individually because:
    1. Single createdb call instead of N
    2. GPU processes all queries in parallel
    3. Amortizes GPU kernel launch overhead

    Example usage:
        batch_searcher = MmseqsBatch(
            binary_path="/usr/local/bin/mmseqs",
            database_path="/data/uniref90_padded",
        )
        results = batch_searcher.search_batch({
            "seq1": "MVLSPADKTNV...",
            "seq2": "MNIFEMLRIDE...",
        })
        # results["seq1"] is MsaToolResult for seq1
    """

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
    ):
        """Initialize batch searcher.

        Args:
            binary_path: Path to mmseqs binary.
            database_path: Path to MMseqs2 padded database.
            e_value: E-value threshold for search.
            sensitivity: Sensitivity parameter (-s flag). Range 1-7.5.
            max_sequences: Maximum sequences per query in MSA.
            gpu_enabled: Whether to use GPU acceleration.
            gpu_device: Specific GPU device to use.
            threads: CPU threads for search.
            temp_dir: Directory for temporary files. If None, uses system default.
                Set to fast local storage on HPC clusters for better performance.
        """
        self._binary_path = binary_path
        self._database_path = database_path
        self._e_value = e_value
        self._sensitivity = sensitivity
        self._max_sequences = max_sequences
        self._gpu_enabled = gpu_enabled
        self._gpu_device = gpu_device
        self._threads = threads
        self._temp_dir = temp_dir

        subprocess_utils.check_binary_exists(path=binary_path, name="MMseqs2")

        if not os.path.exists(f"{database_path}.dbtype"):
            raise ValueError(
                f"MMseqs2 database not found at {database_path}. "
                f"Expected to find {database_path}.dbtype file."
            )

    def _get_env(self) -> dict[str, str] | None:
        """Returns environment variables for subprocess."""
        if self._gpu_device is not None:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self._gpu_device)
            return env
        return None

    def search_batch(
        self,
        sequences: dict[str, str],
    ) -> BatchSearchResult:
        """Search multiple sequences against the database.

        All sequences are processed in a single MMseqs2 search command,
        allowing the GPU to process them in parallel.

        Args:
            sequences: Dict mapping sequence_id -> sequence.

        Returns:
            BatchSearchResult with results and timing stats.
        """
        if not sequences:
            return BatchSearchResult(
                results={},
                timing=BatchTimingStats(
                    batch_size=0,
                    num_sequences=0,
                    createdb_time=0,
                    search_time=0,
                    result2msa_time=0,
                    unpackdb_time=0,
                    total_time=0,
                    sequences_per_second=0,
                ),
            )

        total_start = time.time()

        # Use self._temp_dir for HPC clusters with fast local storage
        with tempfile.TemporaryDirectory(
            prefix="mmseqs_batch_", dir=self._temp_dir
        ) as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)

            query_fasta = tmp_path / "query.fasta"
            query_db = tmp_path / "queryDB"
            result_db = tmp_path / "resultDB"
            msa_db = tmp_path / "msaDB"
            output_dir = tmp_path / "output"
            tmp_search = tmp_path / "tmp"

            output_dir.mkdir()
            tmp_search.mkdir()

            # Step 1: Create batch query FASTA and database
            createdb_start = time.time()
            mmseqs.create_query_fasta(sequences, str(query_fasta))
            self._run_createdb(str(query_fasta), str(query_db))
            createdb_time = time.time() - createdb_start

            # Step 2: Run batch search (GPU processes all queries)
            search_start = time.time()
            self._run_search(
                query_db=str(query_db),
                result_db=str(result_db),
                tmp_dir=str(tmp_search),
            )
            search_time = time.time() - search_start

            # Step 3: Convert results to MSA format
            result2msa_start = time.time()
            self._run_result2msa(
                query_db=str(query_db),
                result_db=str(result_db),
                msa_db=str(msa_db),
            )
            result2msa_time = time.time() - result2msa_start

            # Step 4: Unpack to individual A3M files
            unpackdb_start = time.time()
            self._run_unpackdb(str(msa_db), str(output_dir))
            unpackdb_time = time.time() - unpackdb_start

            # Step 5: Parse results and map back to sequence IDs
            results = self._parse_batch_results(
                sequences=sequences,
                query_db=str(query_db),
                output_dir=output_dir,
            )

        total_time = time.time() - total_start

        timing = BatchTimingStats(
            batch_size=len(sequences),
            num_sequences=len(sequences),
            createdb_time=createdb_time,
            search_time=search_time,
            result2msa_time=result2msa_time,
            unpackdb_time=unpackdb_time,
            total_time=total_time,
            sequences_per_second=len(sequences) / total_time if total_time > 0 else 0,
        )

        logging.info(
            "MMseqs2 batch search completed: %d sequences in %.2f seconds "
            "(%.2f seq/s). Breakdown: createdb=%.2fs, search=%.2fs, "
            "result2msa=%.2fs, unpackdb=%.2fs",
            len(sequences),
            total_time,
            timing.sequences_per_second,
            createdb_time,
            search_time,
            result2msa_time,
            unpackdb_time,
        )

        return BatchSearchResult(results=results, timing=timing)

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
            cmd_name="MMseqs2 createdb (batch)",
            log_stdout=False,
            log_stderr=True,
            log_on_process_error=True,
        )

    def _run_search(
        self,
        query_db: str,
        result_db: str,
        tmp_dir: str,
    ) -> None:
        """Runs MMseqs2 batch search with GPU acceleration."""
        cmd = [
            self._binary_path,
            "search",
            query_db,
            self._database_path,
            result_db,
            tmp_dir,
            "-a",  # Enable alignment backtraces for MSA generation
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
            cmd_name="MMseqs2 search (batch)",
            log_stdout=False,
            log_stderr=True,
            log_on_process_error=True,
            env=self._get_env(),
        )

    def _run_result2msa(
        self,
        query_db: str,
        result_db: str,
        msa_db: str,
    ) -> None:
        """Converts search results to MSA in A3M format."""
        cmd = [
            self._binary_path,
            "result2msa",
            query_db,
            self._database_path,
            result_db,
            msa_db,
            "--msa-format-mode",
            "5",  # A3M format
            # No --threads limit to allow full CPU utilization
        ]
        subprocess_utils.run(
            cmd=cmd,
            cmd_name="MMseqs2 result2msa (batch)",
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
            cmd_name="MMseqs2 unpackdb (batch)",
            log_stdout=False,
            log_stderr=True,
            log_on_process_error=True,
        )

    def _parse_batch_results(
        self,
        sequences: dict[str, str],
        query_db: str,
        output_dir: pathlib.Path,
    ) -> dict[str, msa_tool.MsaToolResult]:
        """Parse batch results and map back to sequence IDs.

        MMseqs2 unpackdb creates files named by internal index (0, 1, 2, ...).
        We need to map these back to the original sequence IDs using the
        query database's .lookup file.

        Args:
            sequences: Original sequences dict {seq_id: sequence}.
            query_db: Path to query database (for .lookup file).
            output_dir: Directory containing unpacked A3M files.

        Returns:
            Dict mapping sequence_id -> MsaToolResult.
        """
        # Parse the .lookup file to get index -> seq_id mapping
        lookup_file = f"{query_db}.lookup"
        index_to_seq_id = {}

        if os.path.exists(lookup_file):
            with open(lookup_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        idx = int(parts[0])
                        seq_id = parts[1]
                        index_to_seq_id[idx] = seq_id
        else:
            # Fallback: assume order matches input order
            logging.warning(
                "Query DB lookup file not found at %s, using input order",
                lookup_file,
            )
            index_to_seq_id = {i: seq_id for i, seq_id in enumerate(sequences.keys())}

        results = {}
        for idx, seq_id in index_to_seq_id.items():
            if seq_id not in sequences:
                continue

            a3m_file = output_dir / str(idx)
            if a3m_file.exists():
                a3m_content = a3m_file.read_text()
            else:
                # No hits found
                logging.warning(
                    "No MSA hits found for sequence %s (index %d)", seq_id, idx
                )
                a3m_content = f">{seq_id}\n{sequences[seq_id]}\n"

            results[seq_id] = msa_tool.MsaToolResult(
                target_sequence=sequences[seq_id],
                e_value=self._e_value,
                a3m=a3m_content,
            )

        # Check for any sequences we missed
        for seq_id in sequences:
            if seq_id not in results:
                logging.warning("Sequence %s not found in batch results", seq_id)
                results[seq_id] = msa_tool.MsaToolResult(
                    target_sequence=sequences[seq_id],
                    e_value=self._e_value,
                    a3m=f">{seq_id}\n{sequences[seq_id]}\n",
                )

        return results


class MmseqsMultiDBBatch:
    """Batch MSA search across multiple databases.

    This class handles searching multiple sequences across multiple target
    databases (uniref90, mgnify, small_bfd, uniprot) with pipelining.
    """

    def __init__(
        self,
        *,
        binary_path: str,
        database_paths: dict[str, str],  # {db_name: db_path}
        e_value: float = 1e-4,
        sensitivity: float = 7.5,
        max_sequences_per_db: dict[str, int] | None = None,
        gpu_enabled: bool = True,
        gpu_device: int | None = None,
        threads: int = 8,
        temp_dir: str | None = None,
    ):
        """Initialize multi-database batch searcher.

        Args:
            binary_path: Path to mmseqs binary.
            database_paths: Dict mapping db_name -> db_path.
            e_value: E-value threshold for search.
            sensitivity: Sensitivity parameter.
            max_sequences_per_db: Optional dict of max sequences per database.
            gpu_enabled: Whether to use GPU acceleration.
            gpu_device: Specific GPU device to use.
            threads: CPU threads for search.
            temp_dir: Directory for temporary files. If None, uses system default.
                Set to fast local storage on HPC clusters for better performance.
        """
        self._binary_path = binary_path
        self._database_paths = database_paths
        self._e_value = e_value
        self._sensitivity = sensitivity
        self._gpu_enabled = gpu_enabled
        self._gpu_device = gpu_device
        self._threads = threads
        self._temp_dir = temp_dir

        # Default max sequences per database
        self._max_sequences_per_db = max_sequences_per_db or {
            "uniref90": 10000,
            "mgnify": 5000,
            "small_bfd": 5000,
            "uniprot": 50000,
        }

        subprocess_utils.check_binary_exists(path=binary_path, name="MMseqs2")

    def search_all_databases(
        self,
        sequences: dict[str, str],
    ) -> dict[str, dict[str, BatchSearchResult]]:
        """Search sequences across all configured databases.

        Uses shared query DB and pipelines GPU searches with CPU post-processing.

        Args:
            sequences: Dict mapping sequence_id -> sequence.

        Returns:
            Nested dict: {db_name: BatchSearchResult}
        """
        if not sequences:
            return {}

        total_start = time.time()

        # Create shared query DB
        # Use self._temp_dir for HPC clusters with fast local storage
        query_db_dir = tempfile.mkdtemp(
            prefix="mmseqs_shared_query_", dir=self._temp_dir
        )
        try:
            query_db, _ = mmseqs.create_query_db(
                binary_path=self._binary_path,
                sequences=sequences,
                output_dir=query_db_dir,
            )

            results = {}

            # Search each database sequentially (GPU), but pipeline post-processing
            with futures.ThreadPoolExecutor() as executor:
                db_futures = {}

                for db_name, db_path in self._database_paths.items():
                    max_seqs = self._max_sequences_per_db.get(db_name, 5000)

                    # Create batch searcher for this database
                    searcher = MmseqsBatch(
                        binary_path=self._binary_path,
                        database_path=db_path,
                        e_value=self._e_value,
                        sensitivity=self._sensitivity,
                        max_sequences=max_seqs,
                        gpu_enabled=self._gpu_enabled,
                        gpu_device=self._gpu_device,
                        threads=self._threads,
                        temp_dir=self._temp_dir,
                    )

                    # Submit search (will use its own temp dir for results)
                    # Note: For true pipelining, we'd need to restructure to
                    # separate search from post-processing
                    db_futures[db_name] = executor.submit(
                        searcher.search_batch, sequences
                    )

                # Collect results
                for db_name, future in db_futures.items():
                    results[db_name] = future.result()

        finally:
            shutil.rmtree(query_db_dir, ignore_errors=True)

        total_time = time.time() - total_start
        logging.info(
            "MMseqs2 multi-DB batch search completed: %d sequences x %d databases "
            "in %.2f seconds",
            len(sequences),
            len(self._database_paths),
            total_time,
        )

        return results

    def search_all_databases_pipelined(
        self,
        sequences: dict[str, str],
    ) -> dict[str, BatchSearchResult]:
        """Search with GPU searches sequential, post-processing parallel.

        This method runs GPU searches for each database sequentially (to avoid
        GPU OOM), but allows CPU post-processing to overlap with subsequent
        GPU searches.

        Args:
            sequences: Dict mapping sequence_id -> sequence.

        Returns:
            Dict mapping db_name -> BatchSearchResult.
        """
        if not sequences:
            return {}

        total_start = time.time()

        # Create shared query DB
        # Use self._temp_dir for HPC clusters with fast local storage
        query_db_dir = tempfile.mkdtemp(
            prefix="mmseqs_shared_query_", dir=self._temp_dir
        )
        try:
            logging.info("Creating shared query DB for %d sequences", len(sequences))
            query_db, _ = mmseqs.create_query_db(
                binary_path=self._binary_path,
                sequences=sequences,
                output_dir=query_db_dir,
            )

            results = {}
            pending_futures: dict[str, futures.Future[BatchSearchResult]] = {}

            with futures.ThreadPoolExecutor() as executor:
                for db_name, db_path in self._database_paths.items():
                    max_seqs = self._max_sequences_per_db.get(db_name, 5000)

                    logging.info(
                        "Starting pipelined search for %s (max_seqs=%d)",
                        db_name,
                        max_seqs,
                    )

                    # Run GPU search synchronously
                    gpu_search_result = self._run_gpu_search_only(
                        query_db=query_db,
                        db_path=db_path,
                        max_seqs=max_seqs,
                    )

                    # Submit post-processing to executor (runs async)
                    pending_futures[db_name] = executor.submit(
                        self._run_postprocess_only,
                        gpu_result=gpu_search_result,
                        sequences=sequences,
                        db_name=db_name,
                    )

                # Wait for all post-processing to complete
                for db_name, future in pending_futures.items():
                    results[db_name] = future.result()

        finally:
            shutil.rmtree(query_db_dir, ignore_errors=True)

        total_time = time.time() - total_start
        logging.info(
            "MMseqs2 pipelined multi-DB search completed: %d sequences x %d databases "
            "in %.2f seconds",
            len(sequences),
            len(self._database_paths),
            total_time,
        )

        return results

    def _run_gpu_search_only(
        self,
        query_db: str,
        db_path: str,
        max_seqs: int,
    ) -> dict:
        """Run only the GPU search phase, return intermediate results.

        Args:
            query_db: Path to shared query database.
            db_path: Path to target database.
            max_seqs: Maximum sequences in results.

        Returns:
            Dict with paths to intermediate files for post-processing.
        """
        search_start = time.time()

        # Create temp directory for this search
        # Use self._temp_dir for HPC clusters with fast local storage
        tmp_dir = tempfile.mkdtemp(prefix="mmseqs_search_", dir=self._temp_dir)
        tmp_path = pathlib.Path(tmp_dir)

        result_db = tmp_path / "resultDB"
        tmp_search = tmp_path / "tmp"
        tmp_search.mkdir()

        # Build search command
        cmd = [
            self._binary_path,
            "search",
            query_db,
            db_path,
            str(result_db),
            str(tmp_search),
            "-a",
            "-s",
            str(self._sensitivity),
            "-e",
            str(self._e_value),
            "--threads",
            str(self._threads),
            "--max-seqs",
            str(max_seqs),
        ]

        if self._gpu_enabled:
            cmd.extend(["--gpu", "1"])

        env = None
        if self._gpu_device is not None:
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(self._gpu_device)

        subprocess_utils.run(
            cmd=cmd,
            cmd_name=f"MMseqs2 search (pipelined)",
            log_stdout=False,
            log_stderr=True,
            log_on_process_error=True,
            env=env,
        )

        search_time = time.time() - search_start
        logging.info("GPU search completed in %.2f seconds", search_time)

        return {
            "tmp_dir": tmp_dir,
            "query_db": query_db,
            "db_path": db_path,
            "result_db": str(result_db),
            "search_time": search_time,
        }

    def _run_postprocess_only(
        self,
        gpu_result: dict,
        sequences: dict[str, str],
        db_name: str,
    ) -> BatchSearchResult:
        """Run post-processing on GPU search results.

        Args:
            gpu_result: Dict from _run_gpu_search_only.
            sequences: Original sequences.
            db_name: Database name (for logging).

        Returns:
            BatchSearchResult with parsed results.
        """
        postprocess_start = time.time()

        tmp_dir = gpu_result["tmp_dir"]
        tmp_path = pathlib.Path(tmp_dir)
        query_db = gpu_result["query_db"]
        result_db = gpu_result["result_db"]
        db_path = gpu_result["db_path"]
        search_time = gpu_result["search_time"]

        msa_db = tmp_path / "msaDB"
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        try:
            # result2msa
            result2msa_start = time.time()
            cmd = [
                self._binary_path,
                "result2msa",
                query_db,
                db_path,
                result_db,
                str(msa_db),
                "--msa-format-mode",
                "5",
            ]
            subprocess_utils.run(
                cmd=cmd,
                cmd_name=f"MMseqs2 result2msa ({db_name})",
                log_stdout=False,
                log_stderr=True,
                log_on_process_error=True,
            )
            result2msa_time = time.time() - result2msa_start

            # unpackdb
            unpackdb_start = time.time()
            cmd = [
                self._binary_path,
                "unpackdb",
                str(msa_db),
                str(output_dir),
            ]
            subprocess_utils.run(
                cmd=cmd,
                cmd_name=f"MMseqs2 unpackdb ({db_name})",
                log_stdout=False,
                log_stderr=True,
                log_on_process_error=True,
            )
            unpackdb_time = time.time() - unpackdb_start

            # Parse results
            results = self._parse_results(sequences, query_db, output_dir)

            postprocess_time = time.time() - postprocess_start
            total_time = search_time + postprocess_time

            timing = BatchTimingStats(
                batch_size=len(sequences),
                num_sequences=len(sequences),
                createdb_time=0,  # Shared, not counted here
                search_time=search_time,
                result2msa_time=result2msa_time,
                unpackdb_time=unpackdb_time,
                total_time=total_time,
                sequences_per_second=len(sequences) / total_time
                if total_time > 0
                else 0,
            )

            logging.info(
                "Post-processing for %s completed in %.2f seconds",
                db_name,
                postprocess_time,
            )

            return BatchSearchResult(results=results, timing=timing)

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _parse_results(
        self,
        sequences: dict[str, str],
        query_db: str,
        output_dir: pathlib.Path,
    ) -> dict[str, msa_tool.MsaToolResult]:
        """Parse batch results from output directory."""
        lookup_file = f"{query_db}.lookup"
        index_to_seq_id = {}

        if os.path.exists(lookup_file):
            with open(lookup_file) as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        idx = int(parts[0])
                        seq_id = parts[1]
                        index_to_seq_id[idx] = seq_id
        else:
            index_to_seq_id = {i: seq_id for i, seq_id in enumerate(sequences.keys())}

        results = {}
        for idx, seq_id in index_to_seq_id.items():
            if seq_id not in sequences:
                continue

            a3m_file = output_dir / str(idx)
            if a3m_file.exists():
                a3m_content = a3m_file.read_text()
            else:
                a3m_content = f">{seq_id}\n{sequences[seq_id]}\n"

            results[seq_id] = msa_tool.MsaToolResult(
                target_sequence=sequences[seq_id],
                e_value=self._e_value,
                a3m=a3m_content,
            )

        for seq_id in sequences:
            if seq_id not in results:
                results[seq_id] = msa_tool.MsaToolResult(
                    target_sequence=sequences[seq_id],
                    e_value=self._e_value,
                    a3m=f">{seq_id}\n{sequences[seq_id]}\n",
                )

        return results
