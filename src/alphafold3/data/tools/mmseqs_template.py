# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Library to run MMseqs2 GPU-accelerated template search from Python."""

from concurrent import futures
import os
import pathlib
import shutil
import tempfile
import time

from absl import logging
from alphafold3.data.tools import subprocess_utils


class MmseqsTemplate:
    """Python wrapper for MMseqs2 template search with GPU support.

    This class provides template search functionality using MMseqs2, serving as
    an alternative to hmmsearch. It searches the query sequence directly against
    the pdb_seqres database using sequence-to-sequence alignment.
    """

    def __init__(
        self,
        *,
        binary_path: str,
        database_path: str,
        e_value: float = 100.0,
        sensitivity: float = 7.5,
        max_hits: int = 1000,
        gpu_enabled: bool = True,
        gpu_device: int | None = None,
        threads: int = 8,
        temp_dir: str | None = None,
    ):
        """Initializes the MMseqs2 template search wrapper.

        Args:
          binary_path: The path to the mmseqs executable.
          database_path: The path to the MMseqs2 padded pdb_seqres database.
          e_value: The E-value threshold for the search. Using high value (100)
            to match hmmsearch behavior for broad template discovery.
          sensitivity: Sensitivity parameter (-s flag). Range 1-7.5, higher values
            find more remote homologs but are slower.
          max_hits: Maximum number of template hits to return.
          gpu_enabled: Whether to use GPU acceleration (--gpu 1 flag).
          gpu_device: Specific GPU device to use. If None, uses all available GPUs.
          threads: Number of CPU threads for non-GPU parts of the search.
          temp_dir: Directory for temporary files. If None, uses system default.
            Set to fast local storage on HPC clusters for better performance.

        Raises:
          RuntimeError: If MMseqs2 binary not found at the path.
          ValueError: If database doesn't exist.
        """
        self._binary_path = binary_path
        subprocess_utils.check_binary_exists(path=self._binary_path, name="MMseqs2")

        self._database_path = database_path
        self._e_value = e_value
        self._sensitivity = sensitivity
        self._max_hits = max_hits
        self._gpu_enabled = gpu_enabled
        self._gpu_device = gpu_device
        self._threads = threads
        self._temp_dir = temp_dir

        # Verify the database exists
        if not os.path.exists(f"{database_path}.dbtype"):
            raise ValueError(
                f"MMseqs2 template database not found at {database_path}. "
                "Run convert_databases_to_mmseqs.sh to create pdb_seqres_padded."
            )

    def query(self, query_sequence: str) -> str:
        """Runs MMseqs2 template search and returns hits in A3M format.

        The workflow is:
        1. Create query database from input sequence
        2. Run GPU-accelerated search with alignment backtraces (-a flag)
        3. Convert alignments to tabular format to get coordinates
        4. Generate MSA database in A3M format
        5. Parse and reformat A3M with proper headers for template parser

        The output A3M format is compatible with the hit description parser in
        templates.py. Headers are reformatted to include alignment coordinates:
        >PDBID_CHAIN/START-END [subseq from] mol:protein length:FULL_LENGTH

        Args:
          query_sequence: The protein sequence to search for templates.

        Returns:
          A3M string containing template hits. Empty string if no hits found.
        """
        logging.info(
            "MMseqs2 template search for sequence: %s",
            query_sequence
            if len(query_sequence) <= 16
            else f"{query_sequence[:16]}... (len {len(query_sequence)})",
        )

        search_start_time = time.time()

        # Use self._temp_dir for HPC clusters with fast local storage
        with tempfile.TemporaryDirectory(dir=self._temp_dir) as tmp_dir:
            tmp_path = pathlib.Path(tmp_dir)

            # Create paths for intermediate files
            query_fasta = tmp_path / "query.fasta"
            query_db = tmp_path / "queryDB"
            result_db = tmp_path / "resultDB"
            aln_tab = tmp_path / "alignments.tab"
            msa_db = tmp_path / "msaDB"
            output_dir = tmp_path / "output"
            tmp_search = tmp_path / "tmp"

            # Create subdirectories
            output_dir.mkdir()
            tmp_search.mkdir()

            # Step 1: Write query sequence to FASTA file
            subprocess_utils.create_query_fasta_file(
                sequence=query_sequence, path=str(query_fasta)
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

            # Step 4: Convert results to tabular format to get alignment coordinates
            alignment_info = self._run_convertalis(
                query_db=str(query_db),
                target_db=self._database_path,
                result_db=str(result_db),
                output_file=str(aln_tab),
            )

            # Step 5: Generate MSA in A3M format
            self._run_result2msa(
                query_db=str(query_db),
                target_db=self._database_path,
                result_db=str(result_db),
                msa_db=str(msa_db),
            )

            # Step 6: Unpack MSA database to files
            self._run_unpackdb(
                msa_db=str(msa_db),
                output_dir=str(output_dir),
            )

            # Step 7: Read and reformat A3M output with proper headers
            a3m_file = output_dir / "0"
            if a3m_file.exists():
                raw_a3m = a3m_file.read_text()
                a3m_content = self._reformat_a3m_headers(raw_a3m, alignment_info)
            else:
                # No hits found
                logging.warning("No template hits found for query sequence.")
                a3m_content = ""

        search_time = time.time() - search_start_time
        logging.info(
            "MMseqs2 template search completed in %.2f seconds, found %d hits",
            search_time,
            len(alignment_info),
        )

        return a3m_content

    def query_pipelined(
        self,
        query_sequence: str,
        executor: futures.ThreadPoolExecutor,
    ) -> futures.Future[str]:
        """Runs MMseqs2 template search with GPU synchronously, post-processing asynchronously.

        This method allows pipelining multiple database searches. The GPU search
        runs synchronously (blocking), but the CPU-bound post-processing (convertalis,
        result2msa, unpackdb) is submitted to the executor and runs in parallel with
        subsequent GPU searches.

        Args:
            query_sequence: The protein sequence to search for templates.
            executor: ThreadPoolExecutor to run post-processing tasks.

        Returns:
            A Future that resolves to A3M string when post-processing completes.
        """
        logging.info(
            "MMseqs2 pipelined template search for sequence: %s",
            query_sequence
            if len(query_sequence) <= 16
            else f"{query_sequence[:16]}... (len {len(query_sequence)})",
        )

        search_start_time = time.time()

        # Create temp directory manually (no context manager) so it persists
        # until post-processing completes
        # Use self._temp_dir for HPC clusters with fast local storage
        tmp_dir = tempfile.mkdtemp(prefix="mmseqs_template_", dir=self._temp_dir)
        tmp_path = pathlib.Path(tmp_dir)

        # Create paths for intermediate files
        query_fasta = tmp_path / "query.fasta"
        query_db = tmp_path / "queryDB"
        result_db = tmp_path / "resultDB"
        aln_tab = tmp_path / "alignments.tab"
        msa_db = tmp_path / "msaDB"
        output_dir = tmp_path / "output"
        tmp_search = tmp_path / "tmp"

        # Create subdirectories
        output_dir.mkdir()
        tmp_search.mkdir()

        # Step 1: Write query sequence to FASTA file
        subprocess_utils.create_query_fasta_file(
            sequence=query_sequence, path=str(query_fasta)
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
            "MMseqs2 GPU template search completed in %.2f seconds for sequence %s",
            search_time,
            query_sequence[:16] if len(query_sequence) > 16 else query_sequence,
        )

        # Submit post-processing to executor (runs async while next search starts)
        return executor.submit(
            self._postprocess_and_cleanup,
            tmp_dir=tmp_dir,
            query_db=str(query_db),
            result_db=str(result_db),
            aln_tab=str(aln_tab),
            msa_db=str(msa_db),
            output_dir=str(output_dir),
            query_sequence=query_sequence,
            search_start_time=search_start_time,
        )

    def _postprocess_and_cleanup(
        self,
        tmp_dir: str,
        query_db: str,
        result_db: str,
        aln_tab: str,
        msa_db: str,
        output_dir: str,
        query_sequence: str,
        search_start_time: float,
    ) -> str:
        """Runs post-processing (convertalis, result2msa, unpackdb) and cleans up.

        This method is designed to be run in a thread pool executor, allowing
        CPU-bound post-processing to run in parallel with GPU searches.

        Args:
            tmp_dir: Path to temporary directory (will be cleaned up).
            query_db: Path to query database.
            result_db: Path to result database.
            aln_tab: Path for alignment tabular output.
            msa_db: Path for MSA database output.
            output_dir: Path for unpacked output files.
            query_sequence: Original query sequence.
            search_start_time: Time when search started (for logging).

        Returns:
            A3M string containing template hits. Empty string if no hits found.
        """
        try:
            # Step 4: Convert results to tabular format to get alignment coordinates
            alignment_info = self._run_convertalis(
                query_db=query_db,
                target_db=self._database_path,
                result_db=result_db,
                output_file=aln_tab,
            )

            # Step 5: Generate MSA in A3M format
            self._run_result2msa(
                query_db=query_db,
                target_db=self._database_path,
                result_db=result_db,
                msa_db=msa_db,
            )

            # Step 6: Unpack MSA database to files
            self._run_unpackdb(
                msa_db=msa_db,
                output_dir=output_dir,
            )

            # Step 7: Read and reformat A3M output with proper headers
            a3m_file = pathlib.Path(output_dir) / "0"
            if a3m_file.exists():
                raw_a3m = a3m_file.read_text()
                a3m_content = self._reformat_a3m_headers(raw_a3m, alignment_info)
            else:
                # No hits found
                logging.warning("No template hits found for query sequence.")
                a3m_content = ""

            total_time = time.time() - search_start_time
            logging.info(
                "MMseqs2 template total (search + postprocess) completed in %.2f seconds, found %d hits",
                total_time,
                len(alignment_info),
            )

            return a3m_content
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
            cmd_name="MMseqs2 createdb (template)",
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
            str(self._max_hits),
        ]

        if self._gpu_enabled:
            cmd.extend(["--gpu", "1"])

        subprocess_utils.run(
            cmd=cmd,
            cmd_name="MMseqs2 template search",
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
            cmd_name="MMseqs2 result2msa (template)",
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
            cmd_name="MMseqs2 unpackdb (template)",
            log_stdout=False,
            log_stderr=True,
            log_on_process_error=True,
        )

    def _run_convertalis(
        self,
        query_db: str,
        target_db: str,
        result_db: str,
        output_file: str,
    ) -> dict[str, tuple[int, int, int]]:
        """Converts search results to tabular format and extracts alignment info.

        Uses MMseqs2 convertalis to get alignment coordinates needed for
        reformatting A3M headers to match the expected template format.

        Args:
          query_db: Path to query database.
          target_db: Path to target database.
          result_db: Path to result database.
          output_file: Path to output tabular file.

        Returns:
          Dictionary mapping target_id to (tstart, tend, tlen) tuple.
          tstart/tend are 1-indexed alignment coordinates in the target.
          tlen is the full length of the target sequence.
        """
        # Output format: target, tstart, tend, tlen
        # tstart and tend are 0-indexed in MMseqs2 output
        cmd = [
            self._binary_path,
            "convertalis",
            query_db,
            target_db,
            result_db,
            output_file,
            "--format-output",
            "target,tstart,tend,tlen",
            # Note: No --threads limit to allow full CPU utilization
        ]
        subprocess_utils.run(
            cmd=cmd,
            cmd_name="MMseqs2 convertalis (template)",
            log_stdout=False,
            log_stderr=True,
            log_on_process_error=True,
        )

        # Parse the tabular output
        alignment_info = {}
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 4:
                        target_id = parts[0]
                        # Convert from 0-indexed to 1-indexed
                        tstart = int(parts[1]) + 1
                        tend = int(parts[2])  # end is already exclusive in MMseqs2
                        tlen = int(parts[3])
                        # Store first occurrence (best hit) for each target
                        if target_id not in alignment_info:
                            alignment_info[target_id] = (tstart, tend, tlen)

        # Log sample of alignment_info keys for debugging
        sample_keys = list(alignment_info.keys())[:5] if alignment_info else []
        logging.info(
            "convertalis alignment_info: %d entries, sample keys: %s",
            len(alignment_info),
            sample_keys,
        )

        return alignment_info

    def _reformat_a3m_headers(
        self, raw_a3m: str, alignment_info: dict[str, tuple[int, int, int]]
    ) -> str:
        """Reformats A3M headers to include alignment coordinates.

        Converts MMseqs2 A3M headers from:
          >101m_A mol:protein length:154
        To the format expected by templates.py:
          >101m_A/1-154 [subseq from] mol:protein length:154

        Args:
          raw_a3m: Raw A3M string from MMseqs2.
          alignment_info: Dictionary mapping target_id to (start, end, full_length).

        Returns:
          Reformatted A3M string with proper headers for template parsing.
        """
        if not raw_a3m:
            return ""

        lines = raw_a3m.split("\n")
        # Log first few lines of raw A3M for debugging
        header_lines = [l for l in lines[:20] if l.startswith(">")]
        logging.info(
            "Reformatting A3M: %d lines, first headers: %s",
            len(lines),
            header_lines[:3],
        )

        # Parse A3M into (header, sequence) pairs
        entries = []
        current_header = None
        current_seq_lines = []

        for line in lines:
            if line.startswith(">"):
                # Save previous entry
                if current_header is not None:
                    entries.append((current_header, "".join(current_seq_lines)))
                current_header = line
                current_seq_lines = []
            elif current_header is not None:
                current_seq_lines.append(line)

        # Don't forget the last entry
        if current_header is not None:
            entries.append((current_header, "".join(current_seq_lines)))

        reformatted_lines = []

        for header_line, sequence in entries:
            header = header_line[1:]  # Remove ">"

            # Skip query sequence
            if header.lower() == "query" or header.lower().startswith("query "):
                continue

            # Extract target ID (first space-separated token)
            parts = header.split(" ", 1)
            full_target_id = parts[0]

            # result2msa may already add /start-end to the header, so strip it
            # for lookup. E.g., "5iby_A/2-23" -> "5iby_A"
            if "/" in full_target_id:
                base_target_id = full_target_id.split("/")[0]
            else:
                base_target_id = full_target_id

            # Look up alignment info - try both full ID (with range) and base ID
            if full_target_id in alignment_info:
                _, _, full_length = alignment_info[full_target_id]
            elif base_target_id in alignment_info:
                _, _, full_length = alignment_info[base_target_id]
            else:
                full_length = len(sequence)  # Fallback to sequence length

            # The template parser validates: len(matching_sequence) == end_index - start_index
            # where matching_sequence = hmmsearch_sequence.replace('-', '').upper()
            # So we must count length the same way (excluding gap characters)
            seq_length = len(sequence.replace("-", ""))

            # Format: >PDBID_CHAIN/START-END [subseq from] mol:protein length:N
            # The template parser does: start_index = start - 1, end_index = end
            # Validation: len(seq) == end_index - start_index == end - (start - 1) == end - start + 1
            # For seq starting at position 1: end - 1 + 1 = end = seq_length
            # So we use /1-{seq_length} to match the processed sequence length
            new_header = f">{base_target_id}/1-{seq_length} [subseq from] mol:protein length:{full_length}"

            logging.info(
                "Reformatted header: %s -> %s (seq_len=%d)",
                header_line[:40],
                new_header[:60],
                seq_length,
            )

            reformatted_lines.append(new_header)
            reformatted_lines.append(sequence)

        return "\n".join(reformatted_lines)
