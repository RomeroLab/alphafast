# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Structure store for fetching and caching AlphaFold Database structures.

This module provides functionality to fetch mmCIF files from the AlphaFold
Database (AFDB) for use as structural templates in the Foldseek mode.
"""

import datetime
import hashlib
import os
import re
import time
from collections.abc import Mapping
from typing import Any

from absl import logging
import requests


# AFDB URL patterns
_AFDB_MMCIF_URL = "https://alphafold.ebi.ac.uk/files/{afdb_id}.cif"
_AFDB_PDB_URL = "https://alphafold.ebi.ac.uk/files/{afdb_id}.pdb"

# Regex to extract UniProt ID from AFDB identifier
# Format: AF-{UniProtID}-F{fragment}-model_v{version}
# Example: AF-Q9Y6K1-F1-model_v4
_AFDB_ID_PATTERN = re.compile(r"AF-([A-Z0-9]+)-F(\d+)-model_v(\d+)")

# Default release date for AFDB structures (v4 release)
_DEFAULT_AFDB_RELEASE_DATE = datetime.date(2022, 7, 22)


class AFDBStructureStore:
    """Fetches and caches AlphaFold Database structures.

    This class provides functionality to fetch mmCIF files from AFDB for
    structural templates identified by Foldseek. It supports optional local
    caching to avoid repeated downloads.
    """

    def __init__(
        self,
        cache_dir: str | None = None,
        download_timeout: int = 30,
        max_retries: int = 3,
    ):
        """Initializes the AFDB structure store.

        Args:
            cache_dir: Optional directory for caching downloaded structures.
                If None, structures are downloaded on-demand without caching.
            download_timeout: Timeout in seconds for HTTP requests.
            max_retries: Maximum number of retry attempts for failed downloads.
        """
        self._cache_dir = cache_dir
        self._download_timeout = download_timeout
        self._max_retries = max_retries

        # In-memory cache for structures fetched in this session
        self._memory_cache: dict[str, str] = {}

        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
            logging.info("AFDB structure cache directory: %s", cache_dir)

    def get_mmcif_str(self, afdb_id: str) -> str:
        """Fetches mmCIF structure for an AFDB entry.

        Args:
            afdb_id: AFDB identifier (e.g., "AF-Q9Y6K1-F1-model_v4").

        Returns:
            mmCIF string of the structure.

        Raises:
            ValueError: If the AFDB ID format is invalid.
            RuntimeError: If download fails after retries.
        """
        # Normalize ID format
        afdb_id = self._normalize_id(afdb_id)

        # Check memory cache first
        if afdb_id in self._memory_cache:
            return self._memory_cache[afdb_id]

        # Check disk cache if available
        if self._cache_dir is not None:
            cache_path = self._get_cache_path(afdb_id)
            if os.path.exists(cache_path):
                logging.debug("Loading AFDB structure from cache: %s", afdb_id)
                with open(cache_path) as f:
                    mmcif_str = f.read()
                self._memory_cache[afdb_id] = mmcif_str
                return mmcif_str

        # Download from AFDB
        mmcif_str = self._download_mmcif(afdb_id)

        # Cache to disk if directory specified
        if self._cache_dir is not None:
            cache_path = self._get_cache_path(afdb_id)
            with open(cache_path, "w") as f:
                f.write(mmcif_str)
            logging.debug("Cached AFDB structure: %s", afdb_id)

        # Cache in memory
        self._memory_cache[afdb_id] = mmcif_str

        return mmcif_str

    def get_pdb_str(self, afdb_id: str) -> str:
        """Fetches PDB structure for an AFDB entry.

        Args:
            afdb_id: AFDB identifier.

        Returns:
            PDB string of the structure.
        """
        afdb_id = self._normalize_id(afdb_id)
        return self._download_pdb(afdb_id)

    def get_uniprot_id(self, afdb_id: str) -> str:
        """Extracts UniProt ID from AFDB identifier.

        Args:
            afdb_id: AFDB identifier (e.g., "AF-Q9Y6K1-F1-model_v4").

        Returns:
            UniProt ID (e.g., "Q9Y6K1").

        Raises:
            ValueError: If the AFDB ID format is invalid.
        """
        match = _AFDB_ID_PATTERN.match(afdb_id)
        if not match:
            raise ValueError(
                f"Invalid AFDB ID format: {afdb_id}. "
                "Expected format: AF-{UniProtID}-F{fragment}-model_v{version}"
            )
        return match.group(1)

    def get_release_date(self, afdb_id: str) -> datetime.date:
        """Returns the release date for an AFDB structure.

        AFDB structures use a common release date based on the database version.
        The model version in the ID indicates which AFDB release it's from.

        Args:
            afdb_id: AFDB identifier.

        Returns:
            Release date of the structure.
        """
        # For now, return the default AFDB v4 release date
        # Could be extended to parse version and return appropriate date
        return _DEFAULT_AFDB_RELEASE_DATE

    def get_metadata(self, afdb_id: str) -> Mapping[str, Any]:
        """Returns metadata for an AFDB structure.

        Args:
            afdb_id: AFDB identifier.

        Returns:
            Dictionary with metadata fields compatible with structure_stores.
        """
        afdb_id = self._normalize_id(afdb_id)
        uniprot_id = self.get_uniprot_id(afdb_id)

        return {
            "afdb_id": afdb_id,
            "uniprot_id": uniprot_id,
            "seq_release_date": self.get_release_date(afdb_id).isoformat(),
            "seq_author_chain_id": "A",  # AFDB models are single-chain
            "seq_unresolved_res_num": "",  # AFDB models are complete
        }

    def _normalize_id(self, afdb_id: str) -> str:
        """Normalizes AFDB identifier format.

        Handles common variations in ID format from different sources.

        Args:
            afdb_id: Raw AFDB identifier.

        Returns:
            Normalized identifier.
        """
        # Strip whitespace
        afdb_id = afdb_id.strip()

        # Handle IDs without "AF-" prefix (e.g., from some Foldseek outputs)
        if not afdb_id.startswith("AF-") and _AFDB_ID_PATTERN.match(f"AF-{afdb_id}"):
            afdb_id = f"AF-{afdb_id}"

        # Validate format
        if not _AFDB_ID_PATTERN.match(afdb_id):
            # Try to extract valid AFDB ID from the string
            match = _AFDB_ID_PATTERN.search(afdb_id)
            if match:
                afdb_id = match.group(0)
            else:
                raise ValueError(f"Invalid AFDB ID format: {afdb_id}")

        return afdb_id

    def _get_cache_path(self, afdb_id: str) -> str:
        """Returns the cache file path for an AFDB ID."""
        # Use hash-based subdirectories to avoid too many files in one directory
        hash_prefix = hashlib.md5(afdb_id.encode()).hexdigest()[:2]
        subdir = os.path.join(self._cache_dir, hash_prefix)
        os.makedirs(subdir, exist_ok=True)
        return os.path.join(subdir, f"{afdb_id}.cif")

    def _download_mmcif(self, afdb_id: str) -> str:
        """Downloads mmCIF file from AFDB.

        Args:
            afdb_id: AFDB identifier.

        Returns:
            mmCIF string.

        Raises:
            RuntimeError: If download fails after retries.
        """
        url = _AFDB_MMCIF_URL.format(afdb_id=afdb_id)
        return self._download_with_retry(url, afdb_id)

    def _download_pdb(self, afdb_id: str) -> str:
        """Downloads PDB file from AFDB.

        Args:
            afdb_id: AFDB identifier.

        Returns:
            PDB string.

        Raises:
            RuntimeError: If download fails after retries.
        """
        url = _AFDB_PDB_URL.format(afdb_id=afdb_id)
        return self._download_with_retry(url, afdb_id)

    def _download_with_retry(self, url: str, afdb_id: str) -> str:
        """Downloads URL content with retry logic.

        Args:
            url: URL to download.
            afdb_id: AFDB identifier (for logging).

        Returns:
            Downloaded content as string.

        Raises:
            RuntimeError: If download fails after all retries.
        """
        last_error = None

        for attempt in range(self._max_retries):
            try:
                logging.debug("Downloading AFDB structure: %s (attempt %d)", afdb_id, attempt + 1)
                response = requests.get(url, timeout=self._download_timeout)
                response.raise_for_status()
                return response.text
            except requests.RequestException as e:
                last_error = e
                if attempt < self._max_retries - 1:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logging.warning(
                        "Failed to download %s: %s. Retrying in %d seconds...",
                        afdb_id,
                        e,
                        wait_time,
                    )
                    time.sleep(wait_time)

        raise RuntimeError(
            f"Failed to download AFDB structure {afdb_id} after {self._max_retries} "
            f"attempts: {last_error}"
        )

    def clear_memory_cache(self) -> None:
        """Clears the in-memory cache."""
        self._memory_cache.clear()
        logging.debug("Cleared AFDB memory cache")
