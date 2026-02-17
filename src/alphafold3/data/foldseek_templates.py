# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Foldseek-based structural template search.

This module provides functionality to:
1. Predict structure from sequence using ESMFold
2. Search predicted structure against AFDB using Foldseek
3. Convert Foldseek hits to AlphaFold 3 Template objects
"""

import time
from collections.abc import Sequence

from absl import logging
from alphafold3.data import afdb_store
from alphafold3.data import folding_input
from alphafold3.data import msa_config
from alphafold3.data.tools import esmfold as esmfold_tool
from alphafold3.data.tools import foldseek as foldseek_tool


def get_foldseek_templates(
    sequence: str,
    foldseek_config: msa_config.FoldseekTemplatesConfig,
) -> Sequence[folding_input.Template]:
    """Gets structural templates using ESMFold + Foldseek.

    Workflow:
    1. Predict structure from sequence using ESMFold
    2. Search predicted structure against AFDB using Foldseek
    3. Fetch mmCIF files for hits from AFDB
    4. Convert to AF3 Template objects

    Args:
        sequence: The protein sequence to find templates for.
        foldseek_config: Configuration for ESMFold and Foldseek.

    Returns:
        List of Template objects for use in AlphaFold 3 prediction.
    """
    logging.info("Starting Foldseek template search for sequence length %d", len(sequence))
    start_time = time.time()

    # Step 1: Predict structure with ESMFold
    esmfold = esmfold_tool.ESMFold(
        model_name=foldseek_config.esmfold_config.model_name,
        device=foldseek_config.esmfold_config.device,
        chunk_size=foldseek_config.esmfold_config.chunk_size,
    )

    try:
        esmfold_result = esmfold.predict(sequence)
    finally:
        # Unload ESMFold to free GPU memory for AF3
        esmfold.unload()

    # Check pLDDT threshold
    if esmfold_result.mean_plddt < foldseek_config.esmfold_config.min_plddt:
        logging.info(
            "ESMFold prediction has low confidence (pLDDT=%.1f < %.1f), "
            "skipping Foldseek search",
            esmfold_result.mean_plddt,
            foldseek_config.esmfold_config.min_plddt,
        )
        return []

    # Step 2: Search with Foldseek
    foldseek = foldseek_tool.Foldseek(
        binary_path=foldseek_config.foldseek_config.binary_path,
        database_path=foldseek_config.foldseek_config.database_path,
        e_value=foldseek_config.foldseek_config.e_value,
        max_hits=foldseek_config.foldseek_config.max_hits,
        alignment_type=foldseek_config.foldseek_config.alignment_type,
        threads=foldseek_config.foldseek_config.threads,
        min_lddt=foldseek_config.foldseek_config.min_lddt,
        gpu_enabled=foldseek_config.foldseek_config.gpu_enabled,
        gpu_device=foldseek_config.foldseek_config.gpu_device,
    )

    foldseek_result = foldseek.search(esmfold_result.pdb_string)

    if not foldseek_result.hits:
        logging.info("No Foldseek hits found")
        return []

    # Step 3: Filter hits
    filtered_hits = _filter_foldseek_hits(
        hits=foldseek_result.hits,
        query_length=len(sequence),
        filter_config=foldseek_config.filter_config,
    )

    if not filtered_hits:
        logging.info("No Foldseek hits passed filtering")
        return []

    # Step 4: Convert to AF3 Templates
    afdb = afdb_store.AFDBStructureStore(
        cache_dir=foldseek_config.afdb_cache_dir,
    )

    templates = []
    for hit in filtered_hits:
        try:
            template = _convert_foldseek_hit_to_template(
                hit=hit,
                query_sequence=sequence,
                afdb_store=afdb,
            )
            templates.append(template)
        except Exception as e:
            logging.warning(
                "Failed to convert Foldseek hit %s to template: %s",
                hit.target_id,
                e,
            )
            continue

    elapsed_time = time.time() - start_time
    logging.info(
        "Foldseek template search completed in %.2f seconds, found %d templates",
        elapsed_time,
        len(templates),
    )

    return templates


def _filter_foldseek_hits(
    hits: Sequence[foldseek_tool.FoldseekHit],
    query_length: int,
    filter_config: msa_config.FoldseekFilterConfig,
) -> list[foldseek_tool.FoldseekHit]:
    """Filters Foldseek hits based on quality criteria.

    Args:
        hits: List of Foldseek hits to filter.
        query_length: Length of the query sequence.
        filter_config: Filtering configuration.

    Returns:
        Filtered list of hits.
    """
    filtered = []
    seen_uniprot_ids = set()

    for hit in hits:
        # Check LDDT threshold
        if hit.lddt < filter_config.min_lddt:
            continue

        # Check sequence identity threshold
        if hit.sequence_identity < filter_config.min_sequence_identity:
            continue

        # Check coverage threshold
        coverage = hit.aligned_length / query_length if query_length > 0 else 0
        if coverage < filter_config.min_coverage:
            continue

        # Deduplicate by UniProt ID
        if filter_config.deduplicate_by_uniprot:
            try:
                uniprot_id = _extract_uniprot_from_afdb_id(hit.target_id)
                if uniprot_id in seen_uniprot_ids:
                    continue
                seen_uniprot_ids.add(uniprot_id)
            except ValueError:
                # If we can't extract UniProt ID, include the hit
                pass

        filtered.append(hit)

        # Stop if we have enough hits
        if len(filtered) >= filter_config.max_hits:
            break

    return filtered


def _extract_uniprot_from_afdb_id(afdb_id: str) -> str:
    """Extracts UniProt ID from AFDB identifier.

    Args:
        afdb_id: AFDB identifier (e.g., "AF-Q9Y6K1-F1-model_v4").

    Returns:
        UniProt ID (e.g., "Q9Y6K1").
    """
    import re
    match = re.search(r"AF-([A-Z0-9]+)-F\d+-model_v\d+", afdb_id)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract UniProt ID from: {afdb_id}")


def _convert_foldseek_hit_to_template(
    hit: foldseek_tool.FoldseekHit,
    query_sequence: str,
    afdb_store: afdb_store.AFDBStructureStore,
) -> folding_input.Template:
    """Converts a Foldseek hit to an AF3 Template object.

    Args:
        hit: Foldseek hit to convert.
        query_sequence: The query protein sequence.
        afdb_store: Store for fetching AFDB structures.

    Returns:
        AF3 Template object.
    """
    # Fetch mmCIF from AFDB
    logging.debug("Fetching AFDB structure: %s", hit.target_id)
    mmcif_str = afdb_store.get_mmcif_str(hit.target_id)

    # Build query-to-template mapping from alignment
    query_to_template_map = _build_alignment_mapping(
        query_aligned=hit.query_aligned,
        target_aligned=hit.target_aligned,
        query_start=hit.query_start,
        target_start=hit.target_start,
    )

    return folding_input.Template(
        mmcif=mmcif_str,
        query_to_template_map=query_to_template_map,
    )


def _build_alignment_mapping(
    query_aligned: str,
    target_aligned: str,
    query_start: int,
    target_start: int,
) -> dict[int, int]:
    """Builds query-to-template residue mapping from alignment.

    Args:
        query_aligned: Aligned query sequence (with gaps).
        target_aligned: Aligned target sequence (with gaps).
        query_start: Start position in query (0-indexed).
        target_start: Start position in target (0-indexed).

    Returns:
        Dictionary mapping query residue indices to template residue indices.
    """
    mapping = {}
    query_pos = query_start
    target_pos = target_start

    for q_char, t_char in zip(query_aligned, target_aligned):
        if q_char != "-" and t_char != "-":
            # Both positions are aligned
            mapping[query_pos] = target_pos

        if q_char != "-":
            query_pos += 1
        if t_char != "-":
            target_pos += 1

    return mapping
