# Copyright 2024 DeepMind Technologies Limited
#
# MSA comparison utilities for benchmark testing.

"""Utilities for comparing MSAs from different search tools.

This module provides functions to compare MSAs produced by different tools
(e.g., Jackhmmer vs MMseqs2-GPU) and compute various similarity metrics.
"""

import re
import string
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from alphafold3.data import msa as msa_module
from alphafold3.data import msa_features


# Table to remove lowercase characters (insertions in A3M format)
_DELETION_TABLE = str.maketrans("", "", string.ascii_lowercase)


@dataclass
class MsaComparisonResult:
    """Result of comparing two MSAs."""

    # Basic depth metrics
    depth_msa1: int
    depth_msa2: int

    # Unique sequences (after removing insertions)
    unique_seqs_msa1: int
    unique_seqs_msa2: int

    # Overlap metrics
    overlap_count: int  # Number of sequences in both MSAs
    jaccard_index: float  # |intersection| / |union|
    overlap_ratio_msa1: float  # overlap / msa1 unique seqs
    overlap_ratio_msa2: float  # overlap / msa2 unique seqs

    # Species diversity
    species_msa1: int  # Unique species in MSA1
    species_msa2: int  # Unique species in MSA2
    species_overlap: int  # Species in both MSAs
    species_jaccard: float  # Species Jaccard index

    # Sequences only in one MSA
    only_in_msa1: int
    only_in_msa2: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "depth_msa1": self.depth_msa1,
            "depth_msa2": self.depth_msa2,
            "unique_seqs_msa1": self.unique_seqs_msa1,
            "unique_seqs_msa2": self.unique_seqs_msa2,
            "overlap_count": self.overlap_count,
            "jaccard_index": self.jaccard_index,
            "overlap_ratio_msa1": self.overlap_ratio_msa1,
            "overlap_ratio_msa2": self.overlap_ratio_msa2,
            "species_msa1": self.species_msa1,
            "species_msa2": self.species_msa2,
            "species_overlap": self.species_overlap,
            "species_jaccard": self.species_jaccard,
            "only_in_msa1": self.only_in_msa1,
            "only_in_msa2": self.only_in_msa2,
        }


def normalize_sequence(seq: str) -> str:
    """Normalize a sequence by removing insertions (lowercase) and gaps.

    Args:
        seq: Sequence string potentially with lowercase insertions.

    Returns:
        Normalized sequence with only uppercase letters.
    """
    return seq.translate(_DELETION_TABLE)


def get_unique_sequences(msa: msa_module.Msa) -> set[str]:
    """Get set of unique sequences from an MSA.

    Sequences are normalized by removing lowercase insertions before comparison.

    Args:
        msa: The MSA object.

    Returns:
        Set of unique normalized sequences.
    """
    return {normalize_sequence(seq) for seq in msa.sequences}


def get_unique_sequences_from_a3m(a3m_string: str) -> set[str]:
    """Get set of unique sequences from an A3M string.

    Args:
        a3m_string: A3M formatted MSA string.

    Returns:
        Set of unique normalized sequences.
    """
    sequences = []
    current_seq = []

    for line in a3m_string.strip().split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if current_seq:
                sequences.append(''.join(current_seq))
                current_seq = []
        else:
            current_seq.append(line)

    if current_seq:
        sequences.append(''.join(current_seq))

    return {normalize_sequence(seq) for seq in sequences}


def get_species_from_descriptions(descriptions: Sequence[str]) -> set[str]:
    """Extract unique species IDs from MSA descriptions.

    Args:
        descriptions: List of description lines from MSA.

    Returns:
        Set of unique species IDs.
    """
    species_ids = msa_features.extract_species_ids(descriptions)
    # Filter out empty strings
    return {s for s in species_ids if s}


def compare_msas(
    msa1: msa_module.Msa,
    msa2: msa_module.Msa,
) -> MsaComparisonResult:
    """Compare two MSAs and compute similarity metrics.

    Args:
        msa1: First MSA (e.g., from Jackhmmer).
        msa2: Second MSA (e.g., from MMseqs2-GPU).

    Returns:
        MsaComparisonResult with comparison metrics.
    """
    # Get unique sequences
    seqs1 = get_unique_sequences(msa1)
    seqs2 = get_unique_sequences(msa2)

    # Calculate overlaps
    overlap = seqs1 & seqs2
    union = seqs1 | seqs2
    only_in_1 = seqs1 - seqs2
    only_in_2 = seqs2 - seqs1

    # Get species diversity
    species1 = get_species_from_descriptions(msa1.descriptions)
    species2 = get_species_from_descriptions(msa2.descriptions)
    species_overlap = species1 & species2
    species_union = species1 | species2

    return MsaComparisonResult(
        depth_msa1=msa1.depth,
        depth_msa2=msa2.depth,
        unique_seqs_msa1=len(seqs1),
        unique_seqs_msa2=len(seqs2),
        overlap_count=len(overlap),
        jaccard_index=len(overlap) / len(union) if union else 0.0,
        overlap_ratio_msa1=len(overlap) / len(seqs1) if seqs1 else 0.0,
        overlap_ratio_msa2=len(overlap) / len(seqs2) if seqs2 else 0.0,
        species_msa1=len(species1),
        species_msa2=len(species2),
        species_overlap=len(species_overlap),
        species_jaccard=len(species_overlap) / len(species_union) if species_union else 0.0,
        only_in_msa1=len(only_in_1),
        only_in_msa2=len(only_in_2),
    )


def compare_a3m_strings(
    a3m1: str,
    a3m2: str,
    query_sequence: str,
    chain_poly_type: str = "polypeptide(L)",
) -> MsaComparisonResult:
    """Compare two A3M strings and compute similarity metrics.

    Args:
        a3m1: First A3M string.
        a3m2: Second A3M string.
        query_sequence: The query sequence used for both MSAs.
        chain_poly_type: Polymer type of the query.

    Returns:
        MsaComparisonResult with comparison metrics.
    """
    msa1 = msa_module.Msa.from_a3m(
        query_sequence=query_sequence,
        chain_poly_type=chain_poly_type,
        a3m=a3m1,
        deduplicate=True,
    )
    msa2 = msa_module.Msa.from_a3m(
        query_sequence=query_sequence,
        chain_poly_type=chain_poly_type,
        a3m=a3m2,
        deduplicate=True,
    )
    return compare_msas(msa1, msa2)


def compute_aggregate_stats(results: list[MsaComparisonResult]) -> dict[str, float]:
    """Compute aggregate statistics from multiple MSA comparisons.

    Args:
        results: List of MsaComparisonResult objects.

    Returns:
        Dictionary with mean and std for each metric.
    """
    if not results:
        return {}

    import numpy as np

    metrics = [
        "depth_msa1", "depth_msa2",
        "unique_seqs_msa1", "unique_seqs_msa2",
        "overlap_count", "jaccard_index",
        "overlap_ratio_msa1", "overlap_ratio_msa2",
        "species_msa1", "species_msa2",
        "species_overlap", "species_jaccard",
        "only_in_msa1", "only_in_msa2",
    ]

    stats = {}
    for metric in metrics:
        values = [getattr(r, metric) for r in results]
        stats[f"{metric}_mean"] = float(np.mean(values))
        stats[f"{metric}_std"] = float(np.std(values))
        stats[f"{metric}_min"] = float(np.min(values))
        stats[f"{metric}_max"] = float(np.max(values))

    return stats


def format_comparison_report(result: MsaComparisonResult, name1: str = "MSA1", name2: str = "MSA2") -> str:
    """Format a comparison result as a human-readable report.

    Args:
        result: The comparison result.
        name1: Name for the first MSA.
        name2: Name for the second MSA.

    Returns:
        Formatted report string.
    """
    lines = [
        f"MSA Comparison Report: {name1} vs {name2}",
        "=" * 50,
        "",
        "Depth:",
        f"  {name1}: {result.depth_msa1} sequences",
        f"  {name2}: {result.depth_msa2} sequences",
        "",
        "Unique Sequences:",
        f"  {name1}: {result.unique_seqs_msa1}",
        f"  {name2}: {result.unique_seqs_msa2}",
        "",
        "Overlap:",
        f"  Shared sequences: {result.overlap_count}",
        f"  Jaccard index: {result.jaccard_index:.3f}",
        f"  % of {name1} in overlap: {result.overlap_ratio_msa1*100:.1f}%",
        f"  % of {name2} in overlap: {result.overlap_ratio_msa2*100:.1f}%",
        f"  Only in {name1}: {result.only_in_msa1}",
        f"  Only in {name2}: {result.only_in_msa2}",
        "",
        "Species Diversity:",
        f"  {name1}: {result.species_msa1} species",
        f"  {name2}: {result.species_msa2} species",
        f"  Shared species: {result.species_overlap}",
        f"  Species Jaccard: {result.species_jaccard:.3f}",
    ]
    return "\n".join(lines)
