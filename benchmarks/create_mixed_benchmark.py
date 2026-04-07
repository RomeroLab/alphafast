#!/usr/bin/env python3
# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Create a mixed-type benchmark test set from PDB mmCIF files.

Extends create_benchmark_test_set.py with RNA and DNA categories.

Samples structures across 5 categories:
  - protein-monomer:  1 protein chain, no RNA/DNA, no meaningful ligands
  - protein-ligand:   1 protein chain + ≥1 real ligand, no RNA/DNA
  - protein-protein:  ≥2 protein chains, no RNA/DNA
  - protein-rna:      ≥1 protein chain + ≥1 RNA chain
  - protein-dna:      ≥1 protein chain + ≥1 DNA chain (no RNA)

Requires alphafold3 module (run inside Docker container or with uv run).

Usage:
    # Inside Docker
    python benchmarks/create_mixed_benchmark.py \
        --mmcif_dir /root/mmcif_files \
        --output_dir benchmarks/benchmark_set_mixed_40 \
        --samples_per_category 8

    # With uv
    uv run python benchmarks/create_mixed_benchmark.py \
        --mmcif_dir /path/to/mmcif_files \
        --output_dir benchmarks/benchmark_set_mixed_40
"""

import argparse
import dataclasses
import datetime
import json
import logging
import os
import random
import sys
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from alphafold3.structure import mmcif as mmcif_lib


# Common crystallization artifacts / ions / buffers to exclude from ligand count.
# Without this filter, almost every PDB structure has ligand_count > 0 and the
# protein-monomer category is effectively empty.
_ARTIFACT_CCD_CODES = frozenset({
    # Ions
    "NA", "CL", "MG", "ZN", "CA", "K", "MN", "FE", "FE2", "CO", "NI",
    "CU", "CU1", "CD", "IOD", "BR", "XE",
    # Common buffers / cryo / crystallization agents
    "SO4", "PO4", "GOL", "EDO", "PEG", "PGE", "MPD", "DMS", "ACT",
    "FMT", "TRS", "CIT", "BME", "MES", "EPE", "IMD", "SCN", "NO3",
    "AZI", "1PE", "P6G", "MLI", "TAR", "SUC", "NH4",
    # Unknown ligand placeholder — zero atoms, crashes inference
    "UNL",
    # Water
    "HOH", "DOD",
})

CATEGORIES = [
    "protein-monomer",
    "protein-ligand",
    "protein-protein",
    "protein-rna",
    "protein-dna",
]


@dataclasses.dataclass
class StructureInfo:
    """Information about a PDB structure for categorization."""

    pdb_id: str
    release_date: datetime.date
    resolution: float | None
    num_protein_chains: int
    num_rna_chains: int
    num_dna_chains: int
    num_ligands: int
    protein_sequences: dict[str, str]  # chain_id -> sequence
    rna_sequences: dict[str, str]
    dna_sequences: dict[str, str]
    ligand_ccd_ids: list[str]  # only real ligands (artifacts excluded)
    total_residues: int
    mmcif_path: str
    category: str = ""


def parse_mmcif_for_info(
    mmcif_path: str, cutoff_date: datetime.date
) -> StructureInfo | None:
    """Parse an mmCIF file and extract structure information."""
    try:
        with open(mmcif_path, "r") as f:
            mmcif_string = f.read()

        mmcif = mmcif_lib.from_string(mmcif_string)

        # Get release date
        release_date_str = mmcif_lib.get_release_date(mmcif)
        if not release_date_str:
            return None

        release_date = datetime.date.fromisoformat(release_date_str)
        if release_date > cutoff_date:
            return None

        # Get resolution
        resolution = mmcif_lib.get_resolution(mmcif)

        # Get entity information
        entity_ids = mmcif.get("_entity.id", [])
        entity_types = mmcif.get("_entity.type", [])
        entity_type_by_id = dict(zip(entity_ids, entity_types))

        # Get polymer types and sequences
        poly_entity_ids = mmcif.get("_entity_poly.entity_id", [])
        poly_types = mmcif.get("_entity_poly.type", [])
        poly_type_by_entity_id = dict(zip(poly_entity_ids, poly_types))

        poly_seqs = mmcif.get("_entity_poly.pdbx_seq_one_letter_code_can", [])
        poly_seq_by_entity_id = dict(zip(poly_entity_ids, poly_seqs))

        # Map chains to entities
        chain_ids = mmcif.get("_struct_asym.id", [])
        chain_entity_ids = mmcif.get("_struct_asym.entity_id", [])
        entity_by_chain = dict(zip(chain_ids, chain_entity_ids))

        # Count chain types and extract sequences
        protein_sequences = {}
        rna_sequences = {}
        dna_sequences = {}
        total_residues = 0

        for chain_id, entity_id in entity_by_chain.items():
            entity_type = entity_type_by_id.get(entity_id, "")
            poly_type = poly_type_by_entity_id.get(entity_id, "").lower()

            if entity_type != "polymer":
                continue

            seq = poly_seq_by_entity_id.get(entity_id, "")
            seq = "".join(c for c in seq if c.isalpha())

            if "polypeptide" in poly_type:
                protein_sequences[chain_id] = seq
                total_residues += len(seq)
            elif "polyribonucleotide" in poly_type and "deoxy" not in poly_type:
                rna_sequences[chain_id] = seq
                total_residues += len(seq)
            elif "polydeoxyribonucleotide" in poly_type:
                dna_sequences[chain_id] = seq
                total_residues += len(seq)

        # Get ligand CCD IDs (excluding artifacts)
        nonpoly_entity_ids = [
            eid for eid, etype in entity_type_by_id.items() if etype == "non-polymer"
        ]
        comp_ids = mmcif.get("_pdbx_entity_nonpoly.comp_id", [])
        entity_nonpoly_ids = mmcif.get("_pdbx_entity_nonpoly.entity_id", [])

        ligand_ccd_ids = []
        for comp_id, entity_id in zip(comp_ids, entity_nonpoly_ids):
            if entity_id in nonpoly_entity_ids:
                if comp_id.upper() not in _ARTIFACT_CCD_CODES:
                    ligand_ccd_ids.append(comp_id)

        pdb_id = Path(mmcif_path).stem.lower()

        return StructureInfo(
            pdb_id=pdb_id,
            release_date=release_date,
            resolution=resolution,
            num_protein_chains=len(protein_sequences),
            num_rna_chains=len(rna_sequences),
            num_dna_chains=len(dna_sequences),
            num_ligands=len(ligand_ccd_ids),
            protein_sequences=protein_sequences,
            rna_sequences=rna_sequences,
            dna_sequences=dna_sequences,
            ligand_ccd_ids=ligand_ccd_ids,
            total_residues=total_residues,
            mmcif_path=mmcif_path,
        )

    except Exception as e:
        logging.debug(f"Error parsing {mmcif_path}: {e}")
        return None


def categorize_structure(info: StructureInfo) -> str | None:
    """Categorize a structure into one of 5 categories."""
    if info.num_protein_chains == 0:
        return None

    # RNA/DNA categories take priority
    if info.num_protein_chains >= 1 and info.num_rna_chains >= 1:
        return "protein-rna"
    if info.num_protein_chains >= 1 and info.num_dna_chains >= 1:
        return "protein-dna"

    # Protein-only categories
    if info.num_protein_chains >= 2:
        return "protein-protein"
    if info.num_protein_chains == 1 and info.num_ligands >= 1:
        return "protein-ligand"
    if info.num_protein_chains == 1 and info.num_ligands == 0:
        return "protein-monomer"

    return None


def filter_structure(
    info: StructureInfo,
    max_resolution: float = 3.0,
    min_seq_length: int = 50,
    max_seq_length: int = 500,
    max_total_residues: int = 1500,
) -> bool:
    """Filter structure based on quality criteria."""
    if info.resolution is not None and info.resolution > max_resolution:
        return False

    # Check protein sequence lengths
    for seq in info.protein_sequences.values():
        if len(seq) < min_seq_length or len(seq) > max_seq_length:
            return False

    if info.total_residues > max_total_residues:
        return False

    return True


def create_af3_input_json(info: StructureInfo) -> dict[str, Any]:
    """Create AF3-compatible input JSON from structure info."""
    sequences = []
    chain_idx = 0

    # Add protein chains
    for chain_id, sequence in sorted(info.protein_sequences.items()):
        chain_label = chr(ord("A") + chain_idx)
        sequences.append({"protein": {"id": [chain_label], "sequence": sequence}})
        chain_idx += 1

    # Add RNA chains
    for chain_id, sequence in sorted(info.rna_sequences.items()):
        chain_label = chr(ord("A") + chain_idx)
        sequences.append({"rna": {"id": [chain_label], "sequence": sequence}})
        chain_idx += 1

    # Add DNA chains
    for chain_id, sequence in sorted(info.dna_sequences.items()):
        chain_label = chr(ord("A") + chain_idx)
        sequences.append({"dna": {"id": [chain_label], "sequence": sequence}})
        chain_idx += 1

    # Add ligands (artifacts already filtered out)
    for ccd_id in info.ligand_ccd_ids:
        if chain_idx >= 26:
            break
        chain_label = chr(ord("A") + chain_idx)
        sequences.append({"ligand": {"id": chain_label, "ccdCodes": [ccd_id]}})
        chain_idx += 1

    return {
        "name": info.pdb_id,
        "modelSeeds": [1],
        "sequences": sequences,
        "dialect": "alphafold3",
        "version": 3,
    }


def scan_mmcif_directory(
    mmcif_dir: str, cutoff_date: datetime.date, num_workers: int = 8
) -> list[StructureInfo]:
    """Scan mmCIF directory and extract structure information in parallel."""
    mmcif_files = list(Path(mmcif_dir).glob("*.cif"))
    logging.info(f"Found {len(mmcif_files)} mmCIF files to scan")

    results = []
    processed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(parse_mmcif_for_info, str(f), cutoff_date): f
            for f in mmcif_files
        }

        for future in as_completed(futures):
            processed += 1
            if processed % 10000 == 0:
                logging.info(
                    f"Processed {processed}/{len(mmcif_files)} files, "
                    f"found {len(results)} valid structures"
                )

            result = future.result()
            if result is not None:
                results.append(result)

    logging.info(f"Scan complete: found {len(results)} structures on or before {cutoff_date}")
    return results


def _compute_per_category_counts(
    total_samples: int, category_names: Sequence[str]
) -> dict[str, int]:
    """Distribute total_samples across categories as evenly as possible."""
    sorted_cats = sorted(category_names)
    base = total_samples // len(sorted_cats)
    remainder = total_samples % len(sorted_cats)
    return {
        cat: base + (1 if i < remainder else 0) for i, cat in enumerate(sorted_cats)
    }


def create_test_set(
    mmcif_dir: str,
    output_dir: str,
    cutoff_date: datetime.date,
    samples_per_category: int | None = None,
    total_samples: int | None = None,
    max_resolution: float = 3.0,
    min_seq_length: int = 50,
    max_seq_length: int = 500,
    max_total_residues: int = 1500,
    num_workers: int = 8,
    seed: int = 42,
):
    """Create the benchmark test set."""
    random.seed(seed)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Scan mmCIF directory
    logging.info("Scanning mmCIF directory...")
    structures = scan_mmcif_directory(mmcif_dir, cutoff_date, num_workers)

    # Categorize and filter
    logging.info("Categorizing and filtering structures...")
    categories: dict[str, list[StructureInfo]] = {cat: [] for cat in CATEGORIES}

    for info in structures:
        category = categorize_structure(info)
        if category is None or category not in categories:
            continue
        if not filter_structure(
            info, max_resolution, min_seq_length, max_seq_length, max_total_residues
        ):
            continue
        info.category = category
        categories[category].append(info)

    logging.info("Categorized structures:")
    for cat in CATEGORIES:
        logging.info(f"  {cat}: {len(categories[cat])} structures")

    # Compute per-category sample counts
    if total_samples is not None:
        per_category_counts = _compute_per_category_counts(
            total_samples, CATEGORIES
        )
        logging.info(f"Distributing {total_samples} total samples: {per_category_counts}")
    else:
        if samples_per_category is None:
            samples_per_category = 8
        per_category_counts = {cat: samples_per_category for cat in CATEGORIES}

    # Sample
    sampled = {}
    for category, items in categories.items():
        target = per_category_counts[category]
        if len(items) <= target:
            sampled[category] = items
            logging.warning(
                f"  {category}: only {len(items)} available, "
                f"using all (requested {target})"
            )
        else:
            sampled[category] = random.sample(items, target)
            logging.info(f"  {category}: sampled {target} from {len(items)}")

    # Save JSON files with category prefix (flat structure)
    index = {
        "cutoff_date": cutoff_date.isoformat(),
        "samples_per_category": per_category_counts,
        "total_samples": total_samples,
        "filters": {
            "max_resolution": max_resolution,
            "min_seq_length": min_seq_length,
            "max_seq_length": max_seq_length,
            "max_total_residues": max_total_residues,
        },
        "categories": {},
        "total_count": 0,
    }

    for category, items in sampled.items():
        # Use underscore prefix for file names (protein-rna -> protein_rna)
        cat_prefix = category.replace("-", "_")
        category_index = []

        for info in items:
            af3_input = create_af3_input_json(info)

            json_filename = f"{cat_prefix}_{info.pdb_id}_input.json"
            json_path = output_path / json_filename
            with open(json_path, "w") as f:
                json.dump(af3_input, f, indent=2)

            entry = {
                "pdb_id": info.pdb_id,
                "category": category,
                "release_date": info.release_date.isoformat(),
                "resolution": info.resolution,
                "num_protein_chains": info.num_protein_chains,
                "num_rna_chains": info.num_rna_chains,
                "num_dna_chains": info.num_dna_chains,
                "num_ligands": info.num_ligands,
                "total_residues": info.total_residues,
                "input_json": json_filename,
                "ground_truth_mmcif": info.mmcif_path,
            }
            category_index.append(entry)

        index["categories"][category] = category_index
        index["total_count"] += len(items)

    # Save master index
    index_path = output_path / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logging.info(f"\nTest set created at {output_dir}")
    logging.info(f"Total structures: {index['total_count']}")
    for cat in CATEGORIES:
        n = len(sampled.get(cat, []))
        logging.info(f"  {cat}: {n}")
    logging.info(f"Index file: {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create a mixed-type benchmark test set from PDB mmCIF files."
    )
    parser.add_argument(
        "--mmcif_dir", required=True,
        help="Directory containing mmCIF files (*.cif)",
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for AF3 JSON files",
    )
    parser.add_argument(
        "--cutoff_date", type=str, default="2021-09-30",
        help="Only include structures released on or before this date (default: 2021-09-30)",
    )
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--samples_per_category", type=int, default=None,
        help="Number of samples per category (default: 8)",
    )
    sample_group.add_argument(
        "--total_samples", type=int, default=None,
        help="Total number of samples, distributed evenly across 5 categories.",
    )
    parser.add_argument(
        "--max_resolution", type=float, default=3.0,
        help="Maximum resolution in Angstroms (default: 3.0)",
    )
    parser.add_argument(
        "--min_seq_length", type=int, default=50,
        help="Minimum protein sequence length per chain (default: 50)",
    )
    parser.add_argument(
        "--max_seq_length", type=int, default=500,
        help="Maximum protein sequence length per chain (default: 500)",
    )
    parser.add_argument(
        "--max_total_residues", type=int, default=1500,
        help="Maximum total residues across all chains (default: 1500)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of parallel workers for scanning (default: 8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cutoff_date = datetime.date.fromisoformat(args.cutoff_date)

    create_test_set(
        mmcif_dir=args.mmcif_dir,
        output_dir=args.output_dir,
        cutoff_date=cutoff_date,
        samples_per_category=args.samples_per_category,
        total_samples=args.total_samples,
        max_resolution=args.max_resolution,
        min_seq_length=args.min_seq_length,
        max_seq_length=args.max_seq_length,
        max_total_residues=args.max_total_residues,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
