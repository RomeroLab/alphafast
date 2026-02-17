# Copyright 2026 Romero Lab, Duke University
#
# Licensed under CC-BY-NC-SA 4.0. This file is part of AlphaFast,
# a derivative work of AlphaFold 3 by DeepMind Technologies Limited.
# https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Library to run ESMFold structure prediction locally from Python.

ESMFold is a fast structure prediction method that can generate protein
structures directly from sequence without MSA. This is useful for the
Foldseek mode where we need a structure to search against AFDB.

Requirements:
  - torch
  - transformers (pip install transformers)
  - GPU with ~16GB VRAM for full model

This implementation uses the Hugging Face transformers library which has
a self-contained ESMFold implementation without external openfold dependency.
"""

import dataclasses
import time
from typing import Sequence

from absl import logging
import numpy as np


@dataclasses.dataclass(frozen=True, slots=True)
class ESMFoldResult:
    """Result from ESMFold structure prediction.

    Attributes:
        sequence: The input protein sequence.
        pdb_string: The predicted structure in PDB format.
        plddt_scores: Per-residue pLDDT confidence scores (0-100).
        mean_plddt: Mean pLDDT score across all residues.
    """

    sequence: str
    pdb_string: str
    plddt_scores: Sequence[float]
    mean_plddt: float


class ESMFold:
    """Wrapper for ESMFold local inference using Hugging Face transformers.

    This class provides structure prediction using the ESMFold model from
    the transformers library. The model is lazily loaded on first use to
    avoid memory allocation when Foldseek mode is not enabled.
    """

    # Valid amino acids
    VALID_AAS = set('ACDEFGHIKLMNPQRSTVWY')

    def __init__(
        self,
        *,
        model_name: str = "facebook/esmfold_v1",
        device: str | None = None,
        chunk_size: int | None = None,
    ):
        """Initializes the ESMFold wrapper.

        Args:
            model_name: The ESMFold model to use from Hugging Face.
                Default is "facebook/esmfold_v1".
            device: Device to run inference on. If None, automatically selects
                CUDA if available, otherwise CPU.
            chunk_size: Optional chunk size for processing long sequences.
                Reduces memory usage but may affect prediction quality.
                If None, processes the full sequence at once.

        Raises:
            ImportError: If torch or transformers packages are not installed.
        """
        self._model_name = model_name
        self._device = device
        self._chunk_size = chunk_size
        self._model = None
        self._tokenizer = None
        self._torch = None

    def _load_model(self) -> None:
        """Lazily loads the ESMFold model from transformers."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import EsmForProteinFolding, EsmTokenizer
        except ImportError as e:
            raise ImportError(
                "ESMFold requires 'torch' and 'transformers' packages. Install with: "
                "pip install torch transformers"
            ) from e

        self._torch = torch

        # Determine device
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logging.info("Loading ESMFold model '%s' on %s...", self._model_name, self._device)
        load_start_time = time.time()

        # Load ESMFold model and tokenizer from transformers
        self._tokenizer = EsmTokenizer.from_pretrained(self._model_name)
        self._model = EsmForProteinFolding.from_pretrained(self._model_name)
        self._model = self._model.to(self._device)
        self._model.eval()

        # Enable chunk processing if specified
        if self._chunk_size is not None:
            self._model.trunk.set_chunk_size(self._chunk_size)

        logging.info(
            "ESMFold model loaded in %.2f seconds",
            time.time() - load_start_time,
        )

    def predict(self, sequence: str) -> ESMFoldResult:
        """Predicts structure from a protein sequence.

        Args:
            sequence: The protein sequence to predict structure for.

        Returns:
            ESMFoldResult containing the predicted structure and confidence scores.

        Raises:
            RuntimeError: If prediction fails.
        """
        self._load_model()

        # Clean sequence - keep only valid amino acids
        clean_seq = ''.join(c for c in sequence.upper() if c in self.VALID_AAS)
        seq_len = len(clean_seq)

        if seq_len == 0:
            raise ValueError("Input sequence contains no valid amino acids")

        logging.info(
            "Running ESMFold prediction for sequence: %s",
            clean_seq if seq_len <= 16 else f"{clean_seq[:16]}... (len {seq_len})",
        )

        predict_start_time = time.time()

        with self._torch.no_grad():
            # Tokenize input
            inputs = self._tokenizer(
                [clean_seq],
                return_tensors="pt",
                add_special_tokens=False
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            # Run prediction
            outputs = self._model(**inputs)

            # Extract pLDDT scores - handle different output shapes
            plddt_raw = outputs.plddt
            if plddt_raw.dim() == 3:
                # Shape: [batch, num_recycles, seq_len] - take last recycle
                plddt = plddt_raw[0, -1].cpu().float().numpy()[:seq_len]
            elif plddt_raw.dim() == 2:
                # Shape: [batch, seq_len]
                plddt = plddt_raw[0].cpu().float().numpy()[:seq_len]
            else:
                # Shape: [seq_len] or flattened
                plddt = plddt_raw.cpu().float().numpy().flatten()[:seq_len]

            plddt_scores = plddt.tolist()
            mean_plddt = float(np.mean(plddt))

            # Convert to PDB format using model's built-in method if available
            try:
                # HuggingFace ESMFold has a convert_outputs_to_pdb method
                pdb_string = self._model.output_to_pdb(outputs)[0]
            except (AttributeError, TypeError):
                # Fallback to manual conversion
                pdb_string = self._outputs_to_pdb(outputs, clean_seq, plddt)

        predict_time = time.time() - predict_start_time
        logging.info(
            "ESMFold prediction completed in %.2f seconds (mean pLDDT: %.1f)",
            predict_time,
            mean_plddt,
        )

        return ESMFoldResult(
            sequence=clean_seq,
            pdb_string=pdb_string,
            plddt_scores=plddt_scores,
            mean_plddt=mean_plddt,
        )

    def _outputs_to_pdb(self, outputs, sequence: str, plddt: np.ndarray) -> str:
        """Convert ESMFold outputs to PDB format.

        Args:
            outputs: Model outputs from ESMFold.
            sequence: The protein sequence.
            plddt: Per-residue pLDDT scores.

        Returns:
            PDB format string.
        """
        try:
            from transformers.models.esm.openfold_utils.protein import to_pdb, Protein
            from transformers.models.esm.openfold_utils import residue_constants
        except ImportError:
            # Fallback: return minimal PDB from CA positions only
            return self._minimal_pdb(outputs, sequence, plddt)

        seq_len = len(sequence)

        # Extract positions from outputs - handle different tensor shapes
        # HuggingFace ESMFold outputs.positions can have different shapes
        positions_raw = outputs.positions
        if positions_raw.dim() == 5:
            # Shape: [batch, num_recycles, seq_len, 14, 3] - take last recycle
            raw_pos = positions_raw[0, -1].cpu().float().numpy()
        elif positions_raw.dim() == 4:
            # Shape: [batch, seq_len, 14, 3]
            raw_pos = positions_raw[0].cpu().float().numpy()
        else:
            # Shape: [seq_len, 14, 3]
            raw_pos = positions_raw.cpu().float().numpy()

        # Ensure we have the right shape [seq_len, 14, 3]
        if raw_pos.shape[0] != seq_len:
            raw_pos = raw_pos[:seq_len]

        # Convert atom14 to atom37 format
        # ESMFold outputs 14 atoms, we need to expand to 37 atom format
        final_positions = np.zeros((seq_len, 37, 3), dtype=np.float32)

        # Map atom14 indices to atom37 indices
        # atom14 order: N, CA, C, O, CB, CG, CG1, CG2, OG, OG1, CD, CD1, CD2, OD1, ...
        # We copy available atoms to their atom37 positions
        num_atoms = min(14, raw_pos.shape[1]) if len(raw_pos.shape) > 1 else 14
        for i in range(num_atoms):
            if i < 37:
                final_positions[:, i, :] = raw_pos[:, i, :]

        # Atom mask - handle different tensor shapes
        atom_exists_raw = outputs.atom37_atom_exists
        if atom_exists_raw.dim() == 3:
            # Shape: [batch, seq_len, 37]
            raw_mask = atom_exists_raw[0].cpu().float().numpy()
        else:
            # Shape: [seq_len, 37]
            raw_mask = atom_exists_raw.cpu().float().numpy()
        final_mask = raw_mask[:seq_len]

        # Create protein object
        aa_order = 'ARNDCQEGHILKMFPSTWYV'
        aatype = np.array([aa_order.index(c) if c in aa_order else 0 for c in sequence], dtype=np.int32)
        b_factors = np.repeat(plddt.reshape(-1, 1), 37, axis=1)

        protein = Protein(
            aatype=aatype,
            atom_positions=final_positions,
            atom_mask=final_mask,
            residue_index=np.arange(seq_len, dtype=np.int32),
            b_factors=b_factors,
            chain_index=np.zeros(seq_len, dtype=np.int32),
        )

        return to_pdb(protein)

    def _minimal_pdb(self, outputs, sequence: str, plddt: np.ndarray) -> str:
        """Create minimal PDB with CA atoms only as fallback.

        Args:
            outputs: Model outputs from ESMFold.
            sequence: The protein sequence.
            plddt: Per-residue pLDDT scores.

        Returns:
            Minimal PDB format string with CA atoms.
        """
        # Three letter codes
        aa_3letter = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU', 'F': 'PHE',
            'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'K': 'LYS', 'L': 'LEU',
            'M': 'MET', 'N': 'ASN', 'P': 'PRO', 'Q': 'GLN', 'R': 'ARG',
            'S': 'SER', 'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
        }

        seq_len = len(sequence)

        # Extract positions from outputs - handle different tensor shapes
        positions_raw = outputs.positions
        if positions_raw.dim() == 5:
            # Shape: [batch, num_recycles, seq_len, 14, 3] - take last recycle
            raw_pos = positions_raw[0, -1].cpu().float().numpy()
        elif positions_raw.dim() == 4:
            # Shape: [batch, seq_len, 14, 3]
            raw_pos = positions_raw[0].cpu().float().numpy()
        else:
            # Shape: [seq_len, 14, 3]
            raw_pos = positions_raw.cpu().float().numpy()

        # Extract CA positions (atom index 1 in atom14 format)
        lines = []
        for i, aa in enumerate(sequence):
            res_name = aa_3letter.get(aa, 'UNK')
            bfactor = plddt[i] if i < len(plddt) else 0.0

            # CA is at index 1 in atom14 format
            try:
                x, y, z = raw_pos[i, 1, :]  # residue, atom (CA=1), xyz
            except (IndexError, ValueError):
                x, y, z = 0.0, 0.0, 0.0

            # PDB ATOM format
            line = (
                f"ATOM  {i+1:5d}  CA  {res_name:3s} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bfactor:6.2f}           C"
            )
            lines.append(line)

        lines.append("END")
        return "\n".join(lines)

    def _extract_plddt_from_pdb(self, pdb_string: str) -> list[float]:
        """Extracts per-residue pLDDT scores from PDB B-factor column.

        ESMFold stores pLDDT scores in the B-factor (temperature factor)
        column of the PDB file.

        Args:
            pdb_string: PDB format string from ESMFold.

        Returns:
            List of pLDDT scores, one per residue.
        """
        plddt_per_residue = {}

        for line in pdb_string.split("\n"):
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                # PDB format: columns 23-26 are residue number, 61-66 are B-factor
                try:
                    res_num = int(line[22:26].strip())
                    bfactor = float(line[60:66].strip())
                    plddt_per_residue[res_num] = bfactor
                except (ValueError, IndexError):
                    continue

        # Return scores in residue order
        if plddt_per_residue:
            max_res = max(plddt_per_residue.keys())
            return [plddt_per_residue.get(i, 0.0) for i in range(1, max_res + 1)]
        return []

    def unload(self) -> None:
        """Unloads the model to free GPU memory.

        Call this before running AlphaFold 3 inference to free up GPU memory
        for the main model.
        """
        if self._model is not None:
            logging.info("Unloading ESMFold model to free GPU memory...")
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None

            if self._torch is not None and "cuda" in str(self._device):
                self._torch.cuda.empty_cache()

            logging.info("ESMFold model unloaded")

    def __del__(self):
        """Cleanup on deletion."""
        self.unload()
