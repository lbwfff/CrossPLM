"""
Shared data-loading utilities.

Centralizes the "load CSV/TSV with separator auto-detection, then shuffle with a
fixed seed and split into shards" logic that must stay IDENTICAL across the
pipeline (extract_embeddings, analyze_features, visualize_features, sequence
analysis). Any divergence here would silently misalign embeddings, labels, and
protein/residue mappings.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def detect_separator(path) -> str:
    """Auto-detect TSV vs CSV separator from the first line."""
    with open(path, "r") as f:
        first = f.readline()
    return "\t" if first.count("\t") > first.count(",") else ","


def load_sequences_df(
    path,
    sequence_column: str = "sequence",
    min_seq_len: int = 0,
    max_seq_len: int = 10_000,
    max_sequences: Optional[int] = None,
) -> pd.DataFrame:
    """Load a sequences CSV/TSV with separator auto-detection, optional length
    filter, and an optional deterministic subset.

    `max_sequences` takes a fixed-seed random sample AFTER the length filter, so
    every pipeline step that calls this (extract + analysis) draws the SAME
    sequences — pass the same value everywhere to keep embeddings/protein
    mappings aligned.
    """
    sep = detect_separator(path)
    df = pd.read_csv(path, sep=sep, low_memory=False)
    if sequence_column not in df.columns:
        raise ValueError(f"'{sequence_column}' column not found in {path}")
    df[sequence_column] = df[sequence_column].fillna("").astype(str)
    if min_seq_len > 0 or max_seq_len < 10_000:
        df = df[df[sequence_column].apply(
            lambda s: min_seq_len <= len(s) <= max_seq_len
        )]
    if max_sequences is not None and len(df) > max_sequences:
        df = df.sample(n=max_sequences, random_state=42).reset_index(drop=True)
    return df


def shuffled_shards(
    df: pd.DataFrame, n_shards: int, seed: int = 42
) -> List[pd.DataFrame]:
    """
    Shuffle a DataFrame with a FIXED seed then split into n_shards sequential chunks.

    MUST match every other call site so embedding shards, concept shards, and
    protein mappings stay aligned.
    """
    if len(df) == 0:
        raise ValueError("shuffled_shards: input DataFrame is empty "
                         "(no sequences after any length filter)")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    shard_size = int(np.ceil(len(df) / n_shards))
    return [df.iloc[i:i + shard_size].reset_index(drop=True)
            for i in range(0, len(df), shard_size)]


def build_residue_positions(
    sequences_csv,
    shard_ids: List[int],
    n_shards: int = 5,
    max_residues: int = 510,
    sequence_column: str = "sequence",
    min_seq_len: int = 0,
    max_seq_len: int = 10_000,
    max_sequences: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[int, List[int]], Dict[int, np.ndarray]]:
    """
    Rebuild the per-shard token -> (protein, residue) mapping.

    Replicates the exact shuffle+shard used by extract_embeddings, so each token
    maps back to (protein index, residue position within protein).

    IMPORTANT: min_seq_len/max_seq_len/max_sequences MUST match whatever filter
    was used at extraction, or the protein/residue mapping silently misaligns.

    Returns:
        df: the shuffled full DataFrame
        shard_proteins: {shard_id: [protein_index per token]}
        shard_respos:   {shard_id: np.ndarray of residue positions per token}
    """
    df = load_sequences_df(sequences_csv, sequence_column=sequence_column,
                           min_seq_len=min_seq_len, max_seq_len=max_seq_len,
                           max_sequences=max_sequences)
    shards = shuffled_shards(df, n_shards)

    shard_proteins: Dict[int, List[int]] = {}
    shard_respos: Dict[int, np.ndarray] = {}
    for sid in shard_ids:
        if sid >= len(shards):
            raise ValueError(f"shard {sid} out of range (0-{len(shards)-1})")
        proteins: List[int] = []
        respos: List[int] = []
        for prot_idx, seq in enumerate(shards[sid][sequence_column].astype(str)):
            seq_len = min(len(seq), max_residues)
            proteins.extend([prot_idx] * seq_len)
            respos.extend(range(seq_len))
        shard_proteins[sid] = proteins
        shard_respos[sid] = np.array(respos, dtype=np.int64)
    return df, shard_proteins, shard_respos
