"""
Shared data-loading utilities.

Centralizes the "load CSV/TSV with separator auto-detection, then shuffle with a
fixed seed and split into shards" logic that must stay IDENTICAL across the
pipeline (extract_embeddings, analyze_features, visualize_features, sequence
analysis). Any divergence here would silently misalign embeddings, labels, and
protein/residue mappings.
"""

import re
import hashlib
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


def validate_sequence_label_lengths(
    df: pd.DataFrame, sequence_column: str, label_column: str
) -> None:
    """Fail before embedding/evaluation if residue labels do not align."""
    missing = [c for c in (sequence_column, label_column) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    lengths = df[sequence_column].astype(str).str.len()
    label_lengths = df[label_column].fillna("").astype(str).str.len()
    bad = lengths != label_lengths
    if bad.any():
        examples = df.loc[bad, [sequence_column, label_column]].head(3)
        raise ValueError(
            f"Found {int(bad.sum())} sequence/label length mismatch(es) in "
            f"{sequence_column!r}/{label_column!r}; refusing to pad or truncate "
            f"labels silently. Examples: {examples.to_dict('records')}"
        )


def sequence_hash(sequence: str) -> str:
    """Stable identity for a normalized protein sequence."""
    normalized = str(sequence).strip().upper()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_residue_mapping_from_metadata(metadata_path, shard_df, sequence_column):
    """Build token mapping from extraction metadata instead of fixed truncation."""
    from collections import Counter
    metadata = pd.read_csv(metadata_path, dtype=str)
    key = "Entry" if "Entry" in metadata.columns and "Entry" in shard_df.columns else "sequence_hash"
    if key == "sequence_hash":
        protein_keys = [sequence_hash(seq) for seq in shard_df[sequence_column].astype(str)]
    else:
        protein_keys = shard_df[key].astype(str).tolist()
    # Occurrence-based keys: duplicated proteins (same sequence_hash/Entry) are
    # common in real datasets. A plain {key: index} dict would silently collapse
    # them to the LAST row and misassign every earlier duplicate's residues. The
    # metadata rows are written in the same protein order as shard_df, so counting
    # occurrences on both sides keeps each duplicate aligned to the correct row.
    shard_occurrences = Counter()
    key_to_index = {}
    for index, value in enumerate(protein_keys):
        occurrence = shard_occurrences[value]
        shard_occurrences[value] += 1
        key_to_index[(value, occurrence)] = index
    if key not in metadata.columns:
        raise ValueError(f"Residue metadata {metadata_path} lacks {key}")
    if not {"position", "sequence_hash"}.issubset(metadata.columns):
        raise ValueError(f"Residue metadata {metadata_path} lacks position/sequence_hash")
    if key == "sequence_hash" and len(shard_occurrences) != len(protein_keys):
        print(
            f"[residue-mapping] {len(protein_keys) - len(shard_occurrences)} duplicated "
            f"sequence(s) in the shard input; aligning residues by occurrence order."
        )
    protein_ids = []
    respos = []
    # The metadata rows are grouped per protein (extract_embeddings /
    # build_concept_matrix write all residues of one protein before the next),
    # with each protein's residue positions running 0..L-1 and then resetting to
    # 0. So a new protein block starts when the key changes OR the position
    # resets to 0 — the latter is required to separate ADJACENT proteins that
    # share the same sequence/hash. Counting per residue row would over-count a
    # single protein's occurrences.
    metadata_occurrences = Counter()
    prev_key = None
    block_occurrence = None
    for _, row in metadata.iterrows():
        protein_key = str(row[key])
        position = int(row["position"])
        if protein_key != prev_key or position == 0:
            block_occurrence = metadata_occurrences[protein_key]
            metadata_occurrences[protein_key] += 1
            prev_key = protein_key
        index = key_to_index.get((protein_key, block_occurrence))
        if index is None:
            raise ValueError(
                f"Residue metadata protein {protein_key!r} (occurrence "
                f"{block_occurrence}) is absent from the shard input; the "
                "length-filter / max_sequences parameters must match "
                "extract_embeddings."
            )
        protein_ids.append(index)
        respos.append(position)
    return protein_ids, np.asarray(respos, dtype=np.int64)


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
) -> Tuple[List[pd.DataFrame], Dict[int, List[int]], Dict[int, np.ndarray]]:
    """
    Rebuild the per-shard token -> (protein, residue) mapping.

    Replicates the exact shuffle+shard used by extract_embeddings, so each token
    maps back to (protein index, residue position within protein).

    IMPORTANT: min_seq_len/max_seq_len/max_sequences MUST match whatever filter
    was used at extraction, or the protein/residue mapping silently misaligns.

    Returns:
        shards: list of shard DataFrames (index by shard id), so callers can
                access shard-local sequences via ``shards[shard][sequence_column]``
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
    return shards, shard_proteins, shard_respos
