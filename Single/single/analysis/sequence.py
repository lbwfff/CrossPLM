"""
Sequence-level analysis of SAE features: sequential Cohen's d + motif enrichment.

Goal: characterize what a feature responds to *along the amino-acid sequence*,
using only sequence data (no 3D structures needed).

Two analyses:

1. sequential Cohen's d
   For a feature, does it activate residues that CLUSTER along the sequence
   (local motif-like) or are SPREAD OUT (global/periodic)?
     - collect per-protein residue indices where the feature activates
     - compute the distribution of pairwise sequence gaps among activated residues
     - compare to the pairwise-gap distribution of randomly sampled positions
       (same number per protein)
     - d = (mean_gap_activated - mean_gap_random) / pooled_sd
   d < 0  → activated residues are closer together than random (local/clustered)
   d ≈ 0  → sequence distribution ~ random
   d > 0  → activated residues are more spread out than random (dispersed)

2. motif enrichment
   Which amino acids / dipeptides are over-represented in a window around the
   feature's activated residues (vs the background sequence composition)?
     - window = [pos - flank, pos + flank] for each activated residue
     - count amino-acid (and optionally dipeptide) frequencies in the window
     - background = global amino-acid composition of all sequences
     - enrichment = log2(observed/expected); report with z-score or Fisher-style
"""

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from single.sae.dictionary import Dictionary
from single.sae.inference import get_sae_feats_in_batches, split_up_feature_list


# Standard amino acids (single-letter)
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


# ---------------------------------------------------------------------------
# Token -> (protein, residue index) mapping
#
# extract_embeddings.py shuffles the CSV (sample(frac=1, random_state=42)) then
# splits into shards. We replicate that exact order so each embedding token can
# be mapped back to (protein, residue-position-in-protein).
# ---------------------------------------------------------------------------

def build_residue_positions(
    sequences_csv: str,
    shard_ids: List[int],
    n_shards: int = 5,
    max_residues: int = 510,
    sequence_column: str = "sequence",
) -> Tuple[List[str], Dict[int, np.ndarray]]:
    """
    Rebuild the per-shard token -> (protein, residue) mapping.

    Returns:
        shard_proteins: for each shard, list of (protein_index) per token
        shard_respos: for each shard, array of residue position within protein
    """
    import re

    df = pd.read_csv(sequences_csv, sep=None, engine="python")
    if sequence_column not in df.columns:
        raise ValueError(f"'{sequence_column}' column not found in {sequences_csv}")
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    shard_size = int(np.ceil(len(df) / n_shards))
    shards = [df.iloc[i:i + shard_size].reset_index(drop=True)
              for i in range(0, len(df), shard_size)]

    shard_proteins: Dict[int, List[int]] = {}
    shard_respos: Dict[int, List[int]] = {}
    for sid in shard_ids:
        if sid >= len(shards):
            raise ValueError(f"shard {sid} out of range (0-{len(shards)-1})")
        proteins: List[int] = []
        respos: List[int] = []
        shard_df = shards[sid]
        for prot_idx, seq in enumerate(shard_df[sequence_column].astype(str)):
            seq_len = min(len(seq), max_residues)
            proteins.extend([prot_idx] * seq_len)
            respos.extend(range(seq_len))
        shard_proteins[sid] = proteins
        shard_respos[sid] = np.array(respos, dtype=np.int64)
    return df, shard_proteins, shard_respos


def feature_activation_positions(
    sae: Dictionary,
    embeddings: torch.Tensor,
    protein_ids: List[int],
    respos: np.ndarray,
    feature_idx: int,
    activation_threshold: float = 0.0,
    batch_size: int = 4096,
    device: str = "cpu",
) -> np.ndarray:
    """
    Return a boolean array over tokens indicating where `feature_idx` activates.

    Args:
        sae, embeddings: trained SAE and the (concatenated) embeddings for one shard
        protein_ids: protein index per token
        respos: residue position (within protein) per token
        feature_idx: the feature to examine
        activation_threshold: activation must be > this to count as "active"

    Returns:
        boolean array [n_tokens]
    """
    feats = get_sae_feats_in_batches(
        sae=sae, aa_embds=embeddings, chunk_size=batch_size,
        feat_list=[feature_idx], normalize_features=True, device=device,
    )
    active = (feats[:, 0] > activation_threshold).cpu().numpy()
    return active


# ---------------------------------------------------------------------------
# Sequential Cohen's d
# ---------------------------------------------------------------------------

def _pairwise_gaps(pos: np.ndarray) -> np.ndarray:
    """Pairwise sequence gaps among a set of 0-based residue positions."""
    if len(pos) < 2:
        return np.array([], dtype=np.float64)
    pos_sorted = np.sort(pos)
    # consecutive gaps capture clustering; also use all pairs for robustness
    diffs = np.diff(pos_sorted)
    return diffs.astype(np.float64)


def sequential_cohens_d(
    active_positions: Dict[int, np.ndarray],
    protein_lengths: List[int],
    n_random_draws: int = 50,
    seed: int = 0,
) -> float:
    """
    Compute sequential Cohen's d for a feature.

    Args:
        active_positions: protein_index -> array of activated residue positions
        protein_lengths: length (in residues) of each protein
        n_random_draws: number of random samples to build the null distribution

    Returns:
        Cohen's d; negative = clustered (local), ~0 = random, positive = dispersed.
    """
    rng = np.random.RandomState(seed)

    # Observed consecutive gaps for the feature's activated residues.
    obs_gaps: List[float] = []
    for prot_idx, pos in active_positions.items():
        if len(pos) >= 2:
            obs_gaps.extend(_pairwise_gaps(pos).tolist())

    if not obs_gaps:
        return 0.0

    # Null: randomly place the same number of residues per protein, same gaps.
    rand_gaps: List[float] = []
    for prot_idx, pos in active_positions.items():
        L = protein_lengths[prot_idx]
        n_act = len(pos)
        if L < 2 or n_act < 2:
            continue
        for _ in range(n_random_draws):
            rand_pos = rng.choice(L, size=min(n_act, L), replace=False)
            rand_gaps.extend(_pairwise_gaps(rand_pos).tolist())

    if not rand_gaps:
        return 0.0

    obs_mean = float(np.mean(obs_gaps))
    rand_mean = float(np.mean(rand_gaps))
    pooled_sd = float(np.sqrt((np.var(obs_gaps) + np.var(rand_gaps)) / 2))
    if pooled_sd < 1e-12:
        return 0.0
    return (obs_mean - rand_mean) / pooled_sd


# ---------------------------------------------------------------------------
# Motif enrichment
# ---------------------------------------------------------------------------

def _background_composition(
    proteins: List[str], max_residues: int = 510
) -> Dict[str, float]:
    """Global single-residue composition of all sequences (for expected counts)."""
    counts: Counter = Counter()
    total = 0
    for seq in proteins:
        for aa in str(seq)[:max_residues]:
            if aa in AA_ALPHABET:
                counts[aa] += 1
                total += 1
    if total == 0:
        return {}
    return {aa: c / total for aa, c in counts.items()}


def motif_enrichment(
    sae: Dictionary,
    embeddings: torch.Tensor,
    proteins: List[str],
    protein_ids: List[int],
    respos: np.ndarray,
    feature_idx: int,
    flank: int = 3,
    activation_threshold: float = 0.0,
    batch_size: int = 4096,
    device: str = "cpu",
) -> Dict:
    """
    Compute amino-acid enrichment in a window around the feature's activated residues.

    Args:
        sae, embeddings: SAE and one shard's embeddings
        proteins: full protein sequences (all proteins in the experiment)
        protein_ids: protein index per token
        respos: residue position per token
        feature_idx: feature to examine
        flank: window radius around each activated residue

    Returns:
        dict with:
          'amino_acid_enrichment': {aa: log2 fold-enrichment}
          'n_active_residues': count of activated residues
          'flank': window radius used
    """
    active = feature_activation_positions(
        sae, embeddings, protein_ids, respos, feature_idx,
        activation_threshold=activation_threshold, device=device,
    )

    # Background composition over all proteins.
    bg = _background_composition(proteins)
    if not bg:
        return {"amino_acid_enrichment": {}, "n_active_residues": 0, "flank": flank}

    # Collect amino acids in windows around activated residues.
    window_counts: Counter = Counter()
    n_active = 0
    for i, is_active in enumerate(active):
        if not is_active:
            continue
        n_active += 1
        prot_idx = protein_ids[i]
        pos = int(respos[i])
        seq = str(proteins[prot_idx])
        for j in range(max(0, pos - flank), min(len(seq), pos + flank + 1)):
            aa = seq[j]
            if aa in AA_ALPHABET:
                window_counts[aa] += 1

    if not window_counts:
        return {"amino_acid_enrichment": {}, "n_active_residues": 0, "flank": flank}

    total_window = sum(window_counts.values())
    enrichment = {}
    for aa in AA_ALPHABET:
        obs = window_counts[aa] / total_window
        exp = bg.get(aa, 0.0)
        if exp <= 0:
            continue
        enrichment[aa] = float(np.log2(obs / exp + 1e-12))

    return {
        "amino_acid_enrichment": enrichment,
        "n_active_residues": n_active,
        "flank": flank,
    }


def summarize_motif(motif: Dict, top_n: int = 5) -> List[Tuple[str, float]]:
    """Return the top-N most enriched amino acids (log2 fold)."""
    enrich = motif.get("amino_acid_enrichment", {})
    ranked = sorted(enrich.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
