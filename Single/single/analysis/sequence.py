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
AA_TO_INDEX = {aa: i for i, aa in enumerate(AA_ALPHABET)}


# ---------------------------------------------------------------------------
# Token -> (protein, residue index) mapping
#
# The shuffle+shard logic is centralized in single.data.build_residue_positions
# (identical to extract_embeddings). We re-export it here for convenience.
# ---------------------------------------------------------------------------

from single.data import build_residue_positions  # noqa: E402

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
    proteins: List[str], max_residues: int = 510,
    protein_lengths: Optional[Dict[int, int]] = None,
) -> Dict[str, float]:
    """Global single-residue composition of all sequences (for expected counts)."""
    counts = _background_counts(
        proteins, max_residues=max_residues, protein_lengths=protein_lengths
    )
    total = 0
    total = sum(counts.values())
    if total == 0:
        return {}
    return {aa: c / total for aa, c in counts.items()}


def _background_counts(
    proteins: List[str], max_residues: int = 510,
    protein_lengths: Optional[Dict[int, int]] = None,
) -> Counter:
    counts: Counter = Counter()
    for index, seq in enumerate(proteins):
        if protein_lengths is not None and index not in protein_lengths:
            # The protein is not covered by the embedding/metadata mapping (e.g.
            # the analysis-side length filter / max_sequences differ from
            # extraction). Counting it would bias the background composition
            # toward proteins that contributed no activations.
            continue
        length = protein_lengths.get(index, max_residues) if protein_lengths else max_residues
        for aa in str(seq)[:length].upper():
            if aa in AA_ALPHABET:
                counts[aa] += 1
    return counts


def motif_enrichment_from_counts(
    window_counts: Dict[str, int],
    background_counts: Dict[str, int],
    flank: int,
    n_active: int,
) -> Dict:
    """Compute motif enrichment from raw counts, allowing shard pooling."""
    window_counts = Counter(window_counts)
    background_counts = Counter(background_counts)
    total_window = sum(window_counts.values())
    total_background = sum(background_counts.values())
    if total_window == 0 or total_background == 0:
        return {
            "amino_acid_enrichment": {},
            "n_active_residues": int(n_active),
            "flank": flank,
            "window_counts": dict(window_counts),
            "background_counts": dict(background_counts),
        }
    enrichment = {}
    for aa in AA_ALPHABET:
        observed = window_counts.get(aa, 0) / total_window
        expected = background_counts.get(aa, 0) / total_background
        if expected <= 0:
            continue
        enrichment[aa] = float(np.log2(observed / expected + 1e-12))
    return {
        "amino_acid_enrichment": enrichment,
        "n_active_residues": int(n_active),
        "flank": flank,
        "window_counts": dict(window_counts),
        "background_counts": dict(background_counts),
    }


def positional_motif_counts(
    proteins: List[str],
    protein_ids: np.ndarray,
    respos: np.ndarray,
    active_mask: np.ndarray,
    flank: int = 5,
    n_permutations: int = 200,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Collect position-specific observed and within-protein null counts.

    Each protein keeps its observed number of active centers in every
    permutation. This preserves protein identity, sequence composition and
    activation density while randomizing the relative positions.
    """
    if flank < 0:
        raise ValueError("flank must be non-negative")
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    protein_ids = np.asarray(protein_ids)
    respos = np.asarray(respos)
    active_mask = np.asarray(active_mask, dtype=bool)
    if not (len(protein_ids) == len(respos) == len(active_mask)):
        raise ValueError("protein_ids, respos and active_mask must have equal length")

    n_positions = 2 * flank + 1
    n_amino_acids = len(AA_ALPHABET)
    observed = np.zeros((n_positions, n_amino_acids), dtype=np.int64)
    null_counts = np.zeros(
        (n_permutations, n_positions, n_amino_acids), dtype=np.int64
    )
    active_by_protein: Dict[int, np.ndarray] = {}
    observed_lengths: Dict[int, int] = {}
    for pid in np.unique(protein_ids):
        token_indices = np.flatnonzero(protein_ids == pid)
        if len(token_indices):
            observed_lengths[int(pid)] = min(
                len(str(proteins[int(pid)])),
                int(respos[token_indices].max()) + 1,
            )
        active_indices = token_indices[active_mask[token_indices]]
        if len(active_indices):
            active_by_protein[int(pid)] = respos[active_indices].astype(np.int64)

    def add_centers(counts: np.ndarray, protein_id: int, centers: np.ndarray):
        sequence = str(proteins[protein_id]).upper()
        sequence_length = observed_lengths[protein_id]
        for center in centers:
            for offset in range(-flank, flank + 1):
                position = int(center) + offset
                if 0 <= position < sequence_length:
                    aa_index = AA_TO_INDEX.get(sequence[position])
                    if aa_index is not None:
                        counts[offset + flank, aa_index] += 1

    for pid, centers in active_by_protein.items():
        add_centers(observed, pid, centers)

    rng = np.random.RandomState(seed)
    for permutation in range(n_permutations):
        for pid, centers in active_by_protein.items():
            sequence_length = observed_lengths[pid]
            n_centers = len(centers)
            random_centers = rng.choice(
                sequence_length, size=n_centers, replace=False
            )
            add_centers(null_counts[permutation], pid, random_centers)

    return {
        "observed_counts": observed,
        "null_counts": null_counts,
        "n_active_centers": np.asarray(sum(len(v) for v in active_by_protein.values())),
    }


def summarize_positional_motif(
    observed_counts: np.ndarray,
    null_counts: np.ndarray,
    flank: int,
    n_active_centers: Optional[int] = None,
) -> Dict:
    """Calculate enrichment, permutation p-values and BH-FDR q-values."""
    observed = np.asarray(observed_counts, dtype=np.float64)
    null = np.asarray(null_counts, dtype=np.float64)
    if null.ndim != 3 or observed.shape != null.shape[1:]:
        raise ValueError("observed/null positional count shapes are incompatible")
    null_mean = null.mean(axis=0)
    log2_enrichment = np.log2((observed + 1e-12) / (null_mean + 1e-12))
    p_values = (1 + (null >= observed[None, ...]).sum(axis=0)) / (null.shape[0] + 1)
    flat_p = p_values.ravel()
    order = np.argsort(flat_p)
    ranked = flat_p[order] * len(flat_p) / np.arange(1, len(flat_p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q_values = np.empty_like(flat_p)
    q_values[order] = np.minimum(ranked, 1.0)
    position_labels = list(range(-flank, flank + 1))
    return {
        "offsets": position_labels,
        "amino_acids": list(AA_ALPHABET),
        "observed_counts": observed.astype(int).tolist(),
        "null_mean_counts": null_mean.tolist(),
        "log2_enrichment": log2_enrichment.tolist(),
        "p_values": p_values.tolist(),
        "q_values": q_values.reshape(p_values.shape).tolist(),
        "n_permutations": int(null.shape[0]),
        "n_active_centers": int(
            n_active_centers
            if n_active_centers is not None
            else observed.sum() / max(1, observed.shape[0])
        ),
    }


def draw_sequence_logo(
    observed_counts: np.ndarray,
    flank: int,
    output_path,
    title: Optional[str] = None,
) -> None:
    """Draw a dependency-free sequence logo from position-specific frequencies."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import PathPatch
    from matplotlib.textpath import TextPath
    from matplotlib.transforms import Affine2D

    counts = np.asarray(observed_counts, dtype=np.float64)
    totals = counts.sum(axis=1, keepdims=True)
    frequencies = counts / np.maximum(totals, 1.0)
    logo_height = np.zeros_like(frequencies)
    for i, row in enumerate(frequencies):
        entropy = -sum(p * np.log2(p) for p in row if p > 0)
        information = max(np.log2(len(AA_ALPHABET)) - entropy, 0.0)
        logo_height[i] = row * information

    fig, ax = plt.subplots(figsize=(max(8, counts.shape[0] * 0.65), 4.5))
    colors = {aa: plt.cm.tab20(j % 20) for j, aa in enumerate(AA_ALPHABET)}
    for col in range(counts.shape[0]):
        y = 0.0
        order = np.argsort(logo_height[col])
        for aa_index in order:
            height = float(logo_height[col, aa_index])
            if height <= 0:
                continue
            aa = AA_ALPHABET[aa_index]
            glyph = TextPath((0, 0), aa, size=1)
            bbox = glyph.get_extents()
            transform = (
                Affine2D()
                .scale(0.72 / max(bbox.width, 1e-6), height / max(bbox.height, 1e-6))
                .translate(col + 0.14, y)
                + ax.transData
            )
            ax.add_patch(PathPatch(glyph, transform=transform, color=colors[aa]))
            y += height
    ax.set_xlim(0, counts.shape[0])
    ax.set_ylim(0, max(1.0, float(logo_height.sum(axis=1).max()) * 1.1))
    ax.set_xticks(np.arange(counts.shape[0]) + 0.5)
    ax.set_xticklabels([str(i) for i in range(-flank, flank + 1)])
    ax.set_xlabel("Position relative to feature activation")
    ax.set_ylabel("Information (bits)")
    if title:
        ax.set_title(title)
    ax.axvline(flank + 0.5, color="black", linewidth=0.8, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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
    active_mask: Optional[np.ndarray] = None,
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
        active_mask: optional precomputed boolean [n_tokens] activation mask. If
            provided, avoids re-running the SAE encode (caller already computed it).

    Returns:
        dict with:
          'amino_acid_enrichment': {aa: log2 fold-enrichment}
          'n_active_residues': count of activated residues
          'flank': window radius used
    """
    # Reuse a precomputed activation mask if the caller provided one (avoids a
    # redundant SAE encode pass).
    if active_mask is not None:
        active = np.asarray(active_mask)
    else:
        active = feature_activation_positions(
            sae, embeddings, protein_ids, respos, feature_idx,
            activation_threshold=activation_threshold, device=device,
        )

    # Background composition over all proteins.
    protein_lengths = {}
    for pid in np.unique(protein_ids):
        token_indices = np.flatnonzero(np.asarray(protein_ids) == pid)
        protein_lengths[int(pid)] = min(
            len(str(proteins[int(pid)])),
            int(np.asarray(respos)[token_indices].max()) + 1,
        ) if len(token_indices) else 0
    background_counts = _background_counts(
        proteins, protein_lengths=protein_lengths
    )
    if not background_counts:
        return motif_enrichment_from_counts({}, {}, flank, 0)

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
        sequence_length = protein_lengths.get(prot_idx, len(seq))
        for j in range(max(0, pos - flank), min(sequence_length, pos + flank + 1)):
            aa = seq[j].upper()
            if aa in AA_ALPHABET:
                window_counts[aa] += 1

    return motif_enrichment_from_counts(
        window_counts, background_counts, flank, n_active
    )


def summarize_motif(motif: Dict, top_n: int = 5) -> List[Tuple[str, float]]:
    """Return the top-N most enriched amino acids (log2 fold)."""
    enrich = motif.get("amino_acid_enrichment", {})
    ranked = sorted(enrich.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]
