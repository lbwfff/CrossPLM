"""
Pairwise feature co-activation / co-localization analysis.

Answers: do two SAE features activate on the SAME residues, or on residues NEAR
each other along the sequence? This reveals whether features work together
(co-localized) or on disjoint subsets.

Metrics (all computed per-protein, then aggregated):
  - overlap_ab        : P(feature B active | feature A active)  (same residue)
  - overlap_ba        : P(feature A active | feature B active)  (same residue)
  - baseline_b        : P(feature B active) overall (unconditional)
  - baseline_a        : P(feature A active) overall
  - enrich_ab         : overlap_ab / baseline_b  (>1 → B is enriched on A's residues)
  - enrich_ba         : overlap_ba / baseline_a
  - neighbor_ab(+k)   : P(B active within ±k residues of an A-active residue),
                        per residue-position, minus overlap; vs baseline_b
  - neighbor_ba(+k)   : the reverse direction

If enrich_ab >> 1 and neighbor_ab >> baseline → features A and B are
co-localized along the sequence (their activated residues cluster together).
If enrich_ab ≈ 1 → they are essentially independent.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from single.sae.dictionary import Dictionary
from single.sae.inference import get_sae_feats_in_batches


def _activation_masks(
    sae: Dictionary,
    embeddings: torch.Tensor,
    feature_indices: List[int],
    activation_threshold: float = 0.0,
    batch_size: int = 4096,
    device: str = "cpu",
) -> np.ndarray:
    """Return boolean [n_tokens, n_features] activation masks."""
    feats = get_sae_feats_in_batches(
        sae=sae, aa_embds=embeddings, chunk_size=batch_size,
        feat_list=feature_indices, normalize_features=True, device=device,
    )
    return (feats.cpu().numpy() > activation_threshold)


def _group_by_protein(mask: np.ndarray, protein_ids: np.ndarray, respos: np.ndarray):
    """
    Group a boolean token mask into per-protein residue-position sets.

    Returns dict protein_idx -> array of residue positions where the mask is True.
    """
    groups: Dict[int, List[int]] = {}
    for i in range(len(mask)):
        if mask[i]:
            pid = protein_ids[i]
            groups.setdefault(pid, []).append(int(respos[i]))
    return {k: np.array(v, dtype=np.int64) for k, v in groups.items()}


def compute_coactivation(
    sae: Dictionary,
    embeddings: torch.Tensor,
    protein_ids: np.ndarray,
    respos: np.ndarray,
    feature_a: int,
    feature_b: int,
    neighborhood: int = 5,
    activation_threshold: float = 0.0,
    batch_size: int = 4096,
    device: str = "cpu",
) -> Dict:
    """
    Compute co-activation / co-localization metrics between two features.

    Args:
        sae: trained SAE
        embeddings: [n_tokens, D] for one shard
        protein_ids: protein index per token
        respos: residue position (within protein) per token
        feature_a, feature_b: the two features to compare
        neighborhood: radius k for the neighbor analysis
        device: torch device

    Returns:
        dict of metrics (see module docstring).
    """
    mask = _activation_masks(
        sae, embeddings, [feature_a, feature_b],
        activation_threshold=activation_threshold, device=device,
    )
    mask_a = mask[:, 0]
    mask_b = mask[:, 1]

    # Overall activation rates (baselines)
    n_valid = len(mask_a)
    p_a = float(mask_a.mean()) if n_valid else 0.0
    p_b = float(mask_b.mean()) if n_valid else 0.0

    # Group active residues per protein
    groups_a = _group_by_protein(mask_a, protein_ids, respos)
    groups_b = _group_by_protein(mask_b, protein_ids, respos)
    # Also need residue -> active flags per protein for the neighbor analysis.
    # Build per-protein boolean arrays over residue indices.
    # respos max within protein:
    max_pos = int(respos.max()) + 1 if len(respos) else 0

    # Same-residue overlap
    both = (mask_a & mask_b).sum()
    n_a = int(mask_a.sum())
    n_b = int(mask_b.sum())
    overlap_ab = both / n_a if n_a > 0 else 0.0
    overlap_ba = both / n_b if n_b > 0 else 0.0

    # Neighborhood analysis: for each A-active residue, is B active within ±k?
    # We operate per protein using residue-position sets.
    neighbor_ab_counts = 0   # A-active positions with B active within ±k (or same)
    neighbor_ab_total = 0
    neighbor_ba_counts = 0
    neighbor_ba_total = 0

    # Build per-protein residue->B-active / A-active maps from token masks.
    # We need, for each protein, the set of positions where B is active.
    b_positions_per_protein = _group_by_protein(mask_b, protein_ids, respos)
    a_positions_per_protein = _group_by_protein(mask_a, protein_ids, respos)

    for pid, a_pos in groups_a.items():
        b_pos = b_positions_per_protein.get(pid, np.array([], dtype=np.int64))
        if len(b_pos) == 0:
            continue
        b_set = set(b_pos.tolist())
        for p in a_pos:
            # window [p-k, p+k]
            hit = any((q in b_set) for q in range(int(p) - neighborhood, int(p) + neighborhood + 1))
            neighbor_ab_counts += int(hit)
            neighbor_ab_total += 1

    for pid, b_pos in groups_b.items():
        a_pos = a_positions_per_protein.get(pid, np.array([], dtype=np.int64))
        if len(a_pos) == 0:
            continue
        a_set = set(a_pos.tolist())
        for p in b_pos:
            hit = any((q in a_set) for q in range(int(p) - neighborhood, int(p) + neighborhood + 1))
            neighbor_ba_counts += int(hit)
            neighbor_ba_total += 1

    neighbor_ab = neighbor_ab_counts / neighbor_ab_total if neighbor_ab_total > 0 else 0.0
    neighbor_ba = neighbor_ba_counts / neighbor_ba_total if neighbor_ba_total > 0 else 0.0

    # Enrichment relative to baseline
    enrich_ab = overlap_ab / p_b if p_b > 0 else 0.0
    enrich_ba = overlap_ba / p_a if p_a > 0 else 0.0

    return {
        "feature_a": feature_a,
        "feature_b": feature_b,
        "n_tokens": n_valid,
        "n_a_active": n_a,
        "n_b_active": n_b,
        "baseline_a": p_a,
        "baseline_b": p_b,
        "overlap_ab": overlap_ab,        # P(B | A) same residue
        "overlap_ba": overlap_ba,        # P(A | B) same residue
        "enrich_ab": enrich_ab,          # overlap_ab / baseline_b
        "enrich_ba": enrich_ba,          # overlap_ba / baseline_a
        "neighborhood": neighborhood,
        "neighbor_ab": neighbor_ab,      # P(B active within ±k | A active), same+window
        "neighbor_ba": neighbor_ba,
        "neighbor_enrich_ab": neighbor_ab / p_b if p_b > 0 else 0.0,
        "neighbor_enrich_ba": neighbor_ba / p_a if p_a > 0 else 0.0,
    }


def interpret(result: Dict) -> str:
    """Return a short human-readable interpretation."""
    enrich_ab = result["enrich_ab"]
    enrich_ba = result["enrich_ba"]
    neigh_enrich_ab = result["neighbor_enrich_ab"]

    lines = []
    if enrich_ab > 1.5 and enrich_ba > 1.5:
        lines.append(f"Both features strongly co-activate on the same residues "
                     f"(enrichment {enrich_ab:.1f}x / {enrich_ba:.1f}x) — likely redundant or co-regulatory.")
    elif enrich_ab > 1.5:
        lines.append(f"Feature #{result['feature_b']} is enriched on Feature #{result['feature_a']}'s "
                     f"residues ({enrich_ab:.1f}x) but not vice versa — asymmetric relationship.")
    elif neigh_enrich_ab > 1.5:
        lines.append(f"Features activate on nearby-but-not-identical residues "
                     f"(neighborhood enrichment {neigh_enrich_ab:.1f}x).")
    else:
        lines.append("No strong co-localization — the two features activate on "
                     "largely independent residue sets.")
    return "\n".join(lines)
