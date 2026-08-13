"""
Track which proteins maximally activate each SAE feature.

Given a trained SAE and protein embedding data, find for each feature:
- The top-N proteins with highest activation
- Per-protein activation frequency
- Activation boundaries (which positions in the protein activate)
"""

import heapq
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from single.sae.dictionary import Dictionary
from single.sae.inference import get_sae_feats_in_batches, split_up_feature_list


class PerProteinActivationTracker:
    def __init__(
        self,
        num_features: int,
        n_top: int = 10,
        activation_threshold: float = 0.05,
    ):
        self.num_features = num_features
        self.n_top = n_top
        self.activation_threshold = activation_threshold

        self.max_heap = [[] for _ in range(num_features)]
        self.total_proteins = 0
        self.unique_proteins = set()
        self.proteins_with_activation = np.zeros(num_features)
        self.max_activation_per_feature = np.zeros(num_features)

    def update(
        self,
        feature_activations: np.ndarray,
        protein_id: str,
        feature_ids: List[int],
    ):
        if protein_id not in self.unique_proteins:
            self.unique_proteins.add(protein_id)
            self.total_proteins += 1

        max_acts = feature_activations.max(axis=0)
        nonzero = (feature_activations > self.activation_threshold).sum(axis=0)

        has_activation = (nonzero > 0).astype(int)
        self.proteins_with_activation[feature_ids] += has_activation
        self.max_activation_per_feature[feature_ids] = np.maximum(
            self.max_activation_per_feature[feature_ids], max_acts
        )

        for i, fid in enumerate(feature_ids):
            if max_acts[i] > 0:
                if len(self.max_heap[fid]) < self.n_top:
                    heapq.heappush(self.max_heap[fid], (max_acts[i], protein_id))
                elif max_acts[i] > self.max_heap[fid][0][0]:
                    heapq.heapreplace(self.max_heap[fid], (max_acts[i], protein_id))

    def get_results(self) -> Dict:
        max_result = {
            i: [p for _, p in sorted(self.max_heap[i], reverse=True)]
            for i in range(self.num_features)
        }
        pct_active = self.proteins_with_activation / max(self.total_proteins, 1) * 100

        return {
            "max": max_result,
            "pct_proteins_with_activation": pct_active.tolist(),
            "max_activation_per_feature": self.max_activation_per_feature.tolist(),
        }


def find_max_activating_proteins(
    sae: Dictionary,
    embeddings: torch.Tensor,
    protein_boundaries: List[Tuple[int, int]],
    protein_ids: List[str],
    feature_chunk_size: int = 200,
    n_top: int = 10,
    activation_threshold: float = 0.05,
    batch_size: int = 1024,
) -> Dict:
    device = embeddings.device
    total_features = sae.dict_size

    tracker = PerProteinActivationTracker(
        total_features, n_top=n_top, activation_threshold=activation_threshold
    )

    for feature_list in tqdm(
        split_up_feature_list(total_features, feature_chunk_size),
        desc="Finding max-activating proteins",
    ):
        feats = get_sae_feats_in_batches(
            sae=sae,
            aa_embds=embeddings,
            chunk_size=batch_size,
            feat_list=feature_list,
            normalize_features=True,
            device=str(device),
        )
        feats_np = feats.cpu().numpy()

        for prot_id, (start, end) in zip(protein_ids, protein_boundaries):
            prot_feats = feats_np[start:end]
            tracker.update(prot_feats, protein_id=str(prot_id), feature_ids=feature_list)

    return tracker.get_results()


def get_feature_activation_boundaries(
    sae: Dictionary,
    embeddings: torch.Tensor,
    feature_idx: int,
    activation_threshold: float = 0.05,
) -> np.ndarray:
    device = embeddings.device
    feats = get_sae_feats_in_batches(
        sae=sae,
        aa_embds=embeddings,
        chunk_size=4096,
        feat_list=[feature_idx],
        normalize_features=True,
        device=str(device),
    )
    return (feats[:, 0] > activation_threshold).cpu().numpy()
