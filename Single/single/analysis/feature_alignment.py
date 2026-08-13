"""
Align SAE features with task-specific labels to discover interpretable patterns.

Core idea:
  1. Run SAE over many residues to get feature activations
  2. Compare each feature's activation pattern against task labels (rigid=0, flexible=1)
  3. For each feature, compute: precision, recall, F1, and AUROC w.r.t. each label class
  4. Features that correlate with task labels are "interpretable" in terms of the task
"""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score
from tqdm import tqdm

from single.sae.dictionary import Dictionary
from single.sae.inference import get_sae_feats_in_batches, split_up_feature_list


def align_features_to_labels(
    sae: Dictionary,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    thresholds: List[float] = None,
    feature_chunk_size: int = 200,
    batch_size: int = 1024,
) -> Dict:
    """
    Align each SAE feature to the task labels.

    For each feature, across multiple activation thresholds, compute:
    - Precision, Recall, F1 for predicting label=1 (flexible)
    - AUROC

    Returns a dict mapping feature_idx -> {best_threshold, precision, recall, f1, auroc}
    """
    if thresholds is None:
        thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    device = embeddings.device
    n_features = sae.dict_size
    n_labels = 2  # binary: rigid (0) vs flexible (1)

    best_metrics = {}
    all_aurocs = np.zeros(n_features)

    for feature_list in tqdm(
        split_up_feature_list(n_features, feature_chunk_size),
        desc="Aligning features to labels",
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
        labels_np = labels.cpu().numpy()

        valid_mask = labels_np != -100
        if not valid_mask.any():
            continue

        labels_binary = (labels_np[valid_mask] == 1).astype(np.float32)

        for feat_idx_in_chunk, global_feat_idx in enumerate(feature_list):
            feat_acts = feats_np[valid_mask, feat_idx_in_chunk]

            if feat_acts.max() == 0:
                best_metrics[int(global_feat_idx)] = {
                    "best_f1": 0.0,
                    "best_precision": 0.0,
                    "best_recall": 0.0,
                    "best_threshold": 0.0,
                    "auroc": 0.5,
                    "mean_activation": 0.0,
                    "activation_freq": 0.0,
                }
                continue

            try:
                auroc = roc_auc_score(labels_binary, feat_acts)
            except ValueError:
                auroc = 0.5
            all_aurocs[int(global_feat_idx)] = auroc

            best_f1 = 0.0
            best_prec = 0.0
            best_rec = 0.0
            best_thresh = 0.0

            for thresh in thresholds:
                preds = (feat_acts > thresh).astype(np.float32)
                tp = (preds * labels_binary).sum()
                fp = preds.sum() - tp
                fn = labels_binary.sum() - tp

                prec = tp / (tp + fp + 1e-10)
                rec = tp / (tp + fn + 1e-10)
                f1 = 2 * prec * rec / (prec + rec + 1e-10)

                if f1 > best_f1:
                    best_f1 = f1
                    best_prec = prec
                    best_rec = rec
                    best_thresh = thresh

            best_metrics[int(global_feat_idx)] = {
                "best_f1": float(best_f1),
                "best_precision": float(best_prec),
                "best_recall": float(best_rec),
                "best_threshold": float(best_thresh),
                "auroc": float(auroc),
                "mean_activation": float(feat_acts.mean()),
                "activation_freq": float((feat_acts > 0).mean()),
            }

    return best_metrics


def find_top_features_for_class(
    metrics: Dict,
    metric: str = "best_f1",
    n_top: int = 20,
    class_label: int = 1,
) -> List[Tuple[int, float]]:
    """
    Return the top-N features ranked by a given metric.
    """
    scores = [(feat_idx, m[metric]) for feat_idx, m in metrics.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n_top]


def compute_feature_label_correlation(
    sae: Dictionary,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 1024,
) -> np.ndarray:
    """
    Compute point-biserial correlation between each feature's activation
    and the binary label (0=rigid, 1=flexible).

    Returns array of shape (n_features,) with correlation coefficients.
    """
    device = embeddings.device
    n_features = sae.dict_size
    correlations = np.zeros(n_features)

    valid_mask = labels != -100
    labels_bin = (labels[valid_mask] == 1).float().cpu().numpy()

    for feat_idx in tqdm(range(n_features), desc="Computing correlations"):
        feats = get_sae_feats_in_batches(
            sae=sae,
            aa_embds=embeddings,
            chunk_size=batch_size,
            feat_list=[feat_idx],
            normalize_features=True,
            device=str(device),
        )
        acts = feats[valid_mask].cpu().numpy().flatten()

        # Point-biserial correlation = Pearson r between binary and continuous
        if acts.std() > 0 and labels_bin.std() > 0:
            corr = np.corrcoef(acts, labels_bin)[0, 1]
            correlations[feat_idx] = corr

    return correlations


def compute_feature_activation_profile(
    sae: Dictionary,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 1024,
    feature_chunk_size: int = 200,
) -> Dict:
    """
    For each feature, compute mean activation on rigid (label=0) vs flexible (label=1) residues.
    """
    device = embeddings.device
    n_features = sae.dict_size

    rigid_means = np.zeros(n_features)
    flexible_means = np.zeros(n_features)
    rigid_max = np.zeros(n_features)
    flexible_max = np.zeros(n_features)

    valid = labels != -100
    rigid_mask = valid & (labels == 0)
    flexible_mask = valid & (labels == 1)

    for feature_list in tqdm(
        split_up_feature_list(n_features, feature_chunk_size),
        desc="Computing activation profiles",
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

        for i, fidx in enumerate(feature_list):
            if rigid_mask.any():
                rigid_means[fidx] = feats_np[rigid_mask, i].mean()
                rigid_max[fidx] = feats_np[rigid_mask, i].max()
            if flexible_mask.any():
                flexible_means[fidx] = feats_np[flexible_mask, i].mean()
                flexible_max[fidx] = feats_np[flexible_mask, i].max()

    return {
        "rigid_mean_activation": rigid_means,
        "flexible_mean_activation": flexible_means,
        "rigid_max_activation": rigid_max,
        "flexible_max_activation": flexible_max,
        "activation_gap": flexible_means - rigid_means,
    }
