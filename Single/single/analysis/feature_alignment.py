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
from scipy import sparse
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
    positive_class: int = 1,
) -> Dict:
    """
    Align each SAE feature to the task labels.

    For each feature, across multiple activation thresholds, compute:
    - Precision, Recall, F1 for predicting the positive class
    - AUROC

    Args:
        positive_class: which integer label is the positive class (default 1)

    Returns a dict mapping feature_idx -> {best_threshold, precision, recall, f1, auroc}
    """
    if thresholds is None:
        thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

    device = embeddings.device
    n_features = sae.dict_size

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

        labels_binary = (labels_np[valid_mask] == positive_class).astype(np.float32)

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
    positive_class: int = 1,
) -> np.ndarray:
    """
    Compute point-biserial correlation between each feature's activation
    and the binary label (0=negative, positive_class=positive).

    Returns array of shape (n_features,) with correlation coefficients.
    """
    device = embeddings.device
    n_features = sae.dict_size
    correlations = np.zeros(n_features)

    valid_mask = labels != -100
    labels_bin = (labels[valid_mask] == positive_class).float().cpu().numpy()

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
    positive_class: int = 1,
    negative_class: int = 0,
    pos_name: str = "positive",
    neg_name: str = "negative",
) -> Dict:
    """
    For each feature, compute mean/max activation on negative vs positive class residues.
    """
    device = embeddings.device
    n_features = sae.dict_size

    pos_means = np.zeros(n_features)
    neg_means = np.zeros(n_features)
    pos_max = np.zeros(n_features)
    neg_max = np.zeros(n_features)

    valid = labels != -100
    neg_mask = valid & (labels == negative_class)
    pos_mask = valid & (labels == positive_class)

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
            if neg_mask.any():
                neg_means[fidx] = feats_np[neg_mask, i].mean()
                neg_max[fidx] = feats_np[neg_mask, i].max()
            if pos_mask.any():
                pos_means[fidx] = feats_np[pos_mask, i].mean()
                pos_max[fidx] = feats_np[pos_mask, i].max()

    return {
        f"{neg_name}_mean_activation": neg_means,
        f"{pos_name}_mean_activation": pos_means,
        f"{neg_name}_max_activation": neg_max,
        f"{pos_name}_max_activation": pos_max,
        "activation_gap": pos_means - neg_means,
    }


# ---------------------------------------------------------------------------
# Multi-concept alignment (Swiss-Prot / UniProtKB)
#
# Unlike the single-binary-label path above, concepts are a 2D sparse matrix
# [n_tokens, n_concepts] where each column is an independent binary label.
# We compute per (feature, concept) precision/recall/F1 across thresholds.
# ---------------------------------------------------------------------------

def align_features_to_concepts(
    sae: Dictionary,
    embeddings: torch.Tensor,
    concept_matrix,
    concept_names: List[str],
    thresholds: Optional[List[float]] = None,
    threshold_percents: Optional[List[float]] = None,
    feature_chunk_size: int = 200,
    batch_size: int = 1024,
    compute_auroc: bool = True,
    compute_domain_f1: bool = True,
    min_positives: int = 10,
) -> Dict:
    """
    Align each SAE feature against every biological concept.

    Two threshold schemes are supported:
    - Fixed absolute thresholds (`thresholds`), e.g. [0, 0.05, 0.1, ...].
    - Percent-of-max thresholds (`threshold_percents`), e.g. [0, 0.15, 0.5, 0.6, 0.8]
      like InterPLM: since features are normalized (max activation = 1), a threshold
      of 0.15 means "activation > 15% of max". Only one scheme is used at a time;
      if `threshold_percents` is given it takes precedence.

    Domain-level F1 (optional, like InterPLM):
    For non-AA-level concepts (domains/regions/secondary structure), each annotation
    instance is a distinct contiguous segment. `recall_per_domain` / `f1_per_domain`
    count *instances* (domains) hit rather than residues, so long proteins don't
    dominate. For AA-level concepts these equal the residue-based metrics.

    Args:
        sae: trained SAE
        embeddings: [n_tokens, d_model]
        concept_matrix: sparse [n_tokens, n_concepts] matrix. Entries are 0 (not
                        annotated) or a positive domain-instance index (1,2,...).
        concept_names: list of n_concepts names
        thresholds: fixed absolute thresholds (ignored if threshold_percents given)
        threshold_percents: InterPLM-style percent-of-max thresholds
        compute_auroc: if False, skips AUROC computation (saves ~50% time)
        compute_domain_f1: if True, computes f1_per_domain / recall_per_domain
        min_positives: skip concepts with fewer than this many positive residues

    Returns dict mapping feature_idx -> { concept_name -> {f1, precision, recall,
        threshold, auroc, recall_per_domain, f1_per_domain, n_domains} }
    """
    from single.analysis.concepts import is_aa_level_concept

    if thresholds is None:
        thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    if threshold_percents is not None:
        thresholds = threshold_percents  # percent-of-max thresholds take precedence

    device = embeddings.device
    n_features = sae.dict_size
    n_concepts = concept_matrix.shape[1]
    n_tokens = embeddings.shape[0]

    # Keep concepts sparse; keep original instance indices for domain counting,
    # and build a binarized copy for residue-level TP/FP.
    if not sparse.issparse(concept_matrix):
        labels = sparse.csr_matrix(concept_matrix)
    else:
        labels = concept_matrix.tocsr()
    labels_bin = labels.copy()
    labels_bin.data = np.ones_like(labels_bin.data)
    labels_bin.eliminate_zeros()
    labels_float = labels_bin.astype(np.float32)

    n_pos = np.asarray(labels_bin.sum(axis=0)).ravel().astype(np.float64)  # [c]
    n_neg = n_tokens - n_pos
    valid_concepts = n_pos >= min_positives
    valid_idx = np.where(valid_concepts)[0]

    # Domain counts per concept (unique positive instance indices)
    if compute_domain_f1:
        n_domains = np.zeros(n_concepts, dtype=np.float64)
        for c in valid_idx:
            col_data = labels.getcol(c).data
            if col_data.size:
                n_domains[c] = len(np.unique(col_data[col_data > 0]))

    # aa-level vs domain-level concepts
    is_aa = [is_aa_level_concept(name) for name in concept_names] if compute_domain_f1 else [True] * n_concepts

    thresholds_arr = np.array(thresholds, dtype=np.float32)

    result = {int(f): {} for f in range(n_features)}

    for feature_list in tqdm(
        split_up_feature_list(n_features, feature_chunk_size),
        desc="Aligning features to concepts",
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
        f_chunk = len(feature_list)

        # Best-metric arrays across all (feature, concept) pairs in this chunk
        best_f1 = np.zeros((f_chunk, n_concepts))
        best_prec = np.zeros((f_chunk, n_concepts))
        best_rec = np.zeros((f_chunk, n_concepts))
        best_thr = np.zeros((f_chunk, n_concepts))
        best_thr_idx = np.zeros((f_chunk, n_concepts), dtype=int)
        best_rec_domain = np.zeros((f_chunk, n_concepts))

        # ---- Vectorized threshold scan ----
        # For each threshold: tp[f, c] = preds[f].T @ labels[c] via one sparse matmul
        for t_idx, t in enumerate(thresholds_arr):
            preds = (feats_np > t).astype(np.float32)  # [n, f_chunk]
            tp = (preds.T @ labels_float).astype(np.float64)  # [f_chunk, c]
            fp = preds.sum(axis=0, keepdims=True).T - tp      # [f_chunk, c]
            fn = n_pos[None, :] - tp
            prec = tp / (tp + fp + 1e-10)
            rec = tp / (tp + fn + 1e-10)
            f1 = 2 * prec * rec / (prec + rec + 1e-10)

            # Domain-level recall at this threshold (only for domain-level concepts)
            rec_dom = rec.copy()
            if compute_domain_f1:
                # Vectorized domain counting: for each concept, group activated
                # residues by domain-instance id, OR-reduce across features, then
                # count how many distinct instances each feature hit.
                preds_bool = preds > 0  # [n, f_chunk]
                for c in valid_idx:
                    if is_aa[c]:
                        continue
                    col = labels.getcol(c).tocoo()
                    ids_arr = col.data.astype(np.int64)
                    rows_arr = col.row
                    if ids_arr.size == 0:
                        continue
                    order = np.argsort(ids_arr, kind="mergesort")
                    s_rows = rows_arr[order]
                    s_ids = ids_arr[order]
                    change = np.concatenate(([0], np.nonzero(np.diff(s_ids))[0] + 1))
                    # OR-reduce activated rows per instance group: [n_groups, f_chunk]
                    group_or = np.maximum.reduceat(preds_bool[s_rows], change, axis=0)
                    hits = group_or.sum(axis=0)  # [f_chunk] unique instances hit per feature
                    rec_dom[:, c] = hits / max(n_domains[c], 1)

            better = f1 > best_f1
            best_f1 = np.where(better, f1, best_f1)
            best_prec = np.where(better, prec, best_prec)
            best_rec = np.where(better, rec, best_rec)
            best_thr = np.where(better, t, best_thr)
            best_thr_idx = np.where(better, t_idx, best_thr_idx)
            best_rec_domain = np.where(better, rec_dom, best_rec_domain)

        # ---- Vectorized AUROC (per-feature rank sum) ----
        if compute_auroc:
            auroc = np.full((f_chunk, n_concepts), 0.5)
            for j in range(f_chunk):
                col = feats_np[:, j]
                if col.max() == 0:
                    continue
                order = np.argsort(col, kind="mergesort")
                ranks = np.empty(n_tokens, dtype=np.float64)
                ranks[order] = np.arange(1, n_tokens + 1)
                rank_sum_pos = np.asarray(labels_float.T @ ranks).ravel()  # [c]
                denom = n_pos * n_neg
                valid = denom > 0
                auc = np.full(n_concepts, 0.5)
                auc[valid] = (rank_sum_pos[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2.0) / denom[valid]
                auroc[j] = auc
        else:
            auroc = np.full((f_chunk, n_concepts), 0.5)

        # ---- Fill results (only for valid concepts with positive F1) ----
        nz_pairs = np.argwhere(best_f1 > 0)
        for j, c in nz_pairs:
            if not valid_concepts[c]:
                continue
            fidx = int(feature_list[j])
            entry = {
                "f1": float(best_f1[j, c]),
                "precision": float(best_prec[j, c]),
                "recall": float(best_rec[j, c]),
                "threshold": float(best_thr[j, c]),
                "auroc": float(auroc[j, c]),
            }
            if compute_domain_f1:
                if is_aa[c]:
                    entry["recall_per_domain"] = entry["recall"]
                    entry["f1_per_domain"] = entry["f1"]
                    entry["n_domains"] = int(n_domains[c])
                else:
                    rec_dom = best_rec_domain[j, c]
                    p = entry["precision"]
                    f1_dom = 2 * p * rec_dom / (p + rec_dom + 1e-10) if rec_dom > 0 else 0.0
                    entry["recall_per_domain"] = float(rec_dom)
                    entry["f1_per_domain"] = float(f1_dom)
                    entry["n_domains"] = int(n_domains[c])
            result[fidx][concept_names[c]] = entry

    return result


def find_top_concept_for_feature(
    feature_concept_metrics: Dict,
    metric: str = "f1",
) -> Tuple[Optional[str], Optional[float]]:
    """Return the (concept, score) with the highest score for a single feature."""
    if not feature_concept_metrics:
        return None, None
    best_concept = max(feature_concept_metrics, key=lambda k: feature_concept_metrics[k].get(metric, 0))
    return best_concept, feature_concept_metrics[best_concept].get(metric, 0)


def find_top_features_for_concept(
    all_metrics: Dict,
    concept_name: str,
    n_top: int = 20,
    metric: str = "f1",
) -> List[Tuple[int, float]]:
    """Return the top-N features for a specific concept, ranked by metric."""
    scores = []
    for fidx, concept_dict in all_metrics.items():
        if concept_name in concept_dict:
            scores.append((fidx, concept_dict[concept_name].get(metric, 0.0)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:n_top]


def summarize_feature_concept_pairs(
    all_metrics: Dict,
    metric: str = "f1",
    min_score: float = 0.3,
) -> List[Tuple[int, str, float]]:
    """
    Return all (feature, concept, score) pairs above min_score, sorted by score.
    This is the main summary for 'feature X encodes biological concept Y'.
    """
    pairs = []
    for fidx, concept_dict in all_metrics.items():
        for concept, scores in concept_dict.items():
            score = scores.get(metric, 0.0)
            if score >= min_score:
                pairs.append((fidx, concept, score))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs
