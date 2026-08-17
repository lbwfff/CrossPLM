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
from scipy.stats import rankdata, t as student_t
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
    device: Optional[str] = None,
    feature_cache: Optional[Dict] = None,
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

    device = device or str(next(sae.parameters()).device)
    n_features = sae.dict_size

    best_metrics = {}
    feature_cache = feature_cache if feature_cache is not None else {}

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
            cache=feature_cache,
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


def _domain_instance_ranges(
    domain_labels: np.ndarray,
    protein_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract contiguous domain-instance spans from a per-residue label column.

    Each positive instance id (>0) occupies a contiguous residue run within a
    single protein (annotations never cross protein boundaries), so runs are split
    both on label change and on protein change. Returns (starts, ends, ids) arrays
    with `end` exclusive and `starts` sorted ascending (ready for searchsorted).
    """
    domain_labels = np.asarray(domain_labels)
    protein_ids = np.asarray(protein_ids)
    change = np.r_[
        True,
        (domain_labels[1:] != domain_labels[:-1])
        | (protein_ids[1:] != protein_ids[:-1]),
        True,
    ]
    run_starts = np.flatnonzero(change[:-1])
    # change[1:] is shifted by one position relative to the original index space,
    # so the +1 recovers the true (exclusive) end positions.
    run_ends = np.flatnonzero(change[1:]) + 1
    labels_at_start = domain_labels[run_starts]
    valid = labels_at_start > 0
    return run_starts[valid], run_ends[valid], labels_at_start[valid]


def _domain_confusion_from_segments(
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    dom_starts: np.ndarray,
    dom_ends: np.ndarray,
    dom_ids: np.ndarray,
) -> Tuple[int, int, int]:
    """Match predicted activation segments to annotated domain instances.

    Same one-to-one greedy overlap semantics as the previous `_domain_confusion`,
    but operates on precomputed segment/instance span tables via searchsorted
    instead of rescanning the token array for every (feature, concept) pair.

    A predicted domain is a contiguous activation run within one protein. A
    predicted run and an annotated instance match when they overlap by at least
    one residue; greedy matching by overlap gives one-to-one TP/FP/FN counts and
    prevents one long prediction from matching several domains.
    """
    seg_starts = np.asarray(seg_starts, dtype=np.int64)
    seg_ends = np.asarray(seg_ends, dtype=np.int64)
    dom_starts = np.asarray(dom_starts, dtype=np.int64)
    dom_ends = np.asarray(dom_ends, dtype=np.int64)
    dom_ids = np.asarray(dom_ids)
    n_domains = len(dom_ids)
    n_segments = len(seg_starts)

    # For every segment at once: the index one-past the last domain instance that
    # starts strictly before the segment end (dom_starts is sorted ascending).
    right = np.searchsorted(dom_starts, seg_ends - 1, side="right")

    candidates = []
    for seg_idx in range(n_segments):
        s, e = seg_starts[seg_idx], seg_ends[seg_idx]
        k = int(right[seg_idx]) - 1
        while k >= 0 and dom_ends[k] > s:
            overlap = min(e, int(dom_ends[k])) - max(s, int(dom_starts[k]))
            if overlap > 0:
                candidates.append((int(overlap), seg_idx, int(dom_ids[k])))
            k -= 1

    candidates.sort(reverse=True)
    matched_segments = set()
    matched_domains = set()
    for _, seg_idx, domain_id in candidates:
        if seg_idx in matched_segments or domain_id in matched_domains:
            continue
        matched_segments.add(seg_idx)
        matched_domains.add(domain_id)

    tp = len(matched_segments)
    fp = len(seg_starts) - tp
    fn = n_domains - tp
    return tp, fp, fn


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg FDR-adjusted q-values."""
    p_values = np.asarray(p_values, dtype=np.float64)
    q_values = np.ones_like(p_values)
    finite = np.isfinite(p_values)
    indices = np.flatnonzero(finite)
    if not len(indices):
        return q_values
    order = indices[np.argsort(p_values[indices])]
    ranked = p_values[order] * len(order) / np.arange(1, len(order) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q_values[order] = np.minimum(ranked, 1.0)
    return q_values


def compute_feature_label_correlation_stats(
    sae: Dictionary,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 1024,
    positive_class: int = 1,
    device: Optional[str] = None,
    feature_cache: Optional[Dict] = None,
    feature_chunk_size: int = 200,
) -> Dict[str, np.ndarray]:
    """
    Compute point-biserial correlation between each feature's activation and the
    binary "is this residue in positive_class?" label (one-vs-rest).

    For multi-class label maps (e.g. ss3), residues of all OTHER classes are
    treated as the negative group; correlation measures how well the feature
    separates positive_class from everything else.

    Returns correlation, raw p-value, FDR q-value and valid-token count arrays.
    """
    device = device or str(next(sae.parameters()).device)
    n_features = sae.dict_size
    correlations = np.zeros(n_features)
    p_values = np.ones(n_features)
    n_valid_values = np.zeros(n_features, dtype=np.int64)
    feature_cache = feature_cache if feature_cache is not None else {}

    valid_mask_np = (labels != -100).cpu().numpy()
    labels_bin = (labels[valid_mask_np] == positive_class).float().cpu().numpy()
    valid_mask_dev = torch.from_numpy(valid_mask_np).to(device)

    # Process features in chunks (one SAE encode per chunk) instead of one encode
    # per feature. Without a dense cache this is the difference between a few full
    # passes and dict_size full passes over the dataset.
    for feature_list in tqdm(
        split_up_feature_list(n_features, feature_chunk_size),
        desc="Computing correlations",
    ):
        feats = get_sae_feats_in_batches(
            sae=sae,
            aa_embds=embeddings,
            chunk_size=batch_size,
            feat_list=feature_list,
            normalize_features=True,
            device=str(device),
            cache=feature_cache,
        )
        # Index on device, then move only the valid rows to CPU.
        acts = feats[valid_mask_dev].cpu().numpy()  # [n_valid, chunk]
        n_obs = acts.shape[0]
        chunk_corr = np.zeros(len(feature_list))
        chunk_p = np.ones(len(feature_list))

        if n_obs >= 3 and labels_bin.std() > 0:
            # Point-biserial correlation = Pearson r between each feature column
            # and the binary label vector, computed vectorized over the chunk.
            centered = acts - acts.mean(axis=0)
            label_centered = labels_bin - labels_bin.mean()
            denom = np.sqrt((centered ** 2).sum(axis=0) * (label_centered ** 2).sum())
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = (centered.T @ label_centered) / denom
            corr = np.nan_to_num(corr, nan=0.0)
            corr = np.clip(corr, -1.0, 1.0)
            t_stat = np.abs(corr) * np.sqrt(
                (n_obs - 2) / np.maximum(1.0 - corr * corr, 1e-12)
            )
            p = np.where(
                np.abs(corr) < 1.0,
                2.0 * student_t.sf(t_stat, n_obs - 2),
                0.0,
            )
            chunk_corr = corr
            chunk_p = p

        correlations[feature_list] = chunk_corr
        p_values[feature_list] = chunk_p
        n_valid_values[feature_list] = n_obs

    return {
        "correlation": correlations,
        "p_value": p_values,
        "q_value": _benjamini_hochberg(p_values),
        "n_valid": n_valid_values,
    }


def compute_feature_label_correlation(
    sae: Dictionary,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int = 1024,
    positive_class: int = 1,
    device: Optional[str] = None,
) -> np.ndarray:
    """Backward-compatible correlation-only wrapper."""
    return compute_feature_label_correlation_stats(
        sae, embeddings, labels, batch_size=batch_size,
        positive_class=positive_class, device=device,
    )["correlation"]


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
    label_map_spec: Optional[dict] = None,
    device: Optional[str] = None,
    feature_cache: Optional[Dict] = None,
) -> Dict:
    """
    For each feature, compute mean/max activation per class.

    With a binary label map (default) this is the classic positive-vs-negative
    profile (activation_gap = pos_mean - neg_mean). With a multi-class label map
    (e.g. ss3: classes 0/1/2), one profile column is produced per class, and
    activation_gap is positive_class mean minus the mean over ALL other classes
    (so no class is silently ignored).
    """
    device = device or str(next(sae.parameters()).device)
    n_features = sae.dict_size

    # Determine classes: if a label_map_spec is given, iterate ALL of its classes;
    # otherwise fall back to the binary positive/negative pair.
    if label_map_spec is not None:
        class_ids = sorted({int(v) for v in label_map_spec.get("mapping", {}).values()})
    else:
        class_ids = sorted({int(negative_class), int(positive_class)})

    means = {c: np.zeros(n_features) for c in class_ids}
    maxes = {c: np.zeros(n_features) for c in class_ids}
    masks = {
        c: ((labels != -100) & (labels == c)).cpu().numpy()
        for c in class_ids
    }
    feature_cache = feature_cache if feature_cache is not None else {}

    valid = (labels != -100).cpu().numpy()
    # "other" = all valid residues not in the positive class (for activation_gap)
    if positive_class in class_ids:
        other_mask = valid & (labels != positive_class)

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
            cache=feature_cache,
        )
        feats_np = feats.cpu().numpy()

        for i, fidx in enumerate(feature_list):
            for c in class_ids:
                if masks[c].any():
                    means[c][fidx] = feats_np[masks[c], i].mean()
                    maxes[c][fidx] = feats_np[masks[c], i].max()

    profile: Dict[str, np.ndarray] = {}
    for c in class_ids:
        name = label_map_spec["class_names"].get(c, str(c)) if label_map_spec else str(c)
        profile[f"{name}_mean_activation"] = means[c]
        profile[f"{name}_max_activation"] = maxes[c]

    # activation_gap: positive class mean minus mean over all other classes.
    if positive_class in class_ids:
        others = [c for c in class_ids if c != positive_class]
        if others:
            other_mean = np.mean([means[c] for c in others], axis=0)
        else:
            other_mean = np.zeros(n_features)
        profile["activation_gap"] = means[positive_class] - other_mean
    else:
        profile["activation_gap"] = np.zeros(n_features)

    # Backward compatibility keys (binary callers may read these directly).
    if negative_class in class_ids and positive_class in class_ids:
        profile[f"{neg_name}_mean_activation"] = means[negative_class]
        profile[f"{pos_name}_mean_activation"] = means[positive_class]
        profile[f"{neg_name}_max_activation"] = maxes[negative_class]
        profile[f"{pos_name}_max_activation"] = maxes[positive_class]

    return profile


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
    fixed_thresholds: Optional[np.ndarray] = None,
    protein_ids: Optional[np.ndarray] = None,
    device: Optional[str] = None,
) -> Dict:
    """
    Align each SAE feature against every biological concept.

    Two threshold schemes are supported:
    - Fixed absolute thresholds (`thresholds`), e.g. [0, 0.05, 0.1, ...].
    - Percent-of-max thresholds (`threshold_percents`), e.g. [0, 0.15, 0.5, 0.6, 0.8]
      like InterPLM: since features are normalized (max activation = 1), a threshold
      of 0.15 means "activation > 15% of max". Only one scheme is used at a time;
      if `threshold_percents` is given it takes precedence.

    Domain-level metrics (optional):
    Each annotation instance is a true domain and each contiguous activation run
    within a protein is a predicted domain. One-to-one overlap matching produces
    domain_tp/domain_fp/domain_fn and the corresponding precision/recall/F1.

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
    if thresholds is None:
        thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    if threshold_percents is not None:
        thresholds = threshold_percents  # percent-of-max thresholds take precedence

    fixed = None
    if fixed_thresholds is not None:
        fixed = np.asarray(fixed_thresholds, dtype=np.float32)
        if fixed.shape != (sae.dict_size, concept_matrix.shape[1]):
            raise ValueError(
                "fixed_thresholds must have shape "
                f"({sae.dict_size}, {concept_matrix.shape[1]}), got {fixed.shape}"
            )
        extra = fixed[np.isfinite(fixed)]
        thresholds = sorted(set(float(t) for t in thresholds) | set(extra.tolist()))

    device = device or str(next(sae.parameters()).device)
    n_features = sae.dict_size
    n_concepts = concept_matrix.shape[1]
    n_tokens = embeddings.shape[0]
    if protein_ids is None:
        # External callers may not have residue metadata. Keep the function
        # usable, but make the fallback conservative: every row is its own
        # protein and therefore cannot form a multi-residue predicted domain.
        protein_ids = np.arange(n_tokens, dtype=np.int64)
    else:
        protein_ids = np.asarray(protein_ids)
        if len(protein_ids) != n_tokens:
            raise ValueError(
                f"protein_ids length {len(protein_ids)} does not match "
                f"embeddings length {n_tokens}"
            )

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

    # Domain instance spans per concept, computed ONCE and reused across every
    # threshold and feature (previously the label column was rescanned inside the
    # threshold loop, making domain-F1 O(thr x concepts x features x tokens) in
    # pure Python).
    domain_tables = {}
    if compute_domain_f1:
        n_domains = np.zeros(n_concepts, dtype=np.float64)
        for c in valid_idx:
            col = labels.getcol(c).toarray().ravel()
            starts, ends, ids = _domain_instance_ranges(col, protein_ids)
            domain_tables[c] = (starts, ends, ids)
            n_domains[c] = len(ids)

    thresholds_arr = np.array(thresholds, dtype=np.float32)

    result = {int(f): {} for f in range(n_features)}
    feature_cache = {}

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
            cache=feature_cache,
        )
        feats_np = feats.cpu().numpy()
        f_chunk = len(feature_list)

        # Best-metric arrays across all (feature, concept) pairs in this chunk
        best_f1 = np.full(
            (f_chunk, n_concepts), -1.0 if fixed is not None else 0.0
        )
        best_prec = np.zeros((f_chunk, n_concepts))
        best_rec = np.zeros((f_chunk, n_concepts))
        best_thr = np.zeros((f_chunk, n_concepts))
        best_thr_idx = np.zeros((f_chunk, n_concepts), dtype=int)
        best_domain_precision = np.zeros((f_chunk, n_concepts))
        best_domain_recall = np.zeros((f_chunk, n_concepts))
        best_domain_f1 = np.zeros((f_chunk, n_concepts))
        best_domain_tp = np.zeros((f_chunk, n_concepts))
        best_domain_fp = np.zeros((f_chunk, n_concepts))
        best_domain_fn = np.zeros((f_chunk, n_concepts))
        best_tp = np.zeros((f_chunk, n_concepts))
        best_fp = np.zeros((f_chunk, n_concepts))
        best_fn = np.zeros((f_chunk, n_concepts))

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

            # Domain-level metrics use one-to-one matching between contiguous
            # predicted activation segments and annotated domain instances.
            domain_precision = np.zeros((f_chunk, n_concepts))
            domain_recall = np.zeros((f_chunk, n_concepts))
            domain_f1 = np.zeros((f_chunk, n_concepts))
            domain_tp = np.zeros((f_chunk, n_concepts))
            domain_fp = np.zeros((f_chunk, n_concepts))
            domain_fn = np.zeros((f_chunk, n_concepts))
            if compute_domain_f1:
                # Predicted activation segments for every feature in the chunk.
                # A segment must stay within one protein (split at protein change)
                # and cannot span inactive tokens.
                preds_bool = preds > 0  # [n, f_chunk]
                prev_pred = np.concatenate(
                    [np.zeros((1, f_chunk), dtype=bool), preds_bool[:-1, :]], axis=0
                )
                prev_same = np.concatenate(
                    [
                        np.zeros((1, 1), dtype=bool),
                        (protein_ids[1:] == protein_ids[:-1])[:, None],
                    ],
                    axis=0,
                )
                is_start = preds_bool & ~(prev_pred & prev_same)
                next_pred = np.concatenate(
                    [preds_bool[1:, :], np.zeros((1, f_chunk), dtype=bool)], axis=0
                )
                next_same = np.concatenate(
                    [
                        (protein_ids[1:] == protein_ids[:-1])[:, None],
                        np.zeros((1, 1), dtype=bool),
                    ],
                    axis=0,
                )
                is_end = preds_bool & ~(next_pred & next_same)
                # Segments depend only on the feature, so compute them once per
                # feature and reuse across concepts.
                for j in range(f_chunk):
                    seg_starts = np.flatnonzero(is_start[:, j])
                    if seg_starts.size == 0:
                        continue
                    seg_ends = np.flatnonzero(is_end[:, j]) + 1
                    for c in valid_idx:
                        starts, ends, ids = domain_tables[c]
                        d_tp, d_fp, d_fn = _domain_confusion_from_segments(
                            seg_starts, seg_ends, starts, ends, ids
                        )
                        domain_tp[j, c] = d_tp
                        domain_fp[j, c] = d_fp
                        domain_fn[j, c] = d_fn
                for c in valid_idx:
                    domain_precision[:, c] = domain_tp[:, c] / np.maximum(
                        domain_tp[:, c] + domain_fp[:, c], 1e-10
                    )
                    domain_recall[:, c] = domain_tp[:, c] / np.maximum(
                        domain_tp[:, c] + domain_fn[:, c], 1e-10
                    )
                    domain_f1[:, c] = (
                        2 * domain_precision[:, c] * domain_recall[:, c]
                        / np.maximum(domain_precision[:, c] + domain_recall[:, c], 1e-10)
                    )

            if fixed is None:
                allowed = True
            else:
                allowed = np.isclose(
                    float(t), fixed[np.asarray(feature_list), :], atol=1e-6
                )
            better = allowed & (f1 > best_f1)
            best_f1 = np.where(better, f1, best_f1)
            best_prec = np.where(better, prec, best_prec)
            best_rec = np.where(better, rec, best_rec)
            best_thr = np.where(better, t, best_thr)
            best_thr_idx = np.where(better, t_idx, best_thr_idx)
            best_domain_precision = np.where(
                better, domain_precision, best_domain_precision
            )
            best_domain_recall = np.where(
                better, domain_recall, best_domain_recall
            )
            best_domain_f1 = np.where(better, domain_f1, best_domain_f1)
            best_tp = np.where(better, tp, best_tp)
            best_fp = np.where(better, fp, best_fp)
            best_fn = np.where(better, fn, best_fn)
            best_domain_tp = np.where(better, domain_tp, best_domain_tp)
            best_domain_fp = np.where(better, domain_fp, best_domain_fp)
            best_domain_fn = np.where(better, domain_fn, best_domain_fn)

        # ---- Vectorized AUROC (per-feature rank sum) ----
        if compute_auroc:
            auroc = np.full((f_chunk, n_concepts), 0.5)
            for j in range(f_chunk):
                col = feats_np[:, j]
                if col.max() == 0:
                    continue
                ranks = rankdata(col, method="average")
                rank_sum_pos = np.asarray(labels_float.T @ ranks).ravel()  # [c]
                denom = n_pos * n_neg
                valid = denom > 0
                auc = np.full(n_concepts, 0.5)
                auc[valid] = (rank_sum_pos[valid] - n_pos[valid] * (n_pos[valid] + 1) / 2.0) / denom[valid]
                auroc[j] = auc
        else:
            auroc = np.full((f_chunk, n_concepts), 0.5)

        # ---- Fill results (only for valid concepts with positive F1) ----
        emit_mask = best_f1 > 0
        if fixed is not None:
            emit_mask |= np.isfinite(fixed[np.asarray(feature_list), :])
        nz_pairs = np.argwhere(emit_mask)
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
                # Sufficient statistics for exact aggregation across shards.
                "tp": float(best_tp[j, c]),
                "fp": float(best_fp[j, c]),
                "fn": float(best_fn[j, c]),
                "n_tokens": int(n_tokens),
                "n_positives": int(n_pos[c]),
            }
            if compute_domain_f1:
                entry["domain_tp"] = float(best_domain_tp[j, c])
                entry["domain_fp"] = float(best_domain_fp[j, c])
                entry["domain_fn"] = float(best_domain_fn[j, c])
                entry["domain_precision"] = float(best_domain_precision[j, c])
                entry["domain_recall"] = float(best_domain_recall[j, c])
                entry["domain_f1"] = float(best_domain_f1[j, c])
                entry["n_domains"] = int(n_domains[c])
                # Deprecated aliases retained for downstream readers.
                entry["recall_per_domain"] = entry["domain_recall"]
                entry["f1_per_domain"] = entry["domain_f1"]
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
