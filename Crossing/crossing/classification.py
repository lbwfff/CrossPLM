"""Classify features as Shared, Task-A-specific, or Task-B-specific."""
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def classify_features(
    feats_a: np.ndarray,
    labels_a: np.ndarray,
    feats_b: np.ndarray,
    labels_b: np.ndarray,
    threshold: float = 0.3,
    val_ratio: float = 0.2,
    method: str = "logistic",
    seed: int = 42,
) -> Dict:
    """Classify each feature as shared, A-specific, or B-specific.

    For each feature, trains a probe on that single feature to predict each task.
    - If it predicts both A and B above threshold → shared
    - If it predicts only A above threshold → A-specific
    - If it predicts only B above threshold → B-specific
    - Otherwise → neither

    Args:
        feats_a: Shape (n_tokens_a, n_features) - SAE features from model A.
        labels_a: Shape (n_tokens_a,) - task A labels.
        feats_b: Shape (n_tokens_b, n_features) - SAE features from model B.
        labels_b: Shape (n_tokens_b,) - task B labels.
        threshold: F1 threshold for "predicts" a task.
        val_ratio: Fraction for validation.
        method: Probe method.
        seed: Random seed.

    Returns:
        Dictionary with classification results per feature.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    rng = np.random.RandomState(seed)
    n_features = feats_a.shape[1]

    # Split data
    def _split(feats, labels):
        n = len(labels)
        idx = rng.permutation(n)
        n_val = int(n * val_ratio)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        return feats[train_idx], labels[train_idx], feats[val_idx], labels[val_idx]

    Xa_tr, ya_tr, Xa_val, ya_val = _split(feats_a, labels_a)
    Xb_tr, yb_tr, Xb_val, yb_val = _split(feats_b, labels_b)

    n_classes_a = len(np.unique(ya_tr))
    n_classes_b = len(np.unique(yb_tr))
    average_a = "macro" if n_classes_a > 2 else "binary"
    average_b = "macro" if n_classes_b > 2 else "binary"

    classifications = []
    f1_a_list = []
    f1_b_list = []

    for feat_idx in range(n_features):
        # Single feature as input
        x_a_tr = Xa_tr[:, feat_idx:feat_idx + 1]
        x_a_val = Xa_val[:, feat_idx:feat_idx + 1]
        x_b_tr = Xb_tr[:, feat_idx:feat_idx + 1]
        x_b_val = Xb_val[:, feat_idx:feat_idx + 1]

        # Probe for task A
        scaler_a = StandardScaler()
        x_a_tr_s = scaler_a.fit_transform(x_a_tr)
        x_a_val_s = scaler_a.transform(x_a_val)
        clf_a = LogisticRegression(max_iter=500, solver="lbfgs")
        clf_a.fit(x_a_tr_s, ya_tr)
        pred_a = clf_a.predict(x_a_val_s)
        f1_a = float(f1_score(ya_val, pred_a, average=average_a, zero_division=0))

        # Probe for task B
        scaler_b = StandardScaler()
        x_b_tr_s = scaler_b.fit_transform(x_b_tr)
        x_b_val_s = scaler_b.transform(x_b_val)
        clf_b = LogisticRegression(max_iter=500, solver="lbfgs")
        clf_b.fit(x_b_tr_s, yb_tr)
        pred_b = clf_b.predict(x_b_val_s)
        f1_b = float(f1_score(yb_val, pred_b, average=average_b, zero_division=0))

        f1_a_list.append(f1_a)
        f1_b_list.append(f1_b)

        # Classify
        predicts_a = f1_a >= threshold
        predicts_b = f1_b >= threshold

        if predicts_a and predicts_b:
            category = "shared"
        elif predicts_a:
            category = "A_specific"
        elif predicts_b:
            category = "B_specific"
        else:
            category = "neither"

        classifications.append({
            "feature_idx": feat_idx,
            "f1_task_a": round(f1_a, 4),
            "f1_task_b": round(f1_b, 4),
            "category": category,
        })

    # Summary
    summary = {
        "shared": 0,
        "A_specific": 0,
        "B_specific": 0,
        "neither": 0,
    }
    for c in classifications:
        summary[c["category"]] += 1

    return {
        "threshold": threshold,
        "n_features": n_features,
        "summary": summary,
        "features": classifications,
        "mean_f1_a": round(float(np.mean(f1_a_list)), 4),
        "mean_f1_b": round(float(np.mean(f1_b_list)), 4),
    }


def classify_features_cross_model(
    sim_matrix: np.ndarray,
    feats_a_f1: np.ndarray,
    feats_b_f1: np.ndarray,
    correlation_threshold: float = 0.5,
    f1_threshold: float = 0.3,
) -> Dict:
    """Classify features based on cross-model similarity and single-task performance.

    This is a simpler approach that uses the similarity matrix directly.

    Args:
        sim_matrix: Shape (n_features_a, n_features_b) - feature similarity.
        feats_a_f1: Shape (n_features_a,) - F1 of each A feature on task A.
        feats_b_f1: Shape (n_features_b,) - F1 of each B feature on task B.
        correlation_threshold: Minimum similarity to consider two features "matched".
        f1_threshold: Minimum F1 to consider a feature "predictive" of its task.

    Returns:
        Classification results.
    """
    n_a, n_b = sim_matrix.shape

    # For each A feature, find best match in B
    best_match_b = np.argmax(sim_matrix, axis=1)
    best_sim = np.max(sim_matrix, axis=1)

    classifications = []
    for i in range(n_a):
        j = best_match_b[i]
        sim = best_sim[i]

        a_predicts = feats_a_f1[i] >= f1_threshold
        b_predicts = feats_b_f1[j] >= f1_threshold
        matched = sim >= correlation_threshold

        if matched and a_predicts and b_predicts:
            category = "shared"
        elif a_predicts and not matched:
            category = "A_specific"
        elif b_predicts and not matched:
            category = "B_specific"
        else:
            category = "neither"

        classifications.append({
            "feature_a": i,
            "feature_b": int(j),
            "similarity": round(float(sim), 4),
            "f1_a": round(float(feats_a_f1[i]), 4),
            "f1_b": round(float(feats_b_f1[j]), 4),
            "category": category,
        })

    summary = {"shared": 0, "A_specific": 0, "B_specific": 0, "neither": 0}
    for c in classifications:
        summary[c["category"]] += 1

    return {
        "correlation_threshold": correlation_threshold,
        "f1_threshold": f1_threshold,
        "n_features_a": n_a,
        "n_features_b": n_b,
        "summary": summary,
        "features": classifications,
    }
