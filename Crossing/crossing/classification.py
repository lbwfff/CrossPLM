"""Classify SAE features as Shared, Task-A-specific, or Task-B-specific.

Aligned mode (recommended): all four arrays have the same token count/order
(n_tokens identical, e.g. both models encoded the same held-out proteins).
Then each feature is tested on BOTH tasks' labels using the same train/val
split, so "shared" is well-defined.

Disjoint mode (legacy, different protein sets): we can only evaluate A
features on task A and B features on task B.  "Shared" via single-feature
probes is not definable – use classify_features_cross_model which uses
the similarity matrix instead.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np


def _f1_single_feature(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    """F1 of a logistic probe on a single scalar feature."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score

    # zero-variance feature -> no signal
    if np.std(x_train) < 1e-12:
        return 0.0
    if len(np.unique(y_train)) < 2:
        return 0.0
    n_classes = len(np.unique(y_train))
    average = "macro" if n_classes > 2 else "binary"
    scaler = StandardScaler()
    xtr_s = scaler.fit_transform(x_train.reshape(-1, 1))
    xva_s = scaler.transform(x_val.reshape(-1, 1))
    clf = LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced")
    try:
        clf.fit(xtr_s, y_train)
        pred = clf.predict(xva_s)
        return float(f1_score(y_val, pred, average=average, zero_division=0))
    except Exception:
        return 0.0


def classify_features(
    feats_a: np.ndarray,
    labels_a: np.ndarray,
    feats_b: np.ndarray,
    labels_b: np.ndarray,
    threshold: float = 0.3,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Dict:
    """Per-feature categorization.

    In aligned mode (same n_tokens) each feature of each model is scored on
    both tasks.

    Returns a dict with keys:
        aligned: bool
        n_features_a, n_features_b
        model_a: { summary, features, mean_f1_a, mean_f1_b }
        model_b: { summary, features, mean_f1_a, mean_f1_b }
      plus legacy top-level summary/features when possible for backwards
      compatibility.
    """
    feats_a = np.asarray(feats_a)
    feats_b = np.asarray(feats_b)
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    n_a_tok = feats_a.shape[0]
    n_b_tok = feats_b.shape[0]
    # Detect aligned: same token count and same label length as feats
    aligned = (
        n_a_tok == n_b_tok == len(labels_a) == len(labels_b)
    )

    if aligned:
        # Joint valid: keep rows where BOTH tasks have real label
        valid = (labels_a != -100) & (labels_b != -100)
        if not np.any(valid):
            raise ValueError("No tokens with valid labels for both tasks after filtering")
        fa = feats_a[valid]
        fb = feats_b[valid]
        ya = labels_a[valid]
        yb = labels_b[valid]

        rng = np.random.RandomState(seed)
        n = len(ya)
        perm = rng.permutation(n)
        n_val = int(round(n * val_ratio))
        n_val = max(1, min(n_val, n - 1)) if n >= 2 else 0
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        ya_tr, ya_val = ya[train_idx], ya[val_idx]
        yb_tr, yb_val = yb[train_idx], yb[val_idx]
        fa_tr, fa_val = fa[train_idx], fa[val_idx]
        fb_tr, fb_val = fb[train_idx], fb[val_idx]

        # Model A features
        n_feat_a = fa.shape[1]
        features_a: List[Dict] = []
        f1_a_on_a, f1_a_on_b = [], []
        for i in range(n_feat_a):
            f1_a = _f1_single_feature(fa_tr[:, i], ya_tr, fa_val[:, i], ya_val)
            f1_b = _f1_single_feature(fa_tr[:, i], yb_tr, fa_val[:, i], yb_val)
            f1_a_on_a.append(f1_a)
            f1_a_on_b.append(f1_b)
            predicts_a = f1_a >= threshold
            predicts_b = f1_b >= threshold
            if predicts_a and predicts_b:
                cat = "shared"
            elif predicts_a:
                cat = "A_specific"
            elif predicts_b:
                cat = "B_specific"
            else:
                cat = "neither"
            features_a.append({
                "feature_idx": int(i),
                "model": "A",
                "f1_task_a": round(float(f1_a), 4),
                "f1_task_b": round(float(f1_b), 4),
                "category": cat,
            })
        # Model B features
        n_feat_b = fb.shape[1]
        features_b: List[Dict] = []
        f1_b_on_a, f1_b_on_b = [], []
        for j in range(n_feat_b):
            f1_a = _f1_single_feature(fb_tr[:, j], ya_tr, fb_val[:, j], ya_val)
            f1_b = _f1_single_feature(fb_tr[:, j], yb_tr, fb_val[:, j], yb_val)
            f1_b_on_a.append(f1_a)
            f1_b_on_b.append(f1_b)
            predicts_a = f1_a >= threshold
            predicts_b = f1_b >= threshold
            if predicts_a and predicts_b:
                cat = "shared"
            elif predicts_a:
                cat = "A_specific"
            elif predicts_b:
                cat = "B_specific"
            else:
                cat = "neither"
            features_b.append({
                "feature_idx": int(j),
                "model": "B",
                "f1_task_a": round(float(f1_a), 4),
                "f1_task_b": round(float(f1_b), 4),
                "category": cat,
            })

        def _summary(feats):
            s = {"shared": 0, "A_specific": 0, "B_specific": 0, "neither": 0}
            for c in feats:
                s[c["category"]] += 1
            return s

        summary_a = _summary(features_a)
        summary_b = _summary(features_b)

        # Combined summary (sum over both models) – useful for paper table
        combined_summary = {
            k: summary_a[k] + summary_b[k] for k in summary_a
        }

        result: Dict = {
            "threshold": float(threshold),
            "aligned": True,
            "n_features_a": int(n_feat_a),
            "n_features_b": int(n_feat_b),
            "n_tokens_valid": int(n),
            "model_a": {
                "n_features": int(n_feat_a),
                "summary": summary_a,
                "features": features_a,
                "mean_f1_a": round(float(np.mean(f1_a_on_a)) if f1_a_on_a else 0.0, 4),
                "mean_f1_b": round(float(np.mean(f1_a_on_b)) if f1_a_on_b else 0.0, 4),
            },
            "model_b": {
                "n_features": int(n_feat_b),
                "summary": summary_b,
                "features": features_b,
                "mean_f1_a": round(float(np.mean(f1_b_on_a)) if f1_b_on_a else 0.0, 4),
                "mean_f1_b": round(float(np.mean(f1_b_on_b)) if f1_b_on_b else 0.0, 4),
            },
            "summary": combined_summary,
        }
        # Legacy fields for callers that expect top-level features list:
        # when dict sizes equal, expose model_a features as top-level for compat.
        if n_feat_a == n_feat_b:
            result["n_features"] = int(n_feat_a)
            result["features"] = features_a
            result["mean_f1_a"] = result["model_a"]["mean_f1_a"]
            result["mean_f1_b"] = result["model_a"]["mean_f1_b"]
        return result

    # Disjoint path – legacy protein sets differ
    warnings.warn(
        "classify_features called on disjoint protein sets (n_tokens differ or "
        "label counts != feats).  Per-feature 'shared' via single-feature probes "
        "is not definable on disjoint sets – this fallback scores A features only "
        "on task A and B features only on task B.  For cross-task shared "
        "detection on disjoint tasks use classify_features_cross_model with a "
        "similarity matrix, or re-extract embeddings on a shared protein set.",
        UserWarning,
    )
    rng = np.random.RandomState(seed)

    def _split(feats, labels):
        valid = labels != -100
        f = feats[valid]
        y = labels[valid]
        n = len(y)
        if n == 0:
            raise ValueError("No valid labels after filtering")
        perm = rng.permutation(n)
        n_val = int(round(n * val_ratio))
        n_val = max(1, min(n_val, n - 1)) if n >= 2 else 0
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]
        return f[train_idx], y[train_idx], f[val_idx], y[val_idx]

    # Use separate RNG streams for A/B to avoid coupling when caller used
    # single RNG sequentially – not critical but deterministic.
    Fa_tr, ya_tr, Fa_val, ya_val = _split(feats_a, labels_a)
    Fb_tr, yb_tr, Fb_val, yb_val = _split(feats_b, labels_b)

    # We can only score each model's features on its own task here.
    # Build separate classifications.
    n_feat_a = feats_a.shape[1]
    n_feat_b = feats_b.shape[1]

    features_a, f1a_list = [], []
    for i in range(n_feat_a):
        f1 = _f1_single_feature(Fa_tr[:, i], ya_tr, Fa_val[:, i], ya_val)
        f1a_list.append(f1)
        cat = "A_specific" if f1 >= threshold else "neither"
        features_a.append({
            "feature_idx": int(i), "model": "A",
            "f1_task_a": round(float(f1), 4), "f1_task_b": 0.0, "category": cat,
        })
    features_b, f1b_list = [], []
    for j in range(n_feat_b):
        f1 = _f1_single_feature(Fb_tr[:, j], yb_tr, Fb_val[:, j], yb_val)
        f1b_list.append(f1)
        cat = "B_specific" if f1 >= threshold else "neither"
        features_b.append({
            "feature_idx": int(j), "model": "B",
            "f1_task_a": 0.0, "f1_task_b": round(float(f1), 4), "category": cat,
        })

    def _summ(lst):
        s = {"shared": 0, "A_specific": 0, "B_specific": 0, "neither": 0}
        for c in lst:
            s[c["category"]] += 1
        return s

    summary_a = _summ(features_a)
    summary_b = _summ(features_b)

    return {
        "threshold": float(threshold),
        "aligned": False,
        "warning": "disjoint protein sets; shared not defined; see per-model results",
        "n_features_a": int(n_feat_a),
        "n_features_b": int(n_feat_b),
        "model_a": {"n_features": int(n_feat_a), "summary": summary_a, "features": features_a,
                    "mean_f1_a": round(float(np.mean(f1a_list)) if f1a_list else 0.0, 4), "mean_f1_b": 0.0},
        "model_b": {"n_features": int(n_feat_b), "summary": summary_b, "features": features_b,
                    "mean_f1_a": 0.0, "mean_f1_b": round(float(np.mean(f1b_list)) if f1b_list else 0.0, 4)},
        "summary": {"shared": 0, "A_specific": summary_a["A_specific"], "B_specific": summary_b["B_specific"],
                    "neither": summary_a["neither"] + summary_b["neither"]},
        "features": features_a,  # legacy compat
        "n_features": int(n_feat_a),
        "mean_f1_a": round(float(np.mean(f1a_list)) if f1a_list else 0.0, 4),
        "mean_f1_b": round(float(np.mean(f1b_list)) if f1b_list else 0.0, 4),
    }


def classify_features_cross_model(
    sim_matrix: np.ndarray,
    feats_a_f1: np.ndarray,
    feats_b_f1: np.ndarray,
    correlation_threshold: float = 0.5,
    f1_threshold: float = 0.3,
) -> Dict:
    """Classify via similarity matrix + per-task predictiveness.

    This is the appropriate method for disjoint protein sets where per-feature
    cross-task probes are not directly comparable.  We match each A feature to
    its best B counterpart (and vice versa) and label based on whether the
    pair is matched AND both members are predictive.

    Args:
        sim_matrix: (n_a, n_b) correlation/cosine similarities.
        feats_a_f1: (n_a,) F1 of each A feature on its task.
        feats_b_f1: (n_b,) F1 of each B feature on its task.
        correlation_threshold: min similarity to count as "matched".
        f1_threshold: min F1 to count as predictive.
    """
    sim_matrix = np.asarray(sim_matrix)
    feats_a_f1 = np.asarray(feats_a_f1)
    feats_b_f1 = np.asarray(feats_b_f1)
    if sim_matrix.ndim != 2:
        raise ValueError("sim_matrix must be 2-D")
    n_a, n_b = sim_matrix.shape
    if len(feats_a_f1) != n_a or len(feats_b_f1) != n_b:
        raise ValueError("F1 vector lengths do not match sim_matrix dims")

    best_match_b = np.argmax(sim_matrix, axis=1)
    best_sim_a = np.max(sim_matrix, axis=1)
    # Also best match for B -> A (to catch B-private)
    best_match_a = np.argmax(sim_matrix, axis=0)
    best_sim_b = np.max(sim_matrix, axis=0)

    # Classify from A's perspective
    classifications_a = []
    for i in range(n_a):
        j = int(best_match_b[i])
        sim = float(best_sim_a[i])
        a_pred = float(feats_a_f1[i]) >= f1_threshold
        b_pred = float(feats_b_f1[j]) >= f1_threshold
        matched = sim >= correlation_threshold
        if matched and a_pred and b_pred:
            cat = "shared"
        elif a_pred and not matched:
            cat = "A_specific"
        elif a_pred:
            # matched but counterpart not predictive -> still A-specific (not shared)
            cat = "A_specific"
        elif b_pred and matched:
            # A feature not predictive but its B match is – not shared for A
            cat = "neither"
        else:
            cat = "neither"
        classifications_a.append({
            "feature_a": int(i),
            "feature_b": int(j),
            "similarity": round(float(sim), 4),
            "f1_a": round(float(feats_a_f1[i]), 4),
            "f1_b": round(float(feats_b_f1[j]), 4),
            "category": cat,
        })

    # Classify from B's perspective (for B-specific counts)
    classifications_b = []
    for j in range(n_b):
        i = int(best_match_a[j])
        sim = float(best_sim_b[j])
        b_pred = float(feats_b_f1[j]) >= f1_threshold
        a_pred = float(feats_a_f1[i]) >= f1_threshold
        matched = sim >= correlation_threshold
        if matched and a_pred and b_pred:
            cat = "shared"
        elif b_pred and not matched:
            cat = "B_specific"
        elif b_pred:
            cat = "B_specific"
        else:
            cat = "neither"
        classifications_b.append({
            "feature_b": int(j),
            "feature_a": int(i),
            "similarity": round(float(sim), 4),
            "f1_a": round(float(feats_a_f1[i]), 4),
            "f1_b": round(float(feats_b_f1[j]), 4),
            "category": cat,
        })

    def _sum(lst):
        s = {"shared": 0, "A_specific": 0, "B_specific": 0, "neither": 0}
        for c in lst:
            s[c["category"]] += 1
        return s

    summary_a = _sum(classifications_a)
    summary_b = _sum(classifications_b)

    # Primary summary is from A's perspective (as before) for backward compat
    summary = dict(summary_a)

    return {
        "correlation_threshold": float(correlation_threshold),
        "f1_threshold": float(f1_threshold),
        "n_features_a": int(n_a),
        "n_features_b": int(n_b),
        "summary": summary,
        "summary_a": summary_a,
        "summary_b": summary_b,
        "features": classifications_a,
        "features_a": classifications_a,
        "features_b": classifications_b,
    }
