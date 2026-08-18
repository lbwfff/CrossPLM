"""Cross-task probing: train a probe on one model's features to predict another task."""
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "logistic",
    max_iter: int = 1000,
) -> Dict:
    """Train a linear probe and evaluate on validation set.

    Args:
        X_train: Shape (n_samples, n_features) - SAE activations.
        y_train: Shape (n_samples,) - task labels.
        X_val: Validation features.
        y_val: Validation labels.
        method: "logistic" for logistic regression, "linear" for linear SVM.
        max_iter: Maximum iterations for optimizer.

    Returns:
        Dictionary with model, metrics, and coefficients.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    if method == "logistic":
        clf = LogisticRegression(
            max_iter=max_iter, solver="lbfgs", multi_class="auto"
        )
        clf.fit(X_train_s, y_train)
        train_pred = clf.predict(X_train_s)
        val_pred = clf.predict(X_val_s)
        val_proba = clf.predict_proba(X_val_s)
    elif method == "linear":
        clf = LinearSVC(max_iter=max_iter)
        clf.fit(X_train_s, y_train)
        train_pred = clf.predict(X_train_s)
        val_pred = clf.predict(X_val_s)
        val_proba = None
    else:
        raise ValueError(f"Unknown method: {method}")

    n_classes = len(np.unique(y_train))
    average = "macro" if n_classes > 2 else "binary"

    results = {
        "method": method,
        "n_classes": n_classes,
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_f1": float(f1_score(y_val, val_pred, average=average, zero_division=0)),
    }

    if val_proba is not None and n_classes == 2:
        try:
            results["val_auroc"] = float(roc_auc_score(y_val, val_proba[:, 1]))
        except ValueError:
            results["val_auroc"] = 0.5
    elif val_proba is not None and n_classes > 2:
        try:
            results["val_auroc"] = float(
                roc_auc_score(y_val, val_proba, average="macro", multi_class="ovr")
            )
        except ValueError:
            results["val_auroc"] = 0.5

    if hasattr(clf, "coef_"):
        results["coef_shape"] = list(clf.coef_.shape)

    return {"model": clf, "scaler": scaler, "metrics": results}


def cross_task_evaluate(
    feats_a: np.ndarray,
    labels_a: np.ndarray,
    feats_b: np.ndarray,
    labels_b: np.ndarray,
    val_ratio: float = 0.2,
    method: str = "logistic",
    seed: int = 42,
) -> Dict:
    """Evaluate cross-task information transfer.

    Trains probes: A→A, A→B, B→B, B→A.

    Args:
        feats_a: Shape (n_tokens_a, n_features_a) - SAE features from model A.
        labels_a: Shape (n_tokens_a,) - task A labels.
        feats_b: Shape (n_tokens_b, n_features_b) - SAE features from model B.
        labels_b: Shape (n_tokens_b,) - task B labels.
        val_ratio: Fraction of data for validation.
        method: Probe method.
        seed: Random seed.

    Returns:
        Dictionary with results for all four directions.
    """
    rng = np.random.RandomState(seed)

    def _split(feats, labels):
        n = len(labels)
        idx = rng.permutation(n)
        n_val = int(n * val_ratio)
        val_idx, train_idx = idx[:n_val], idx[n_val:]
        return feats[train_idx], labels[train_idx], feats[val_idx], labels[val_idx]

    Xa_tr, ya_tr, Xa_val, ya_val = _split(feats_a, labels_a)
    Xb_tr, yb_tr, Xb_val, yb_val = _split(feats_b, labels_b)

    results = {}

    # A → A
    probe_a = train_probe(Xa_tr, ya_tr, Xa_val, ya_val, method=method)
    results["A_to_A"] = probe_a["metrics"]

    # B → B
    probe_b = train_probe(Xb_tr, yb_tr, Xb_val, yb_val, method=method)
    results["B_to_B"] = probe_b["metrics"]

    # A → B (train on A, test on B)
    probe_a_to_b = train_probe(Xa_tr, ya_tr, Xb_val, yb_val, method=method)
    results["A_to_B"] = probe_a_to_b["metrics"]

    # B → A (train on B, test on A)
    probe_b_to_a = train_probe(Xb_tr, yb_tr, Xa_val, ya_val, method=method)
    results["B_to_A"] = probe_b_to_a["metrics"]

    return results


def compute_transfer_matrix(results: Dict) -> np.ndarray:
    """Extract the 2x2 transfer matrix from cross-task results.

    Returns:
        2x2 array where [i][j] = performance of probe trained on task i, tested on task j.
    """
    matrix = np.array([
        [results["A_to_A"]["val_f1"], results["A_to_B"]["val_f1"]],
        [results["B_to_A"]["val_f1"], results["B_to_B"]["val_f1"]],
    ])
    return matrix
