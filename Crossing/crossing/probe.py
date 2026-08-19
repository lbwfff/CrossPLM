"""Cross-task probing: train linear probes on SAE features to predict task labels.

Two regimes:

- Aligned (recommended): feats_a and feats_b are computed on the *same*
  proteins/tokens in the same order (n_tokens identical).  Labels for both
  tasks are available on those tokens (or on an overlapping subset).  This
  directly measures whether one model's representation contains information
  about the other task.

- Disjoint (fallback): feats_a/labels_a and feats_b/labels_b come from
  different protein sets.  We still report per-task baselines but cross-task
  numbers are not comparable and a warning is emitted.  For a meaningful
  cross-task test on disjoint tasks, re-extract embeddings: run model A on
  task B's proteins (and vice versa) and then probe in aligned mode.
"""

import warnings
from typing import Dict, Optional, Tuple

import numpy as np


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "logistic",
    max_iter: int = 1000,
    class_weight: Optional[str] = "balanced",
) -> Dict:
    """Train a linear probe and evaluate on validation set.

    Args:
        X_train: (n_train, n_features).
        y_train: (n_train,)  integer labels (-100 already removed).
        X_val:   (n_val, n_features).
        y_val:   (n_val,)
        method: "logistic" or "linear" (LinearSVC).
        max_iter: optimizer iterations.
        class_weight: "balanced" (default) or None.  Balanced is more
            appropriate for skewed per-residue labels.

    Returns:
        {"model", "scaler", "metrics"} dict.  Metrics always include
        train_accuracy, val_accuracy, val_f1; val_auroc when probas available.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    if X_train.shape[0] == 0 or X_val.shape[0] == 0:
        raise ValueError("Empty train/val split")
    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Feature dim mismatch between train and val")
    if len(np.unique(y_train)) < 2:
        # Single-class degenerate split – report chance level.
        return {
            "model": None,
            "scaler": None,
            "metrics": {
                "method": method,
                "n_classes": int(len(np.unique(y_train))),
                "train_accuracy": 1.0,
                "val_accuracy": float(np.mean(y_val == y_train[0])) if len(y_val) else 0.0,
                "val_f1": 0.0,
                "val_auroc": 0.5,
                "warning": "single-class train split",
            },
        }

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    if method == "logistic":
        clf = LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            class_weight=class_weight,
        )
        clf.fit(X_train_s, y_train)
        train_pred = clf.predict(X_train_s)
        val_pred = clf.predict(X_val_s)
        try:
            val_proba = clf.predict_proba(X_val_s)
        except Exception:
            val_proba = None
    elif method == "linear":
        clf = LinearSVC(max_iter=max_iter, class_weight=class_weight)
        clf.fit(X_train_s, y_train)
        train_pred = clf.predict(X_train_s)
        val_pred = clf.predict(X_val_s)
        val_proba = None
    else:
        raise ValueError(f"Unknown method {method!r} (expected 'logistic' or 'linear')")

    n_classes = len(np.unique(y_train))
    average = "macro" if n_classes > 2 else "binary"

    metrics = {
        "method": method,
        "n_classes": int(n_classes),
        "train_accuracy": float(accuracy_score(y_train, train_pred)),
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_f1": float(f1_score(y_val, val_pred, average=average, zero_division=0)),
    }
    if val_proba is not None:
        try:
            if n_classes == 2:
                metrics["val_auroc"] = float(roc_auc_score(y_val, val_proba[:, 1]))
            else:
                metrics["val_auroc"] = float(
                    roc_auc_score(y_val, val_proba, average="macro", multi_class="ovr")
                )  # type: ignore[call-arg]
        except ValueError:
            metrics["val_auroc"] = 0.5
    if hasattr(clf, "coef_"):
        metrics["coef_shape"] = list(clf.coef_.shape)
    return {"model": clf, "scaler": scaler, "metrics": metrics}


def _train_val_split(n: int, val_ratio: float, rng: np.random.RandomState):
    idx = rng.permutation(n)
    n_val = int(round(n * val_ratio))
    n_val = max(1, min(n_val, n - 1)) if n >= 2 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx


def cross_task_evaluate_aligned(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    val_ratio: float = 0.2,
    method: str = "logistic",
    seed: int = 42,
) -> Dict:
    """Aligned evaluation: same tokens, same row order for both models.

    All four probes use the SAME train/val split indices so results are
    directly comparable:

        A->A : feats_a[train] / labels_a[train]  ->  feats_a[val] / labels_a[val]
        B->B : feats_b[train] / labels_b[train]  ->  feats_b[val] / labels_b[val]
        A->B : feats_a[train] / labels_b[train]  ->  feats_a[val] / labels_b[val]
        B->A : feats_b[train] / labels_a[train]  ->  feats_b[val] / labels_a[val]

    Requires feats_a.shape[0] == feats_b.shape[0] == len(labels_a) == len(labels_b).
    Rows where labels_a==-100 or labels_b==-100 are dropped jointly so the
    remaining tokens stay aligned.

    Returns dict with keys A_to_A, B_to_B, A_to_B, B_to_A, transfer_matrix,
    n_tokens, n_tokens_valid, and an alignment flag.
    """
    feats_a = np.asarray(feats_a)
    feats_b = np.asarray(feats_b)
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    n = feats_a.shape[0]
    if not (feats_b.shape[0] == n == len(labels_a) == len(labels_b)):
        raise ValueError(
            f"Aligned mode requires n_tokens equal: feats_a {feats_a.shape[0]}, "
            f"feats_b {feats_b.shape[0]}, labels_a {len(labels_a)}, labels_b {len(labels_b)}"
        )
    # Joint valid mask (keep rows where BOTH tasks have a real label)
    # If tasks have different ignore patterns, keep intersection so probes run
    # on the same tokens.
    valid = (labels_a != -100) & (labels_b != -100)
    if not np.any(valid):
        raise ValueError("No tokens with valid labels for both tasks after filtering (-100)")
    feats_a = feats_a[valid]
    feats_b = feats_b[valid]
    labels_a = labels_a[valid]
    labels_b = labels_b[valid]

    rng = np.random.RandomState(seed)
    train_idx, val_idx = _train_val_split(len(labels_a), val_ratio, rng)

    results: Dict = {}
    results["n_tokens"] = int(n)
    results["n_tokens_valid"] = int(len(labels_a))
    results["aligned"] = True

    # A -> A
    r = train_probe(
        feats_a[train_idx], labels_a[train_idx],
        feats_a[val_idx], labels_a[val_idx],
        method=method,
    )
    results["A_to_A"] = r["metrics"]

    # B -> B
    r = train_probe(
        feats_b[train_idx], labels_b[train_idx],
        feats_b[val_idx], labels_b[val_idx],
        method=method,
    )
    results["B_to_B"] = r["metrics"]

    # A -> B  (A representation predicts B label)
    r = train_probe(
        feats_a[train_idx], labels_b[train_idx],
        feats_a[val_idx], labels_b[val_idx],
        method=method,
    )
    results["A_to_B"] = r["metrics"]

    # B -> A
    r = train_probe(
        feats_b[train_idx], labels_a[train_idx],
        feats_b[val_idx], labels_a[val_idx],
        method=method,
    )
    results["B_to_A"] = r["metrics"]

    results["transfer_matrix"] = compute_transfer_matrix(results).tolist()
    return results


def cross_task_evaluate(
    feats_a: np.ndarray,
    labels_a: np.ndarray,
    feats_b: np.ndarray,
    labels_b: np.ndarray,
    val_ratio: float = 0.2,
    method: str = "logistic",
    seed: int = 42,
) -> Dict:
    """Legacy disjoint evaluation (kept for compatibility).

    If feats_a/feats_b have the same n_tokens this is likely a mis-aligned
    call – delegate to the aligned version which does meaningful cross-task
    probes.  Otherwise, with truly disjoint protein sets, we report per-task
    baselines (A->A, B->B) and degraded cross probes that train/test on
    different distributions (train on A distribution, test on B distribution).
    Those cross numbers are NOT comparable to the baselines and are marked
    as such.

    Prefer `cross_task_evaluate_aligned` for publishable results.
    """
    feats_a = np.asarray(feats_a)
    feats_b = np.asarray(feats_b)
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)

    # Fast path: same n -> assume caller forgot to use aligned mode.
    if feats_a.shape[0] == feats_b.shape[0] == len(labels_a) == len(labels_b):
        warnings.warn(
            "cross_task_evaluate called with equal n_tokens; delegating to "
            "cross_task_evaluate_aligned (same proteins, same order). "
            "For disjoint tasks use cross-encoded embeddings.",
            UserWarning,
        )
        return cross_task_evaluate_aligned(feats_a, feats_b, labels_a, labels_b, val_ratio, method, seed)

    # Truly disjoint: keep old behavior but fix the cross logic to at least
    # be explicit.  We still train on source distribution and test on target
    # distribution – report with a warning and don't pretend it's the same as
    # the aligned information-transfer test.
    warnings.warn(
        "Cross-task evaluation on DISJOINT protein sets: A_to_B / B_to_A "
        "train/test on different distributions and are not directly comparable "
        "to A_to_A / B_to_B.  For a valid information-transfer test, "
        "re-extract embeddings so both models encode the SAME proteins and use "
        "cross_task_evaluate_aligned.",
        UserWarning,
    )
    rng = np.random.RandomState(seed)

    def _split(feats, labels):
        # filter -100 already? keep defense
        valid = labels != -100
        feats = feats[valid]
        labels = labels[valid]
        train_idx, val_idx = _train_val_split(len(labels), val_ratio, rng)
        return feats[train_idx], labels[train_idx], feats[val_idx], labels[val_idx]

    Xa_tr, ya_tr, Xa_val, ya_val = _split(feats_a, labels_a)
    Xb_tr, yb_tr, Xb_val, yb_val = _split(feats_b, labels_b)

    results: Dict = {}
    results["aligned"] = False
    results["warning"] = "disjoint protein sets; cross-task numbers not comparable"
    results["n_tokens_a"] = int(feats_a.shape[0])
    results["n_tokens_b"] = int(feats_b.shape[0])

    # Baselines always valid
    results["A_to_A"] = train_probe(Xa_tr, ya_tr, Xa_val, ya_val, method=method)["metrics"]
    results["B_to_B"] = train_probe(Xb_tr, yb_tr, Xb_val, yb_val, method=method)["metrics"]
    # Degraded cross: source model space -> target labels (different spaces)
    # We keep it but flag.
    try:
        # Need same feature dim to even attempt; if dict sizes differ, skip.
        if Xa_tr.shape[1] != Xb_val.shape[1]:
            results["A_to_B"] = {"val_f1": 0.0, "val_accuracy": 0.0, "error": "feature dim mismatch"}
            results["B_to_A"] = {"val_f1": 0.0, "val_accuracy": 0.0, "error": "feature dim mismatch"}
        else:
            results["A_to_B"] = train_probe(Xa_tr, ya_tr, Xb_val, yb_val, method=method)["metrics"]
            results["B_to_A"] = train_probe(Xb_tr, yb_tr, Xa_val, ya_val, method=method)["metrics"]
    except Exception as e:
        results["A_to_B"] = {"val_f1": 0.0, "error": str(e)}
        results["B_to_A"] = {"val_f1": 0.0, "error": str(e)}

    # Only fill transfer_matrix when comparable
    try:
        results["transfer_matrix"] = compute_transfer_matrix(results).tolist()
    except Exception:
        pass
    return results


def compute_transfer_matrix(results: Dict) -> np.ndarray:
    """Extract 2×2 transfer matrix [train_task, test_task] from results.

    Returns:
        2x2 array where [0,0]=A->A, [0,1]=A->B, [1,0]=B->A, [1,1]=B->B.
        Values are val_f1.
    """
    return np.array([
        [results["A_to_A"]["val_f1"], results["A_to_B"]["val_f1"]],
        [results["B_to_A"]["val_f1"], results["B_to_B"]["val_f1"]],
    ], dtype=np.float32)
