"""Cross-model feature similarity computation (CKA, correlation, cosine, MI)."""
import numpy as np
import warnings
from typing import Literal, Optional


def linear_cka(X: np.ndarray, Y: np.ndarray, center: bool = True) -> float:
    """Compute linear Centered Kernel Alignment (CKA).

    Correct formulation for linear kernels:
        K = X X^T , L = Y Y^T  (Gram matrices)
        HSIC(K,L) = <K_c, L_c>_F   where K_c, L_c are centered Grams.
    For the *linear* kernel this has a closed form without building n×n Grams:
        CKA = ||Y^T X||_F^2 / ( ||X^T X||_F * ||Y^T Y||_F )

    Args:
        X: (n_samples, n_features_a).
        Y: (n_samples, n_features_b). n_samples must match.
        center: whether to column-center X/Y before computing.

    Returns:
        CKA score in [0, 1].  Returns 0 if either side is degenerate.
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"X/Y sample count mismatch: {X.shape[0]} vs {Y.shape[0]}")
    if X.shape[0] < 2:
        return 0.0

    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
        Y = Y - Y.mean(axis=0, keepdims=True)

    # Frobenius norms via covariance – avoids n×n Gram.
    # ||X^T X||_F  and ||Y^T Y||_F
    xty = X.T @ Y  # (p_a, p_b)
    xtx = X.T @ X
    yty = Y.T @ Y

    hsic_xy = float(np.sum(xty ** 2))
    hsic_xx = float(np.sum(xtx ** 2))
    hsic_yy = float(np.sum(yty ** 2))

    denom = np.sqrt(hsic_xx * hsic_yy) + 1e-12
    if denom < 1e-12 or hsic_xx < 1e-12 or hsic_yy < 1e-12:
        return 0.0
    return float(np.clip(hsic_xy / denom, 0.0, 1.0))


def centered_kernel_alignment(X: np.ndarray, Y: np.ndarray) -> float:
    """Alias for linear_cka."""
    return linear_cka(X, Y)


def _standardize_columns(mat: np.ndarray, eps: float = 1e-10):
    """Center each column; return centered and per-column L2 norms."""
    m = mat - mat.mean(axis=0, keepdims=True)
    # L2 after centering
    n = np.sqrt(np.sum(m ** 2, axis=0))
    return m, n


def compute_feature_similarity_matrix(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    method: str = "correlation",
) -> np.ndarray:
    """Pairwise similarity between every feature of A and every feature of B.

    Vectorized (no n_a × n_b Python loops).

    Args:
        feats_a: (n_tokens, n_features_a).
        feats_b: (n_tokens, n_features_b). n_tokens must match.
        method: "correlation" (Pearson) or "cosine".

    Returns:
        (n_features_a, n_features_b) matrix.
    """
    feats_a = np.asarray(feats_a)
    feats_b = np.asarray(feats_b)
    if feats_a.shape[0] != feats_b.shape[0]:
        raise ValueError(
            f"Token count mismatch: feats_a {feats_a.shape[0]} vs feats_b {feats_b.shape[0]}"
        )
    if feats_a.ndim != 2 or feats_b.ndim != 2:
        raise ValueError("feats_a/b must be 2-D (n_tokens, n_features)")
    if feats_a.shape[0] == 0:
        raise ValueError("No tokens to compare")

    if method == "correlation":
        # Pearson: center then L2-normalize per column, then corr = A_norm.T @ B_norm
        a_centered, a_n = _standardize_columns(feats_a.astype(np.float64))
        b_centered, b_n = _standardize_columns(feats_b.astype(np.float64))
        # columns with ~0 variance -> zero out so corr=0
        a_valid = a_n > 1e-10
        b_valid = b_n > 1e-10
        # divide safely
        a_norm = np.zeros_like(a_centered)
        b_norm = np.zeros_like(b_centered)
        a_norm[:, a_valid] = a_centered[:, a_valid] / a_n[a_valid]
        b_norm[:, b_valid] = b_centered[:, b_valid] / b_n[b_valid]
        # corr = dot product (since vectors already unit-norm)
        # (n_a, n_b)
        sim = a_norm.T @ b_norm
        # Clamp numerical noise
        sim = np.clip(sim, -1.0, 1.0).astype(np.float32)
        return sim

    elif method == "cosine":
        a_n = np.linalg.norm(feats_a.astype(np.float64), axis=0)
        b_n = np.linalg.norm(feats_b.astype(np.float64), axis=0)
        a_valid = a_n > 1e-10
        b_valid = b_n > 1e-10
        a_norm = np.zeros_like(feats_a, dtype=np.float64)
        b_norm = np.zeros_like(feats_b, dtype=np.float64)
        a_norm[:, a_valid] = feats_a[:, a_valid] / a_n[a_valid]
        b_norm[:, b_valid] = feats_b[:, b_valid] / b_n[b_valid]
        sim = (a_norm.T @ b_norm).astype(np.float32)
        sim = np.clip(sim, -1.0, 1.0)
        return sim

    else:
        raise ValueError(f"Unknown method: {method!r} (expected 'correlation' or 'cosine')")


def estimate_mutual_information(
    X: np.ndarray,
    Y: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Histogram-based mutual information between two 1-D feature activations.

    Uses fixed-width bins spanning the joint min/max.  Zero-variance inputs
    return 0.  Result in nats.
    """
    X = np.asarray(X).ravel()
    Y = np.asarray(Y).ravel()
    if X.size != Y.size:
        raise ValueError("X/Y length mismatch")
    if X.size == 0:
        return 0.0
    if np.std(X) < 1e-12 or np.std(Y) < 1e-12:
        return 0.0

    # Bin edges inclusive of max via small epsilon
    eps = 1e-10
    x_min, x_max = float(X.min()), float(X.max())
    y_min, y_max = float(Y.min()), float(Y.max())
    # handle degenerate range
    if x_max - x_min < 1e-12 or y_max - y_min < 1e-12:
        return 0.0
    x_bins = np.linspace(x_min, x_max + eps, n_bins + 1)
    y_bins = np.linspace(y_min, y_max + eps, n_bins + 1)

    p_xy, _, _ = np.histogram2d(X, Y, bins=[x_bins, y_bins])
    total = p_xy.sum()
    if total == 0:
        return 0.0
    p_xy = p_xy / total
    # add small to avoid log(0), but keep normalization ≈1
    p_xy = p_xy + 1e-12
    p_xy = p_xy / p_xy.sum()

    p_x = p_xy.sum(axis=1, keepdims=True)  # (n_bins, 1)
    p_y = p_xy.sum(axis=0, keepdims=True)  # (1, n_bins)
    # outer already via broadcasting
    mi = np.sum(p_xy * np.log(p_xy / (p_x * p_y + 1e-12) + 1e-12))
    return float(max(0.0, mi))


def compute_mi_matrix(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    n_bins: int = 20,
    max_features: int = 200,
) -> np.ndarray:
    """Pairwise MI matrix, capped to max_features per side for cost.

    Returns shape (min(n_a, max_features), min(n_b, max_features)).
    """
    n_a = min(int(feats_a.shape[1]), int(max_features))
    n_b = min(int(feats_b.shape[1]), int(max_features))
    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=np.float32)
    mi_matrix = np.zeros((n_a, n_b), dtype=np.float32)
    for i in range(n_a):
        for j in range(n_b):
            mi_matrix[i, j] = estimate_mutual_information(
                feats_a[:, i], feats_b[:, j], n_bins=n_bins
            )
    return mi_matrix
