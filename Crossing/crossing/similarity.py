"""Cross-model feature similarity computation (CKA and Mutual Information)."""
import numpy as np
import torch
from typing import Optional, Tuple


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Linear Centered Kernel Alignment (CKA) between two feature matrices.

    Args:
        X: Shape (n_samples, n_features_a) - activations from model A.
        Y: Shape (n_samples, n_features_b) - activations from model B.

    Returns:
        CKA similarity score in [0, 1].
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    XT_X = X @ X.T
    YT_Y = Y @ Y.T
    XT_Y = X @ Y.T

    hsic_xy = torch.tensor(XT_X * YT_Y).sum() + torch.tensor(XT_Y ** 2).sum() \
              - 2 * torch.tensor(XT_Y * YT_Y).sum()
    hsic_xx = torch.tensor(XT_X * XT_X).sum()
    hsic_yy = torch.tensor(YT_Y * YT_Y).sum()

    cka = hsic_xy / (torch.sqrt(hsic_xx * hsic_yy) + 1e-10)
    return cka.item()


def centered_kernel_alignment(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute CKA (alias for linear_cka)."""
    return linear_cka(X, Y)


def compute_feature_similarity_matrix(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    method: str = "correlation",
) -> np.ndarray:
    """Compute pairwise similarity between features of two models.

    Args:
        feats_a: Shape (n_tokens, n_features_a) - SAE features from model A.
        feats_b: Shape (n_tokens, n_features_b) - SAE features from model B.
        method: "correlation" (Pearson) or "cosine".

    Returns:
        Similarity matrix of shape (n_features_a, n_features_b).
    """
    n_a = feats_a.shape[1]
    n_b = feats_b.shape[1]
    sim_matrix = np.zeros((n_a, n_b), dtype=np.float32)

    if method == "correlation":
        for i in range(n_a):
            for j in range(n_b):
                a, b = feats_a[:, i], feats_b[:, j]
                if np.std(a) < 1e-10 or np.std(b) < 1e-10:
                    sim_matrix[i, j] = 0.0
                else:
                    sim_matrix[i, j] = np.corrcoef(a, b)[0, 1]
    elif method == "cosine":
        a_norm = feats_a / (np.linalg.norm(feats_a, axis=0, keepdims=True) + 1e-10)
        b_norm = feats_b / (np.linalg.norm(feats_b, axis=0, keepdims=True) + 1e-10)
        sim_matrix = a_norm.T @ b_norm
    else:
        raise ValueError(f"Unknown method: {method}")

    return sim_matrix


def estimate_mutual_information(
    X: np.ndarray,
    Y: np.ndarray,
    n_bins: int = 20,
) -> float:
    """Estimate mutual information between two feature vectors using histogram.

    Args:
        X: Shape (n_samples,).
        Y: Shape (n_samples,).
        n_bins: Number of bins for histogram.

    Returns:
        Estimated mutual information in nats.
    """
    x_bins = np.linspace(X.min(), X.max() + 1e-10, n_bins + 1)
    y_bins = np.linspace(Y.min(), Y.max() + 1e-10, n_bins + 1)

    p_xy, _, _ = np.histogram2d(X, Y, bins=[x_bins, y_bins])
    p_xy = p_xy / p_xy.sum() + 1e-10

    p_x = p_xy.sum(axis=1)
    p_y = p_xy.sum(axis=0)

    mi = np.sum(p_xy * np.log(p_xy / (np.outer(p_x, p_y) + 1e-10)))
    return float(mi)


def compute_mi_matrix(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    n_bins: int = 20,
    max_features: int = 200,
) -> np.ndarray:
    """Compute MI matrix between features of two models.

    Args:
        feats_a: Shape (n_tokens, n_features_a).
        feats_b: Shape (n_tokens, n_features_b).
        n_bins: Number of bins for MI estimation.
        max_features: Maximum features per model to compute (for efficiency).

    Returns:
        MI matrix of shape (min(n_features_a, max_features), min(n_features_b, max_features)).
    """
    n_a = min(feats_a.shape[1], max_features)
    n_b = min(feats_b.shape[1], max_features)

    mi_matrix = np.zeros((n_a, n_b), dtype=np.float32)
    for i in range(n_a):
        for j in range(n_b):
            mi_matrix[i, j] = estimate_mutual_information(
                feats_a[:, i], feats_b[:, j], n_bins=n_bins
            )

    return mi_matrix
