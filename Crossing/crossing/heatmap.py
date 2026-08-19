"""Feature × Feature correspondence heatmap & clustering.

Inputs: similarity matrices S_act, S_sem, S_cross  (n_a, n_b).
Outputs:
  - heatmap PNG (optional reordering by hierarchical clustering)
  - row/col cluster assignments
  - feature communities (joint clustering on augmented space)

Uses only dependencies already in requirements.txt (numpy, scipy, matplotlib, sklearn).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _hclust_order(sim: np.ndarray, axis: str = "row") -> Tuple[np.ndarray, np.ndarray]:
    """Hierarchical clustering order for one axis.

    We cluster features by their similarity profile: row vectors for A,
    column vectors for B.  Returns (order, linkage_matrix).

    Single-axis distance = 1 - cosine similarity between profile vectors
    (robust to scale).  Uses scipy.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    if axis == "row":
        profiles = sim  # (n_a, n_b) -> cluster n_a items
    else:
        profiles = sim.T  # (n_b, n_a) -> cluster n_b items

    n = profiles.shape[0]
    if n <= 1:
        return np.arange(n), np.zeros((0, 4))
    if n == 2:
        return np.array([0, 1]), np.zeros((0, 4))

    # Handle zero rows/cols: pdist cosine gives NaN for zero vectors.
    norms = np.linalg.norm(profiles, axis=1)
    zero_mask = norms < 1e-12
    # Replace zero profiles with small random to avoid NaN (they'll cluster together)
    if np.any(zero_mask):
        rng = np.random.RandomState(0)
        profiles = profiles.copy()
        profiles[zero_mask] = rng.randn(int(zero_mask.sum()), profiles.shape[1]) * 1e-6

    try:
        dists = pdist(profiles, metric="cosine")
        dists = np.nan_to_num(dists, nan=1.0)
        dists = np.clip(dists, 0.0, 2.0)
        Z = linkage(dists, method="average")
        order = leaves_list(Z)
    except Exception:
        # Fallback: sort by max similarity
        if axis == "row":
            order = np.argsort(-sim.max(axis=1))
        else:
            order = np.argsort(-sim.max(axis=0))
        Z = np.zeros((0, 4))
    return order, Z


def reorder_by_clustering(
    S: np.ndarray,
    reorder_rows: bool = True,
    reorder_cols: bool = True,
) -> Dict:
    """Reorder S by hierarchical clustering on row/column profiles.

    Returns:
        S_reordered, row_order, col_order, row_linkage, col_linkage
    """
    row_order = np.arange(S.shape[0])
    col_order = np.arange(S.shape[1])
    row_Z = np.zeros((0, 4))
    col_Z = np.zeros((0, 4))
    if reorder_rows and S.shape[0] > 1:
        row_order, row_Z = _hclust_order(S, axis="row")
    if reorder_cols and S.shape[1] > 1:
        col_order, col_Z = _hclust_order(S, axis="col")
    return {
        "S_reordered": S[row_order][:, col_order],
        "row_order": row_order,
        "col_order": col_order,
        "row_linkage": row_Z,
        "col_linkage": col_Z,
    }


def assign_communities(
    S: np.ndarray,
    n_row_clusters: int = 8,
    n_col_clusters: int = 8,
    method: str = "kmeans",
) -> Dict:
    """Assign each feature to a community/cluster.

    Clusters A-features and B-features separately based on their
    cross-similarity profiles.  This is a lightweight alternative to
    biclustering and fits the "feature communities" story in the roadmap.

    Args:
        S: (n_a, n_b) similarity (typically S_cross or S_act).
        n_row_clusters, n_col_clusters: number of clusters.
        method: "kmeans" (default) or "hclust".

    Returns:
        row_labels (n_a,), col_labels (n_b,), plus cluster centers/maps.
    """
    from sklearn.cluster import KMeans

    def _cluster(profiles: np.ndarray, k: int) -> np.ndarray:
        n = profiles.shape[0]
        k = max(1, min(k, n))
        if n <= 1:
            return np.zeros(n, dtype=int)
        # Row-normalize for cosine-ish clustering
        norms = np.linalg.norm(profiles, axis=1, keepdims=True) + 1e-12
        normed = profiles / norms
        # Handle zero rows already normalized to ~0
        if method == "hclust":
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import pdist

            dists = pdist(normed, metric="cosine")
            dists = np.nan_to_num(dists, nan=1.0)
            Z = linkage(dists, method="average")
            labels = fcluster(Z, t=k, criterion="maxclust") - 1
            return labels.astype(int)
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            return km.fit_predict(normed).astype(int)

    row_labels = _cluster(S, n_row_clusters)
    col_labels = _cluster(S.T, n_col_clusters)
    return {
        "row_labels": row_labels,
        "col_labels": col_labels,
        "n_row_clusters": int(len(np.unique(row_labels))),
        "n_col_clusters": int(len(np.unique(col_labels))),
    }


def save_heatmap(
    S: np.ndarray,
    output_path: str | Path,
    title: str = "Feature × Feature Similarity",
    xlabel: str = "Task B features",
    ylabel: str = "Task A features",
    cmap: str = "viridis",
    reorder: bool = True,
    annotate_top_k: int = 0,
    S_raw: Optional[np.ndarray] = None,
) -> Path:
    """Save a heatmap PNG.  Optionally reorder rows/cols by clustering.

    Args:
        S: (n_a, n_b) similarity to display (already chosen: act/sem/cross).
        output_path: PNG path.
        reorder: if True, cluster-reorder before plotting.
        annotate_top_k: if >0, annotate top-k matches (by raw or displayed matrix).
        S_raw: if given, top-k is selected from S_raw then mapped through reorder.

    Returns:
        Path to saved file.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if reorder:
        info = reorder_by_clustering(S, reorder_rows=True, reorder_cols=True)
        S_disp = info["S_reordered"]
        row_order = info["row_order"]
        col_order = info["col_order"]
    else:
        S_disp = S
        row_order = np.arange(S.shape[0])
        col_order = np.arange(S.shape[1])

    # Cap display size for readability: if huge, we show the reordered matrix
    # as-is (matplotlib handles downsampling); ticks are thinned.
    n_a, n_b = S_disp.shape
    figsize = (max(6, min(16, n_b / 64 + 6)), max(5, min(14, n_a / 64 + 5)))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(S_disp, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Thin ticks for large matrices
    if n_b > 100:
        step = max(1, n_b // 20)
        ax.set_xticks(np.arange(0, n_b, step))
        ax.set_xticklabels(col_order[::step], fontsize=6)
    if n_a > 100:
        step = max(1, n_a // 20)
        ax.set_yticks(np.arange(0, n_a, step))
        ax.set_yticklabels(row_order[::step], fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if annotate_top_k and annotate_top_k > 0:
        src = S_raw if S_raw is not None else S
        # Map top matches through reorder: find (i,j) in original, then locate in reordered
        flat = np.argsort(src.ravel())[::-1]
        rank = {}
        # Build inverse order maps
        inv_row = {old: new for new, old in enumerate(row_order)}
        inv_col = {old: new for new, old in enumerate(col_order)}
        count = 0
        for f in flat:
            i, j = divmod(int(f), src.shape[1])
            if src[i, j] <= 0:
                break
            ax.plot(inv_col[j], inv_row[i], marker="o", color="red", markersize=3, markerfacecolor="none")
            # Avoid clutter: only annotate first 5
            if count < 5:
                ax.text(inv_col[j], inv_row[i], f"{i}->{j}", color="white", fontsize=5, ha="center", va="bottom")
            count += 1
            if count >= annotate_top_k:
                break

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path
