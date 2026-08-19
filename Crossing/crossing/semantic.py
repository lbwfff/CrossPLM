"""Biological semantic similarity between cross-model SAE features.

Each SAE feature can be annotated by its *concept hit profile*: which
biological concepts it predicts above threshold and with what strength.

We reuse Single's ``align_features_to_concepts`` (F1/precision/recall/AUROC)
so that the biological semantics are grounded in the same computation that
Single uses for single-model interpretability.  The semantic similarity
between a feature from model A and a feature from model B is then the
similarity of their concept-hit profiles.

Supports:

- Concept-F1 vector cosine (default, robust and graded).
- Binary Jaccard over "hit" concept sets (interpretable).
- Combined score S_cross = α * S_activation + β * S_semantic  (ROADMAP 1.4).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import sparse

try:
    import pandas as pd  # noqa: F401
except Exception:  # soft dep
    pd = None


# ---------------------------------------------------------------------------
# Core helpers – concept-F1 matrices
# ---------------------------------------------------------------------------

def _load_sae(model_dir: str | Path, device: str = "cpu"):
    from single.sae.inference import load_sae  # type: ignore

    return load_sae(str(model_dir), device=device)


def _discover_shard_ids(path: Path) -> List[int]:
    ids: List[int] = []
    for child in path.glob("shard_*"):
        m = re.fullmatch(r"shard_(\d+)", child.name)
        if m:
            ids.append(int(m.group(1)))
    ids = sorted(set(ids))
    if not ids:
        raise ValueError(f"No numeric shard directories found in {path}")
    return ids


def _load_embeddings_dir(embeddings_dir: Path, device: str = "cpu") -> Tuple[torch.Tensor, List[Path]]:
    """Load all shards concatenated – caller must ensure shards are aligned."""
    shard_ids = _discover_shard_ids(embeddings_dir)
    all_embs: List[torch.Tensor] = []
    shard_paths: List[Path] = []
    for sid in shard_ids:
        p = embeddings_dir / f"shard_{sid}" / "embeddings.pt"
        if not p.exists():
            # try recursive
            cands = sorted(embeddings_dir.glob(f"shard_{sid}/**/embeddings.pt"))
            if not cands:
                raise FileNotFoundError(f"Missing embeddings for shard {sid} in {embeddings_dir}")
            p = cands[0]
        data = torch.load(p, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            emb = data.get("embeddings", next(iter(data.values())))
        else:
            emb = data
        all_embs.append(emb.float())
        shard_paths.append(p)
    return torch.cat(all_embs, dim=0), shard_paths


def compute_concept_f1_matrix(
    sae_dir: str | Path,
    embeddings_dir: str | Path,
    concepts_dir: str | Path,
    device: str = "cpu",
    feature_chunk_size: int = 200,
    batch_size: int = 1024,
    min_positives: int = 10,
    threshold_percents: Optional[List[float]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Compute per-feature concept-F1 matrix.

    Returns:
        F: (n_features, n_concepts)  F1 scores (0 if no alignment above threshold).
        concept_names: length n_concepts.
    """
    from single.analysis.concepts import load_concept_names  # type: ignore
    from single.analysis.feature_alignment import align_features_to_concepts  # type: ignore
    from single.scripts.analyze_concepts import (  # type: ignore
        _load_combined_shards,
        _discover_shard_ids as _discover_concept_shards,
    )

    sae_dir = Path(sae_dir)
    embeddings_dir = Path(embeddings_dir)
    concepts_dir = Path(concepts_dir)

    sae = _load_sae(sae_dir, device=device)
    concept_names = load_concept_names(concepts_dir)
    if not concept_names:
        raise ValueError(f"No concept_columns.txt in {concepts_dir}")

    # Try pooled combined shards (aligned residue identity via metadata)
    try:
        shard_ids = _discover_concept_shards(concepts_dir)
        pooled_matrix, pooled_embeddings, pooled_protein_ids = _load_combined_shards(
            embeddings_dir, concepts_dir, shard_ids, device
        )
        concept_matrix = pooled_matrix
        embeddings = pooled_embeddings
        protein_ids = pooled_protein_ids
    except Exception as e:
        # Fallback: naive concat (for unvalidated builds / unit tests)
        print(f"[semantic] pooled shard load failed ({e}); falling back to concat embeddings")
        embeddings, _ = _load_embeddings_dir(embeddings_dir, device=device)
        # Build a single sparse concept matrix by stacking shards naively
        from single.analysis.concepts import load_concept_shards  # type: ignore

        mats = []
        for sid in _discover_shard_ids(concepts_dir):
            m, _ = load_concept_shards(concepts_dir, sid)
            mats.append(m.tocsr())
        concept_matrix = sparse.vstack(mats, format="csr")
        protein_ids = None

    n_features = int(sae.dict_size)
    n_concepts = len(concept_names)

    metrics = align_features_to_concepts(
        sae=sae,
        embeddings=embeddings.to(device) if hasattr(embeddings, "to") else embeddings,
        concept_matrix=concept_matrix,
        concept_names=concept_names,
        feature_chunk_size=feature_chunk_size,
        batch_size=batch_size,
        compute_auroc=False,
        compute_domain_f1=False,
        min_positives=min_positives,
        threshold_percents=threshold_percents,
        protein_ids=protein_ids,
        device=device,
    )

    # Flatten to dense F matrix
    F = np.zeros((n_features, n_concepts), dtype=np.float32)
    name_to_idx = {n: i for i, n in enumerate(concept_names)}
    for fidx, concept_dict in metrics.items():
        for concept, entry in concept_dict.items():
            j = name_to_idx.get(concept)
            if j is not None:
                F[int(fidx), j] = float(entry.get("f1", 0.0))
    return F, concept_names


# ---------------------------------------------------------------------------
# Semantic similarity from concept-F1 matrices
# ---------------------------------------------------------------------------

def semantic_similarity_matrix(
    F_a: np.ndarray,
    F_b: np.ndarray,
    mode: str = "cosine",
    hit_threshold: float = 0.3,
) -> np.ndarray:
    """Pairwise semantic similarity between features of two models.

    Args:
        F_a: (n_a, n_concepts) F1 scores for model A.
        F_b: (n_b, n_concepts) F1 scores for model B.  Concepts must be
             aligned (same column ordering / same concept set).  If the two
             runs used the same TSV & categorical options they will be.
        mode: "cosine" (graded, cosine between F1 vectors) or "jaccard"
              (binary Jaccard over hit-sets defined by hit_threshold) or
              "pearson".
        hit_threshold: for "jaccard", F1 >= threshold counts as a hit.

    Returns:
        (n_a, n_b) similarity matrix in [0, 1] (jaccard/cosine non-negative
        after ReLU).  For Pearson the range is [-1, 1] (clipped).
    """
    if F_a.shape[1] != F_b.shape[1]:
        raise ValueError(
            f"Concept count mismatch: F_a has {F_a.shape[1]} concepts, "
            f"F_b has {F_b.shape[1]}.  Build both concept matrices from the same "
            "TSV / categorical options so columns align."
        )
    n_a, n_b = F_a.shape[0], F_b.shape[0]
    if n_a == 0 or n_b == 0:
        return np.zeros((n_a, n_b), dtype=np.float32)

    if mode == "cosine":
        # L2-normalize each feature's concept vector, then dot
        a = F_a.astype(np.float64)
        b = F_b.astype(np.float64)
        # Replace NaN
        a = np.nan_to_num(a)
        b = np.nan_to_num(b)
        a_n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
        b_n = np.linalg.norm(b, axis=1, keepdims=True) + 1e-12
        # zero-vector features -> remain zeros -> similarity 0
        a_norm = a / a_n
        b_norm = b / b_n
        # zero out originally-zero rows (they divided by eps, not zero)
        a_zero = np.linalg.norm(a, axis=1) < 1e-10
        b_zero = np.linalg.norm(b, axis=1) < 1e-10
        a_norm[a_zero] = 0.0
        b_norm[b_zero] = 0.0
        sim = (a_norm @ b_norm.T).astype(np.float32)
        # cosine of non-negative F1 vectors is in [0, 1]; clamp numerical
        sim = np.clip(sim, 0.0, 1.0)
        return sim

    elif mode == "jaccard":
        # Binarize hits
        A_hit = (F_a >= hit_threshold)
        B_hit = (F_b >= hit_threshold)
        # Vectorized Jaccard: for each (i,j), |A_i cap B_j| / |A_i cup B_j|
        # Could be large (n_a * n_b * n_concepts) – do in chunks
        sim = np.zeros((n_a, n_b), dtype=np.float32)
        chunk = 512
        for i0 in range(0, n_a, chunk):
            chunk_a = A_hit[i0:i0+chunk]  # (c, K)
            for j0 in range(0, n_b, chunk):
                chunk_b = B_hit[j0:j0+chunk]  # (d, K)
                # intersection: (c, d) = sum_k  A[i,k] & B[j,k]
                inter = (chunk_a[:, None, :] & chunk_b[None, :, :]).sum(axis=2).astype(np.float32)
                union = (chunk_a[:, None, :] | chunk_b[None, :, :]).sum(axis=2).astype(np.float32)
                # Both empty -> 1.0 if we treat "no concept" as trivially similar,
                # but for SAE semantics it's more honest to return 0 (no evidence).
                # Use 0 when union==0.
                out = np.zeros_like(inter)
                nz = union > 0
                out[nz] = inter[nz] / union[nz]
                sim[i0:i0+chunk, j0:j0+chunk] = out
        return sim

    elif mode == "pearson":
        # Pearson between F1 vectors – useful when many concepts
        # Center per feature
        a = F_a.astype(np.float64)
        b = F_b.astype(np.float64)
        a_c = a - a.mean(axis=1, keepdims=True)
        b_c = b - b.mean(axis=1, keepdims=True)
        a_n = np.sqrt(np.sum(a_c ** 2, axis=1)) + 1e-12
        b_n = np.sqrt(np.sum(b_c ** 2, axis=1)) + 1e-12
        a_norm = a_c / a_n[:, None]
        b_norm = b_c / b_n[:, None]
        # zero-var rows -> 0
        a_zero = np.sqrt(np.sum(a_c ** 2, axis=1)) < 1e-10
        b_zero = np.sqrt(np.sum(b_c ** 2, axis=1)) < 1e-10
        a_norm[a_zero] = 0.0
        b_norm[b_zero] = 0.0
        sim = (a_norm @ b_norm.T).astype(np.float32)
        return np.clip(sim, -1.0, 1.0)

    else:
        raise ValueError(f"Unknown semantic mode {mode!r} (expected 'cosine'/'jaccard'/'pearson')")


def combined_similarity(
    S_activation: np.ndarray,
    S_semantic: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5,
    normalize: bool = True,
) -> np.ndarray:
    """ROADMAP S_cross = α S_activation + β S_semantic.

    If normalize=True, each matrix is min-max normalized to [0,1] before
    combination (so a Pearson activation in [-1,1] doesn't dominate a
    Jaccard semantic in [0,1]).  Already-[0,1] inputs are unaffected.
    """
    if S_activation.shape != S_semantic.shape:
        raise ValueError("Activation and semantic matrices must have same shape")
    if normalize:
        def _norm(m: np.ndarray) -> np.ndarray:
            mn, mx = float(m.min()), float(m.max())
            if mx - mn < 1e-12:
                return np.zeros_like(m, dtype=np.float32)
            return ((m - mn) / (mx - mn)).astype(np.float32)

        S_activation = _norm(S_activation)
        S_semantic = _norm(S_semantic)
    total = alpha + beta
    if total < 1e-12:
        raise ValueError("alpha+beta must be >0")
    alpha_n = alpha / total
    beta_n = beta / total
    return (alpha_n * S_activation + beta_n * S_semantic).astype(np.float32)


def build_semantic_matrices(
    sae_a_dir: str | Path,
    embeddings_a_dir: str | Path,
    concepts_a_dir: str | Path,
    sae_b_dir: str | Path,
    embeddings_b_dir: str | Path,
    concepts_b_dir: str | Path,
    device: str = "cpu",
    mode: str = "cosine",
    hit_threshold: float = 0.3,
    feature_chunk_size: int = 200,
    batch_size: int = 1024,
    min_positives: int = 10,
) -> Dict:
    """High-level helper: given two experiments' dirs, build semantic matrices.

    Returns dict with:
        F_a, F_b, concept_names, S_semantic, S_semantic_mode, hit_threshold
    If the two models share the same concept build (recommended), concepts
    align directly.  Otherwise we intersect concept names.
    """
    F_a, names_a = compute_concept_f1_matrix(
        sae_a_dir, embeddings_a_dir, concepts_a_dir,
        device=device, feature_chunk_size=feature_chunk_size,
        batch_size=batch_size, min_positives=min_positives,
    )
    F_b, names_b = compute_concept_f1_matrix(
        sae_b_dir, embeddings_b_dir, concepts_b_dir,
        device=device, feature_chunk_size=feature_chunk_size,
        batch_size=batch_size, min_positives=min_positives,
    )

    if names_a == names_b:
        S = semantic_similarity_matrix(F_a, F_b, mode=mode, hit_threshold=hit_threshold)
        names = names_a
    else:
        # Intersect concepts
        common = sorted(set(names_a) & set(names_b))
        if not common:
            raise ValueError("No common concepts between the two builds – ensure same TSV/options")
        idx_a = [names_a.index(n) for n in common]
        idx_b = [names_b.index(n) for n in common]
        S = semantic_similarity_matrix(F_a[:, idx_a], F_b[:, idx_b], mode=mode, hit_threshold=hit_threshold)
        names = common
        F_a = F_a[:, idx_a]
        F_b = F_b[:, idx_b]

    return {
        "F_a": F_a,
        "F_b": F_b,
        "concept_names": names,
        "S_semantic": S,
        "mode": mode,
        "hit_threshold": hit_threshold,
        "n_features_a": int(F_a.shape[0]),
        "n_features_b": int(F_b.shape[0]),
        "n_concepts": len(names),
    }
