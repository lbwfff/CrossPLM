"""Controls for cross-model feature similarity.

Reviewer-critical: "shared activation != shared biology".  We must show that
high activation correlation is not just due to generic covariates:

- protein length
- amino-acid composition
- protein family (if available)
- secondary structure content (if concepts include HELIX/STRAND)
- sequence similarity / redundancy
- random feature pairing / label shuffling (permutation null)
- unrelated-task control (concept hit comparison)

This module provides *partial correlation / residualization* and *permutation
null* utilities that can be applied to either the global CKA or the per-feature
Pearson matrix.

Usage pattern:
    1. Build per-token covariates dataframe (n_tokens rows).
    2. Residualize feats_a/b against covariates -> corrected sim matrix.
    3. Compare raw vs residualized sim; large drop = confounded.
    4. Permutation null: shuffle pairings / labels within length-matched bins.

No new dependencies beyond requirements.txt.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Covariates
# ---------------------------------------------------------------------------

def build_length_covariate(
    protein_ids: np.ndarray,
    protein_lengths: Optional[np.ndarray] = None,
    token_protein: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-token scalar covariate: protein length repeated per residue.

    Args:
        protein_ids: (n_tokens,) int protein index per token.
        protein_lengths: (n_proteins,) lengths.  If None, inferred from token counts.
        token_protein: unused, kept for API compat if caller has residues.csv.

    Returns:
        (n_tokens, 1) array with per-token protein length.
    """
    protein_ids = np.asarray(protein_ids)
    if protein_lengths is None:
        # Infer from token counts (approx, but preserves ordering)
        uniq, counts = np.unique(protein_ids, return_counts=True)
        length_map = dict(zip(uniq, counts))
        per_token = np.array([length_map[pid] for pid in protein_ids], dtype=np.float64)
    else:
        protein_lengths = np.asarray(protein_lengths)
        per_token = protein_lengths[protein_ids].astype(np.float64)
    return per_token[:, None]


def build_aa_composition_covariates(
    residue_aas: np.ndarray,
    protein_ids: np.ndarray,
    window: str = "protein",
) -> np.ndarray:
    """Per-token AA composition covariates expanded to token level.

    Args:
        residue_aas: (n_tokens,) single-letter AA strings.
        protein_ids: (n_tokens,) protein index per token.
        window: "protein" (use full-protein composition for each token)
                or "token" (one-hot of the residue itself – for strict control).

    Returns:
        (n_tokens, K)  where K = 20 canonical AAs (A,R,N,...).  Columns are
        normalized frequencies (protein window) or one-hot (token window).
    """
    aas = np.asarray(residue_aas, dtype=str)
    pids = np.asarray(protein_ids)
    canonical = np.array(list("ARNDCQEGHILKMFPSTWYV"))
    aa_to_idx = {aa: i for i, aa in enumerate(canonical)}
    n_tokens = len(aas)
    if window == "token":
        out = np.zeros((n_tokens, len(canonical)), dtype=np.float64)
        for i, aa in enumerate(aas):
            aa = aa.upper()
            if aa in aa_to_idx:
                out[i, aa_to_idx[aa]] = 1.0
        return out
    else:
        # protein-level composition broadcast to tokens
        out = np.zeros((n_tokens, len(canonical)), dtype=np.float64)
        for pid in np.unique(pids):
            mask = pids == pid
            prot_aas = aas[mask]
            counts = np.zeros(len(canonical), dtype=np.float64)
            for aa in prot_aas:
                aa = aa.upper()
                if aa in aa_to_idx:
                    counts[aa_to_idx[aa]] += 1
            freq = counts / max(1, len(prot_aas))
            out[mask] = freq
        return out


# ---------------------------------------------------------------------------
# Residualization (partial correlation)
# ---------------------------------------------------------------------------

def residualize(matrix: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """Residualize each column of `matrix` against covariates via OLS.

    Args:
        matrix: (n_tokens, n_features) to residualize.
        covariates: (n_tokens, n_cov).  Intercept is added internally.

    Returns:
        (n_tokens, n_features) residuals, same shape.
    """
    X = np.asarray(matrix, dtype=np.float64)  # (n, p)
    C = np.asarray(covariates, dtype=np.float64)  # (n, k)
    if X.shape[0] != C.shape[0]:
        raise ValueError(f"Row mismatch: X {X.shape[0]} vs C {C.shape[0]}")
    n = X.shape[0]
    # Design: [1, C]  (n, k+1)
    Design = np.concatenate([np.ones((n, 1), dtype=np.float64), C], axis=1)
    # Least squares: beta = (D^T D)^{-1} D^T X  -> (k+1, p)
    # Use lstsq for stability
    beta, *_ = np.linalg.lstsq(Design, X, rcond=None)
    X_pred = Design @ beta
    resid = X - X_pred
    return resid.astype(np.float32)


def partial_correlation_matrix(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    covariates: np.ndarray,
) -> np.ndarray:
    """Activation similarity after residualizing both sides against covariates.

    Returns:
        (n_a, n_b) Pearson correlation matrix of residuals – i.e. partial
        correlation controlling for covariates (up to linear effect).
    """
    from .similarity import compute_feature_similarity_matrix

    ra = residualize(feats_a, covariates)
    rb = residualize(feats_b, covariates)
    return compute_feature_similarity_matrix(ra, rb, method="correlation")


def length_stratified_sampling_controls(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    protein_lengths: np.ndarray,
    protein_ids: np.ndarray,
    n_bins: int = 4,
    seed: int = 0,
) -> Dict:
    """Assess length confounding via stratified sampling.

    Splits proteins into n_bins by length, computes within-bin vs cross-bin
    similarity contrast.  If high similarity is driven by length, within-bin
    (same-length proteins) will be inflated vs overall.

    Returns dict with bin stats and a simple length-confound score.
    """
    uniq_prots = np.unique(protein_ids)
    n_prots = len(uniq_prots)
    # Map protein id -> length (infer from token counts if needed)
    if protein_lengths is not None and len(protein_lengths) == n_prots:
        # Assume protein_lengths aligned to sorted uniq_prots
        sorted_pids = np.sort(uniq_prots)
        pid_to_len = dict(zip(sorted_pids, protein_lengths))
        # If not aligned, fallback to token counts
        if set(pid_to_len.keys()) != set(uniq_prots):
            _, counts = np.unique(protein_ids, return_counts=True)
            pid_to_len = dict(zip(np.unique(protein_ids), counts))
    else:
        _, counts = np.unique(protein_ids, return_counts=True)
        pid_to_len = dict(zip(np.unique(protein_ids), counts))

    sorted_pids = sorted(uniq_prots, key=lambda pid: pid_to_len.get(pid, 0))
    # Bin proteins by rank (equal count per bin)
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    for rank, pid in enumerate(sorted_pids):
        bins[int(rank * n_bins / max(1, n_prots))].append(int(pid))

    from .similarity import compute_feature_similarity_matrix

    overall = float(np.abs(compute_feature_similarity_matrix(feats_a, feats_b, method="correlation")).mean())
    bin_means: List[float] = []
    for bi, plist in enumerate(bins):
        if not plist:
            bin_means.append(0.0)
            continue
        mask = np.isin(protein_ids, plist)
        if mask.sum() < 10:
            bin_means.append(0.0)
            continue
        m = float(np.abs(compute_feature_similarity_matrix(feats_a[mask], feats_b[mask], method="correlation")).mean())
        bin_means.append(m)

    # Length-confound score: (max within-bin mean) / (overall mean + eps)
    score = float(max(bin_means) / (overall + 1e-12)) if overall > 1e-12 else 0.0
    return {
        "overall_mean_abs_corr": overall,
        "bin_mean_abs_corr": bin_means,
        "bin_protein_counts": [len(b) for b in bins],
        "length_confound_score": score,
        "note": "High score (>1) suggests length stratification inflates similarity.",
    }


# ---------------------------------------------------------------------------
# Permutation nulls
# ---------------------------------------------------------------------------

def permutation_null(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    n_permutations: int = 200,
    method: str = "correlation",
    seed: int = 0,
    per_token: bool = True,
) -> Dict:
    """Permutation null for activation similarity.

    Two modes:

    - per_token=True (default, stronger): for each permutation, independently
      shuffle the TOKEN order of feats_a (or feats_b) and recompute the full
      similarity matrix.  The observed max/mean is compared to the null
      distribution of that statistic.

    - per_token=False: shuffle protein assignments (block shuffle) to preserve
      within-protein autocorrelation structure – more conservative.

    Args:
        feats_a, feats_b: (n_tokens, n_features_*) – must have same n_tokens.
        n_permutations: number of shuffles.
        method: "correlation" or "cosine" for the similarity metric.
        seed: RNG seed.
        per_token: if True token-level shuffle, else protein-block shuffle
                   requires protein_ids (not yet – reserves API).

    Returns:
        Dict with null distribution summary and empirical p-values for
        observed mean and max absolute similarity.
    """
    from .similarity import compute_feature_similarity_matrix

    feats_a = np.asarray(feats_a)
    feats_b = np.asarray(feats_b)
    if feats_a.shape[0] != feats_b.shape[0]:
        raise ValueError("Token count must match for permutation null")
    n = feats_a.shape[0]
    rng = np.random.RandomState(seed)

    obs = compute_feature_similarity_matrix(feats_a, feats_b, method=method)
    obs_mean = float(np.abs(obs).mean())
    obs_max = float(np.abs(obs).max())

    null_means: List[float] = []
    null_maxs: List[float] = []
    for _ in range(n_permutations):
        perm = rng.permutation(n)
        fa_perm = feats_a[perm]
        cur = compute_feature_similarity_matrix(fa_perm, feats_b, method=method)
        null_means.append(float(np.abs(cur).mean()))
        null_maxs.append(float(np.abs(cur).max()))

    null_means = np.array(null_means, dtype=np.float64)
    null_maxs = np.array(null_maxs, dtype=np.float64)

    # Empirical one-sided p-value: P(null >= obs)
    p_mean = float((np.sum(null_means >= obs_mean) + 1) / (len(null_means) + 1))
    p_max = float((np.sum(null_maxs >= obs_max) + 1) / (len(null_maxs) + 1))

    return {
        "observed_mean_abs": obs_mean,
        "observed_max_abs": obs_max,
        "null_mean_abs": {
            "mean": float(null_means.mean()),
            "std": float(null_means.std()),
            "p_value": p_mean,
            "values": null_means.tolist(),
        },
        "null_max_abs": {
            "mean": float(null_maxs.mean()),
            "std": float(null_maxs.std()),
            "p_value": p_max,
            "values": null_maxs.tolist(),
        },
        "n_permutations": int(n_permutations),
        "method": method,
    }


def random_pairing_control(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    method: str = "correlation",
    n_pairs: int = 1000,
    seed: int = 0,
) -> Dict:
    """Control: compare observed similarity distribution to random feature pairs.

    We sample random (i,j) pairs uniformly and report their similarity quantiles
    vs the distribution of best-match similarities.  Helps show that top matches
    are outliers vs the random-pair background.

    Returns quantiles of full similarity matrix vs random-pair subset (which for
    uniform sampling is the same distribution – but the function makes it easy
    to swap in length-matched or family-matched random pairing later).
    """
    from .similarity import compute_feature_similarity_matrix

    S = compute_feature_similarity_matrix(feats_a, feats_b, method=method)
    full = S.ravel()
    rng = np.random.RandomState(seed)
    # sample n_pairs random pairs (uniform)
    idx_a = rng.randint(0, S.shape[0], size=n_pairs)
    idx_b = rng.randint(0, S.shape[1], size=n_pairs)
    sampled = S[idx_a, idx_b]
    # best-match per A feature
    best = np.max(S, axis=1) if S.size else np.array([])
    return {
        "full": {
            "mean": float(np.mean(full)) if full.size else 0.0,
            "std": float(np.std(full)) if full.size else 0.0,
            "q50": float(np.median(full)) if full.size else 0.0,
            "q90": float(np.quantile(full, 0.90)) if full.size else 0.0,
            "q95": float(np.quantile(full, 0.95)) if full.size else 0.0,
            "q99": float(np.quantile(full, 0.99)) if full.size else 0.0,
        },
        "sampled_random_pairs": {
            "mean": float(np.mean(sampled)) if sampled.size else 0.0,
            "std": float(np.std(sampled)) if sampled.size else 0.0,
        },
        "best_per_a": {
            "mean": float(np.mean(best)) if best.size else 0.0,
            "q50": float(np.median(best)) if best.size else 0.0,
            "q90": float(np.quantile(best, 0.90)) if best.size else 0.0,
            "max": float(np.max(best)) if best.size else 0.0,
        },
    }
