#!/usr/bin/env python3
"""Compute cross-model feature similarity (CKA, correlation, MI).

Usage:
  python crossplm.py crossing compute_feature_similarity \
      --sae_a Outputs/exp_a/sae/model.pt \
      --sae_b Outputs/exp_b/sae/model.pt \
      --embeddings_a Outputs/exp_a/mbmrb/embeddings/layer_6 \
      --embeddings_b Outputs/exp_b/mbmrb/embeddings/layer_6 \
      --experiment my_crossing
"""
import argparse
import json
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_embeddings(embed_dir):
    """Load all shard embeddings from a directory."""
    all_feats = []
    shard_dirs = sorted(
        [d for d in os.listdir(embed_dir) if d.startswith("shard_")],
        key=lambda x: int(x.split("_")[1]),
    )
    for shard_dir in shard_dirs:
        emb_path = os.path.join(embed_dir, shard_dir, "embeddings.pt")
        if os.path.exists(emb_path):
            emb = torch.load(emb_path, map_location="cpu", weights_only=True)
            if isinstance(emb, dict):
                emb = emb.get("hidden_states", emb.get("embeddings", list(emb.values())[0]))
            if isinstance(emb, list):
                emb = emb[0] if len(emb) == 1 else torch.cat(emb, dim=0)
            all_feats.append(emb.numpy() if hasattr(emb, "numpy") else np.array(emb))
    if not all_feats:
        raise FileNotFoundError(f"No embeddings found in {embed_dir}")
    return np.concatenate(all_feats, axis=0)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute cross-model feature similarity"
    )
    parser.add_argument(
        "--sae_a", required=True,
        help="Path to SAE model for task A (e.g. Outputs/exp_a/sae/model.pt)"
    )
    parser.add_argument(
        "--sae_b", required=True,
        help="Path to SAE model for task B"
    )
    parser.add_argument(
        "--embeddings_a", required=True,
        help="Directory with embeddings from task A model"
    )
    parser.add_argument(
        "--embeddings_b", required=True,
        help="Directory with embeddings from task B model"
    )
    parser.add_argument(
        "--experiment", required=True,
        help="Experiment name (output saved to Outputs/<experiment>/crossing/)"
    )
    parser.add_argument(
        "--source_a", default=None,
        help="Source ID for model A (e.g. mbmrb)"
    )
    parser.add_argument(
        "--source_b", default=None,
        help="Source ID for model B (e.g. swissprot)"
    )
    parser.add_argument(
        "--method", default="correlation",
        choices=["correlation", "cosine"],
        help="Similarity method for feature matrix (default: correlation)"
    )
    parser.add_argument(
        "--use_cka", action="store_true",
        help="Also compute global CKA score"
    )
    parser.add_argument(
        "--use_mi", action="store_true",
        help="Also compute mutual information matrix"
    )
    parser.add_argument(
        "--mi_max_features", type=int, default=200,
        help="Max features per model for MI computation (default: 200)"
    )
    parser.add_argument(
        "--normalize", action="store_true",
        help="Normalize SAE features before computing similarity"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=None,
        help="Max tokens to use (for speed; default: all)"
    )

    args = parser.parse_args(argv)

    # Load SAEs
    print("[Loading SAEs]")
    from single.sae.inference import load_sae

    sae_a = load_sae(os.path.dirname(args.sae_a), device="cpu")
    sae_b = load_sae(os.path.dirname(args.sae_b), device="cpu")
    print(f"  SAE A: dict_size={sae_a.dict_size}, activation_dim={sae_a.activation_dim}")
    print(f"  SAE B: dict_size={sae_b.dict_size}, activation_dim={sae_b.activation_dim}")

    # Load embeddings
    print("[Loading embeddings]")
    emb_a = _load_embeddings(args.embeddings_a)
    emb_b = _load_embeddings(args.embeddings_b)
    print(f"  Embeddings A: {emb_a.shape}")
    print(f"  Embeddings B: {emb_b.shape}")

    if args.max_tokens:
        emb_a = emb_a[:args.max_tokens]
        emb_b = emb_b[:args.max_tokens]
        print(f"  Subsampled to {args.max_tokens} tokens")

    # Compute SAE features
    print("[Computing SAE features]")
    with torch.no_grad():
        feats_a = sae_a.encode(torch.tensor(emb_a, dtype=torch.float32)).numpy()
        feats_b = sae_b.encode(torch.tensor(emb_b, dtype=torch.float32)).numpy()

    if args.normalize:
        max_a = np.abs(feats_a).max(axis=0, keepdims=True) + 1e-8
        max_b = np.abs(feats_b).max(axis=0, keepdims=True) + 1e-8
        feats_a = feats_a / max_a
        feats_b = feats_b / max_b
        print("  Features normalized")

    print(f"  Features A: {feats_a.shape}")
    print(f"  Features B: {feats_b.shape}")

    # Compute similarity
    print(f"[Computing feature similarity ({args.method})]")
    from crossing.similarity import compute_feature_similarity_matrix

    sim_matrix = compute_feature_similarity_matrix(feats_a, feats_b, method=args.method)
    print(f"  Similarity matrix: {sim_matrix.shape}")
    print(f"  Mean: {sim_matrix.mean():.4f}, Max: {sim_matrix.max():.4f}")

    # Build results
    results = {
        "method": args.method,
        "sae_a": args.sae_a,
        "sae_b": args.sae_b,
        "embeddings_a": args.embeddings_a,
        "embeddings_b": args.embeddings_b,
        "n_tokens_a": len(emb_a),
        "n_tokens_b": len(emb_b),
        "n_features_a": int(feats_a.shape[1]),
        "n_features_b": int(feats_b.shape[1]),
        "mean_similarity": float(sim_matrix.mean()),
        "max_similarity": float(sim_matrix.max()),
    }

    # CKA
    if args.use_cka:
        from crossing.similarity import linear_cka
        print("[Computing CKA]")
        cka_score = linear_cka(emb_a, emb_b)
        results["cka_score"] = cka_score
        print(f"  CKA: {cka_score:.4f}")

    # MI
    if args.use_mi:
        from crossing.similarity import compute_mi_matrix
        print(f"[Computing MI matrix (max {args.mi_max_features} features)]")
        mi_matrix = compute_mi_matrix(
            feats_a, feats_b, n_bins=20, max_features=args.mi_max_features
        )
        results["mi_mean"] = float(mi_matrix.mean())
        results["mi_max"] = float(mi_matrix.max())
        print(f"  MI matrix: {mi_matrix.shape}")
        print(f"  MI mean: {mi_matrix.mean():.4f}, Max: {mi_matrix.max():.4f}")

    # Save outputs
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "..", "Outputs", args.experiment, "crossing")
    os.makedirs(base_dir, exist_ok=True)

    # Save similarity matrix
    np.save(os.path.join(base_dir, "feature_similarity_matrix.npy"), sim_matrix)

    # Save top matches
    top_matches = []
    n_a, n_b = sim_matrix.shape
    flat_indices = np.argsort(sim_matrix.ravel())[::-1]
    seen_a, seen_b = set(), set()
    for flat_idx in flat_indices:
        i, j = divmod(int(flat_idx), n_b)
        if i in seen_a or j in seen_b:
            continue
        top_matches.append({
            "feature_a": int(i),
            "feature_b": int(j),
            "similarity": round(float(sim_matrix[i, j]), 4),
        })
        seen_a.add(i)
        seen_b.add(j)
        if len(top_matches) >= 50:
            break
    results["top_matches"] = top_matches

    # Save JSON
    with open(os.path.join(base_dir, "feature_similarity.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Done] Results saved to {base_dir}/")
    print(f"  feature_similarity_matrix.npy")
    print(f"  feature_similarity.json")
    print(f"\nTop 5 matches:")
    for m in top_matches[:5]:
        print(f"  A[{m['feature_a']}] <-> B[{m['feature_b']}] sim={m['similarity']:.4f}")


if __name__ == "__main__":
    main()
