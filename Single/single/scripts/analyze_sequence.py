#!/usr/bin/env python
"""
Sequence-level analysis of SAE features (no 3D structures needed).

For each requested feature, computes:
  1. sequential Cohen's d  — are the feature's activated residues CLUSTERED along
     the sequence (local/motif-like) or SPREAD OUT (global/periodic)?
  2. motif enrichment      — which amino acids are over-represented in a window
     around the activated residues (the amino-acid "signature" of the feature).

Usage:
    python -m single.scripts.analyze_sequence \
        --sae_dir ../Outputs/mb/sae \
        --embeddings_dir ../Outputs/mb/embeddings/layer_6 \
        --sequences_csv ../Dataset/mBMRB.csv \
        --experiment mb \
        --feature_indices 375 42 234
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

from single.sae.inference import load_sae
from single.analysis.sequence import (
    build_residue_positions,
    sequential_cohens_d,
    motif_enrichment,
    summarize_motif,
)


def analyze(
    sae_dir: Path,
    embeddings_dir: Path,
    sequences_csv: Path,
    feature_indices: List[int],
    experiment: Optional[str] = None,
    exp_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    layer: int = 6,
    shard: int = 0,
    n_shards: int = 5,
    max_length: int = 512,
    flank: int = 3,
    activation_threshold: float = 0.0,
):
    from single.paths import resolve_experiment

    if output_dir is None:
        exp = resolve_experiment(exp_dir=exp_dir, name=experiment)
        output_dir = exp.analysis_dir
        print(f"Experiment dir: {exp.dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_residues = max_length - 2

    print("=" * 60)
    print("SAE SEQUENCE ANALYSIS (Cohen's d + motif enrichment)")
    print("=" * 60)

    print(f"\nLoading SAE from {sae_dir}...")
    sae = load_sae(sae_dir, device=device)
    print(f"  {sae.__class__.__name__}: {sae.dict_size} features, {sae.activation_dim}D")

    for f in feature_indices:
        if f >= sae.dict_size:
            raise ValueError(f"feature_idx {f} out of range [0, {sae.dict_size})")

    # Load embeddings for the requested shard
    emb_path = Path(embeddings_dir) / f"shard_{shard}" / "activations.pt"
    if not emb_path.exists():
        cands = sorted(Path(embeddings_dir).glob(f"shard_{shard}/**/activations.pt"))
        if not cands:
            raise FileNotFoundError(f"No embeddings found for shard {shard} in {embeddings_dir}")
        emb_path = cands[0]
    data = torch.load(emb_path, map_location="cpu", weights_only=True)
    if isinstance(data, dict):
        embeddings = data["embeddings"]
    else:
        embeddings = data
    embeddings = embeddings.to(device)
    print(f"\nEmbeddings shard_{shard}: {embeddings.shape}")

    # Rebuild token -> (protein, residue) mapping (same shuffle+shard as extraction)
    print(f"\nRebuilding protein/residue mapping from {sequences_csv}...")
    df, shard_proteins, shard_respos = build_residue_positions(
        sequences_csv, [shard], n_shards=n_shards, max_residues=max_residues,
    )
    protein_ids = shard_proteins[shard]
    respos = shard_respos[shard]
    proteins = df["sequence"].astype(str).tolist()
    print(f"  {len(protein_ids):,} tokens mapped to {len(proteins)} proteins")

    # Compute protein lengths (for the random null distribution in Cohen's d)
    protein_lengths = [min(len(s), max_residues) for s in proteins]

    results = {"layer": layer, "shard": shard, "features": {}}
    for fidx in feature_indices:
        print(f"\n--- Feature #{fidx} ---")

        # Active residues per protein
        active = torch.zeros(len(protein_ids), dtype=torch.bool)
        # We already compute activations inside motif_enrichment; to avoid
        # recomputing, gather them once here and pass to both analyses.
        from single.sae.inference import get_sae_feats_in_batches
        feats = get_sae_feats_in_batches(
            sae=sae, aa_embds=embeddings, chunk_size=4096,
            feat_list=[fidx], normalize_features=True, device=device,
        )
        active = (feats[:, 0] > activation_threshold).cpu().numpy()

        # Group active tokens by protein -> residue positions
        active_positions: dict = {}
        active_np = active
        for i in range(len(active_np)):
            if active_np[i]:
                prot_idx = protein_ids[i]
                active_positions.setdefault(prot_idx, []).append(int(respos[i]))
        active_positions = {k: np.array(v, dtype=np.int64) for k, v in active_positions.items()}
        n_active = sum(len(v) for v in active_positions.values())

        # 1) Sequential Cohen's d
        d = sequential_cohens_d(active_positions, protein_lengths)
        cluster_label = "clustered (local)" if d < -0.1 else ("dispersed" if d > 0.1 else "~random")
        print(f"  Sequential Cohen's d: {d:+.3f}  → {cluster_label}")

        # 2) Motif enrichment (reuse the same activation mask)
        active_t = torch.from_numpy(active_np)
        motif = motif_enrichment(
            sae, embeddings, proteins, protein_ids, respos, fidx,
            flank=flank, activation_threshold=activation_threshold, device=device,
        )
        top = summarize_motif(motif, top_n=5)
        print(f"  Active residues: {motif['n_active_residues']}")
        print("  Top enriched amino acids (log2 fold):")
        for aa, v in top:
            print(f"    {aa}: {v:+.2f}")

        results["features"][str(fidx)] = {
            "sequential_cohens_d": round(d, 4),
            "cluster_label": cluster_label,
            "n_active_residues": int(n_active),
            "motif_enrichment": {aa: round(v, 4) for aa, v in motif["amino_acid_enrichment"].items()},
            "flank": flank,
        }

    # Save
    out_path = output_dir / f"sequence_analysis_shard{shard}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sequence-level SAE feature analysis")
    parser.add_argument("--sae_dir", type=Path, required=True)
    parser.add_argument("--embeddings_dir", type=Path, required=True)
    parser.add_argument("--sequences_csv", type=Path, required=True)
    parser.add_argument("--feature_indices", type=int, nargs="+", required=True)
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--exp_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--flank", type=int, default=3,
                        help="Window radius around each activated residue for motif enrichment")
    parser.add_argument("--activation_threshold", type=float, default=0.0)
    args = parser.parse_args()
    analyze(**vars(args))
