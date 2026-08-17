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

import os
import sys

# Allow running directly from the repository root, e.g.
#   python Single/single/scripts/analyze_sequence.py ...
# without `cd Single` or installing the package (the `single` package lives at
# Single/single/, two levels up from this file).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import json
import re
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
    motif_enrichment_from_counts,
    positional_motif_counts,
    summarize_positional_motif,
    draw_sequence_logo,
    summarize_motif,
)
from single.data import load_residue_mapping_from_metadata


def analyze(
    embeddings_dir: Path,
    sequences_csv: Path,
    feature_indices: List[int],
    experiment: Optional[str] = None,
    exp_dir: Optional[Path] = None,
    source: Optional[str] = None,
    output_dir: Optional[Path] = None,
    sequence_column: str = "sequence",
    label_map: Optional[str] = None,
    layer: int = 6,
    shard: Optional[int] = None,
    n_shards: int = 5,
    max_length: int = 512,
    flank: int = 5,
    activation_threshold: float = 0.0,
    min_seq_len: int = 0,
    max_seq_len: int = 10_000,
    max_sequences: Optional[int] = None,
    sae_dir: Optional[Path] = None,
    motif_permutations: int = 200,
    motif_seed: int = 0,
):
    from single.paths import resolve_experiment

    # If a label map is given, use its sequence_column unless explicitly set.
    if label_map:
        from single.label_maps import get_label_map, resolve_columns
        sequence_column, _ = resolve_columns(
            get_label_map(label_map), sequence_column, None
        )

    # --sae_dir and --output_dir default into Outputs/<experiment>/.
    exp = None
    if sae_dir is None or output_dir is None:
        exp = resolve_experiment(exp_dir=exp_dir, name=experiment, source=source)
    if sae_dir is None:
        sae_dir = exp.sae_dir
        print(f"  SAE dir (inferred): {sae_dir}")
    if output_dir is None:
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
        if f < 0 or f >= sae.dict_size:
            raise ValueError(f"feature_idx {f} out of range [0, {sae.dict_size})")

    if shard is None:
        shard_ids = []
        for path in Path(embeddings_dir).glob("shard_*"):
            match = re.fullmatch(r"shard_(\d+)", path.name)
            if match:
                shard_ids.append(int(match.group(1)))
        shard_ids = sorted(set(shard_ids))
        if not shard_ids:
            raise ValueError(f"No numeric shard directories found in {embeddings_dir}")
        if shard_ids != list(range(shard_ids[-1] + 1)):
            raise ValueError(f"Shard IDs are not contiguous: {shard_ids}")
    else:
        if shard < 0:
            raise ValueError("shard must be non-negative")
        shard_ids = [shard]

    # Rebuild token -> (protein, residue) mappings for all selected shards.
    print(f"\nRebuilding protein/residue mapping from {sequences_csv}...")
    shards, shard_proteins, shard_respos = build_residue_positions(
        sequences_csv, shard_ids, n_shards=n_shards, max_residues=max_residues,
        sequence_column=sequence_column,
        min_seq_len=min_seq_len, max_seq_len=max_seq_len,
        max_sequences=max_sequences,
    )
    from single.sae.inference import get_sae_feats_in_batches
    active_positions_by_feature = {f: {} for f in feature_indices}
    window_counts_by_feature = {f: {} for f in feature_indices}
    background_counts_by_feature = {f: {} for f in feature_indices}
    positional_observed_by_feature = {
        f: None for f in feature_indices
    }
    positional_null_by_feature = {
        f: None for f in feature_indices
    }
    positional_active_by_feature = {f: 0 for f in feature_indices}
    protein_lengths = []
    protein_offset = 0

    for sid in shard_ids:
        emb_path = Path(embeddings_dir) / f"shard_{sid}" / "embeddings.pt"
        if not emb_path.exists():
            cands = sorted(Path(embeddings_dir).glob(f"shard_{sid}/**/embeddings.pt"))
            if not cands:
                raise FileNotFoundError(f"No embeddings found for shard {sid} in {embeddings_dir}")
            emb_path = cands[0]
        data = torch.load(emb_path, map_location="cpu", weights_only=True)
        embeddings = data["embeddings"] if isinstance(data, dict) else data
        embeddings = embeddings.to(device)
        metadata_path = Path(embeddings_dir) / f"shard_{sid}" / "residues.csv"
        if metadata_path.exists():
            mapped_ids, mapped_positions = load_residue_mapping_from_metadata(
                metadata_path, shards[sid], sequence_column
            )
            protein_ids = np.asarray(mapped_ids, dtype=np.int64)
            respos = mapped_positions
        else:
            protein_ids = np.asarray(shard_proteins[sid], dtype=np.int64)
            respos = shard_respos[sid]
        proteins = shards[sid][sequence_column].astype(str).tolist()
        if len(protein_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Embedding/mapping token count mismatch for shard {sid}: "
                f"embeddings={embeddings.shape[0]}, mapping={len(protein_ids)}"
            )
        print(f"\nEmbeddings shard_{sid}: {embeddings.shape}")
        protein_lengths.extend([min(len(s), max_residues) for s in proteins])

        feats = get_sae_feats_in_batches(
            sae=sae, aa_embds=embeddings, chunk_size=4096,
            feat_list=feature_indices, normalize_features=True, device=device,
        ).cpu().numpy()
        for col, fidx in enumerate(feature_indices):
            active = feats[:, col] > activation_threshold
            local_positions = {}
            for i in np.flatnonzero(active):
                global_pid = int(protein_ids[i]) + protein_offset
                local_positions.setdefault(global_pid, []).append(int(respos[i]))
            for pid, positions in local_positions.items():
                active_positions_by_feature[fidx][pid] = np.asarray(
                    positions, dtype=np.int64
                )
            motif = motif_enrichment(
                sae, embeddings, proteins, protein_ids, respos, fidx,
                flank=flank, activation_threshold=activation_threshold,
                device=device, active_mask=active,
            )
            for aa, count in motif.get("window_counts", {}).items():
                window_counts_by_feature[fidx][aa] = (
                    window_counts_by_feature[fidx].get(aa, 0) + count
                )
            for aa, count in motif.get("background_counts", {}).items():
                background_counts_by_feature[fidx][aa] = (
                    background_counts_by_feature[fidx].get(aa, 0) + count
                )
            positional = positional_motif_counts(
                proteins=proteins,
                protein_ids=protein_ids,
                respos=respos,
                active_mask=active,
                flank=flank,
                n_permutations=motif_permutations,
                seed=motif_seed + sid,
            )
            if positional_observed_by_feature[fidx] is None:
                positional_observed_by_feature[fidx] = positional["observed_counts"]
                positional_null_by_feature[fidx] = positional["null_counts"]
            else:
                positional_observed_by_feature[fidx] += positional["observed_counts"]
                positional_null_by_feature[fidx] += positional["null_counts"]
            positional_active_by_feature[fidx] += int(positional["n_active_centers"])
        protein_offset += len(proteins)

    results = {"layer": layer, "shard": shard, "shards": shard_ids, "features": {}}
    for fidx in feature_indices:
        active_positions = active_positions_by_feature[fidx]
        d = sequential_cohens_d(active_positions, protein_lengths)
        cluster_label = "clustered (local)" if d < -0.1 else ("dispersed" if d > 0.1 else "~random")
        motif = motif_enrichment_from_counts(
            window_counts_by_feature[fidx],
            background_counts_by_feature[fidx],
            flank=flank,
            n_active=sum(len(v) for v in active_positions.values()),
        )
        top = summarize_motif(motif, top_n=5)
        positional_motif = summarize_positional_motif(
            positional_observed_by_feature[fidx],
            positional_null_by_feature[fidx],
            flank=flank,
            n_active_centers=positional_active_by_feature[fidx],
        )
        logo_path = output_dir / f"sequence_logo_feature_{fidx}.png"
        draw_sequence_logo(
            np.asarray(positional_observed_by_feature[fidx]),
            flank=flank,
            output_path=logo_path,
            title=f"Feature #{fidx} positional motif",
        )
        print(f"\n--- Feature #{fidx} ---")
        print(f"  Sequential Cohen's d: {d:+.3f}  → {cluster_label}")
        print(f"  Active residues: {motif['n_active_residues']}")
        print("  Top enriched amino acids (log2 fold):")
        for aa, value in top:
            print(f"    {aa}: {value:+.2f}")
        results["features"][str(fidx)] = {
            "sequential_cohens_d": round(d, 4),
            "cluster_label": cluster_label,
            "n_active_residues": int(motif["n_active_residues"]),
            "motif_enrichment": {
                aa: round(value, 4)
                for aa, value in motif["amino_acid_enrichment"].items()
            },
            "flank": flank,
            "positional_motif": positional_motif,
            "sequence_logo": str(logo_path),
        }

    # Save
    out_name = "sequence_analysis.json" if shard is None else f"sequence_analysis_shard{shard}.json"
    out_path = output_dir / out_name
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sequence-level SAE feature analysis")
    parser.add_argument("--sae_dir", type=Path, default=None,
                        help="Trained SAE dir (default: Outputs/<experiment>/sae)")
    parser.add_argument("--embeddings_dir", type=Path, required=True)
    parser.add_argument("--sequences_csv", type=Path, required=True)
    parser.add_argument("--sequence_column", type=str, default="sequence",
                        help="Column holding the protein sequence")
    parser.add_argument("--label_map", type=str, default=None,
                        help="Label-map preset/YAML (uses its sequence_column)")
    parser.add_argument("--feature_indices", type=int, nargs="+", required=True)
    parser.add_argument("--source", type=str, default=None,
                        help="Data-source id; nests outputs under Outputs/<experiment>/<source> (default: flat)")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--exp_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--shard", type=int, default=None,
                        help="Analyze one shard for a quick test; default aggregates all shards")
    parser.add_argument("--n_shards", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--flank", type=int, default=5,
                        help="Window radius around each activated residue for motif enrichment")
    parser.add_argument("--motif_permutations", type=int, default=200,
                        help="Within-protein permutations for positional motif null")
    parser.add_argument("--motif_seed", type=int, default=0)
    parser.add_argument("--activation_threshold", type=float, default=0.0)
    parser.add_argument("--min_seq_len", type=int, default=0,
                        help="Must match extract_embeddings --min_seq_len")
    parser.add_argument("--max_sequences", type=int, default=None,
                        help="Deterministic subset; must match extract_embeddings --max_sequences")
    parser.add_argument("--max_seq_len", type=int, default=10000,
                        help="Must match extract_embeddings --max_seq_len")
    args = parser.parse_args(argv)
    analyze(**vars(args))


if __name__ == "__main__":
    main()
