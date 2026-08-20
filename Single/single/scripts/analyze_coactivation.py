#!/usr/bin/env python
"""
Pairwise feature co-activation / co-localization analysis.

For two SAE features, measures whether they activate on the SAME residues or on
residues NEAR each other along the sequence (vs the baseline activation rate).

Usage:
    python -m single.scripts.analyze_coactivation \
        --sae_dir ../Outputs/mb/sae \
        --embeddings_dir ../Outputs/mb/embeddings/layer_6 \
        --sequences_csv ../Dataset/mBMRB.csv \
        --experiment mb \
        --feature_a 375 --feature_b 42
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
from typing import Optional

import numpy as np
import torch

from single.sae.inference import load_sae
from single.analysis.sequence import build_residue_positions
from single.analysis.co_activation import (
    aggregate_coactivation_results,
    compute_coactivation,
    interpret,
)
from single.data import load_residue_mapping_from_metadata


def analyze(
    embeddings_dir: Path,
    sequences_csv: Path,
    feature_a: int,
    feature_b: int,
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
    neighborhood: int = 5,
    activation_threshold: float = 0.0,
    min_seq_len: int = 0,
    max_seq_len: int = 10_000,
    max_sequences: Optional[int] = None,
    sae_dir: Optional[Path] = None,
):
    from single.paths import resolve_experiment

    # If a label map is given, use its sequence_column/label_column unless explicitly set.
    label_column_for_drop = None
    if label_map:
        from single.label_maps import get_label_map, resolve_columns
        sequence_column, label_column_resolved = resolve_columns(
            get_label_map(label_map), sequence_column, None
        )
        label_column_for_drop = label_column_resolved
    elif "label" in str(sequences_csv):
        label_column_for_drop = "label"

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
    print("SAE PAIRWISE CO-ACTIVATION ANALYSIS")
    print("=" * 60)

    print(f"\nLoading SAE from {sae_dir}...")
    sae = load_sae(sae_dir, device=device)
    print(f"  {sae.__class__.__name__}: {sae.dict_size} features, {sae.activation_dim}D")
    for f in (feature_a, feature_b):
        if f < 0 or f >= sae.dict_size:
            raise ValueError(f"feature index {f} out of range [0, {sae.dict_size})")

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
    # Must mirror extract_embeddings's drop_mismatched_lengths so shard composition matches residues.csv
    print(f"\nRebuilding protein/residue mapping from {sequences_csv}...")
    shards, shard_proteins, shard_respos = build_residue_positions(
        sequences_csv, shard_ids, n_shards=n_shards, max_residues=max_residues,
        sequence_column=sequence_column,
        label_column=label_column_for_drop,
        min_seq_len=min_seq_len, max_seq_len=max_seq_len,
        max_sequences=max_sequences,
    )
    shard_results = []
    for sid in shard_ids:
        emb_path = Path(embeddings_dir) / f"shard_{sid}" / "embeddings.pt"
        if not emb_path.exists():
            cands = sorted(Path(embeddings_dir).glob(f"shard_{sid}/**/embeddings.pt"))
            if not cands:
                raise FileNotFoundError(f"No embeddings for shard {sid} in {embeddings_dir}")
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
            protein_ids = np.array(shard_proteins[sid], dtype=np.int64)
            respos = shard_respos[sid]
        if len(protein_ids) != embeddings.shape[0]:
            raise ValueError(
                f"Embedding/mapping token count mismatch for shard {sid}: "
                f"embeddings={embeddings.shape[0]}, mapping={len(protein_ids)}"
            )
        print(f"\nComputing shard_{sid} co-activation: {embeddings.shape}")
        shard_results.append(compute_coactivation(
            sae=sae, embeddings=embeddings,
            protein_ids=protein_ids, respos=respos,
            feature_a=feature_a, feature_b=feature_b,
            neighborhood=neighborhood,
            activation_threshold=activation_threshold,
            device=device,
        ))
    result = aggregate_coactivation_results(shard_results)
    result["shard"] = shard
    result["shards"] = shard_ids

    # Report
    print("\n" + "=" * 60)
    print("CO-ACTIVATION RESULTS")
    print("=" * 60)
    print(f"  Tokens analyzed:       {result['n_tokens']:,}")
    print(f"  Feature #{feature_a} active on: {result['n_a_active']:,} "
          f"({result['baseline_a']*100:.2f}%)")
    print(f"  Feature #{feature_b} active on: {result['n_b_active']:,} "
          f"({result['baseline_b']*100:.2f}%)")
    print(f"  Same-residue co-activation:")
    print(f"    P(B | A) = {result['overlap_ab']*100:.2f}%   (baseline {result['baseline_b']*100:.2f}%)"
          f"  enrich={result['enrich_ab']:.2f}x")
    print(f"    P(A | B) = {result['overlap_ba']*100:.2f}%   (baseline {result['baseline_a']*100:.2f}%)"
          f"  enrich={result['enrich_ba']:.2f}x")
    print(f"  Neighborhood (±{neighborhood} residues):")
    print(f"    P(B nearby, excluding same residue | A) = {result['neighbor_ab']*100:.2f}%"
          f"  (independence null {result['neighbor_ab_null']*100:.2f}%)"
          f"  enrich={result['neighbor_enrich_ab']:.2f}x")
    print(f"    P(A nearby, excluding same residue | B) = {result['neighbor_ba']*100:.2f}%"
          f"  (independence null {result['neighbor_ba_null']*100:.2f}%)"
          f"  enrich={result['neighbor_enrich_ba']:.2f}x")
    print("=" * 60)
    print("Interpretation:")
    print(interpret(result))
    print("=" * 60)

    out_name = (
        f"coactivation_{feature_a}_{feature_b}.json"
        if shard is None else f"coactivation_{feature_a}_{feature_b}_shard{shard}.json"
    )
    out_path = output_dir / out_name
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pairwise SAE feature co-activation")
    parser.add_argument("--sae_dir", type=Path, default=None,
                        help="Trained SAE dir (default: Outputs/<experiment>/sae)")
    parser.add_argument("--embeddings_dir", type=Path, required=True)
    parser.add_argument("--sequences_csv", type=Path, required=True)
    parser.add_argument("--sequence_column", type=str, default="sequence",
                        help="Column holding the protein sequence")
    parser.add_argument("--label_map", type=str, default=None,
                        help="Label-map preset/YAML (uses its sequence_column)")
    parser.add_argument("--feature_a", type=int, required=True)
    parser.add_argument("--feature_b", type=int, required=True)
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
    parser.add_argument("--neighborhood", type=int, default=5,
                        help="Radius ±k for the neighborhood co-activation")
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
