#!/usr/bin/env python
"""
Align SAE features against Swiss-Prot / UniProtKB biological concepts.

Two steps:

1. Build concept matrices from a UniProtKB TSV export:
     python -m single.scripts.analyze_concepts build \
         --annotations_tsv ../Dataset/uniprotkb_swissprot.tsv.gz \
         --concepts_dir ../Outputs/concepts

2. Align a trained SAE's features to those concepts:
     python -m single.scripts.analyze_concepts align \
         --sae_dir ../Outputs/sae/esm2_8m_l6_d640_<ts> \
         --embeddings_dir ../Outputs/embeddings/esm2_8m/layer_6 \
         --concepts_dir ../Outputs/concepts \
         --output_dir ../Outputs/analysis/concepts

For each feature × concept pair, computes F1 / precision / recall / AUROC
across activation thresholds and saves a summary CSV.
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import pandas as pd

from single.sae.inference import load_sae
from single.analysis.feature_alignment import align_features_to_concepts


def cmd_build(
    annotations_tsv: Path,
    concepts_dir: Optional[Path] = None,
    experiment: Optional[str] = None,
    exp_dir: Optional[Path] = None,
    n_shards: int = 5,
    min_seq_len: int = 30,
    max_seq_len: int = 1022,
):
    from single.analysis.concepts import build_concept_matrix
    from single.paths import resolve_experiment

    if concepts_dir is None:
        exp = resolve_experiment(exp_dir=exp_dir, name=experiment)
        concepts_dir = exp.concepts_dir
        print(f"Experiment dir: {exp.dir}")
    build_concept_matrix(
        annotations_tsv=annotations_tsv,
        output_dir=concepts_dir,
        n_shards=n_shards,
        min_seq_len=min_seq_len,
        max_seq_len=max_seq_len,
    )


def cmd_align(
    sae_dir: Path,
    embeddings_dir: Path,
    concepts_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    experiment: Optional[str] = None,
    exp_dir: Optional[Path] = None,
    shard: Optional[int] = None,
    threshold_min_f1: float = 0.0,
    n_top_per_concept: int = 20,
    feature_chunk_size: int = 200,
    batch_size: int = 1024,
    compute_auroc: bool = True,
    compute_domain_f1: bool = True,
    min_positives: int = 10,
    threshold_percents: Optional[List[float]] = None,
):
    from scipy import sparse
    from single.analysis.concepts import load_concept_names, load_concept_shards
    from single.paths import resolve_experiment

    if concepts_dir is None or output_dir is None:
        exp = resolve_experiment(exp_dir=exp_dir, name=experiment)
        print(f"Experiment dir: {exp.dir}")
        if concepts_dir is None:
            concepts_dir = exp.concepts_dir
        if output_dir is None:
            output_dir = exp.analysis_dir

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("SAE Feature × Concept Alignment")
    print("=" * 60)

    # Load SAE
    print("\nLoading SAE...")
    sae = load_sae(sae_dir, device=device)
    print(f"  {sae.__class__.__name__}: {sae.dict_size} features, {sae.activation_dim}D")

    # Load concept names
    concept_names = load_concept_names(concepts_dir)
    if not concept_names:
        raise ValueError(f"No concept columns found in {concepts_dir}/aa_concepts_columns.txt")

    # Load embeddings
    print("\nLoading embeddings...")
    all_pairs = []
    shards_to_process = [shard] if shard is not None else list(range(len(list(Path(concepts_dir).glob("shard_*")))))

    for shard_id in shards_to_process:
        print(f"  Shard {shard_id}...")
        concept_matrix, metadata = load_concept_shards(concepts_dir, shard_id)

        # Load THIS shard's own embeddings from its shard directory, so token
        # counts align exactly with the concept matrix (same sharding logic).
        shard_emb_path = Path(embeddings_dir) / f"shard_{shard_id}" / "activations.pt"
        if not shard_emb_path.exists():
            # Fallback: look for per-shard activations.pt anywhere in embeddings_dir
            cands = sorted(Path(embeddings_dir).glob(f"shard_{shard_id}/**/activations.pt"))
            if not cands:
                raise FileNotFoundError(
                    f"No per-shard embeddings found for shard {shard_id} in {embeddings_dir}. "
                    f"Expected {shard_emb_path}. Re-extract embeddings with the same "
                    f"--n_shards used for concept building."
                )
            shard_emb_path = cands[0]
        shard_emb = torch.load(shard_emb_path, map_location="cpu", weights_only=True)
        if isinstance(shard_emb, dict):
            shard_emb = shard_emb.get("embeddings", shard_emb)
        shard_embeddings = shard_emb.to(device)
        print(f"  Embeddings shard_{shard_id}: {shard_embeddings.shape}")

        if concept_matrix.shape[0] != shard_embeddings.shape[0]:
            print(f"  ⚠️  Concept matrix ({concept_matrix.shape[0]}) ≠ embeddings "
                  f"({shard_embeddings.shape[0]}). Using min length.")
            n = min(concept_matrix.shape[0], shard_embeddings.shape[0])
            concept_matrix = concept_matrix[:n]
            shard_embeddings = shard_embeddings[:n]

        # Align
        metrics = align_features_to_concepts(
            sae=sae,
            embeddings=shard_embeddings,
            concept_matrix=concept_matrix,
            concept_names=concept_names,
            feature_chunk_size=feature_chunk_size,
            batch_size=batch_size,
            compute_auroc=compute_auroc,
            compute_domain_f1=compute_domain_f1,
            min_positives=min_positives,
            threshold_percents=threshold_percents,
        )

        # Collect ALL pairs (f1 > 0); threshold_min_f1 only filters console output.
        for fidx, concept_dict in metrics.items():
            for concept, entry in concept_dict.items():
                row = {
                    "feature": fidx,
                    "concept": concept,
                    "f1": entry["f1"],
                    "precision": entry["precision"],
                    "recall": entry["recall"],
                    "auroc": entry["auroc"],
                    "threshold": entry["threshold"],
                    "shard": shard_id,
                }
                if compute_domain_f1 and "f1_per_domain" in entry:
                    row["f1_per_domain"] = entry["f1_per_domain"]
                    row["recall_per_domain"] = entry["recall_per_domain"]
                    row["n_domains"] = entry["n_domains"]
                all_pairs.append(row)

    # Save full metrics
    output_path = output_dir / "feature_concept_metrics.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved full metrics to {output_path}")

    # Save summary CSV (all pairs, no threshold filtering)
    df = pd.DataFrame(all_pairs)
    if not df.empty:
        csv_path = output_dir / "feature_concept_pairs.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved ALL {len(df)} feature-concept pairs to {csv_path}")

        # Print top pairs (filtered by threshold_min_f1 for readability)
        has_domain = "f1_per_domain" in df.columns
        top_df = df[df["f1"] >= threshold_min_f1] if threshold_min_f1 > 0 else df
        print(f"\nTop feature-concept associations (F1 >= {threshold_min_f1}, showing {len(top_df)}):")
        for _, row in top_df.nlargest(30, "f1").iterrows():
            extra = f"  F1dom={row['f1_per_domain']:.3f}" if has_domain else ""
            print(f"  Feature #{row['feature']:<4d} → {row['concept']:<30s} "
                  f"F1={row['f1']:.3f} AUROC={row['auroc']:.3f} "
                  f"P={row['precision']:.3f} R={row['recall']:.3f}{extra}")
    else:
        print("\nNo feature-concept pairs found.")

    # Per-concept best features
    print(f"\nTop {n_top_per_concept} features for each concept:")
    for concept in concept_names:
        concept_df = df[df["concept"] == concept]
        if not concept_df.empty:
            top = concept_df.nlargest(n_top_per_concept, "f1")
            best = top.iloc[0]
            print(f"  {concept}: Feature #{best['feature']} (F1={best['f1']:.3f})")

    print(f"\n{'=' * 60}")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(description="SAE feature × Swiss-Prot concept alignment")
    sub = parser.add_subparsers(dest="command", required=True)

    # build
    p_build = sub.add_parser("build", help="Build concept matrices from UniProt TSV")
    p_build.add_argument("--annotations_tsv", type=Path, required=True,
                         help="UniProtKB/Swiss-Prot TSV export")
    p_build.add_argument("--experiment", type=str, default=None,
                         help="Experiment name; creates Outputs/<experiment>_<ts>/")
    p_build.add_argument("--exp_dir", type=Path, default=None,
                         help="Reuse an existing experiment directory")
    p_build.add_argument("--concepts_dir", type=Path, default=None,
                         help="Explicit concepts dir (overrides experiment routing)")
    p_build.add_argument("--n_shards", type=int, default=5)
    p_build.add_argument("--min_seq_len", type=int, default=30)
    p_build.add_argument("--max_seq_len", type=int, default=1022)

    # align
    p_align = sub.add_parser("align", help="Align SAE features to concepts")
    p_align.add_argument("--sae_dir", type=Path, required=True)
    p_align.add_argument("--embeddings_dir", type=Path, required=True)
    p_align.add_argument("--experiment", type=str, default=None,
                         help="Experiment name; creates Outputs/<experiment>_<ts>/")
    p_align.add_argument("--exp_dir", type=Path, default=None,
                         help="Reuse an existing experiment directory")
    p_align.add_argument("--concepts_dir", type=Path, default=None,
                         help="Explicit concepts dir (overrides experiment routing)")
    p_align.add_argument("--output_dir", type=Path, default=None,
                         help="Explicit output dir (overrides experiment routing)")
    p_align.add_argument("--shard", type=int, default=None,
                         help="Process only one concept shard")
    p_align.add_argument("--threshold_min_f1", type=float, default=0.0,
                         help="Only affects console top-pairs display (CSV always contains all pairs)")
    p_align.add_argument("--n_top_per_concept", type=int, default=20)
    p_align.add_argument("--feature_chunk_size", type=int, default=200)
    p_align.add_argument("--batch_size", type=int, default=1024)
    p_align.add_argument("--no_auroc", action="store_true",
                         help="Skip AUROC computation (faster)")
    p_align.add_argument("--no_domain_f1", action="store_true",
                         help="Skip domain-level F1 (faster)")
    p_align.add_argument("--min_positives", type=int, default=10,
                         help="Skip concepts with fewer positive residues")
    p_align.add_argument("--threshold_percents", type=float, nargs="+", default=None,
                         help="InterPLM-style percent-of-max thresholds, e.g. 0 0.15 0.5 0.6 0.8")

    args = parser.parse_args()
    if args.command == "build":
        cmd_build(args.annotations_tsv, args.concepts_dir, args.experiment,
                  args.exp_dir, args.n_shards, args.min_seq_len, args.max_seq_len)
    else:
        cmd_align(args.sae_dir, args.embeddings_dir, args.concepts_dir,
                  args.output_dir, args.experiment, args.exp_dir,
                  args.shard, args.threshold_min_f1, args.n_top_per_concept,
                  args.feature_chunk_size, args.batch_size,
                  compute_auroc=not args.no_auroc,
                  compute_domain_f1=not args.no_domain_f1,
                  min_positives=args.min_positives,
                  threshold_percents=args.threshold_percents)


if __name__ == "__main__":
    main()
