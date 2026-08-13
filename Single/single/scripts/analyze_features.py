#!/usr/bin/env python
"""
Analyze SAE features: align with task labels and find interpretable patterns.

Usage:
    python scripts/analyze_features.py \
        --sae_dir ../models/sae/layer_6 \
        --embeddings_dir ../data/embeddings/esm2_8m/layer_6 \
        --output_dir ../analysis_results \
        --label_column label
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import json
import yaml
import pandas as pd
from tqdm import tqdm

from single.sae.inference import load_sae, get_sae_feats_in_batches, normalize_sae_features
from single.analysis.feature_alignment import (
    align_features_to_labels,
    find_top_features_for_class,
    compute_feature_label_correlation,
    compute_feature_activation_profile,
)
from single.analysis.per_protein_tracking import find_max_activating_proteins


def analyze_features(
    sae_dir: Path,
    embeddings_dir: Path,
    output_dir: Path,
    label_column: Optional[str] = None,
    sequences_csv: Optional[Path] = None,
    activation_threshold: float = 0.05,
    n_top_features: int = 50,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("SAE Feature Analysis")
    print("=" * 60)

    # 1. Load SAE
    print("\n1. Loading SAE...")
    sae = load_sae(sae_dir, device=device)
    print(f"   SAE: {sae.__class__.__name__}, {sae.dict_size} features, {sae.activation_dim}D")

    # 2. Load embeddings
    print("\n2. Loading embeddings...")
    data_path = Path(embeddings_dir)
    labels = None

    if data_path.is_file():
        data = torch.load(data_path, map_location=device, weights_only=True)
        if isinstance(data, dict):
            embeddings = data["embeddings"].to(device)
            labels = data.get("labels")
            if labels is not None:
                labels = labels.to(device)
        else:
            embeddings = data.to(device)
    else:
        pt_files = sorted(data_path.glob("**/activations.pt"))
        all_embeddings, all_labels = [], []
        for f in pt_files:
            d = torch.load(f, map_location="cpu", weights_only=True)
            if isinstance(d, dict):
                lbl = d.get("labels")
                if lbl is not None:
                    all_labels.append(lbl)
                d = d.get("embeddings", d)
            all_embeddings.append(d.float())
        embeddings = torch.cat(all_embeddings, dim=0).to(device)
        if all_labels:
            labels = torch.cat(all_labels, dim=0).to(device)
            print(f"   Labels loaded from {len(pt_files)} shard files (already aligned)")

    print(f"   Embeddings shape: {embeddings.shape}")

    # 3. Load labels from CSV if not in shards
    if labels is None and sequences_csv is not None and label_column is not None:
        print("\n3. Parsing labels from CSV...")
        df = pd.read_csv(sequences_csv)
        df[label_column] = df[label_column].fillna("").astype(str)

        MBM_LABEL = {"A": 0, ".": 1, "0": 0, "1": 1}
        all_labels = []
        for seq_idx, label_str in enumerate(tqdm(df[label_column], desc="Parsing labels")):
            label_str = label_str.strip()
            for c in label_str:
                all_labels.append(MBM_LABEL.get(c, -100))
        labels = torch.tensor(all_labels, device=device)

    if labels is not None and labels.shape[0] != embeddings.shape[0]:
        min_len = min(labels.shape[0], embeddings.shape[0])
        print(f"\n⚠️  WARNING: labels ({labels.shape[0]}) ≠ embeddings ({embeddings.shape[0]}) tokens")
        print(f"   Truncating both to {min_len} tokens...")
        labels = labels[:min_len]
        embeddings = embeddings[:min_len]
    print(f"   Labels shape: {labels.shape if labels is not None else 'N/A'}")

    # 4. Feature-label alignment
    if labels is not None:
        print("\n4. Aligning features to task labels...")
        metrics = align_features_to_labels(sae, embeddings, labels)

        top_flexible = find_top_features_for_class(metrics, "best_f1", n_top_features, class_label=1)
        print(f"\n   Top {n_top_features} features for 'flexible' (label=1):")
        for feat_idx, f1 in top_flexible[:10]:
            m = metrics[feat_idx]
            print(f"     Feature #{feat_idx}: F1={f1:.3f}, AUROC={m['auroc']:.3f}, "
                  f"Prec={m['best_precision']:.3f}, Rec={m['best_recall']:.3f}")

        # Save metrics
        metrics_serializable = {str(k): v for k, v in metrics.items()}
        output_path = output_dir / "feature_label_metrics.json"
        with open(output_path, "w") as f:
            json.dump(metrics_serializable, f, indent=2)
        print(f"\n   Metrics saved to {output_path}")

        # Correlation analysis
        print("\n5. Computing feature-label correlations...")
        correlations = compute_feature_label_correlation(sae, embeddings, labels)

        top_corr = np.argsort(np.abs(correlations))[::-1][:20]
        print("   Top features by |correlation| with label:")
        for fidx in top_corr:
            print(f"     Feature #{fidx}: r={correlations[fidx]:.4f}")

        np.save(output_dir / "feature_label_correlations.npy", correlations)

        # Activation profile
        print("\n6. Computing activation profiles (rigid vs flexible)...")
        profile = compute_feature_activation_profile(sae, embeddings, labels)

        gap = profile["activation_gap"]
        top_flex_features = np.argsort(gap)[::-1][:20]
        print("   Features with highest activation gap (flexible - rigid):")
        for fidx in top_flex_features:
            print(f"     Feature #{fidx}: gap={gap[fidx]:.4f} "
                  f"(rigid_mean={profile['rigid_mean_activation'][fidx]:.4f}, "
                  f"flex_mean={profile['flexible_mean_activation'][fidx]:.4f})")

        np.savez(output_dir / "activation_profile.npz", **profile)

    # 5. Compute max activation per feature (for normalization)
    print("\n7. Computing max activation per feature (for normalization)...")
    n_features = sae.dict_size
    max_per_feat = torch.zeros(n_features, device=device)

    from single.sae.inference import split_up_feature_list
    for feat_list in tqdm(
        split_up_feature_list(n_features, 200),
        desc="Computing max activations",
    ):
        feats = get_sae_feats_in_batches(
            sae=sae, aa_embds=embeddings,
            chunk_size=4096, feat_list=feat_list,
            normalize_features=False, device=str(device),
        )
        max_per_feat[feat_list] = torch.max(
            max_per_feat[feat_list], feats.max(dim=0)[0]
        )

    torch.save(max_per_feat.cpu(), output_dir / "max_activations_per_feature.pt")

    # Save normalized model
    sae_normalized = normalize_sae_features(sae, max_per_feat)
    torch.save(sae_normalized.state_dict(), sae_dir / "ae_normalized.pt")
    print(f"   Normalized model saved to {sae_dir / 'ae_normalized.pt'}")

    print(f"\n{'=' * 60}")
    print(f"Analysis complete! Results in {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = Path(f"../Outputs/analysis/{timestamp}")

    parser = argparse.ArgumentParser(description="Analyze SAE features against task labels")
    parser.add_argument("--sae_dir", type=Path, required=True)
    parser.add_argument("--embeddings_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=default_output)
    parser.add_argument("--sequences_csv", type=Path, default=None, help="CSV with label column")
    parser.add_argument("--label_column", type=str, default=None)
    parser.add_argument("--activation_threshold", type=float, default=0.05)
    parser.add_argument("--n_top_features", type=int, default=50)
    args = parser.parse_args()
    analyze_features(**{k: v for k, v in vars(args).items() if v is not None})
