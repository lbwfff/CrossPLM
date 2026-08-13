#!/usr/bin/env python
"""
Visualize SAE feature activations on protein sequences.

Usage:
    python scripts/visualize_features.py \
        --sae_dir ../models/sae/layer_6 \
        --embeddings_path ../data/embeddings/esm2_8m/layer_6/activations.pt \
        --output_dir ../analysis_results/visualizations
"""

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from single.sae.inference import load_sae


def visualize_feature_on_sequence(
    sae,
    embedding: torch.Tensor,
    sequence: str,
    feature_idx: int,
    ax=None,
    title: Optional[str] = None,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))

    with torch.no_grad():
        feats = sae.encode(embedding.unsqueeze(0), normalize_features=True)
    activations = feats[0, :, feature_idx].cpu().numpy()

    ax.bar(range(len(activations)), activations, color="steelblue", width=0.8)
    ax.set_xlabel("Residue Position")
    ax.set_ylabel("Activation")
    ax.set_title(title or f"Feature #{feature_idx} Activation on Sequence")

    if len(sequence) <= 60:
        ax.set_xticks(range(len(sequence)))
        ax.set_xticklabels(list(sequence), fontsize=8, rotation=90)

    return ax


def plot_feature_label_overlap(
    sae,
    embedding: torch.Tensor,
    labels: torch.Tensor,
    sequence: str,
    feature_idx: int,
    save_path: Optional[Path] = None,
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    with torch.no_grad():
        feats = sae.encode(embedding.unsqueeze(0), normalize_features=True)
    activations = feats[0, :, feature_idx].cpu().numpy()

    n = min(len(activations), len(labels))
    positions = np.arange(n)

    ax1.bar(positions, activations[:n], color="steelblue", width=0.8)
    ax1.set_ylabel("Feature Activation")
    ax1.set_title(f"Feature #{feature_idx} Activation")

    colors = ["white" if l == -100 else ("#ff6b6b" if l == 1 else "#4ecdc4") for l in labels[:n].tolist()]
    ax2.bar(positions, np.ones(n), color=colors, edgecolor="gray", linewidth=0.5, width=0.8)
    ax2.set_ylabel("Ground Truth")
    ax2.set_xlabel("Residue Position")
    ax2.set_yticks([])

    red_patch = mpatches.Patch(color="#ff6b6b", label="Flexible (1)")
    green_patch = mpatches.Patch(color="#4ecdc4", label="Rigid (0)")
    ax2.legend(handles=[red_patch, green_patch], loc="upper right")

    if len(sequence) <= 60:
        ax2.set_xticks(positions)
        ax2.set_xticklabels(list(sequence[:n]), fontsize=8, rotation=90)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_feature_summary(
    metrics: dict,
    save_path: Path,
    metric: str = "best_f1",
    n_top: int = 30,
):
    scores = [(int(k), v[metric]) for k, v in metrics.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:n_top]

    fig, ax = plt.subplots(figsize=(10, 6))
    features, values = zip(*top)
    colors = plt.cm.RdYlGn(np.array(values) / max(values))
    ax.barh(range(len(features)), values, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([f"#{f}" for f in features])
    ax.set_xlabel(metric)
    ax.set_title(f"Top {n_top} Features by {metric}")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_features(
    sae_dir: Path,
    embeddings_path: Path,
    output_dir: Path,
    sequences_csv: Optional[Path] = None,
    sequence_column: str = "sequence",
    label_column: Optional[str] = None,
    feature_indices: Optional[List[int]] = None,
    n_features: int = 10,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading SAE...")
    sae = load_sae(sae_dir, device=device)

    print("Loading embeddings...")
    data = torch.load(embeddings_path, map_location=device, weights_only=True)
    if isinstance(data, dict):
        embeddings = data["embeddings"].to(device)
        labels = data.get("labels")
    else:
        embeddings = data.to(device)
        labels = None

    if sequences_csv:
        import pandas as pd
        df = pd.read_csv(sequences_csv)
        sequences = df[sequence_column].tolist()
    else:
        sequences = ["M" * embeddings.shape[0]]

    if feature_indices is None:
        from single.analysis.feature_alignment import align_features_to_labels
        if labels is not None:
            metrics = align_features_to_labels(sae, embeddings, labels)
            top = sorted(metrics.items(), key=lambda x: x[1]["best_f1"], reverse=True)
            feature_indices = [int(k) for k, _ in top[:n_features]]
            plot_feature_summary(metrics, output_dir / "feature_summary.png")
        else:
            feature_indices = list(range(min(n_features, sae.dict_size)))

    print(f"Visualizing {len(feature_indices)} features...")
    for fidx in feature_indices:
        save_path = output_dir / f"feature_{fidx}_overlap.png"
        plot_feature_label_overlap(
            sae, embeddings, labels if labels is not None else torch.zeros(embeddings.shape[0]),
                sequences[0] if sequences else "",
                fidx, save_path=save_path,
            )
        print(f"  Feature #{fidx} → {save_path}")

    print(f"\nVisualizations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SAE feature activations on proteins")
    parser.add_argument("--sae_dir", type=Path, required=True)
    parser.add_argument("--embeddings_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sequences_csv", type=Path, default=None)
    parser.add_argument("--sequence_column", type=str, default="sequence")
    parser.add_argument("--label_column", type=str, default=None)
    parser.add_argument("--feature_indices", type=int, nargs="+", default=None)
    parser.add_argument("--n_features", type=int, default=10)
    args = parser.parse_args()
    visualize_features(**{k: v for k, v in vars(args).items() if v is not None})
