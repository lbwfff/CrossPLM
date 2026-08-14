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
    label_names: Optional[dict] = None,
):
    label_names = label_names or {0: "Negative", 1: "Positive"}

    def color_for(label):
        return {1: "#ff6b6b", 0: "#4ecdc4"}.get(label, "#cccccc")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    with torch.no_grad():
        feats = sae.encode(embedding.unsqueeze(0), normalize_features=True)
    activations = feats[0, :, feature_idx].cpu().numpy()

    n = min(len(activations), len(labels))
    positions = np.arange(n)

    ax1.bar(positions, activations[:n], color="steelblue", width=0.8)
    ax1.set_ylabel("Feature Activation")
    ax1.set_title(f"Feature #{feature_idx} Activation")

    colors = [
        "white" if l == -100 else (color_for(l))
        for l in labels[:n].tolist()
    ]
    ax2.bar(positions, np.ones(n), color=colors, edgecolor="gray", linewidth=0.5, width=0.8)
    ax2.set_ylabel("Ground Truth")
    ax2.set_xlabel("Residue Position")
    ax2.set_yticks([])

    red_patch = mpatches.Patch(color=color_for(1), label=f"{label_names.get(1, 'Positive')} (1)")
    green_patch = mpatches.Patch(color=color_for(0), label=f"{label_names.get(0, 'Negative')} (0)")
    ax2.legend(handles=[red_patch, green_patch], loc="upper right")

    if len(sequence) <= 60 and n <= 60:
        ax2.set_xticks(positions)
        ax2.set_xticklabels(list(sequence[:n]), fontsize=8, rotation=90)
    else:
        ax2.set_xlim(-0.5, n - 0.5)

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
    output_dir: Optional[Path] = None,
    experiment: Optional[str] = None,
    exp_dir: Optional[Path] = None,
    sequences_csv: Optional[Path] = None,
    sequence_column: str = "sequence",
    label_column: Optional[str] = None,
    feature_indices: Optional[List[int]] = None,
    n_features: int = 10,
    max_proteins: int = 3,
    label_map: str = "mBMRB",
):
    from single.paths import resolve_experiment

    # Prefer explicit output_dir (legacy); else route into the experiment dir.
    if output_dir is None:
        exp = resolve_experiment(exp_dir=exp_dir, name=experiment)
        output_dir = exp.visualizations_dir()
        print(f"Experiment dir: {exp.dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from single.label_maps import get_label_map
    label_map_spec = get_label_map(label_map)
    label_names = label_map_spec["class_names"]

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

    # Build boundaries from CSV: each sequence's token range in the concatenated tensor
    boundaries = []
    if sequences_csv:
        import pandas as pd
        with open(sequences_csv, "r") as _f:
            _first = _f.readline()
        _sep = "\t" if _first.count("\t") > _first.count(",") else ","
        df = pd.read_csv(sequences_csv, sep=_sep, low_memory=False)
        sequences = df[sequence_column].tolist()
        if label_column and label_column in df.columns:
            df[label_column] = df[label_column].fillna("").astype(str)
            seq_labels = df[label_column].tolist()
        else:
            seq_labels = None
        start = 0
        for seq in sequences[:max_proteins]:
            seq_len = min(len(seq), 510)
            end = start + seq_len
            boundaries.append((start, end, seq, seq_labels[len(boundaries)] if seq_labels else None))
            start = end
    else:
        sequences = ["M" * embeddings.shape[0]]
        boundaries = [(0, min(200, embeddings.shape[0]), sequences[0], None)]

    if feature_indices is None:
        from single.analysis.feature_alignment import align_features_to_labels
        if labels is not None:
            metrics = align_features_to_labels(
                sae, embeddings, labels, positive_class=label_map_spec["positive_class"],
            )
            top = sorted(metrics.items(), key=lambda x: x[1]["best_f1"], reverse=True)
            feature_indices = [int(k) for k, _ in top[:n_features]]
            plot_feature_summary(metrics, output_dir / "feature_summary.png")
        else:
            feature_indices = list(range(min(n_features, sae.dict_size)))

    print(f"Visualizing {len(feature_indices)} features across {len(boundaries)} proteins...")
    for fidx in feature_indices:
        for prot_idx, (start, end, seq, lbl_str) in enumerate(boundaries):
            prot_emb = embeddings[start:end]
            if labels is not None:
                prot_labels = labels[start:end]
            else:
                prot_labels = torch.zeros(end - start, device=device)

            save_path = output_dir / f"feature_{fidx}_protein_{prot_idx}.png"
            plot_feature_label_overlap(
                sae, prot_emb, prot_labels, seq, fidx, save_path=save_path,
                label_names=label_names,
            )
        print(f"  Feature #{fidx} → {output_dir}/feature_{fidx}_protein_*.png")

    print(f"\nVisualizations saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SAE feature activations on proteins")
    parser.add_argument("--sae_dir", type=Path, required=True)
    parser.add_argument("--embeddings_path", type=Path, required=True)
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name; creates Outputs/<experiment>_<ts>/")
    parser.add_argument("--exp_dir", type=Path, default=None,
                        help="Reuse an existing experiment directory")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Explicit output dir (overrides experiment routing)")
    parser.add_argument("--sequences_csv", type=Path, default=None)
    parser.add_argument("--sequence_column", type=str, default="sequence")
    parser.add_argument("--label_column", type=str, default=None)
    parser.add_argument("--feature_indices", type=int, nargs="+", default=None)
    parser.add_argument("--n_features", type=int, default=10)
    parser.add_argument("--label_map", type=str, default="mBMRB",
                        help="Label encoding preset name or path to YAML label-map file")
    args = parser.parse_args()
    visualize_features(**{k: v for k, v in vars(args).items() if v is not None})
