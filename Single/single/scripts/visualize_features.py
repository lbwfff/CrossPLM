#!/usr/bin/env python
"""
Visualize SAE feature activations on protein sequences.

Usage:
    python scripts/visualize_features.py \
        --sae_dir ../models/sae/layer_6 \
        --embeddings_path ../data/embeddings/esm2_8m/layer_6/embeddings.pt \
        --output_dir ../analysis_results/visualizations
"""

import os
import sys

# Allow running directly from the repository root, e.g.
#   python Single/single/scripts/analyze_sequence.py ...
# without `cd Single` or installing the package (the `single` package lives at
# Single/single/, two levels up from this file).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe: figures are saved, never shown
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
    sae_dir: Optional[Path] = None,
    embeddings_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    experiment: Optional[str] = None,
    exp_dir: Optional[Path] = None,
    source: Optional[str] = None,
    sequences_csv: Optional[Path] = None,
    sequence_column: str = "sequence",
    label_column: Optional[str] = None,
    feature_indices: Optional[List[int]] = None,
    n_features: int = 10,
    max_proteins: int = 3,
    label_map: str = "mBMRB",
    layer: int = 6,
    shard: int = 0,
    max_length: int = 512,
    min_seq_len: int = 0,
    max_seq_len: int = 10_000,
    max_sequences: Optional[int] = None,
    filter_sequence: Optional[str] = None,
):
    from single.paths import resolve_experiment

    if experiment is None and exp_dir is None and embeddings_path is None and output_dir is None:
        raise ValueError("Must provide --experiment/--exp_dir (or an explicit path)")
    exp = resolve_experiment(exp_dir=exp_dir, name=experiment, source=source) if (experiment or exp_dir) else None

    # --sae_dir defaults into Outputs/<experiment>/sae but stays overridable.
    if sae_dir is None:
        if exp is None:
            raise ValueError("--sae_dir required when no experiment is given")
        sae_dir = exp.sae_dir
        print(f"  SAE dir (inferred): {sae_dir}")

    # Prefer explicit output_dir; else route into the experiment dir.
    if output_dir is None:
        if exp is None:
            raise ValueError("--output_dir required when no experiment is given")
        output_dir = exp.visualizations_dir()
        print(f"Experiment dir: {exp.dir}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Infer embeddings path from experiment if not given:
    # Outputs/<exp>/embeddings/layer_<N>/shard_<S>/embeddings.pt
    # Fallback: if source-nested path doesn't exist but flat does, use flat
    # (common when task embeddings were extracted flat but analysis runs with --source)
    if embeddings_path is None:
        emb_dir = exp.embeddings_dir(layer=layer) / f"shard_{shard}"
        embeddings_path = emb_dir / "embeddings.pt"
        if not embeddings_path.exists() and source:
            from pathlib import Path as _Path
            # Try flat location: Outputs/<exp>/embeddings/layer_<N>/shard_<S>/embeddings.pt
            flat_exp = resolve_experiment(exp_dir=exp_dir, name=experiment, source=None) if (experiment or exp_dir) else None
            if flat_exp is not None:
                flat_path = flat_exp.embeddings_dir(layer=layer) / f"shard_{shard}" / "embeddings.pt"
                if flat_path.exists():
                    print(f"[Fallback] source-nested {embeddings_path} not found, using flat {flat_path}")
                    embeddings_path = flat_path
        print(f"Inferred embeddings path: {embeddings_path}")
    embeddings_path = Path(embeddings_path)

    from single.label_maps import get_label_map, resolve_columns
    label_map_spec = get_label_map(label_map)
    # The label map describes the dataset's columns; use them unless the user
    # explicitly overrode them on the command line.
    sequence_column, label_column = resolve_columns(
        label_map_spec, sequence_column, label_column
    )

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

    # Build boundaries from the CSV. IMPORTANT: extract_embeddings.py shuffles the
    # DataFrame (sample(frac=1, random_state=42)) before sharding, so we must
    # replicate that exact order to align proteins with embedding tokens. If a
    # given shard contains no labels, we skip label overlay for that protein.
    boundaries = []
    if sequences_csv:
        from single.data import load_sequences_df, shuffled_shards

        df = load_sequences_df(sequences_csv, sequence_column=sequence_column,
                               min_seq_len=min_seq_len, max_seq_len=max_seq_len,
                               max_sequences=max_sequences)

        # Determine shard count from the embeddings directory (for the given layer).
        n_shards = len(list(exp.embeddings_dir(layer=layer).glob("shard_*"))) if exp else 1
        max_residues = max_length - 2

        filt = None
        if filter_sequence is not None:
            filt = str(filter_sequence).strip().upper()
            if not filt:
                raise ValueError("--filter_sequence requires a non-empty sequence string")

        # Shared shuffle+shard (fixed seed) — identical to extract_embeddings.
        if n_shards > 1:
            shards = shuffled_shards(df, n_shards)
            if filt is not None:
                # Search all shards for the exact sequence; keep original sharding
                # so token offsets remain aligned with the stored embeddings.
                found_shard = None
                found_idx_in_shard = None
                for sid, sdf in enumerate(shards):
                    mask = sdf[sequence_column].astype(str).str.strip().str.upper() == filt
                    if mask.any():
                        found_shard = sid
                        found_idx_in_shard = int(np.where(mask)[0][0])
                        break
                if found_shard is None:
                    raise ValueError(
                        f"No protein with the given sequence found in {sequences_csv} "
                        f"(normalized length {len(filt)}). Check whitespace/case."
                    )
                if found_shard != shard:
                    print(f"[Filter] sequence found in shard {found_shard}, overriding --shard {shard} → {found_shard}")
                    shard = found_shard
                    # Reload embeddings from the correct shard
                    new_emb_dir = exp.embeddings_dir(layer=layer) / f"shard_{shard}"
                    new_path = new_emb_dir / "embeddings.pt"
                    if not new_path.exists() and source:
                        flat_exp = resolve_experiment(exp_dir=exp_dir, name=experiment, source=None) if (experiment or exp_dir) else None
                        if flat_exp is not None:
                            flat_path = flat_exp.embeddings_dir(layer=layer) / f"shard_{shard}" / "embeddings.pt"
                            if flat_path.exists():
                                new_path = flat_path
                    print(f"[Filter] reloading embeddings from {new_path}")
                    data = torch.load(new_path, map_location=device, weights_only=True)
                    if isinstance(data, dict):
                        embeddings = data["embeddings"].to(device)
                        labels = data.get("labels")
                        if labels is not None:
                            labels = labels.to(device)
                    else:
                        embeddings = data.to(device)
                        labels = None
                    print(f"  Reloaded embeddings shard_{shard}: {embeddings.shape}")
                # Keep the full shard for offset calculation, but isolate the matched row for boundaries
                full_shard_df = shards[shard]
                # Compute token offset of the matched protein within its shard
                # (sum of truncated lengths of preceding proteins in that shard)
                offset = 0
                for idx, seq in enumerate(full_shard_df[sequence_column].astype(str).tolist()):
                    if idx == found_idx_in_shard:
                        break
                    offset += min(len(str(seq)), max_residues)
                shard_df = full_shard_df.iloc[[found_idx_in_shard]].reset_index(drop=True)
                # For filtered case, boundaries must start at the true offset within the shard's embedding tensor
                seq_labels = None
                if label_column and label_column in shard_df.columns:
                    shard_df[label_column] = shard_df[label_column].fillna("").astype(str)
                    seq_labels = shard_df[label_column].tolist()
                seq = shard_df[sequence_column].iloc[0]
                seq_len = min(len(str(seq)), max_residues)
                boundaries.append((offset, offset + seq_len, str(seq), seq_labels[0] if seq_labels else None))
            else:
                shard_df = shards[shard] if shard < len(shards) else df.iloc[:0]
                seq_labels = None
                if label_column and label_column in shard_df.columns:
                    shard_df[label_column] = shard_df[label_column].fillna("").astype(str)
                    seq_labels = shard_df[label_column].tolist()
                start = 0
                for i, seq in enumerate(shard_df[sequence_column].tolist()[:max_proteins]):
                    seq_len = min(len(str(seq)), max_residues)
                    end = start + seq_len
                    lbl = seq_labels[i] if seq_labels is not None else None
                    boundaries.append((start, end, str(seq), lbl))
                    start = end
        else:
            shard_df = df
            if filt is not None:
                mask = shard_df[sequence_column].astype(str).str.strip().str.upper() == filt
                if not mask.any():
                    raise ValueError(f"Filtered sequence not in shard {shard}")
                # Single-shard case with filter: find offset similarly
                found_idx = int(np.where(mask)[0][0])
                offset = 0
                for idx, seq in enumerate(shard_df[sequence_column].astype(str).tolist()):
                    if idx == found_idx:
                        break
                    offset += min(len(str(seq)), max_residues)
                seq = shard_df[sequence_column].iloc[found_idx]
                lbl = shard_df[label_column].iloc[found_idx] if label_column and label_column in shard_df.columns else None
                if lbl is not None:
                    lbl = str(lbl)
                seq_len = min(len(str(seq)), max_residues)
                boundaries.append((offset, offset + seq_len, str(seq), lbl))
            else:
                seq_labels = None
                if label_column and label_column in shard_df.columns:
                    shard_df[label_column] = shard_df[label_column].fillna("").astype(str)
                    seq_labels = shard_df[label_column].tolist()
                start = 0
                for i, seq in enumerate(shard_df[sequence_column].tolist()[:max_proteins]):
                    seq_len = min(len(str(seq)), max_residues)
                    end = start + seq_len
                    lbl = seq_labels[i] if seq_labels is not None else None
                    boundaries.append((start, end, str(seq), lbl))
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="Visualize SAE feature activations on proteins")
    parser.add_argument("--sae_dir", type=Path, default=None,
                        help="Trained SAE dir (default: Outputs/<experiment>/sae)")
    parser.add_argument("--embeddings_path", type=Path, default=None,
                        help="Embeddings shard file (default: inferred from experiment "
                             "as embeddings/layer_<N>/shard_<S>/embeddings.pt)")
    parser.add_argument("--source", type=str, default=None,
                        help="Data-source id; nests outputs under Outputs/<experiment>/<source> (default: flat)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name; routes outputs into Outputs/<experiment>/")
    parser.add_argument("--exp_dir", type=Path, default=None,
                        help="Reuse an existing experiment directory")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Explicit output dir (overrides experiment routing)")
    parser.add_argument("--layer", type=int, default=6,
                        help="Embedding layer (for inferred embeddings_path)")
    parser.add_argument("--shard", type=int, default=0,
                        help="Embedding shard (for inferred embeddings_path)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Embedder max_length used at extraction (boundaries truncated "
                             "to max_length-2)")
    parser.add_argument("--min_seq_len", type=int, default=0,
                        help="Must match extract_embeddings --min_seq_len")
    parser.add_argument("--max_sequences", type=int, default=None,
                        help="Deterministic subset; must match extract_embeddings --max_sequences")
    parser.add_argument("--max_seq_len", type=int, default=10000,
                        help="Must match extract_embeddings --max_seq_len")
    parser.add_argument("--sequences_csv", type=Path, default=None)
    parser.add_argument("--sequence_column", type=str, default="sequence")
    parser.add_argument("--label_column", type=str, default=None)
    parser.add_argument("--feature_indices", type=int, nargs="+", default=None)
    parser.add_argument("--n_features", type=int, default=10)
    parser.add_argument("--filter_sequence", type=str, default=None,
                        help="Exact protein sequence to visualize (single sequence, case-insensitive, whitespace trimmed). "
                             "When given, only that protein is visualized and --shard is auto-corrected to its shard. "
                             "Requires --sequences_csv.")
    parser.add_argument("--label_map", type=str, default="mBMRB",
                        help="Label encoding preset name or path to YAML label-map file")
    args = parser.parse_args(argv)
    visualize_features(**{k: v for k, v in vars(args).items() if v is not None})


if __name__ == "__main__":
    main()
