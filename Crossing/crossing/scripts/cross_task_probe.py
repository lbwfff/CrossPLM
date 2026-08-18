#!/usr/bin/env python3
"""Cross-task probing: evaluate whether one model's features predict another task.

Usage:
  python crossplm.py crossing cross_task_probe \
      --sae_a Outputs/exp_a/sae/model.pt \
      --sae_b Outputs/exp_b/sae/model.pt \
      --embeddings_a Outputs/exp_a/mbmrb/embeddings/layer_6 \
      --embeddings_b Outputs/exp_b/mbmrb/embeddings/layer_6 \
      --labels_a Dataset/mBMRB.csv \
      --labels_b Dataset/relaxdb.csv \
      --label_column_a label \
      --label_column_b label \
      --label_map_a mBMRB \
      --label_map_b relaxdb \
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


def _load_labels_from_csv(csv_path, seq_col, lbl_col):
    import pandas as pd
    sep = "\t" if csv_path.endswith(".tsv") else ","
    df = pd.read_csv(csv_path, sep=sep)
    return df[seq_col].tolist(), df[lbl_col].tolist()


def _load_label_map(name, base_dir=None):
    from single.label_maps import get_label_map
    p = os.path.Path(name)
    if p.suffix in {".yaml", ".yml"} and not p.is_absolute() and base_dir:
        p = os.path.Path(base_dir) / p
    return get_label_map(str(p))


def _encode_labels(labels_list, label_map, max_length):
    encoded = []
    for lbl_str in labels_list:
        row = []
        for ch in lbl_str[:max_length]:
            if ch in label_map:
                row.append(label_map[ch])
            else:
                row.append(-100)
        encoded.append(row)
    return encoded


def _align_embeddings_to_labels(emb, labels_encoded):
    aligned_emb, aligned_lbl = [], []
    for i, lbl in enumerate(labels_encoded):
        n = len(lbl)
        aligned_emb.append(emb[i, :n])
        aligned_lbl.append(lbl[:n])
    return np.concatenate(aligned_emb), np.concatenate(aligned_lbl)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cross-task probing")
    parser.add_argument("--sae_a", required=True)
    parser.add_argument("--sae_b", required=True)
    parser.add_argument("--embeddings_a", required=True)
    parser.add_argument("--embeddings_b", required=True)
    parser.add_argument("--labels_a", required=True, help="CSV with task A labels")
    parser.add_argument("--labels_b", required=True, help="CSV with task B labels")
    parser.add_argument("--sequence_column_a", default="sequence")
    parser.add_argument("--sequence_column_b", default="sequence")
    parser.add_argument("--label_column_a", default="label")
    parser.add_argument("--label_column_b", default="label")
    parser.add_argument("--label_map_a", default=None, help="Label map preset or YAML for A")
    parser.add_argument("--label_map_b", default=None, help="Label map preset or YAML for B")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--source_a", default=None)
    parser.add_argument("--source_b", default=None)
    parser.add_argument("--method", default="logistic", choices=["logistic", "linear"])
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true")

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
    print(f"  Raw embeddings A: {emb_a.shape}, B: {emb_b.shape}")

    # Load labels
    print("[Loading labels]")
    seqs_a, lbls_a = _load_labels_from_csv(args.labels_a, args.sequence_column_a, args.label_column_a)
    seqs_b, lbls_b = _load_labels_from_csv(args.labels_b, args.sequence_column_b, args.label_column_b)
    print(f"  Samples A: {len(seqs_a)}, B: {len(seqs_b)}")

    # Load label maps
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    if args.label_map_a:
        spec_a = _load_label_map(args.label_map_a, base_dir)
        lm_a = dict(spec_a["mapping"])
    else:
        from single.label_maps import build_label_map
        lm_a = build_label_map(lbls_a)
    if args.label_map_b:
        spec_b = _load_label_map(args.label_map_b, base_dir)
        lm_b = dict(spec_b["mapping"])
    else:
        from single.label_maps import build_label_map
        lm_b = build_label_map(lbls_b)
    print(f"  Label map A: {lm_a}")
    print(f"  Label map B: {lm_b}")

    # Encode labels
    labels_enc_a = _encode_labels(lbls_a, lm_a, args.max_seq_length)
    labels_enc_b = _encode_labels(lbls_b, lm_b, args.max_seq_length)

    # Align embeddings to label lengths
    print("[Aligning embeddings to labels]")
    feats_a_flat, labels_a_flat = _align_embeddings_to_labels(emb_a, labels_enc_a)
    feats_b_flat, labels_b_flat = _align_embeddings_to_labels(emb_b, labels_enc_b)
    print(f"  Aligned A: {feats_a_flat.shape}, B: {feats_b_flat.shape}")

    if args.max_tokens:
        feats_a_flat = feats_a_flat[:args.max_tokens]
        labels_a_flat = labels_a_flat[:args.max_tokens]
        feats_b_flat = feats_b_flat[:args.max_tokens]
        labels_b_flat = labels_b_flat[:args.max_tokens]

    # Remove -100 labels
    valid_a = labels_a_flat >= 0
    valid_b = labels_b_flat >= 0
    feats_a_flat = feats_a_flat[valid_a]
    labels_a_flat = labels_a_flat[valid_a]
    feats_b_flat = feats_b_flat[valid_b]
    labels_b_flat = labels_b_flat[valid_b]
    print(f"  Valid tokens A: {len(labels_a_flat)}, B: {len(labels_b_flat)}")

    # Compute SAE features
    print("[Computing SAE features]")
    with torch.no_grad():
        sae_feats_a = sae_a.encode(torch.tensor(feats_a_flat, dtype=torch.float32)).numpy()
        sae_feats_b = sae_b.encode(torch.tensor(feats_b_flat, dtype=torch.float32)).numpy()
    if args.normalize:
        max_a = np.abs(sae_feats_a).max(axis=0, keepdims=True) + 1e-8
        max_b = np.abs(sae_feats_b).max(axis=0, keepdims=True) + 1e-8
        sae_feats_a = sae_feats_a / max_a
        sae_feats_b = sae_feats_b / max_b
    print(f"  SAE features A: {sae_feats_a.shape}, B: {sae_feats_b.shape}")

    # Cross-task probing
    print("[Running cross-task probes]")
    from crossing.probe import cross_task_evaluate, compute_transfer_matrix
    results = cross_task_evaluate(
        sae_feats_a, labels_a_flat,
        sae_feats_b, labels_b_flat,
        method=args.method,
        seed=args.seed,
    )

    matrix = compute_transfer_matrix(results)
    results["transfer_matrix"] = matrix.tolist()

    print(f"\n  Transfer matrix (rows=train, cols=test):")
    print(f"             Task A    Task B")
    print(f"  Trained A  {matrix[0,0]:.4f}    {matrix[0,1]:.4f}")
    print(f"  Trained B  {matrix[1,0]:.4f}    {matrix[1,1]:.4f}")

    for direction in ["A_to_A", "A_to_B", "B_to_B", "B_to_A"]:
        m = results[direction]
        print(f"  {direction}: F1={m['val_f1']:.4f}  Acc={m['val_accuracy']:.4f}")

    # Save
    output_dir = os.path.join(
        os.path.dirname(script_dir), "..", "Outputs", args.experiment, "crossing"
    )
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "cross_task_probe.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Done] Saved to {output_dir}/cross_task_probe.json")


if __name__ == "__main__":
    main()
