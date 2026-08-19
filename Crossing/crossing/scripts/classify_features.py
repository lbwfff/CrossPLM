#!/usr/bin/env python3
"""Classify features as Shared / A-specific / B-specific (Phase 2).

Aligned mode (same proteins / same token order): each feature of each model
is scored on both tasks via single-feature logistic probes on a shared
train/val split – "shared" means the feature predicts both tasks.

Disjoint mode (different protein sets): per-feature cross-task probes are not
definable.  Use --mode cross_sim which classifies via the similarity matrix
+ per-task F1 (ROADMAP 2.2).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
_REPO = Path(__file__).resolve().parents[3]
for _sub in ("Single", "Crossing"):
    p = str(_REPO / _sub)
    if p not in sys.path:
        sys.path.insert(0, p)


def _resolve_sae_path(p: str) -> str:
    path = Path(p)
    return str(path.parent) if path.is_file() else str(path)


def _encode_sae(sae, embeddings: torch.Tensor, batch_size: int = 2048) -> np.ndarray:
    from single.sae.inference import get_sae_feats_in_batches  # type: ignore

    feats = get_sae_feats_in_batches(
        sae=sae, aa_embds=embeddings, chunk_size=batch_size,
        feat_list=None, normalize_features=False, device="cpu", cache=None,
    )
    arr = feats.detach().cpu().numpy() if isinstance(feats, torch.Tensor) else np.asarray(feats)
    return arr


def _load_label_spec(name, base_dir=None):
    from single.label_maps import get_label_map  # type: ignore

    p = Path(name)
    if p.suffix in {".yaml", ".yml"} and not p.is_absolute() and base_dir:
        p = Path(base_dir) / p
    return get_label_map(str(p))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Classify features as Shared / A-specific / B-specific")
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
    parser.add_argument("--label_map_a", default=None)
    parser.add_argument("--label_map_b", default=None)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--exp_dir", type=str, default=None)
    parser.add_argument("--mode", choices=["auto", "aligned", "cross_sim"], default="auto",
                        help="aligned (single-feature probes on shared proteins) | cross_sim (similarity+F1, for disjoint)")
    parser.add_argument("--threshold", type=float, default=0.3, help="F1 threshold for 'predicts' a task")
    parser.add_argument("--correlation_threshold", type=float, default=0.5, help="For cross_sim: similarity threshold for 'matched'")
    parser.add_argument("--f1_threshold", type=float, default=0.3, help="For cross_sim: F1 threshold for predictive")
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--batch_size", type=int, default=2048)

    args = parser.parse_args(argv)

    if args.exp_dir:
        output_dir = Path(args.exp_dir)
        if output_dir.name != "crossing":
            output_dir = output_dir / "crossing"
    else:
        output_dir = _REPO / "Outputs" / args.experiment / "crossing"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SAEs
    print("[Loading SAEs]")
    from single.sae.inference import load_sae  # type: ignore

    sae_a = load_sae(_resolve_sae_path(args.sae_a), device="cpu")
    sae_b = load_sae(_resolve_sae_path(args.sae_b), device="cpu")
    print(f"  SAE A: dict_size={sae_a.dict_size}, act_dim={sae_a.activation_dim}")
    print(f"  SAE B: dict_size={sae_b.dict_size}, act_dim={sae_b.activation_dim}")

    # Detect mode
    from crossing.io import load_embeddings_with_residues, align_two_embeddings  # type: ignore

    emb_a_raw, residues_a = load_embeddings_with_residues(args.embeddings_a)
    emb_b_raw, residues_b = load_embeddings_with_residues(args.embeddings_b)
    print(f"  Raw A: {tuple(emb_a_raw.shape)} residues={'yes' if residues_a is not None else 'no'}")
    print(f"  Raw B: {tuple(emb_b_raw.shape)} residues={'yes' if residues_b is not None else 'no'}")

    mode = args.mode
    if mode == "auto":
        # Prefer aligned if residues overlap; otherwise cross_sim
        try:
            _a, _b, _info = align_two_embeddings(emb_a_raw, residues_a, emb_b_raw, residues_b, max_tokens=None)
            mode = "aligned"
        except Exception:
            mode = "cross_sim"
    print(f"  Mode: {mode}")

    if mode == "cross_sim":
        # Disjoint-style: build per-task F1 via single-model probes, then sim matrix, then classify
        print("[cross_sim mode: similarity + per-task F1]")
        # For disjoint tasks, embeddings are already per-model (different n_tokens) – encode separately
        feats_a_raw = _encode_sae(sae_a, emb_a_raw, batch_size=args.batch_size)
        feats_b_raw = _encode_sae(sae_b, emb_b_raw, batch_size=args.batch_size)
        if args.normalize:
            feats_a_raw = feats_a_raw / (np.abs(feats_a_raw).max(axis=0, keepdims=True) + 1e-8)
            feats_b_raw = feats_b_raw / (np.abs(feats_b_raw).max(axis=0, keepdims=True) + 1e-8)

        # Build per-token labels per model via residues (separately, not jointly)
        import pandas as pd

        script_base = Path(__file__).resolve().parents[1]
        spec_a = _load_label_spec(args.label_map_a, base_dir=str(script_base)) if args.label_map_a else None
        spec_b = _load_label_spec(args.label_map_b, base_dir=str(script_base)) if args.label_map_b else None
        if spec_a is None or spec_b is None:
            raise ValueError("cross_sim mode requires --label_map_a and --label_map_b")

        def _labels_for_model(residues, emb, label_csv, seq_col, lbl_col, spec):
            if residues is None:
                raise ValueError("Need residues.csv for cross_sim label mapping")
            mapping = dict(spec.get("mapping", {}))
            ignore = set(spec.get("ignore", ""))
            csv = Path(label_csv)
            with open(csv, "r") as f:
                first = f.readline()
            sep = "\t" if first.count("\t") > first.count(",") else ","
            df = pd.read_csv(csv, sep=sep, dtype=str)
            if "Entry" in residues.columns and "Entry" in df.columns:
                key_to_lbl = dict(zip(df["Entry"].astype(str), df[lbl_col].astype(str).fillna("")))
                key_col = "Entry"
            else:
                from single.data import sequence_hash  # type: ignore
                key_to_lbl = {}
                for _, r in df.iterrows():
                    seq = str(r[seq_col]).upper().strip()
                    key_to_lbl[sequence_hash(seq)] = str(r[lbl_col]) if pd.notna(r[lbl_col]) else ""
                key_col = "sequence_hash"
            enc = []
            for _, row in residues.iterrows():
                lbl_str = key_to_lbl.get(str(row[key_col]), "")
                pos = int(row["position"])
                ch = lbl_str[pos] if pos < len(lbl_str) else "_"
                if ch in ignore:
                    enc.append(-100)
                elif ch in mapping:
                    enc.append(int(mapping[ch]))
                else:
                    enc.append(-100)
            return np.array(enc, dtype=np.int64)

        labels_a = _labels_for_model(residues_a, emb_a_raw, args.labels_a, args.sequence_column_a, args.label_column_a, spec_a)
        labels_b = _labels_for_model(residues_b, emb_b_raw, args.labels_b, args.sequence_column_b, args.label_column_b, spec_b)

        # Per-task single-feature F1: use same helper as Single's feature_alignment but single-feature probes
        # Quick via train_probe per feature – reuse classify path but compute f1 vectors first
        # Simpler: use similarity-based classification's required f1 vectors via per-model single-feature probes
        # We can get per-feature F1 on its own task via single-feature probes on each model's tokens
        from crossing.classification import classify_features as _legacy  # to get f1 vectors, but we do directly

        def _per_feature_f1(feats, labels):
            # single-feature logistic probes on same train/val split
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import f1_score

            valid = labels != -100
            f = feats[valid]
            y = labels[valid]
            n = len(y)
            rng = np.random.RandomState(args.seed)
            perm = rng.permutation(n)
            n_val = int(round(n * args.val_ratio))
            n_val = max(1, min(n_val, n - 1)) if n >= 2 else 0
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
            f_tr, f_val = f[train_idx], f[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            n_classes = len(np.unique(y_tr)) if len(y_tr) else 0
            average = "macro" if n_classes > 2 else "binary"
            f1s = []
            for i in range(feats.shape[1]):
                xtr = f_tr[:, i:i+1]
                xva = f_val[:, i:i+1]
                if np.std(xtr) < 1e-12 or len(np.unique(y_tr)) < 2:
                    f1s.append(0.0)
                    continue
                scaler = StandardScaler()
                xtr_s = scaler.fit_transform(xtr)
                xva_s = scaler.transform(xva)
                clf = LogisticRegression(max_iter=500, solver="lbfgs", class_weight="balanced")
                try:
                    clf.fit(xtr_s, y_tr)
                    pred = clf.predict(xva_s)
                    f1s.append(float(f1_score(y_val, pred, average=average, zero_division=0)))
                except Exception:
                    f1s.append(0.0)
            return np.array(f1s, dtype=np.float32)

        f1_a = _per_feature_f1(feats_a_raw, labels_a)
        f1_b = _per_feature_f1(feats_b_raw, labels_b)
        print(f"  Per-feature F1 on own task: A mean {f1_a.mean():.4f} max {f1_a.max():.4f}, "
              f"B mean {f1_b.mean():.4f} max {f1_b.max():.4f}")

        # Need similarity matrix; for disjoint tasks token sets don't align, so compute on
        # separate feature distributions (not comparable) is not meaningful. Instead, recommend aligned.
        # But for cross_sim we use the feature-F1 + similarity among feature vectors independent of token alignment?
        # Similarity between features requires aligned tokens – disjoint fails. So we must align via common proteins subsampled?
        # Try to align a subset of proteins that appear in both datasets (intersection via residues)
        try:
            emb_a_al, emb_b_al, _ = align_two_embeddings(emb_a_raw, residues_a, emb_b_raw, residues_b, max_tokens=args.max_tokens, seed=args.seed)
            feats_a_al = _encode_sae(sae_a, emb_a_al, batch_size=args.batch_size)
            feats_b_al = _encode_sae(sae_b, emb_b_al, batch_size=args.batch_size)
            if args.normalize:
                feats_a_al = feats_a_al / (np.abs(feats_a_al).max(axis=0, keepdims=True) + 1e-8)
                feats_b_al = feats_b_al / (np.abs(feats_b_al).max(axis=0, keepdims=True) + 1e-8)
            from crossing.similarity import compute_feature_similarity_matrix  # type: ignore
            S = compute_feature_similarity_matrix(feats_a_al, feats_b_al, method="correlation")
            print(f"  Aligned similarity for cross_sim: {S.shape}, mean {S.mean():.4f}, max {S.max():.4f}")
        except Exception as e:
            raise ValueError(
                f"cross_sim requires overlapping proteins to compute similarity: {e}. "
                "Provide evaluation embeddings on a shared protein set (same residues.csv overlap)."
            )

        from crossing.classification import classify_features_cross_model  # type: ignore

        # Use threshold as F1 threshold, since _per_feature_f1 is single-feature probe on own task
        results = classify_features_cross_model(
            S, f1_a, f1_b,
            correlation_threshold=args.correlation_threshold,
            f1_threshold=args.threshold if args.threshold != 0.3 or args.f1_threshold == 0.3 else args.f1_threshold,
        )
        # Respect explicit f1_threshold if given
        if args.f1_threshold != 0.3:
            results = classify_features_cross_model(S, f1_a, f1_b, correlation_threshold=args.correlation_threshold, f1_threshold=args.f1_threshold)

        results["mode"] = "cross_sim"
        results["f1_a"] = f1_a.tolist()
        results["f1_b"] = f1_b.tolist()
        # Save similarity as well for inspection
        np.save(output_dir / "feature_similarity_matrix.npy", S)
        with open(output_dir / "feature_similarity.json", "w") as f:
            json.dump({"mode": "cross_sim", "mean": float(S.mean()), "max": float(S.max()), "correlation_threshold": float(args.correlation_threshold)}, f, indent=2)

    else:
        # Aligned mode – joint token set, single-feature probes on both tasks
        print("[aligned mode: per-feature cross-task probes on shared proteins]")
        emb_a, emb_b, align_info = align_two_embeddings(emb_a_raw, residues_a, emb_b_raw, residues_b, max_tokens=args.max_tokens, seed=args.seed)
        print(f"  Aligned via {align_info.get('aligned_via')} -> {emb_a.shape[0]} tokens")
        feats_a = _encode_sae(sae_a, emb_a, batch_size=args.batch_size)
        feats_b = _encode_sae(sae_b, emb_b, batch_size=args.batch_size)
        if args.normalize:
            feats_a = feats_a / (np.abs(feats_a).max(axis=0, keepdims=True) + 1e-8)
            feats_b = feats_b / (np.abs(feats_b).max(axis=0, keepdims=True) + 1e-8)

        # Labels on aligned tokens via residues join
        import pandas as pd

        script_base = Path(__file__).resolve().parents[1]
        spec_a = _load_label_spec(args.label_map_a, base_dir=str(script_base)) if args.label_map_a else None
        spec_b = _load_label_spec(args.label_map_b, base_dir=str(script_base)) if args.label_map_b else None
        if spec_a is None or spec_b is None:
            raise ValueError("aligned mode requires --label_map_a and --label_map_b")

        # Build aligned residues (same merge as io)
        def _aligned_residues(ra, rb):
            cols_a = set(ra.columns); cols_b = set(rb.columns)
            key_cols = []
            if "Entry" in cols_a and "Entry" in cols_b:
                key_cols.append("Entry")
            for c in ["sequence_hash", "position"]:
                if c in cols_a and c in cols_b and c not in key_cols:
                    key_cols.append(c)
            ra2 = ra.copy(); rb2 = rb.copy()
            for df in (ra2, rb2):
                if "position" in df.columns:
                    df["position"] = df["position"].astype(str)
            ra2["_idx_a"] = np.arange(len(ra2))
            rb2["_idx_b"] = np.arange(len(rb2))
            merged = ra2.merge(rb2, on=key_cols, how="inner", suffixes=("_a", "_b"))
            aligned = ra.iloc[merged["_idx_a"].to_numpy()].reset_index(drop=True)
            if args.max_tokens is not None and args.max_tokens < len(aligned):
                rng = np.random.RandomState(args.seed)
                idx = np.sort(rng.choice(len(aligned), size=args.max_tokens, replace=False))
                aligned = aligned.iloc[idx].reset_index(drop=True)
            aligned["position"] = aligned["position"].astype(int)
            return aligned

        aligned_residues = _aligned_residues(residues_a, residues_b)

        def _labels_on_aligned(aligned, label_csv, seq_col, lbl_col, spec):
            mapping = dict(spec.get("mapping", {}))
            ignore = set(spec.get("ignore", ""))
            csv = Path(label_csv)
            with open(csv, "r") as f:
                first = f.readline()
            sep = "\t" if first.count("\t") > first.count(",") else ","
            df = pd.read_csv(csv, sep=sep, dtype=str)
            if "Entry" in aligned.columns and "Entry" in df.columns:
                key_to_lbl = dict(zip(df["Entry"].astype(str), df[lbl_col].astype(str).fillna("")))
                key_col = "Entry"
            else:
                from single.data import sequence_hash  # type: ignore
                key_to_lbl = {}
                for _, r in df.iterrows():
                    seq = str(r[seq_col]).upper().strip()
                    key_to_lbl[sequence_hash(seq)] = str(r[lbl_col]) if pd.notna(r[lbl_col]) else ""
                key_col = "sequence_hash"
            enc = []
            for _, row in aligned.iterrows():
                lbl_str = key_to_lbl.get(str(row[key_col]), "")
                pos = int(row["position"])
                ch = lbl_str[pos] if pos < len(lbl_str) else "_"
                if ch in ignore:
                    enc.append(-100)
                elif ch in mapping:
                    enc.append(int(mapping[ch]))
                else:
                    enc.append(-100)
            return np.array(enc, dtype=np.int64)

        labels_a = _labels_on_aligned(aligned_residues, args.labels_a, args.sequence_column_a, args.label_column_a, spec_a)
        labels_b = _labels_on_aligned(aligned_residues, args.labels_b, args.sequence_column_b, args.label_column_b, spec_b)
        print(f"  Aligned labels: A valid {(labels_a!=-100).sum()}/{len(labels_a)}, B valid {(labels_b!=-100).sum()}/{len(labels_b)}")

        from crossing.classification import classify_features  # type: ignore

        results = classify_features(
            feats_a, labels_a, feats_b, labels_b,
            threshold=args.threshold, val_ratio=args.val_ratio, seed=args.seed,
        )
        results["align_info"] = align_info

    # Print summary
    # Handle both aligned and cross_sim result shapes
    if "model_a" in results:
        summary_a = results.get("model_a", {}).get("summary", {})
        summary_b = results.get("model_b", {}).get("summary", {})
        print(f"\n  Model A features: {summary_a}")
        print(f"  Model B features: {summary_b}")
        if "summary" in results:
            print(f"  Combined: {results['summary']}")
            summary = results["summary"]
        else:
            summary = summary_a
        mean_a = results.get("model_a", {}).get("mean_f1_a", results.get("mean_f1_a", 0))
        mean_b = results.get("model_a", {}).get("mean_f1_b", results.get("mean_f1_b", 0))
        print(f"  Mean F1 (A-model on A/B): {mean_a:.4f} / {mean_b:.4f}")
    else:
        summary = results.get("summary", {})
        print(f"\n  Summary: {summary}")
        print(f"  Mean F1 A/B on own task available in per-feature arrays")

    for cat in ["shared", "A_specific", "B_specific"]:
        feats_in_cat = [f for f in results.get("features", []) if f.get("category") == cat][:5]
        if feats_in_cat:
            print(f"\n  Top {cat} (A-side):")
            for f in feats_in_cat:
                # Handle both classify schemas
                if "f1_task_a" in f:
                    print(f"    Feature {f['feature_idx']}: F1_A={f['f1_task_a']:.4f}, F1_B={f['f1_task_b']:.4f}")
                elif "feature_a" in f:
                    print(f"    A[{f['feature_a']}]<->B[{f['feature_b']}] sim={f['similarity']:.4f} F1_A={f['f1_a']:.4f} F1_B={f['f1_b']:.4f}")

    with open(output_dir / "feature_classification.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Done] Saved to {output_dir / 'feature_classification.json'}")


if __name__ == "__main__":
    main()
