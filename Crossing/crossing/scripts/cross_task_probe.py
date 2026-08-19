#!/usr/bin/env python3
"""Cross-task probing (Phase 2).

Information-transfer test: does model A's representation contain information
about task B?  Two regimes:

- Aligned (recommended, publishable): both models encode the SAME proteins
  (same token set / same order).  Labels for tasks A and B are available
  on those tokens (either from the same evaluation set or from joining
  datasets by protein identity).  Then A→B / B→A are directly comparable to
  A→A / B→B with a shared train/val split.

- Legacy disjoint fallback: feats_a/labels_a and feats_b/labels_b are from
  different protein sets (old single CLI that passed two different CSVs).
  Baselines still run; cross-task numbers are flagged as not comparable.
  To make them comparable, re-extract cross-encoded embeddings: run model A
  on task B's proteins and vice versa, then use --mode aligned.

Usage (aligned, recommended):
  # First, ensure embeddings for evaluation proteins overlap:
  #   exp_shared = same eval CSV for both models (or intersection)
  python crossplm.py crossing cross_task_probe \
      --sae_a Outputs/exp_a/sae --sae_b Outputs/exp_b/sae \
      --embeddings_a Outputs/exp_shared/eval_on_shared/layer_6 \
      --embeddings_b Outputs/exp_shared/eval_on_shared/layer_6 \
      --labels_a Dataset/mBMRB.csv --labels_b Dataset/relaxdb.csv \
      --label_column_a label --label_column_b label \
      --label_map_a mBMRB --label_map_b relaxdb \
      --mode aligned --experiment my_crossing

Usage (back-compat, disjoint embeddings):
  python crossplm.py crossing cross_task_probe \
      --sae_a ... --sae_b ... \
      --embeddings_a Outputs/exp_a/... --embeddings_b Outputs/exp_b/... \
      --labels_a Dataset/a.csv --labels_b Dataset/b.csv \
      --experiment my_crossing  # will emit warning

New mode (re-extract not needed for quick test): if you already have
residues.csv, the script can align embeddings/lables via residue identity
and run aligned probes on the intersection.
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


def _encode_labels_simple(labels_list, mapping: dict, ignore_chars: str):
    """Encode raw label strings char-by-char -> list of ints (-100 ignore)."""
    ignore = set(ignore_chars)
    encoded = []
    for s in labels_list:
        s = str(s)
        row = []
        for ch in s:
            if ch in ignore:
                row.append(-100)
            elif ch in mapping:
                row.append(int(mapping[ch]))
            else:
                row.append(-100)
        encoded.append(row)
    return encoded


def main(argv=None):
    parser = argparse.ArgumentParser(description="Cross-task probing (Phase 2)")
    # Core
    parser.add_argument("--sae_a", required=True)
    parser.add_argument("--sae_b", required=True)
    parser.add_argument("--embeddings_a", required=True)
    parser.add_argument("--embeddings_b", required=True)
    # Labels – either provide both CSVs (aligned via residues) or rely on embeddings.labels
    parser.add_argument("--labels_a", type=str, default=None, help="CSV with task A labels (per-residue string)")
    parser.add_argument("--labels_b", type=str, default=None, help="CSV with task B labels")
    parser.add_argument("--sequence_column_a", default="sequence")
    parser.add_argument("--sequence_column_b", default="sequence")
    parser.add_argument("--label_column_a", default="label")
    parser.add_argument("--label_column_b", default="label")
    parser.add_argument("--label_map_a", default=None, help="Preset or YAML for task A")
    parser.add_argument("--label_map_b", default=None, help="Preset or YAML for task B")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--exp_dir", type=str, default=None)
    # Probe params
    parser.add_argument("--mode", choices=["auto", "aligned", "disjoint"], default="auto",
                        help="auto (detect via token counts/residues) | aligned (require residues overlap) | disjoint (legacy)")
    parser.add_argument("--method", default="logistic", choices=["logistic", "linear"])
    parser.add_argument("--max_tokens", type=int, default=None, help="Subsample aligned tokens (deterministic)")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--normalize", action="store_true", help="L_inf normalize SAE features")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--max_seq_length", type=int, default=512, help="Truncation length for CSV labels (max_length-2 residues)")

    args = parser.parse_args(argv)

    # Resolve output dir
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

    # Load embeddings with residues
    print("[Loading embeddings]")
    from crossing.io import load_embeddings_with_residues, align_two_embeddings  # type: ignore

    emb_a_raw, residues_a = load_embeddings_with_residues(args.embeddings_a)
    emb_b_raw, residues_b = load_embeddings_with_residues(args.embeddings_b)
    print(f"  Raw A: {tuple(emb_a_raw.shape)} residues={'yes' if residues_a is not None else 'no'}")
    print(f"  Raw B: {tuple(emb_b_raw.shape)} residues={'yes' if residues_b is not None else 'no'}")

    # Decide mode
    mode = args.mode
    if mode == "auto":
        # If token counts differ and residues available -> still can align via residues overlap
        # We'll attempt aligned first; fallback to disjoint if no overlap
        try:
            _a, _b, _info = align_two_embeddings(emb_a_raw, residues_a, emb_b_raw, residues_b, max_tokens=None)
            mode = "aligned"
        except Exception:
            mode = "disjoint"
    print(f"  Mode: {mode}")

    # Aligned path – token-joint via residues
    if mode == "aligned":
        emb_a, emb_b, align_info = align_two_embeddings(
            emb_a_raw, residues_a, emb_b_raw, residues_b,
            max_tokens=args.max_tokens, seed=args.seed,
        )
        print(f"  Aligned via {align_info.get('aligned_via')} -> {emb_a.shape[0]} tokens")
        # Load & encode labels on the ALIGNED token set via residue identity
        # This is critical: labels must correspond to the overlapping tokens, not raw CSV order
        # We load CSVs and map via residues after alignment
        # For that we need post-alignment residues
        # Re-derive aligned residues by inner join indices (align_two_embeddings already filtered)
        # Easiest: reload aligned residues via same merge
        # Build per-token labels for both tasks on the aligned set
        labels_a_aligned = None
        labels_b_aligned = None

        # Helper to get per-token label array on aligned residues
        def _labels_on_aligned(residues_aligned, label_csv, seq_col, lbl_col, spec):
            if label_csv is None:
                return None
            import pandas as pd

            mapping = dict(spec.get("mapping", {})) if isinstance(spec, dict) else {}
            ignore = spec.get("ignore", "") if isinstance(spec, dict) else ""
            ignore_set = set(ignore)
            # Load CSV
            csv = Path(label_csv)
            with open(csv, "r") as f:
                first = f.readline()
            sep = "\t" if first.count("\t") > first.count(",") else ","
            df = pd.read_csv(csv, sep=sep, dtype=str)
            if seq_col not in df.columns or lbl_col not in df.columns:
                raise ValueError(f"CSV missing {seq_col}/{lbl_col}")
            # Build key -> label string
            # Prefer Entry if present in both
            if "Entry" in residues_aligned.columns and "Entry" in df.columns:
                key_to_lbl = dict(zip(df["Entry"].astype(str), df[lbl_col].astype(str).fillna("")))
                key_col = "Entry"
            else:
                # fallback: sequence -> label
                # Use normalized sequence as key; need sequence column
                # This is slower but works for synthetic tests
                from single.data import sequence_hash  # type: ignore
                key_to_lbl = {}
                for _, r in df.iterrows():
                    seq = str(r[seq_col]).upper().strip()
                    key_to_lbl[sequence_hash(seq)] = str(r[lbl_col]) if pd.notna(r[lbl_col]) else ""
                # residues_aligned has sequence_hash
                key_col = "sequence_hash"
            enc = []
            for _, row in residues_aligned.iterrows():
                key = str(row[key_col])
                lbl_str = key_to_lbl.get(key, "")
                pos = int(row["position"])
                ch = lbl_str[pos] if pos < len(lbl_str) else "_"
                if ch in ignore_set:
                    enc.append(-100)
                elif ch in mapping:
                    enc.append(int(mapping[ch]))
                else:
                    enc.append(-100)
            return np.array(enc, dtype=np.int64)

        # Need aligned residues DataFrames
        if residues_a is not None and residues_b is not None:
            import pandas as pd
            # Reconstruct aligned residues frames matching emb_a/emb_b order after align_two_embeddings
            # Do the same merge as io.align_two_embeddings to get ordering
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
                # Preserve merged order as the aligned embeddings order
                # merged is in ra order (since inner); need to sort by _idx_a if subsampled
                # For determinism, keep merged order
                # Now recover aligned residues in merged order
                # Use ra's residues in merged order
                # Build DataFrame with canonical columns
                aligned = ra.iloc[merged["_idx_a"].to_numpy()].reset_index(drop=True)
                # Keep position as int for label indexing
                aligned["position"] = aligned["position"].astype(int).astype(str)
                # But we lost truncation subsampling (max_tokens) – apply same RNG selection
                if args.max_tokens is not None and args.max_tokens < len(aligned):
                    rng = np.random.RandomState(args.seed)
                    idx = rng.choice(len(aligned), size=args.max_tokens, replace=False)
                    idx = np.sort(idx)
                    aligned = aligned.iloc[idx].reset_index(drop=True)
                aligned["position"] = aligned["position"].astype(str)
                # Convert back to int where needed in caller; keep as string then int
                return aligned

            if args.labels_a or args.labels_b:
                aligned_residues = _aligned_residues(residues_a, residues_b)
                # Ensure position is int-ish for slicing
                aligned_residues["position"] = aligned_residues["position"].astype(int).astype(str)
                # But _labels_on_aligned expects position as str convertible to int
                # Fix: keep int
                aligned_residues["position"] = aligned_residues["position"].astype(int)

            # Load specs
            script_base = Path(__file__).resolve().parents[1]
            spec_a = _load_label_spec(args.label_map_a, base_dir=str(script_base)) if args.label_map_a else None
            spec_b = _load_label_spec(args.label_map_b, base_dir=str(script_base)) if args.label_map_b else None

            # Try to use residues->labels mapping; fallback to embeddings sidecar labels if CSV missing
            if args.labels_a and spec_a is not None:
                labels_a_aligned = _labels_on_aligned(aligned_residues, args.labels_a, args.sequence_column_a, args.label_column_a, spec_a)
            if args.labels_b and spec_b is not None:
                labels_b_aligned = _labels_on_aligned(aligned_residues, args.labels_b, args.sequence_column_b, args.label_column_b, spec_b)

            # If CSV not provided but embeddings contain sidecar labels, use those (filtered to aligned indices)
            # Sidecar labels are per-shard; we need to map (not yet). For now, require CSV when not trivial.
            if labels_a_aligned is None or labels_b_aligned is None:
                print("  WARNING: one task has no label CSV – cannot run aligned probe for that task. "
                      "Provide --labels_a/b with --label_map_a/b for full 2x2 matrix.")
                # Try to salvage: if embeddings were extracted with labels sidecar, we could load those
                # But they were for their original tasks only (A sidecar = A labels). Not comparable.
                # So we keep aligned only for tasks where CSV was provided.
        else:
            print("  WARNING: residues.csv missing – cannot residue-align labels to tokens.")

        print(f"  Labels aligned: A {'yes' if labels_a_aligned is not None else 'no'} ({len(labels_a_aligned) if labels_a_aligned is not None else 0}), "
              f"B {'yes' if labels_b_aligned is not None else 'no'} ({len(labels_b_aligned) if labels_b_aligned is not None else 0})")

        # Encode SAE features on aligned embeddings
        print("[Computing SAE features (aligned)]")
        feats_a = _encode_sae(sae_a, emb_a, batch_size=args.batch_size)
        feats_b = _encode_sae(sae_b, emb_b, batch_size=args.batch_size)
        if args.normalize:
            feats_a = feats_a / (np.abs(feats_a).max(axis=0, keepdims=True) + 1e-8)
            feats_b = feats_b / (np.abs(feats_b).max(axis=0, keepdims=True) + 1e-8)

        # Run aligned probe
        from crossing.probe import cross_task_evaluate_aligned, compute_transfer_matrix  # type: ignore

        # Need both label arrays of same length as feats
        if labels_a_aligned is None or labels_b_aligned is None:
            raise ValueError(
                "Aligned mode requires --labels_a and --labels_b (and residues.csv) to provide per-token labels "
                "for BOTH tasks on the aligned token set.  Missing one side -> 2x2 transfer is not definable."
            )
        if len(labels_a_aligned) != feats_a.shape[0] or len(labels_b_aligned) != feats_b.shape[0]:
            raise ValueError(
                f"Label length mismatch after alignment: feats A {feats_a.shape[0]} vs labels_a {len(labels_a_aligned)}, "
                f"feats B {feats_b.shape[0]} vs labels_b {len(labels_b_aligned)}"
            )

        print("[Running aligned cross-task probes (shared train/val split)]")
        results = cross_task_evaluate_aligned(
            feats_a, feats_b, labels_a_aligned, labels_b_aligned,
            val_ratio=args.val_ratio, method=args.method, seed=args.seed,
        )
        results["align_info"] = align_info
        results["mode"] = "aligned"
        results["n_tokens_aligned"] = int(feats_a.shape[0])
        results["sae_a"] = args.sae_a
        results["sae_b"] = args.sae_b
        results["labels_a"] = args.labels_a
        results["labels_b"] = args.labels_b
        results["val_ratio"] = float(args.val_ratio)
        results["method"] = args.method

    else:
        # Disjoint legacy path – per-model token sets differ; use old alignment
        # We reproduce the old per-model truncation but keep the regression
        print("[Disjoint mode: per-model token sets – cross numbers not comparable]")
        # For disjoint we can reuse io load but keep raw
        # Encode SAE features per model independently
        # Need to map labels per model: use CSV per-model truncation (old helper)
        import pandas as pd
        from single.data import sequence_hash  # type: ignore

        script_base = Path(__file__).resolve().parents[1]
        spec_a = _load_label_spec(args.label_map_a, base_dir=str(script_base)) if args.label_map_a else None
        spec_b = _load_label_spec(args.label_map_b, base_dir=str(script_base)) if args.label_map_b else None

        # Load labels trivially: per-protein label string truncated to max_seq_length-2 handled via residues
        # For disjoint we keep the old path: load CSV and encode truncated
        # But we already have residues per model; use them to slice correctly per protein
        def _labels_disjoint(residues, emb, label_csv, seq_col, lbl_col, spec):
            if label_csv is None or spec is None:
                return None, None
            mapping = dict(spec.get("mapping", {}))
            ignore = set(spec.get("ignore", ""))
            csv = Path(label_csv)
            with open(csv, "r") as f:
                first = f.readline()
            sep = "\t" if first.count("\t") > first.count(",") else ","
            df = pd.read_csv(csv, sep=sep, dtype=str)
            if "Entry" in df.columns and residues is not None and "Entry" in residues.columns:
                key_to_lbl = dict(zip(df["Entry"].astype(str), df[lbl_col].astype(str).fillna("")))
                enc = []
                for _, row in residues.iterrows():
                    lbl_str = key_to_lbl.get(str(row["Entry"]), "")
                    pos = int(row["position"])
                    ch = lbl_str[pos] if pos < len(lbl_str) else "_"
                    if ch in ignore:
                        enc.append(-100)
                    elif ch in mapping:
                        enc.append(int(mapping[ch]))
                    else:
                        enc.append(-100)
                return np.array(enc, dtype=np.int64), emb
            else:
                # fallback: hash
                from single.data import sequence_hash  # type: ignore
                key_to_lbl = {}
                for _, r in df.iterrows():
                    seq = str(r[seq_col]).upper().strip()
                    key_to_lbl[sequence_hash(seq)] = str(r[lbl_col]) if pd.notna(r[lbl_col]) else ""
                enc = []
                if residues is not None:
                    for _, row in residues.iterrows():
                        lbl_str = key_to_lbl.get(str(row["sequence_hash"]), "")
                        pos = int(row["position"])
                        ch = lbl_str[pos] if pos < len(lbl_str) else "_"
                        if ch in ignore:
                            enc.append(-100)
                        elif ch in mapping:
                            enc.append(int(mapping[ch]))
                        else:
                            enc.append(-100)
                    return np.array(enc, dtype=np.int64), emb
                else:
                    # No residues: naive per-protein expand truncated
                    # Build per-protein label arrays in CSV order (risky)
                    all_lbl = []
                    for _, r in df.iterrows():
                        seq = str(r[seq_col])
                        lbl_str = str(r[lbl_col]) if pd.notna(r[lbl_col]) else ""
                        seq_len = min(len(seq), args.max_seq_length - 2)
                        lbl_trunc = lbl_str[:seq_len]
                        for ch in lbl_trunc:
                            if ch in ignore:
                                all_lbl.append(-100)
                            elif ch in mapping:
                                all_lbl.append(int(mapping[ch]))
                            else:
                                all_lbl.append(-100)
                    arr = np.array(all_lbl, dtype=np.int64)
                    # truncate to emb length if mismatch
                    if len(arr) != emb.shape[0]:
                        arr = arr[:emb.shape[0]]
                    return arr, emb

        # Truncate residues/emb to max_tokens consistently per model if needed (separate RNG would break comparability but disjoint anyway)
        la, _ = _labels_disjoint(residues_a, emb_a_raw, args.labels_a, args.sequence_column_a, args.label_column_a, spec_a)
        lb, _ = _labels_disjoint(residues_b, emb_b_raw, args.labels_b, args.sequence_column_b, args.label_column_b, spec_b)

        # Subsample
        if args.max_tokens is not None:
            # per-model independent subsample (legacy)
            rng = np.random.RandomState(args.seed)
            if la is not None and len(la) > args.max_tokens:
                idx = np.sort(rng.choice(len(la), size=args.max_tokens, replace=False))
                residues_a = residues_a.iloc[idx] if residues_a is not None else None  # not needed
                # Need to filter emb as well – but we haven't encoded yet
                pass

        print("[Computing SAE features (disjoint)]")
        # For disjoint we encode without alignment
        feats_a_raw = _encode_sae(sae_a, emb_a_raw, batch_size=args.batch_size)
        feats_b_raw = _encode_sae(sae_b, emb_b_raw, batch_size=args.batch_size)
        if args.normalize:
            feats_a_raw = feats_a_raw / (np.abs(feats_a_raw).max(axis=0, keepdims=True) + 1e-8)
            feats_b_raw = feats_b_raw / (np.abs(feats_b_raw).max(axis=0, keepdims=True) + 1e-8)

        # Filter -100 before probe (cross_task_evaluate handles but do here for consistency)
        # Keep per-model valid tokens
        def _valid_filter(feats, labels):
            if labels is None:
                return feats, labels
            valid = labels != -100
            return feats[valid], labels[valid]

        feats_a_f, la_f = _valid_filter(feats_a_raw, la)
        feats_b_f, lb_f = _valid_filter(feats_b_raw, lb)

        # If disjoint and labels differ in token count, subsample to same count for legacy cross (flagged)
        from crossing.probe import cross_task_evaluate  # type: ignore
        # cross_task_evaluate will handle disjoint warning and attempt cross
        # It expects feats/labels per model possibly different n_tokens – that's ok in disjoint branch
        results = cross_task_evaluate(
            feats_a_f, la_f if la_f is not None else np.zeros(feats_a_f.shape[0], dtype=np.int64),
            feats_b_f, lb_f if lb_f is not None else np.zeros(feats_b_f.shape[0], dtype=np.int64),
            val_ratio=args.val_ratio, method=args.method, seed=args.seed,
        )
        results["mode"] = "disjoint"
        results["sae_a"] = args.sae_a
        results["sae_b"] = args.sae_b

    # Print transfer matrix
    try:
        from crossing.probe import compute_transfer_matrix  # type: ignore
        mat = compute_transfer_matrix(results)
        print("\n  Transfer matrix (rows=train task, cols=test task) val_f1:")
        print(f"             Task A    Task B")
        print(f"  Trained A  {mat[0,0]:.4f}    {mat[0,1]:.4f}")
        print(f"  Trained B  {mat[1,0]:.4f}    {mat[1,1]:.4f}")
        results["transfer_matrix"] = mat.tolist()
    except Exception:
        pass

    for direction in ["A_to_A", "A_to_B", "B_to_B", "B_to_A"]:
        if direction in results and isinstance(results[direction], dict) and "val_f1" in results[direction]:
            m = results[direction]
            print(f"  {direction}: F1={m['val_f1']:.4f} Acc={m.get('val_accuracy', 0):.4f} AUROC={m.get('val_auroc', 0):.4f}")

    if results.get("mode") == "disjoint" or not results.get("aligned", True):
        print("\n  NOTE: disjoint embeddings – A_to_B / B_to_A are not comparable to baselines. "
              "For publishable information-transfer, provide overlapping embeddings + labels_a/b.")

    with open(output_dir / "cross_task_probe.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[Done] Saved to {output_dir / 'cross_task_probe.json'}")


if __name__ == "__main__":
    main()
