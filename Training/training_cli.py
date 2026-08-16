#!/usr/bin/env python3
"""CrossPLM Training CLI (invoked via the unified `crossplm training` command).

  python crossplm.py training init --task_name my_experiment
  python crossplm.py training train --config Outputs/my_experiment/config.yaml
  python crossplm.py training eval --checkpoint Outputs/my_experiment/checkpoints/xxx --csv Dataset/mBMRB.csv
"""
import os
import sys
import argparse
from pathlib import Path


def _load_label_map(name):
    """Load a label-map spec (preset name or YAML file path).

    Uses the SAME label maps as the interpretability module
    (Single/single/label_maps.py, made importable by training/__init__.py) so
    Training and Single interpret a dataset's per-residue labels identically.
    The spec carries the sequence/label column names, the char->class mapping,
    and ignore characters.

    Relative YAML paths are resolved against the Training module directory (like
    csv_data_path), so the result does not depend on the current working dir.
    """
    from single.label_maps import get_label_map

    p = Path(name)
    if p.suffix in {".yaml", ".yml"} and not p.is_absolute():
        p = Path(os.path.dirname(os.path.abspath(__file__))) / p
    return get_label_map(str(p))


def cmd_labelmap(args):
    """Create an empty label-map YAML template in the repo's Dataset/ dir."""
    import training  # noqa: F401  (training/__init__.py puts Single/ on sys.path)
    from single.label_maps import generate_template

    if args.output:
        out = Path(args.output)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out = Path(script_dir).parent / "Dataset" / f"{args.name}.yaml"
    path = generate_template(out)
    rel = os.path.relpath(path, os.path.dirname(os.path.abspath(__file__)))
    print(f"[Label Map Template] {path}")
    print(f"Edit it, then reference it via config 'label_map: {rel}' or "
          f"'--label_map {rel}' (relative to Training/).")


def cmd_init(args):
    from training import TrainingConfig, create_task_folder

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Shared top-level output root (same as the interpretability module):
    # <repo>/Outputs/<task_name>/  (verbatim, no timestamp)
    output_dir = os.path.join(script_dir, "..", "Outputs")
    task_dir = create_task_folder(output_dir, args.task_name)

    config_path = os.path.join(task_dir, "config.yaml")
    if os.path.exists(config_path):
        print(f"[WARNING] {task_dir} already exists — overwriting config.yaml")
    TrainingConfig.generate_template(config_path)

    print(f"[Task Created] {task_dir}")
    print(f"[Config Template] {config_path}")
    print(f"Edit the config, then run (from the repository root):")
    print(f"  python crossplm.py training train --config {config_path}")


def cmd_train(args):
    import random
    import numpy as np
    import torch

    from training import (
        TrainingConfig, TokenClassificationDataset, load_data_from_csv,
        split_dataset, build_label_map, build_id2label, label_map_n_classes,
        compute_class_weights, PLMModel, Trainer,
    )

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    config = TrainingConfig.from_yaml(args.config)
    set_seed(config.seed)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(args.config)
    task_dir = os.path.dirname(config_path)

    print(f"[Config Loaded] {config_path}")
    print(f"[Task Dir] {task_dir}")

    csv_path = config.csv_data_path
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(script_dir, csv_path)

    # Unified label map (same as Single): a preset name ("mBMRB", "relaxdb",
    # "ss3") or a YAML file. The spec can also define which columns hold the
    # sequence and the label string. Without it we fall back to the config's
    # columns and infer the map from the CSV (legacy behavior).
    spec = None
    label_map = None
    if config.label_map:
        spec = _load_label_map(config.label_map)
        label_map = dict(spec["mapping"])
        seq_col = spec.get("sequence_column") or config.sequence_column
        lbl_col = spec.get("label_column") or config.label_column
        print(f"  Label map (from '{config.label_map}'): {label_map}")
    else:
        seq_col, lbl_col = config.sequence_column, config.label_column

    print(f"[Loading CSV] {csv_path}")
    sequences, labels = load_data_from_csv(
        csv_path,
        sequence_column=seq_col,
        label_column=lbl_col,
    )
    print(f"  Total samples: {len(sequences)}")
    if len(sequences) == 0:
        print("ERROR: No valid data loaded. Check CSV path and column names.")
        sys.exit(1)

    if label_map is None:
        label_map = build_label_map(labels)
        print(f"  Label map (inferred from CSV): {label_map}")

    print(f"[Loading backbone model] {config.backbone_model_id}")
    n_classes = label_map_n_classes(label_map)
    plm = PLMModel(
        backbone_model_id=config.backbone_model_id,
        task_type=config.task_type,
        num_labels=n_classes,
    )

    # Persist the training-time label map into the model config so downstream
    # evaluation can reuse the exact same label -> id mapping instead of
    # rebuilding it from the eval CSV (which may have different characters).
    # build_id2label dedupes by class id (many-to-one char maps like mBMRB).
    plm.config.id2label = build_id2label(label_map)
    plm.config.label2id = {c: i for c, i in label_map.items()}

    class_weights = compute_class_weights(labels, label_map, config.class_weight_method)

    print(f"[Building dataset...]")
    full_dataset = TokenClassificationDataset(
        sequences=sequences,
        labels=labels,
        tokenizer=plm.tokenizer,
        max_length=config.max_seq_length,
        label_map=label_map,
    )
    print(f"  Model params: {plm.get_num_params():,}")

    train_dataset, eval_dataset = split_dataset(full_dataset, config.train_ratio, config.seed)
    print(f"  Train: {len(train_dataset)}  Eval: {len(eval_dataset)}")

    print("[Training...]")
    trainer = Trainer(
        model=plm,
        tokenizer=plm.tokenizer,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        task_dir=task_dir,
        class_weights=class_weights,
        n_classes=n_classes,
    )
    trainer.train()
    print(f"[Done] -> {task_dir}")


def cmd_eval(args):
    import json
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from training.data.dataset import TokenClassificationDataset, load_data_from_csv, build_label_map, label_map_n_classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.abspath(args.checkpoint)
    csv_path = os.path.abspath(args.csv)

    print(f"[Loading checkpoint] {ckpt_path}")
    # The fine-tuned backbones (e.g. Synthyra/ESM2-8M via FastPLMs) ship custom
    # code, so trust_remote_code=True is required to load their checkpoints.
    tokenizer = AutoTokenizer.from_pretrained(
        ckpt_path, local_files_only=True, trust_remote_code=True,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        ckpt_path, local_files_only=True, trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"

    print(f"[Loading CSV] {csv_path}")
    seq_col = args.sequence_column
    lbl_col = args.label_column

    # Explicit --label_map (preset/YAML, same as Single) wins; it may also define
    # the CSV columns. Otherwise use the training-time map persisted in the
    # checkpoint, falling back to the eval CSV only for legacy checkpoints.
    label_map = None
    if args.label_map:
        spec = _load_label_map(args.label_map)
        label_map = dict(spec["mapping"])
        seq_col = spec.get("sequence_column") or seq_col
        lbl_col = spec.get("label_column") or lbl_col
        print(f"  Label map (from '{args.label_map}'): {label_map}")

    sequences, labels = load_data_from_csv(csv_path, sequence_column=seq_col, label_column=lbl_col)
    print(f"  samples: {len(sequences)}")

    if label_map is None:
        # Priority: sidecar label_map.json (written by train) > model.config.label2id
        # > rebuild from the eval CSV (legacy). The FastEsm runtime drops
        # num_labels/id2label/label2id from config.json, so the sidecar is the
        # reliable source for the training-time mapping.
        sidecar = os.path.join(ckpt_path, "label_map.json")
        if os.path.exists(sidecar):
            with open(sidecar) as f:
                lm = json.load(f)
            label_map = {str(k): int(v) for k, v in lm.get("label2id", {}).items()}
            print(f"  Label map (from checkpoint label_map.json): {label_map}")
        else:
            # label2id is the complete {char: class_id} map persisted at training time
            # (many chars may map to one class), so it is the correct source.
            # id2label is only class_id -> a single representative char and is NOT
            # sufficient to re-derive the full mapping.
            cfg_label2id = getattr(model.config, "label2id", None)
            # Guard against HF placeholder maps like {"LABEL_0":0, "LABEL_1":1}, which
            # are auto-generated and are not the training-time mapping.
            is_hf_placeholder = (
                cfg_label2id is not None
                and all(str(k).startswith("LABEL_") for k in cfg_label2id)
            )
            if cfg_label2id and not is_hf_placeholder and all(str(v) != "None" for v in cfg_label2id.values()):
                label_map = {str(k): int(v) for k, v in cfg_label2id.items()}
                print(f"  Label map (from checkpoint): {label_map}")
            else:
                label_map = build_label_map(labels)
                print(f"  Label map (rebuilt from eval CSV): {label_map}")

    n_classes = label_map_n_classes(label_map)
    dataset = TokenClassificationDataset(
        sequences=sequences, labels=labels, tokenizer=tokenizer,
        max_length=args.max_seq_length, label_map=label_map,
    )

    def collate_fn(batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels_batch = [item["labels"] for item in batch]
        padded = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True, return_tensors="pt",
        )
        max_len = padded["input_ids"].size(1)
        padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, lbl in enumerate(labels_batch):
            padded_labels[i, :len(lbl)] = lbl
        return {"input_ids": padded["input_ids"], "attention_mask": padded["attention_mask"], "labels": padded_labels}

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    print("[Evaluating...]")
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits, dim=-1)
            n_classes = probs.size(-1)
            for i in range(logits.size(0)):
                mask = batch["labels"][i] != -100
                all_preds.extend(logits[i][mask].argmax(dim=-1).cpu().tolist())
                all_labels.extend(batch["labels"][i][mask].cpu().tolist())
                all_probs.extend(probs[i][mask].cpu().tolist())  # keep ALL class probs

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    # n_classes is computed above from label_map_n_classes(label_map).
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", labels=list(range(n_classes)), zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(n_classes)))

    # AUPRC: macro-avg of per-class PR-AUC (handles binary and >2 classes).
    # We also keep the last per-class (precision, recall) curves for plotting.
    from sklearn.metrics import precision_recall_curve, auc
    auprc = 0.0
    per_class_pr = []  # list of (precision, recall) arrays for plotting
    try:
        for cls in range(n_classes):
            y_bin = [1 if l == cls else 0 for l in all_labels]
            p, r, _ = precision_recall_curve(y_bin, [prob[cls] for prob in all_probs])
            auprc += auc(r, p)
            per_class_pr.append((p, r))
        auprc /= n_classes
    except Exception as e:
        print(f"  (AUPRC skipped: {e})")
        auprc = 0.0

    print(f"\n  Accuracy:  {accuracy:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  AUPRC (macro): {auprc:.4f}")
    print(f"  Num classes: {n_classes}")
    print("\n  Confusion Matrix (rows=actual, cols=pred):")
    print("        " + " ".join(f"Pred{i:>5}" for i in range(n_classes)))
    for i in range(n_classes):
        print(f"  Act{i:>3}  " + " ".join(f"{cm[i][j]:>9d}" for j in range(n_classes)))

    metrics = {"accuracy": round(accuracy, 4), "f1_macro": round(f1, 4),
               "auprc_macro": round(auprc, 4), "n_classes": n_classes,
               "confusion_matrix": cm.tolist(), "n_positions": len(all_labels),
               "checkpoint": args.checkpoint, "csv": args.csv}

    output_dir = args.output
    if output_dir is None:
        # Default: <experiment>/evaluations/<csv_name>/  (the checkpoint is
        # <experiment>/checkpoints/<ckpt>, so the experiment dir is two levels up).
        ckpt_dir = os.path.dirname(os.path.normpath(args.checkpoint))
        task_dir = os.path.dirname(ckpt_dir)
        csv_name = os.path.splitext(os.path.basename(args.csv))[0]
        output_dir = os.path.join(task_dir, "evaluations", csv_name)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Confusion matrix (supports any n_classes >= 1)
        plt.figure(figsize=(max(4, 1.2 * n_classes), max(3, 1.0 * n_classes)))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix", fontsize=14)
        plt.colorbar()
        cm_max = cm.max() if cm.size else 1
        for i in range(n_classes):
            for j in range(n_classes):
                plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                         color="white" if cm[i, j] > cm_max / 2 else "black")
        tick_labels = [str(i) for i in range(n_classes)]
        plt.xticks(range(n_classes), tick_labels, fontsize=12)
        plt.yticks(range(n_classes), tick_labels, fontsize=12)
        plt.xlabel("Predicted", fontsize=13)
        plt.ylabel("True", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

        # Precision-Recall curve: one curve per class (macro-AUPRC in legend)
        plt.figure(figsize=(7, 5))
        if per_class_pr:
            for cls, (p, r) in enumerate(per_class_pr):
                plt.plot(r, p, linewidth=1.5, label=f"class {cls}")
            plt.plot([], [], " ", label=f"macro-AUPRC = {auprc:.4f}")
        plt.xlabel("Recall", fontsize=13)
        plt.ylabel("Precision", fontsize=13)
        plt.title("Precision-Recall Curve", fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "auprc_curve.png"), dpi=150)
        plt.close()
    except ImportError:
        pass

    print(f"  Results saved to {output_dir}/")

    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description="CrossPLM Training CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create task folder and config template")
    p_init.add_argument("--task_name", type=str, required=True)

    p_labelmap = sub.add_parser("labelmap",
                                help="Create an empty label-map YAML template (Dataset/)")
    p_labelmap.add_argument("--name", type=str, required=True,
                            help="Base name, e.g. my_dataset -> Dataset/my_dataset.yaml")
    p_labelmap.add_argument("--output", type=str, default=None,
                            help="Explicit output path (default: Dataset/<name>.yaml)")

    p_train = sub.add_parser("train", help="Start training from a config.yaml")
    p_train.add_argument("--config", type=str, required=True)

    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint on a CSV")
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--csv", type=str, required=True)
    p_eval.add_argument("--output", type=str, default=None,
                        help="Output directory (default: <experiment>/evaluations/<csv_name>)")
    p_eval.add_argument("--batch_size", type=int, default=8)
    p_eval.add_argument("--max_seq_length", type=int, default=512)
    p_eval.add_argument("--label_map", type=str, default=None,
                        help="Label-map preset (mBMRB/relaxdb/ss3) or YAML file, "
                             "same as the interpretability module. Overrides the "
                             "checkpoint's persisted label map.")
    p_eval.add_argument("--sequence_column", type=str, default="sequence")
    p_eval.add_argument("--label_column", type=str, default="label")

    args = parser.parse_args(argv)

    if args.command == "init":
        cmd_init(args)
    elif args.command == "labelmap":
        cmd_labelmap(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
