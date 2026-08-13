#!/usr/bin/env python3
"""CrossPLM Training CLI

Usage:
  python crossplm.py init --task_name my_experiment
  python crossplm.py train --config outputs/tasks/my_experiment_<ts>/config.yaml
  python crossplm.py eval --checkpoint outputs/tasks/xxx/checkpoints/xxx --csv ./data.csv
"""
import os
import sys
import argparse


def cmd_init(args):
    from crossplm import TrainingConfig, create_task_folder

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs", "tasks")
    task_dir = create_task_folder(output_dir, args.task_name)

    config_path = os.path.join(task_dir, "config.yaml")
    TrainingConfig.generate_template(config_path)

    print(f"[Task Created] {task_dir}")
    print(f"[Config Template] {config_path}")
    print(f"Edit the config, then run:")
    print(f"  python crossplm.py train --config {config_path}")


def cmd_train(args):
    import random
    import numpy as np
    import torch

    from crossplm import (
        TrainingConfig, TokenClassificationDataset, load_data_from_csv,
        split_dataset, build_label_map, compute_class_weights, PLMModel, Trainer,
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

    print(f"[Loading CSV] {csv_path}")
    sequences, labels = load_data_from_csv(
        csv_path,
        sequence_column=config.sequence_column,
        label_column=config.label_column,
    )
    print(f"  Total samples: {len(sequences)}")
    if len(sequences) == 0:
        print("ERROR: No valid data loaded. Check CSV path and column names.")
        sys.exit(1)

    label_map = build_label_map(labels)
    print(f"  Label map: {label_map}")

    print(f"[Loading backbone model] {config.backbone_model_id}")
    plm = PLMModel(
        backbone_model_id=config.backbone_model_id,
        task_type=config.task_type,
        num_labels=len(label_map),
    )

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
    )
    trainer.train()
    print(f"[Done] -> {task_dir}")


def cmd_eval(args):
    import json
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from crossplm.data.dataset import TokenClassificationDataset, load_data_from_csv, build_label_map

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = os.path.abspath(args.checkpoint)
    csv_path = os.path.abspath(args.csv)

    print(f"[Loading checkpoint] {ckpt_path}")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path, local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(ckpt_path, local_files_only=True)
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<pad>"

    print(f"[Loading CSV] {csv_path}")
    sequences, labels = load_data_from_csv(csv_path)
    print(f"  samples: {len(sequences)}")

    label_map = build_label_map(labels)
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
            for i in range(logits.size(0)):
                mask = batch["labels"][i] != -100
                all_preds.extend(logits[i][mask].argmax(dim=-1).cpu().tolist())
                all_labels.extend(batch["labels"][i][mask].cpu().tolist())
                all_probs.extend(probs[i][mask, 1].cpu().tolist())

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_curve, auc

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    precision, recall, _ = precision_recall_curve(all_labels, all_probs, pos_label=1)
    auprc = auc(recall, precision)

    print(f"\n  Accuracy:  {accuracy:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  AUPRC:      {auprc:.4f}")
    print(f"\n  Confusion Matrix:               Pred 0   Pred 1")
    print(f"                    Actual 0    {cm[0][0]:>7d}  {cm[0][1]:>7d}")
    print(f"                    Actual 1    {cm[1][0]:>7d}  {cm[1][1]:>7d}")

    metrics = {"accuracy": round(accuracy, 4), "f1_macro": round(f1, 4), "auprc": round(auprc, 4),
               "confusion_matrix": cm.tolist(), "n_positions": len(all_labels),
               "checkpoint": args.checkpoint, "csv": args.csv}

    output_dir = args.output
    if output_dir is None:
        ckpt_dir = os.path.dirname(os.path.normpath(args.checkpoint))
        csv_name = os.path.splitext(os.path.basename(args.csv))[0]
        output_dir = os.path.join(ckpt_dir, f"eval_on_{csv_name}")
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title("Confusion Matrix", fontsize=14)
        plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.xticks([0, 1], ["0", "1"], fontsize=12)
        plt.yticks([0, 1], ["0", "1"], fontsize=12)
        plt.xlabel("Predicted", fontsize=13)
        plt.ylabel("True", fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(recall, precision, linewidth=2, label=f"AUPRC = {auprc:.4f}")
        plt.fill_between(recall, precision, alpha=0.15)
        plt.xlabel("Recall", fontsize=13)
        plt.ylabel("Precision", fontsize=13)
        plt.title("Precision-Recall Curve", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "auprc_curve.png"), dpi=150)
        plt.close()
    except ImportError:
        pass

    print(f"  Results saved to {output_dir}/")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="CrossPLM Training CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="Create task folder and config template")
    p_init.add_argument("--task_name", type=str, required=True)

    p_train = sub.add_parser("train", help="Start training from a config.yaml")
    p_train.add_argument("--config", type=str, required=True)

    p_eval = sub.add_parser("eval", help="Evaluate a checkpoint on a CSV")
    p_eval.add_argument("--checkpoint", type=str, required=True)
    p_eval.add_argument("--csv", type=str, required=True)
    p_eval.add_argument("--output", type=str, default=None,
                        help="Output directory (default: <checkpoint_dir>/eval_on_<csv_name>)")
    p_eval.add_argument("--batch_size", type=int, default=8)
    p_eval.add_argument("--max_seq_length", type=int, default=512)

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
