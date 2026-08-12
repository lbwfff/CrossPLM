#!/usr/bin/env python3
import os
import sys
import argparse

from config import TrainingConfig
from utils import create_task_folder


def cmd_generate(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "tasks")
    task_dir = create_task_folder(output_dir, args.task_name)

    config_path = os.path.join(task_dir, "config.yaml")
    TrainingConfig.generate_template(config_path)

    print(f"[Task Created] {task_dir}")
    print(f"[Config Template] {config_path}")
    print(f"Please edit the config.yaml, then run:")
    print(f"  python launch.py --config {config_path}")


def cmd_train(args):
    import random
    import numpy as np
    import torch

    from data import TokenClassificationDataset, load_data_from_csv, split_dataset, build_label_map, compute_class_weights
    from models import PLMModel
    from trainers import Trainer

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
        print("ERROR: No valid data loaded. Check your CSV file and column names.")
        sys.exit(1)

    label_map = build_label_map(labels)
    print(f"  Label map: {label_map}")
    print(f"  Num classes: {len(label_map)}")

    print(f"[Loading backbone model] {config.backbone_model_id}")
    plm = PLMModel(
        backbone_model_id=config.backbone_model_id,
        task_type=config.task_type,
        num_labels=len(label_map),
    )

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

    class_weights = compute_class_weights(labels, label_map, config.class_weight_method)

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


def main():
    parser = argparse.ArgumentParser(description="CrossPLM Training Launcher")
    parser.add_argument("--task_name", type=str, default=None,
                        help="Create task folder + config template")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml, start training")

    args = parser.parse_args()

    if args.task_name:
        cmd_generate(args)
    elif args.config:
        cmd_train(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
