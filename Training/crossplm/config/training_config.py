from dataclasses import dataclass, asdict
from typing import List, Optional


@dataclass
class TrainingConfig:
    task_name: str = "my_plm_task"
    model_name: str = "esm2_t6_8M"
    backbone_model_id: str = "facebook/esm2_t6_8M_UR50D"
    csv_data_path: str = ""
    sequence_column: str = "sequence"
    label_column: str = "label"
    train_ratio: float = 0.9

    task_type: str = "token_classification"
    max_seq_length: int = 512
    mlm_probability: float = 0.15

    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear"
    warmup_ratio: float = 0.06
    num_train_epochs: int = 3
    max_steps: int = -1

    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    seed: int = 42
    dataloader_num_workers: int = 2
    fp16: bool = False
    bf16: bool = False
    class_weight_method: str = "inverse"
    ignore_pad_token_for_loss: bool = True
    resume_from_checkpoint: Optional[str] = None

    def __post_init__(self):
        for field_name, field_type in self.__annotations__.items():
            if field_type is int:
                setattr(self, field_name, int(getattr(self, field_name)))
            elif field_type is float:
                setattr(self, field_name, float(getattr(self, field_name)))
            elif field_type is bool:
                if isinstance(getattr(self, field_name), str):
                    setattr(self, field_name, getattr(self, field_name).lower() in ("true", "1", "yes"))

    def to_yaml(self, path: str):
        import yaml
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        import yaml
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**{k: v for k, v in raw.items() if not k.startswith("#") and v is not None})

    @staticmethod
    def generate_template(path: str):
        template = """# ============================================================
# CrossPLM Training Config Template
# ============================================================
# Edit this file, then run:
#   python crossplm.py train --config <path>
# ============================================================

# Task name (for identification only)
task_name: my_plm_task

# Name used when saving the model
model_name: esm2_t6_8M

# HuggingFace backbone model ID
backbone_model_id: facebook/esm2_t6_8M_UR50D

# Path to training CSV (required, relative or absolute)
csv_data_path: ./examples/sample.csv

# Column name for sequences in the CSV
sequence_column: sequence

# Column name for per-residue labels in the CSV
label_column: label

# Train/eval split ratio (0~1), the rest is used for eval
train_ratio: 0.9

# Task type: token_classification | mlm
task_type: token_classification

# Max sequence length; longer sequences are truncated
max_seq_length: 512

# Only used when task_type=mlm, mask probability
mlm_probability: 0.15

# ----- Training hyperparameters -----
per_device_train_batch_size: 8
per_device_eval_batch_size: 8
gradient_accumulation_steps: 1
learning_rate: 2.0e-5
weight_decay: 0.01
num_train_epochs: 3
max_steps: -1           # -1 means determined by num_train_epochs

# ----- Logging & saving -----
logging_steps: 10
eval_steps: 500
save_steps: 1000
save_total_limit: 3

# ----- Class weights (address imbalance) -----
#   none     = no weighting
#   inverse  = inverse frequency normalization (recommended)
#   log      = log scaling (gentler)
class_weight_method: inverse

# ----- Other -----
seed: 42
dataloader_num_workers: 2
fp16: false
bf16: false
"""
        with open(path, "w") as f:
            f.write(template.lstrip())
