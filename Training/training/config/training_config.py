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
    mlm_probability: float = 0.15  # reserved (MLM task not implemented)

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
    fp16: bool = False  # mixed precision (AMP) — requires a CUDA GPU
    bf16: bool = False  # bfloat16 AMP — requires a CUDA GPU (Ampere+)
    class_weight_method: str = "inverse"
    ignore_pad_token_for_loss: bool = True
    resume_from_checkpoint: Optional[str] = None  # reserved (not implemented)
    # Optional label-map preset (from Single's single/label_maps.py) or a path to
    # a YAML label-map file. This is the SAME label encoding used by the
    # interpretability module, so Training and Single always interpret a dataset
    # identically. A preset/YAML may also define the sequence/label CSV columns;
    # when set it takes precedence over sequence_column/label_column below.
    # E.g. "mBMRB" for {A:0, .:1}, "relaxdb" for the raw relaxdb chars, "ss3"
    # for 3-class secondary structure. Leave empty to infer the map from the CSV.
    label_map: str = ""

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
        # Note: YAML comments are already stripped by the parser, so only drop
        # explicitly-null values (YAML allows `key:` with no value).
        return cls(**{k: v for k, v in raw.items() if v is not None})

    @staticmethod
    def generate_template(path: str):
        template = """# ============================================================
# CrossPLM Training Config Template
# ============================================================
# Edit this file, then run:
#   python crossplm.py training train --config <path>
# ============================================================

# Task name (for identification only)
task_name: my_plm_task

# Name used when saving the model
model_name: esm2_t6_8M

# HuggingFace backbone model ID
backbone_model_id: facebook/esm2_t6_8M_UR50D

# Path to training CSV (required). Relative paths are resolved against the
# Training/ module directory (not the CWD), so "../Dataset/mBMRB.csv" means the
# repo's Dataset/ folder no matter where you run the command from.
# You can point directly at a RAW dataset (e.g. ../Dataset/mBMRB.csv) — no
# separate preprocessing step is needed when a label_map is set below.
csv_data_path: ../Dataset/mBMRB.csv

# Unified label map (same as the interpretability module):
#   a preset name: mBMRB | relaxdb | ss3   (from Single/single/label_maps.py)
#   or a path to a YAML file.
# The preset/YAML defines which columns hold the sequence and label strings,
# how each label character maps to a class, and which characters are ignored.
# Leave empty to infer the mapping from the CSV (legacy).
label_map: mBMRB

# Column names (only used when label_map is empty, or to override a preset's)
sequence_column: sequence
label_column: label

# Train/eval split ratio (0~1), the rest is used for eval
train_ratio: 0.9

# Task type: token_classification (mlm not implemented)
task_type: token_classification

# Max sequence length; longer sequences are truncated
max_seq_length: 512

# Reserved for a future MLM task (currently unused)
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
#   inverse  = inverse frequency normalization (aggressive, can over-predict the
#              rare class -> high recall / low precision)
#   sqrt     = sqrt of inverse ratio (gentler; try this if precision is low)
#   log      = log scaling (gentler)
class_weight_method: inverse

# ----- Other -----
seed: 42
dataloader_num_workers: 2
fp16: false          # mixed precision (AMP) — requires a CUDA GPU
bf16: false          # bfloat16 AMP — requires a CUDA GPU (Ampere+); prefer over fp16
"""
        with open(path, "w") as f:
            f.write(template.lstrip())
