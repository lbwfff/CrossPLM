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
# 请根据实际需求修改此文件，然后运行:
#   python launch.py --config <此文件路径>
# ============================================================

# 任务名称（仅用于标识）
task_name: my_plm_task

# 保存模型时的名字
model_name: esm2_t6_8M

# HuggingFace 上的 backbone 模型 ID
backbone_model_id: facebook/esm2_t6_8M_UR50D

# 训练数据 CSV 文件路径（必填，支持相对路径或绝对路径）
csv_data_path: ./examples/sample.csv

# CSV 中序列列的列名
sequence_column: sequence

# CSV 中标签列的列名（每个字符对应一个氨基酸位置的标签）
label_column: label

# 训练集划分比例（0~1），剩余为验证集
train_ratio: 0.9

# 任务类型: token_classification | mlm
task_type: token_classification

# 序列最大长度，超过则截断
max_seq_length: 512

# 仅在 task_type=mlm 时生效，掩码比例
mlm_probability: 0.15

# ----- 训练超参数 -----
per_device_train_batch_size: 8
per_device_eval_batch_size: 8
gradient_accumulation_steps: 1
learning_rate: 2.0e-5
weight_decay: 0.01
num_train_epochs: 3
max_steps: -1           # -1 表示由 num_train_epochs 决定

# ----- 日志与保存 -----
logging_steps: 10
eval_steps: 500
save_steps: 1000
save_total_limit: 3

# ----- 类别权重（缓解不平衡） -----
#   none     = 不加权
#   inverse  = 逆频率归一化（推荐）
#   log      = 对数缩放（更温和）
class_weight_method: inverse

# ----- 其他 -----
seed: 42
dataloader_num_workers: 2
fp16: false
bf16: false
"""
        with open(path, "w") as f:
            f.write(template.lstrip())
