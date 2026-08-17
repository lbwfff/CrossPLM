from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SAEConfig:
    activation_dim: int = 320
    dict_size: int = 1280
    expansion_factor: int = 4
    normalize_to_sqrt_d: bool = False

    def __post_init__(self):
        if self.dict_size is None and self.expansion_factor is not None:
            self.dict_size = self.expansion_factor * self.activation_dim


@dataclass
class TrainingConfig:
    activation_dim: int = 320
    dict_size: int = 1280
    expansion_factor: int = 4
    lr: float = 2e-4
    steps: int = 10_000
    batch_size: int = 512
    warmup_steps: int = 1000
    decay_start: int = 8000
    l1_penalty: float = 0.06
    reconstruction_loss: str = "l2"  # l2 (legacy) or mse
    l1_penalty_warmup_steps: int = 500
    resample_steps: Optional[int] = None
    grad_clip_norm: float = 1.0
    normalize_to_sqrt_d: bool = False
    seed: int = 42


@dataclass
class DataConfig:
    plm_embd_dir: Path
    eval_embd_dir: Optional[Path] = None
    n_shards_to_include: Optional[int] = None
    samples_to_skip: int = 0
    target_dtype: str = "float32"
    shard: Optional[int] = None  # None = use all shards; else only that shard


@dataclass
class EvalConfig:
    eval_seq_path: Optional[Path] = None
    eval_labels_path: Optional[Path] = None
    eval_steps: int = 500
    eval_batch_size: int = 8


@dataclass
class CheckpointConfig:
    save_dir: Path
    save_steps: int = 2000
    max_ckpts_to_keep: int = 3


@dataclass
class AnalysisConfig:
    activation_threshold: float = 0.05
    n_top_proteins: int = 10
    feature_chunk_size: int = 200


@dataclass
class EmbedderConfig:
    ckpt_path: Path
    model_name: str = "Synthyra/ESM2-8M"
    layer: int = 6
    batch_size: int = 8
    max_length: int = 512
    device: Optional[str] = None
