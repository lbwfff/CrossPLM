from abc import ABC, abstractmethod
from pathlib import Path

import torch as t


class SAETrainer(ABC):
    def __init__(self, trainer_config, logging_parameters=None):
        self.config = trainer_config
        self.logging_parameters = logging_parameters or []

    @abstractmethod
    def loss(self, x, step=None, logging=False):
        pass

    @abstractmethod
    def update(self, step, x):
        pass

    @property
    def current_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def save_checkpoint(self, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        t.save(self.ae.state_dict(), save_dir / "checkpoint.pt")
        optimizer_state = {
            "optimizer": self.optimizer.state_dict() if hasattr(self, "optimizer") else None,
            "scheduler": self.scheduler.state_dict() if hasattr(self, "scheduler") else None,
        }
        t.save(optimizer_state, save_dir / "optimizer.pt")

    def update_from_checkpoint(self, checkpoint_dir: Path):
        self.ae.load_state_dict(t.load(checkpoint_dir / "checkpoint.pt"))
        optimizer_state = t.load(checkpoint_dir / "optimizer.pt")
        self.optimizer.load_state_dict(optimizer_state["optimizer"])
        self.scheduler.load_state_dict(optimizer_state["scheduler"])
