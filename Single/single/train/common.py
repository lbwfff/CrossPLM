"""ConstrainedAdam optimizer and LR scheduling utilities."""

from typing import Callable, Optional

import torch as t


class ConstrainedAdam(t.optim.Adam):
    """
    Adam variant that maintains unit-norm constraints on decoder weights.
    Projects away gradient components parallel to weight directions,
    then renormalizes after each step.
    """

    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr)
        self.constrained_params = list(constrained_params)

    def step(self, closure=None):
        with t.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=0, keepdim=True)
                p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure)
        with t.no_grad():
            for p in self.constrained_params:
                p /= p.norm(dim=0, keepdim=True)


def get_lr_schedule(
    total_steps: int,
    warmup_steps: int,
    decay_start: Optional[int] = None,
) -> Callable[[int], float]:
    if decay_start is not None:
        assert decay_start > warmup_steps, "decay_start must be > warmup_steps"

    def lr_schedule(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        if decay_start is not None and step >= decay_start:
            if decay_start >= total_steps:
                return 1.0
            return (total_steps - step) / (total_steps - decay_start)
        return 1.0

    return lr_schedule


def get_sparsity_warmup_fn(
    total_steps: int, sparsity_warmup_steps: Optional[int] = None
) -> Callable[[int], float]:
    if sparsity_warmup_steps is None:
        sparsity_warmup_steps = 0

    def scale_fn(step: int) -> float:
        if sparsity_warmup_steps == 0:
            return 1.0
        return min(step / sparsity_warmup_steps, 1.0)

    return scale_fn
