from collections import namedtuple

import torch as t

from single.sae.dictionary import ReLUSAE
from single.train.trainers import SAETrainer
from single.train.common import ConstrainedAdam, get_lr_schedule, get_sparsity_warmup_fn


class ReLUTrainer(SAETrainer):
    def __init__(self, trainer_config):
        super().__init__(
            trainer_config=trainer_config,
            logging_parameters=["training/learning_rate", "training/l1_penalty"],
        )
        cfg = trainer_config

        self.ae = ReLUSAE(
            activation_dim=cfg.activation_dim,
            dict_size=cfg.dict_size,
            normalize_to_sqrt_d=cfg.normalize_to_sqrt_d,
        )

        self.lr = cfg.lr
        self.steps = cfg.steps
        self.warmup_steps = cfg.warmup_steps
        self.decay_start = cfg.decay_start
        self.grad_clip_norm = cfg.grad_clip_norm

        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.ae.to(self.device)

        if cfg.l1_penalty_warmup_steps is None:
            cfg.l1_penalty_warmup_steps = int(self.steps * 0.05)

        self.resample_steps = cfg.resample_steps
        self.steps_since_active = t.zeros(self.ae.dict_size, dtype=int).to(self.device)

        self.optimizer = ConstrainedAdam(
            params=self.ae.parameters(),
            constrained_params=self.ae.decoder.parameters(),
            lr=self.lr,
        )

        lr_fn = get_lr_schedule(
            total_steps=self.steps,
            warmup_steps=self.warmup_steps,
            decay_start=self.decay_start,
        )
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)

        self.l1_penalty = cfg.l1_penalty
        self.reconstruction_loss_name = str(cfg.reconstruction_loss).lower()
        if self.reconstruction_loss_name not in {"l2", "mse"}:
            raise ValueError("reconstruction_loss must be 'l2' or 'mse'")
        self.l1_penalty_warmup_steps = cfg.l1_penalty_warmup_steps
        self.l1_penalty_warmup_fn = get_sparsity_warmup_fn(
            self.steps, self.l1_penalty_warmup_steps
        )

    def loss(self, x, step=None, logging=False, **kwargs):
        x_hat, f = self.ae(x, output_features=True)

        reconstruction_loss = self.reconstruction_loss(x, x_hat)
        l1_loss = f.norm(p=1, dim=-1).mean()

        deads = (f == 0).all(dim=0)
        self.steps_since_active[deads] += 1
        self.steps_since_active[~deads] = 0

        self.current_l1_penalty_scale = self.l1_penalty * self.l1_penalty_warmup_fn(step)
        loss = reconstruction_loss + l1_loss * self.current_l1_penalty_scale

        if not logging:
            return loss

        return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
            x,
            x_hat,
            f,
            {
                "loss/reconstruction": reconstruction_loss.item(),
                "loss/sparsity": l1_loss.item(),
                "loss/total": loss.item(),
            },
        )

    def reconstruction_loss(self, x, x_hat):
        residual = x - x_hat
        if self.reconstruction_loss_name == "mse":
            return residual.pow(2).mean()
        return t.linalg.norm(residual, dim=-1).mean()

    def update(self, step, x):
        x = x.to(self.device)
        self.optimizer.zero_grad()
        loss = self.loss(x, step=step)
        loss.backward()

        if self.grad_clip_norm is not None:
            t.nn.utils.clip_grad_norm_(self.ae.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(
                self.steps_since_active > self.resample_steps // 2, x
            )

        return loss.item()

    def resample_neurons(self, dead_neurons: t.Tensor, activations: t.Tensor):
        if not dead_neurons.any():
            return
        n_dead = dead_neurons.sum().item()
        print(f"Resampling {n_dead} dead neurons...")
        with t.no_grad():
            if activations.shape[0] >= n_dead:
                sampled = activations[t.randperm(activations.shape[0])[:n_dead]]
            else:
                sampled = activations[t.randint(0, activations.shape[0], (n_dead,))]
            noise = 0.01
            new_weights = sampled + noise * t.randn_like(sampled)
            self.ae.encoder.weight.data[dead_neurons] = (
                new_weights / new_weights.norm(dim=1, keepdim=True)
            )
            self.ae.encoder.bias.data[dead_neurons] = 0.0
            self.ae.decoder.weight.data[:, dead_neurons] = (
                self.ae.encoder.weight.data[dead_neurons].T
            )
            self.steps_since_active[dead_neurons] = 0

            # Reset the corresponding Adam optimizer state (exp_avg / exp_avg_sq /
            # step) for the resampled rows/columns. Otherwise the newly-initialized
            # neurons keep the stale momentum from their previous (dead) weights,
            # which slows convergence of the resampled features.
            self._reset_optimizer_state_for(dead_neurons)

    def _reset_optimizer_state_for(self, dead_neurons: t.Tensor):
        """Zero the Adam state for the encoder rows / decoder columns being resampled."""
        dead_idx = dead_neurons.nonzero(as_tuple=False).flatten().tolist()
        if not dead_idx:
            return
        # optimizer.params order matches ae.parameters() order:
        #   0: bias, 1: encoder.weight, 2: encoder.bias, 3: decoder.weight
        for p_idx, p in enumerate(self.optimizer.param_groups[0]["params"]):
            state = self.optimizer.state.get(p)
            if not state:
                continue
            # Only reset state for parameters that actually had rows/columns
            # resampled (encoder.weight / encoder.bias / decoder.weight). The
            # pre-encoding bias (p_idx 0) is untouched and keeps its state.
            if p_idx not in (1, 2, 3):
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if key not in state:
                    continue
                s = state[key]
                if p_idx == 1:  # encoder.weight [dict_size, act_dim] — reset rows
                    s[dead_idx] = 0.0
                elif p_idx == 2:  # encoder.bias [dict_size] — reset rows
                    s[dead_idx] = 0.0
                elif p_idx == 3:  # decoder.weight [act_dim, dict_size] — reset cols
                    s[:, dead_idx] = 0.0
            # Reset the step counter in-place (it is a torch scalar tensor, NOT a
            # Python int; assigning an int breaks optimizer.step()).
            if "step" in state:
                state["step"].zero_()
