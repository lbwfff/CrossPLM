from pathlib import Path

import torch as t
from tqdm import tqdm

from single.train.data_loader import ActivationDataLoader
from single.train.trainers.relu import ReLUTrainer
from single.configs import TrainingConfig, DataConfig, SAEConfig, CheckpointConfig


class SAETrainingRun:
    def __init__(
        self,
        data: ActivationDataLoader,
        trainer: ReLUTrainer,
        checkpoint_cfg: CheckpointConfig,
        log_steps: int = 500,
    ):
        self.data = data
        self.trainer = trainer
        self.checkpoint_cfg = checkpoint_cfg
        self.log_steps = log_steps

        self.training_state = {
            "n_tokens_total": 0,
            "current_step": 0,
        }

    def compute_metrics(self, batch):
        with t.no_grad():
            x_hat, f = self.trainer.ae(batch, output_features=True)
            recon_loss = t.linalg.norm(batch - x_hat, dim=-1).mean().item()
            sparsity_loss = f.norm(p=1, dim=-1).mean().item()

            l0 = (f > 0).float().sum(dim=-1).mean().item()
            dead_pct = ((f == 0).all(dim=0).sum().item() / f.shape[1]) * 100
            active_pct = (f > 0).float().mean(dim=0)
            rarely_active = (active_pct < 0.01).sum().item()
            rarely_pct = (rarely_active / f.shape[1]) * 100

        return {
            "recon_loss": recon_loss,
            "sparsity_loss": sparsity_loss,
            "l0": l0,
            "dead_pct": dead_pct,
            "rarely_active_pct": rarely_pct,
        }

    def run(self):
        trainer = self.trainer
        steps = trainer.steps
        save_steps = self.checkpoint_cfg.save_steps
        log_steps = self.log_steps
        save_dir = Path(self.checkpoint_cfg.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Resume support: if we resumed from a checkpoint, start at the saved
        # step instead of 0, and skip the optimizer/scheduler warm-up that was
        # already applied. If no checkpoint was loaded, start from step 0.
        start_step = self.training_state["current_step"]
        if start_step >= steps:
            print(f"Already trained {start_step}/{steps} steps; nothing to do.")
            return

        progress = tqdm(range(start_step, steps), desc="Training SAE",
                        unit="step", initial=start_step)
        data_iter = iter(self.data)
        for step in progress:
            try:
                batch = next(data_iter)
            except StopIteration:
                # One pass over the dataset finished: start a new epoch and
                # continue sampling fresh batches (so all tokens get used).
                data_iter = iter(self.data)
                batch = next(data_iter)
            loss_val = trainer.update(step, batch)
            self.training_state["n_tokens_total"] += batch.shape[0]
            self.training_state["current_step"] = step

            if step % log_steps == 0:
                metrics = self.compute_metrics(batch)
                progress.set_postfix({
                    "loss": f"{metrics['recon_loss']:.4f}",
                    "L0": f"{metrics['l0']:.1f}",
                    "dead%": f"{metrics['dead_pct']:.1f}",
                    "LR": f"{trainer.current_lr:.2e}",
                })
            else:
                progress.set_postfix(loss=f"{loss_val:.4f}")

            if step > 0 and step % save_steps == 0:
                ckpt_dir = save_dir / f"checkpoint_{step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                trainer.save_checkpoint(ckpt_dir)
                t.save(trainer.ae.state_dict(), save_dir / f"ae_step_{step}.pt")
                # Persist training_state so --resume_from can pick up the step.
                t.save(self.training_state, ckpt_dir / "training_state.pt")

        t.save(trainer.ae.state_dict(), save_dir / "ae.pt")
        print(f"\n{'='*50}")
        print("Training complete. Final metrics on last batch:")
        final = self.compute_metrics(batch)
        for k, v in final.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"{'='*50}")

    @classmethod
    def from_configs(
        cls,
        data_cfg: DataConfig,
        sae_cfg: SAEConfig,
        train_cfg: TrainingConfig,
        checkpoint_cfg: CheckpointConfig,
    ):
        train_cfg.activation_dim = sae_cfg.activation_dim
        train_cfg.dict_size = sae_cfg.dict_size
        train_cfg.normalize_to_sqrt_d = sae_cfg.normalize_to_sqrt_d

        data_loader = ActivationDataLoader(
            data_path=data_cfg.plm_embd_dir,
            batch_size=train_cfg.batch_size,
            seed=train_cfg.seed,
            samples_to_skip=data_cfg.samples_to_skip,
            shard=data_cfg.shard,
        )

        trainer = ReLUTrainer(train_cfg)

        return cls(data=data_loader, trainer=trainer, checkpoint_cfg=checkpoint_cfg)

    def resume_from_checkpoint(self, checkpoint_dir: Path):
        self.trainer.update_from_checkpoint(checkpoint_dir)
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            self.training_state.update(t.load(state_path, weights_only=True))
            print(f"Resumed from step {self.training_state['current_step']}")
        else:
            # Legacy checkpoint saved before training_state.pt existed; start from 0.
            print(f"Checkpoint {checkpoint_dir} has no training_state.pt; starting from step 0.")
