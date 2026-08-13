import os
import json
import shutil
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
from typing import Optional


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        config,
        train_dataset,
        eval_dataset=None,
        task_dir: Optional[str] = None,
        class_weights=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.task_dir = task_dir
        self.class_weights = class_weights
        self.top_k = 3

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.model.to(self.device)

        self.optimizer = AdamW(
            self.model.model.parameters(),
            lr=float(config.learning_rate),
            betas=(float(config.adam_beta1), float(config.adam_beta2)),
            eps=float(config.adam_epsilon),
            weight_decay=float(config.weight_decay),
        )

        self.global_step = 0
        self.epoch = 0
        self.best_f1 = 0.0
        self.best_checkpoints = []
        self.eval_history = []

    def _get_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.config.per_device_train_batch_size if shuffle else self.config.per_device_eval_batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            num_workers=int(self.config.dataloader_num_workers),
            pin_memory=True,
        )

    def _collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        padded = self.tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors="pt",
        )

        max_len = padded["input_ids"].size(1)
        padded_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, lbl in enumerate(labels):
            padded_labels[i, :len(lbl)] = lbl

        return {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": padded_labels,
        }

    def train(self):
        train_loader = self._get_dataloader(self.train_dataset, shuffle=True)

        num_update_steps_per_epoch = len(train_loader) // int(self.config.gradient_accumulation_steps)
        if int(self.config.max_steps) > 0:
            num_training_steps = int(self.config.max_steps)
            num_epochs = int(self.config.num_train_epochs)
        else:
            num_training_steps = num_update_steps_per_epoch * int(self.config.num_train_epochs)
            num_epochs = int(self.config.num_train_epochs)

        lr_scheduler = get_scheduler(
            str(self.config.lr_scheduler_type),
            self.optimizer,
            num_warmup_steps=int(num_training_steps * float(self.config.warmup_ratio)),
            num_training_steps=num_training_steps,
        )

        self.model.model.zero_grad()
        train_loss = 0.0
        epochs_to_run = num_epochs if int(self.config.max_steps) <= 0 else 9999

        loss_fct = None
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=self.class_weights.to(self.device), ignore_index=-100
            )

        for epoch in range(int(epochs_to_run)):
            self.epoch = epoch
            self.model.model.train()
            epoch_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

            for step, batch in enumerate(progress_bar):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model.model(**batch)
                logits = outputs.logits
                if loss_fct is not None:
                    loss = loss_fct(
                        logits.view(-1, logits.size(-1)),
                        batch["labels"].view(-1),
                    )
                else:
                    loss = outputs.loss
                loss = loss / int(self.config.gradient_accumulation_steps)
                loss.backward()
                train_loss += loss.item()
                epoch_loss += loss.item()

                if (step + 1) % int(self.config.gradient_accumulation_steps) == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.model.parameters(), float(self.config.max_grad_norm)
                    )
                    self.optimizer.step()
                    lr_scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                    current_epoch = self.global_step / num_update_steps_per_epoch

                    if self.global_step % int(self.config.logging_steps) == 0:
                        avg_loss = train_loss / max(1, int(self.config.logging_steps) * int(self.config.gradient_accumulation_steps))
                        progress_bar.set_postfix({"loss": f"{avg_loss:.4f}"})
                        train_loss = 0.0

                    if self.global_step % int(self.config.eval_steps) == 0 and self.eval_dataset:
                        eval_metrics = self.evaluate()
                        self.eval_history.append({
                            "epoch": round(current_epoch, 2),
                            "step": self.global_step,
                            **eval_metrics,
                        })
                        self.model.model.train()

                        f1 = eval_metrics.get("f1", 0.0)
                        self._update_best_checkpoints(f1, self.global_step, current_epoch)

                    if int(self.config.max_steps) > 0 and self.global_step >= int(self.config.max_steps):
                        break

            if int(self.config.max_steps) > 0 and self.global_step >= int(self.config.max_steps):
                break

        self._save_checkpoint("final", self.best_f1, self.epoch)
        self._plot_training_curve()

    def evaluate(self):
        eval_loader = self._get_dataloader(self.eval_dataset, shuffle=False)
        self.model.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model.model(**batch)
                total_loss += outputs.loss.item()

                preds = outputs.logits.argmax(dim=-1)
                for i in range(preds.size(0)):
                    mask = batch["labels"][i] != -100
                    all_preds.extend(preds[i][mask].cpu().tolist())
                    all_labels.extend(batch["labels"][i][mask].cpu().tolist())

        avg_loss = total_loss / len(eval_loader)

        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / max(1, len(all_preds))
        f1 = self._compute_f1(all_preds, all_labels)

        print(f"  loss: {avg_loss:.4f}  acc: {accuracy:.4f}  f1: {f1:.4f}")

        metrics = {"eval_loss": avg_loss, "accuracy": accuracy, "f1": f1}
        if self.task_dir:
            metrics_path = os.path.join(self.task_dir, "eval_metrics.jsonl")
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metrics) + "\n")

        return metrics

    def _compute_f1(self, predictions, labels):
        num_classes = max(max(predictions), max(labels)) + 1
        tp = [0] * num_classes
        fp = [0] * num_classes
        fn = [0] * num_classes
        for p, l in zip(predictions, labels):
            if p == l:
                tp[p] += 1
            else:
                fp[p] += 1
                fn[l] += 1
        f1_list = []
        for i in range(num_classes):
            precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
            recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_list.append(f1)
        return sum(f1_list) / num_classes

    def _update_best_checkpoints(self, f1, step, epoch):
        if f1 <= self.best_f1 and len(self.best_checkpoints) >= self.top_k:
            return

        tag = f"epoch_{epoch:.2f}_f1_{int(f1 * 10000):04d}"
        self._save_checkpoint(tag, f1, epoch)

        self.best_checkpoints.append((f1, step, tag))
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        self.best_checkpoints = self.best_checkpoints[:self.top_k]

        if f1 > self.best_f1:
            self.best_f1 = f1

        kept_tags = {c[2] for c in self.best_checkpoints}
        if self.task_dir:
            ckpt_root = os.path.join(self.task_dir, "checkpoints")
            if os.path.isdir(ckpt_root):
                for d in os.listdir(ckpt_root):
                    if d not in kept_tags:
                        path = os.path.join(ckpt_root, d)
                        if os.path.isdir(path):
                            shutil.rmtree(path)

    def _save_checkpoint(self, tag: str, f1: float, epoch: float):
        if self.task_dir is None:
            return
        ckpt_dir = os.path.join(self.task_dir, "checkpoints", tag)
        os.makedirs(ckpt_dir, exist_ok=True)

        self.model.save_pretrained(ckpt_dir)

        training_state = {
            "global_step": self.global_step,
            "epoch": epoch,
            "f1": f1,
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(training_state, os.path.join(ckpt_dir, "training_state.pt"))

    def _plot_training_curve(self):
        if not self.eval_history or self.task_dir is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Plot] matplotlib not installed, skipping training curve")
            return

        epochs = [h["epoch"] for h in self.eval_history]
        f1_scores = [h["f1"] for h in self.eval_history]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, f1_scores, marker="o", linestyle="-", linewidth=2, markersize=4)
        plt.xlabel("Epoch", fontsize=14)
        plt.ylabel("F1-score (macro)", fontsize=14)
        plt.title(f"Training Curve — {self.config.backbone_model_id}", fontsize=15)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.task_dir, "training_curve.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[Plot] training curve saved to {save_path}")
