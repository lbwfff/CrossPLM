from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


class FineTunedESMEmbedder:
    """
    Extracts hidden states from a fine-tuned ESM2-8M model for SAE training.

    Unlike InterPLM which uses the base ESM, this embedder loads the fine-tuned
    checkpoint and extracts intermediate layer activations from the backbone,
    bypassing the classification head.
    """

    def __init__(
        self,
        ckpt_path: Union[str, Path],
        model_name: str = "Synthyra/ESM2-8M",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.ckpt_path = Path(ckpt_path)
        self.model_name = model_name
        self.max_length = max_length

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.ckpt_path, local_files_only=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.ckpt_path, local_files_only=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

    def extract_embeddings(
        self,
        sequences: List[str],
        layer: Optional[int] = None,
        batch_size: int = 8,
        return_labels: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if layer is None:
            layer = self.num_layers

        if layer < 0 or layer > self.num_layers:
            raise ValueError(f"Layer {layer} out of range [0, {self.num_layers}]")

        all_embeddings = []
        all_labels = [] if return_labels else None
        all_boundaries = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            batch_labels = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in batch_labels.items()}

            with torch.no_grad():
                outputs = self.model(
                    **inputs, output_hidden_states=True, return_dict=True
                )
                hidden_states = outputs.hidden_states

            layer_output = hidden_states[layer].detach().cpu()
            logits = outputs.logits.detach().cpu()

            for seq_idx, seq in enumerate(batch_seqs):
                seq_len = len(seq)
                seq_emb = layer_output[seq_idx, 1 : seq_len + 1, :]
                all_embeddings.append(seq_emb)

                if return_labels:
                    seq_logits = logits[seq_idx, 1 : seq_len + 1, :]
                    seq_preds = seq_logits.argmax(dim=-1)
                    all_labels.append(seq_preds)

                boundaries = (i + seq_idx, len(all_embeddings) - 1)
                all_boundaries.append(boundaries)

        embeddings = torch.cat(all_embeddings, dim=0)

        if return_labels:
            return {
                "embeddings": embeddings,
                "labels": torch.cat(all_labels, dim=0) if all_labels else None,
                "boundaries": all_boundaries,
            }

        return embeddings

    def extract_embeddings_with_labels(
        self,
        sequences: List[str],
        labels: List[str],
        layer: Optional[int] = None,
        batch_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        if layer is None:
            layer = self.num_layers

        all_embeddings = []
        all_label_ids = []

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]

            inputs = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(
                    **inputs, output_hidden_states=True, return_dict=True
                )
                hidden_states = outputs.hidden_states

            layer_output = hidden_states[layer].detach().cpu()

            for seq_idx, (seq, label_str) in enumerate(zip(batch_seqs, batch_labels)):
                seq_len = min(len(seq), self.max_length - 2)
                seq_emb = layer_output[seq_idx, 1 : seq_len + 1, :]
                all_embeddings.append(seq_emb)

                label_str_clean = str(label_str) if not isinstance(label_str, str) else label_str
                label_chars = list(label_str_clean[:seq_len])
                if len(label_chars) < seq_len:
                    label_chars += ["_"] * (seq_len - len(label_chars))
                MBM_LABEL = {"A": 0, ".": 1, "0": 0, "1": 1}
                label_ids = torch.tensor(
                    [MBM_LABEL.get(c, -100) for c in label_chars],
                    dtype=torch.long,
                )
                all_label_ids.append(label_ids)

        return {
            "embeddings": torch.cat(all_embeddings, dim=0),
            "labels": torch.cat(all_label_ids, dim=0),
        }

    @property
    def embedding_dim(self) -> int:
        return self.hidden_size

    @property
    def n_layers(self) -> int:
        return self.num_layers
