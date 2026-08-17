from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer


def _residue_positions(batch_encoding, tokenizer, sequences):
    """Return residue positions while excluding padding and special tokens."""
    special = batch_encoding.pop("special_tokens_mask", None)
    if special is None:
        special_ids = set(tokenizer.all_special_ids)
        special = torch.tensor([
            [int(token_id in special_ids) for token_id in row]
            for row in batch_encoding["input_ids"].tolist()
        ])
    attention = batch_encoding["attention_mask"]
    positions = []
    for index, sequence in enumerate(sequences):
        residue = torch.nonzero(
            (attention[index] == 1) & (special[index] == 0)
        ).flatten()
        expected = min(
            len(sequence), int(attention[index].sum().item()) - int(special[index].sum().item())
        )
        if len(residue) != expected:
            raise ValueError(
                f"Tokenizer produced {len(residue)} residue tokens, expected "
                f"{expected}; one-token-per-residue alignment is required."
            )
        positions.append(residue)
    return positions


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
        # The fine-tuned backbones (e.g. Synthyra/ESM2-8M via FastPLMs) ship custom
        # code in config.json (auto_map), so trust_remote_code=True is required.
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.ckpt_path, local_files_only=True, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "<pad>"
        if getattr(self.tokenizer, "padding_side", "right") != "right":
            raise ValueError("Embedding extraction requires tokenizer.padding_side='right'")

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.ckpt_path, local_files_only=True, trust_remote_code=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers

    def _forward_target_layer(self, inputs, layer: int, need_logits: bool):
        """Run the model while retaining only the requested final hidden layer."""
        final_layer = getattr(self.model, "esm", None)
        if final_layer is not None:
            final_layer = getattr(final_layer.encoder, "emb_layer_norm_after", None)
        if layer == self.num_layers and final_layer is not None:
            captured = {}

            def capture(_, __, output):
                captured["hidden"] = output

            handle = final_layer.register_forward_hook(capture)
            try:
                outputs = self.model(
                    **inputs, output_hidden_states=False, return_dict=True
                )
            finally:
                handle.remove()
            if "hidden" not in captured:
                raise RuntimeError("Failed to capture the requested final hidden layer")
            hidden = captured["hidden"]
        else:
            outputs = self.model(
                **inputs, output_hidden_states=True, return_dict=True
            )
            hidden = outputs.hidden_states[layer]
        logits = outputs.logits if need_logits else None
        return hidden, logits

    def residue_lengths(self, sequences: List[str], batch_size: int = 8) -> List[int]:
        """Return the actual residue-token count kept for every sequence."""
        lengths = []
        for i in range(0, len(sequences), batch_size):
            batch = self.tokenizer(
                sequences[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )
            lengths.extend(
                len(pos) for pos in _residue_positions(
                    batch, self.tokenizer, sequences[i : i + batch_size]
                )
            )
        return lengths

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

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            batch_labels = self.tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )
            residue_positions = _residue_positions(
                batch_labels, self.tokenizer, batch_seqs
            )
            inputs = {k: v.to(self.device) for k, v in batch_labels.items()}

            with torch.no_grad():
                layer_output, logits = self._forward_target_layer(
                    inputs, layer, return_labels
                )

            layer_output = layer_output.detach().cpu()
            logits = logits.detach().cpu() if return_labels else None

            for seq_idx, seq in enumerate(batch_seqs):
                seq_positions = residue_positions[seq_idx]
                seq_emb = layer_output[seq_idx, seq_positions, :]
                all_embeddings.append(seq_emb)

                if return_labels:
                    seq_logits = logits[seq_idx, seq_positions, :]
                    seq_preds = seq_logits.argmax(dim=-1)
                    all_labels.append(seq_preds)

        embeddings = torch.cat(all_embeddings, dim=0)

        if return_labels:
            return {
                "embeddings": embeddings,
                "labels": torch.cat(all_labels, dim=0) if all_labels else None,
            }

        return embeddings

    def extract_embeddings_with_labels(
        self,
        sequences: List[str],
        labels: List[str],
        layer: Optional[int] = None,
        batch_size: int = 8,
        label_map: Optional[dict] = None,
    ) -> Dict[str, torch.Tensor]:
        if label_map is None:
            from single.label_maps import get_label_map
            label_map = get_label_map("mBMRB")
        if layer is None:
            layer = self.num_layers

        if len(sequences) != len(labels):
            raise ValueError(
                f"Received {len(sequences)} sequences but {len(labels)} labels"
            )

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
                return_special_tokens_mask=True,
            )
            residue_positions = _residue_positions(
                inputs, self.tokenizer, batch_seqs
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                layer_output, _ = self._forward_target_layer(
                    inputs, layer, False
                )

            layer_output = layer_output.detach().cpu()

            for seq_idx, (seq, label_str) in enumerate(zip(batch_seqs, batch_labels)):
                if len(str(label_str)) != len(seq):
                    raise ValueError(
                        "Sequence/label length mismatch reached the embedder; "
                        "validate the input table before extraction."
                    )
                seq_positions = residue_positions[seq_idx]
                seq_len = len(seq_positions)
                seq_emb = layer_output[seq_idx, seq_positions, :]
                all_embeddings.append(seq_emb)

                label_str_clean = str(label_str) if not isinstance(label_str, str) else label_str
                label_chars = list(label_str_clean[:seq_len])
                if len(label_chars) < seq_len:
                    label_chars += ["_"] * (seq_len - len(label_chars))
                from single.label_maps import encode_label_string
                label_ids = torch.tensor(
                    encode_label_string("".join(label_chars), label_map),
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
