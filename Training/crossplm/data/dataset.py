import torch
from torch.utils.data import Dataset, random_split
from typing import List, Tuple, Optional, Dict
from transformers import PreTrainedTokenizer


IGNORE_CHAR = "_"


def build_label_map(labels: List[str]) -> Dict[str, int]:
    unique = set()
    for label_str in labels:
        unique.update(label_str)
    unique.discard(IGNORE_CHAR)
    return {lbl: i for i, lbl in enumerate(sorted(unique))}


class TokenClassificationDataset(Dataset):
    def __init__(
        self,
        sequences: List[str],
        labels: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        label_map: Optional[Dict[str, int]] = None,
    ):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_map = label_map or build_label_map(labels)
        self.num_labels = len(self.label_map)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label_str = self.labels[idx]

        encoded = self.tokenizer(
            seq,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        label_ids = self._align_labels(input_ids, label_str)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }

    def _align_labels(self, input_ids: List[int], label_str: str) -> List[int]:
        label_ids = []
        seq_idx = 0
        for token_id in input_ids:
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            if token in self.tokenizer.all_special_tokens:
                label_ids.append(-100)
            else:
                cleaned = token.replace(" ", "").replace("▁", "")
                if seq_idx < len(label_str):
                    ch = label_str[seq_idx]
                    if ch == IGNORE_CHAR:
                        label_ids.append(-100)
                    else:
                        label_ids.append(self.label_map[ch])
                    seq_idx += 1
                else:
                    label_ids.append(-100)
        return label_ids


def compute_class_weights(
    labels: List[str],
    label_map: Dict[str, int],
    method: str = "inverse",
) -> torch.Tensor:
    counts = [0] * len(label_map)
    for label_str in labels:
        for ch in label_str:
            if ch == IGNORE_CHAR:
                continue
            counts[label_map[ch]] += 1

    total = sum(counts)
    n_classes = len(counts)

    if method == "none":
        weights = [1.0] * n_classes
    elif method == "inverse":
        weights = [total / (n_classes * c) if c > 0 else 1.0 for c in counts]
    elif method == "log":
        import math
        weights = [math.log(total / c) / math.log(2) if c > 0 else 1.0 for c in counts]
    else:
        raise ValueError(f"Unknown class_weight_method: {method}")

    print(f"  Class counts: {counts}")
    print(f"  Class weights ({method}): {[round(w, 4) for w in weights]}")
    return torch.tensor(weights, dtype=torch.float)


def load_data_from_csv(
    csv_path: str,
    sequence_column: str = "sequence",
    label_column: str = "label",
) -> Tuple[List[str], List[str]]:
    import csv
    sequences, labels = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq = row[sequence_column].strip().upper()
            lbl = row[label_column].strip()
            if len(seq) != len(lbl):
                continue
            sequences.append(seq)
            labels.append(lbl)
    return sequences, labels


def split_dataset(dataset: Dataset, train_ratio: float, seed: int = 42):
    total = len(dataset)
    train_size = int(total * train_ratio)
    eval_size = total - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, eval_size], generator=generator)
