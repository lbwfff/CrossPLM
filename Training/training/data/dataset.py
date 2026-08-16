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


def label_map_n_classes(label_map: Dict[str, int]) -> int:
    """Number of distinct classes in a {char: class_id} label map.

    Several characters may map to the same class id (e.g. mBMRB 'A' and '0'
    both -> 0), so the class count is max(id) + 1, NOT len(label_map).
    """
    if not label_map:
        return 0
    return max(label_map.values()) + 1


def build_id2label(label_map: Dict[str, int]) -> Dict[str, str]:
    """Build config.id2label (class_id -> representative char) from a {char: id} map.

    Deduplicates by class id so that many-to-one mappings (e.g. {'A':0, '0':0})
    do not produce duplicate keys. The representative char is the lexicographically
    smallest char for that class (matches build_label_map's sorted order).
    """
    by_class: Dict[int, str] = {}
    for ch, cid in label_map.items():
        if cid not in by_class or ch < by_class[cid]:
            by_class[cid] = ch
    return {str(cid): ch for cid, ch in sorted(by_class.items())}


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
        # Number of DISTINCT classes (many chars may map to one class), NOT the
        # number of characters in the map.
        self.num_labels = label_map_n_classes(self.label_map)

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
                    # Unknown characters (e.g. a residue not seen in training)
                    # are ignored, consistent with Single's encode_label_string.
                    if ch == IGNORE_CHAR or ch not in self.label_map:
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
    n_classes = label_map_n_classes(label_map)
    counts = [0] * n_classes
    for label_str in labels:
        for ch in label_str:
            if ch == IGNORE_CHAR or ch not in label_map:
                continue
            counts[label_map[ch]] += 1

    total = sum(counts)

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
    """Load sequences + labels from a CSV/TSV using the given column names.

    Delegates to single.label_maps.load_labeled_sequences so Training and the
    interpretability module read datasets with the exact same logic (separator
    auto-detection, uppercased sequences, rows with mismatched lengths dropped).
    """
    from single.label_maps import load_labeled_sequences

    return load_labeled_sequences(
        csv_path, {"sequence_column": sequence_column, "label_column": label_column}
    )


def split_dataset(dataset: Dataset, train_ratio: float, seed: int = 42):
    total = len(dataset)
    train_size = int(total * train_ratio)
    eval_size = total - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, eval_size], generator=generator)
