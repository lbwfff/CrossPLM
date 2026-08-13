from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np


class ActivationDataset(Dataset):
    """
    Dataset for loading pre-extracted PLM activations.

    Supports:
    - Single tensor file (.pt)
    - Sharded directory structure (shard_N/activations.pt)
    - Memory-mapped .dat files
    """

    def __init__(
        self,
        data_path: Path,
        shuffle: bool = True,
        seed: int = 42,
    ):
        self.data_path = Path(data_path)
        self.shuffle = shuffle
        self.seed = seed

        if seed is not None:
            torch.manual_seed(seed)

        if self.data_path.is_file():
            self._load_single_file()
        elif self.data_path.is_dir():
            self._load_sharded()
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")

    def _load_single_file(self):
        data = torch.load(self.data_path, map_location="cpu", weights_only=True)
        if isinstance(data, dict):
            data = data["embeddings"]
        self.data = data.float()
        self.d_model = self.data.shape[1]
        self.total_tokens = self.data.shape[0]

    def _load_sharded(self):
        shards = []
        subdirs = sorted([d for d in self.data_path.iterdir() if d.is_dir()])
        if not subdirs:
            pt_files = sorted(self.data_path.glob("*.pt"))
            if pt_files:
                for f in pt_files:
                    data = torch.load(f, map_location="cpu", weights_only=True)
                    if isinstance(data, dict):
                        data = data["embeddings"]
                    shards.append(data.float())
            else:
                raise FileNotFoundError(f"No shard data found in {self.data_path}")
        else:
            for subdir in subdirs:
                pt_file = subdir / "activations.pt"
                if pt_file.exists():
                    data = torch.load(pt_file, map_location="cpu", weights_only=True)
                    if isinstance(data, dict):
                        data = data["embeddings"]
                    shards.append(data.float())

        if not shards:
            raise FileNotFoundError(f"No activation data found in {self.data_path}")

        self.data = torch.cat(shards, dim=0)
        self.d_model = self.data.shape[1]
        self.total_tokens = self.data.shape[0]

    def __len__(self):
        return self.total_tokens

    def __getitem__(self, idx):
        return self.data[idx]


class ActivationDataLoader(DataLoader):
    def __init__(
        self,
        data_path: Path,
        batch_size: int = 512,
        shuffle: bool = True,
        seed: int = 42,
        samples_to_skip: int = 0,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dataset = ActivationDataset(data_path, shuffle=shuffle, seed=seed)
        self.config = type('obj', (object,), {
            'batch_size': batch_size,
            'plm_embd_dir': data_path,
        })

        if samples_to_skip > 0:
            dataset = torch.utils.data.Subset(
                dataset, range(samples_to_skip, len(dataset))
            )
            dataset.d_model = dataset.dataset.d_model

        def collate_fn(batch):
            return torch.stack(batch).to(device)

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
