import re
from bisect import bisect_right
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
    - Sharded directory structure (shard_N/embeddings.pt)

    Shards are memory-mapped (`torch.load(..., mmap=True)`) and kept as separate
    tensors instead of being concatenated into one big in-RAM tensor, so large
    datasets do not blow up memory. Rows are resolved lazily across shards on
    access (only the current batch is materialized).
    """

    def __init__(
        self,
        data_path: Path,
        shard: Optional[int] = None,
        mmap: bool = True,
    ):
        # NOTE: this dataset does not seed any RNG. Shuffling is handled by the
        # DataLoader via a dedicated torch.Generator so the global RNG is not
        # polluted.
        self.data_path = Path(data_path)
        self.shard = shard  # None = use all shards; else only that shard
        self.mmap = mmap

        self._shards = []       # list of (mmap'd) tensors, one per shard
        self._offsets = []      # cumulative token count before each shard
        self.total_tokens = 0
        self.d_model = None

        if self.data_path.is_file():
            self._load_one(self.data_path)
        elif self.data_path.is_dir():
            self._load_sharded()
        else:
            raise FileNotFoundError(f"Data path not found: {data_path}")

        if not self._shards:
            raise FileNotFoundError(
                f"No activation data found in {self.data_path}"
                + (f" for shard {self.shard}" if self.shard is not None else "")
            )
        self.d_model = self._shards[0].shape[1]

    def _load_one(self, path):
        data = torch.load(path, map_location="cpu", weights_only=True,
                          mmap=self.mmap)
        if isinstance(data, dict):
            data = data["embeddings"]
        # Convert only if needed (avoids copying an already-float32 mmap).
        if data.dtype != torch.float32:
            data = data.float()
        self._shards.append(data)
        self._offsets.append(self.total_tokens)
        self.total_tokens += data.shape[0]

    def _load_single_file(self):
        self._load_one(self.data_path)

    def _load_sharded(self):
        subdirs = [
            d for d in self.data_path.iterdir()
            if d.is_dir() and re.fullmatch(r"shard_\d+", d.name)
        ]
        subdirs.sort(key=lambda d: int(d.name.split("_")[1]))
        if not subdirs:
            pt_files = list(self.data_path.glob("*.pt"))
            pt_files.sort(key=lambda p: int(re.search(r"shard_(\d+)", p.stem).group(1))
                          if re.search(r"shard_(\d+)", p.stem) else p.name)
            if not pt_files:
                raise FileNotFoundError(f"No shard data found in {self.data_path}")
            if self.shard is not None:
                selected = []
                for path in pt_files:
                    match = re.search(r"(?:^|/)shard_(\d+)(?:\.pt)?$", str(path))
                    if match and int(match.group(1)) == self.shard:
                        selected.append(path)
                if not selected:
                    raise FileNotFoundError(
                        f"No flat .pt file for shard {self.shard} found in {self.data_path}"
                    )
                pt_files = selected
            for f in pt_files:
                self._load_one(f)
        else:
            shard_ids = [int(d.name.split("_")[1]) for d in subdirs]
            if self.shard is None and shard_ids != list(range(shard_ids[-1] + 1)):
                raise ValueError(f"Shard directories are not contiguous: {shard_ids}")
            for subdir in subdirs:
                # If a specific shard was requested, skip all others.
                if self.shard is not None:
                    m = re.match(r"shard_(\d+)", subdir.name)
                    if not m or int(m.group(1)) != self.shard:
                        continue
                pt_file = subdir / "embeddings.pt"
                if pt_file.exists():
                    self._load_one(pt_file)

    def __len__(self):
        return self.total_tokens

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_tokens:
            raise IndexError(idx)
        shard_idx = bisect_right(self._offsets, idx) - 1
        return self._shards[shard_idx][idx - self._offsets[shard_idx]]


class ActivationDataLoader(DataLoader):
    def __init__(
        self,
        data_path: Path,
        batch_size: int = 512,
        shuffle: bool = True,
        seed: int = 42,
        samples_to_skip: int = 0,
        shard: Optional[int] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        dataset = ActivationDataset(data_path, shard=shard)

        if samples_to_skip > 0:
            dataset = torch.utils.data.Subset(
                dataset, range(samples_to_skip, len(dataset))
            )
            dataset.d_model = dataset.dataset.d_model

        def collate_fn(batch):
            return torch.stack(batch).to(device)

        # Use a dedicated generator seeded once per DataLoader so shuffling is
        # reproducible AND the global RNG is not disturbed. `reseed()` advances
        # it each epoch so consecutive epochs do not repeat the same order.
        self.seed = seed
        self._epoch = 0
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=self.generator,
            collate_fn=collate_fn,
        )

    def reseed(self):
        """Advance the shuffle seed so the next epoch uses a different order."""
        self._epoch += 1
        self.generator.manual_seed(self.seed + self._epoch)
