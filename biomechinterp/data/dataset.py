from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..utils.devices import resolve_device


@dataclass
class ActivationManifest:
    root: Path
    dtype: str
    flatten_tokens: bool
    shards: List[dict]
    num_items: int
    num_tokens: int
    metadata: dict

    @property
    def shard_offsets(self) -> List[int]:
        offsets: List[int] = []
        total = 0
        for shard in self.shards:
            offsets.append(total)
            total += shard["num_items"]
        return offsets


def load_manifest(path: str | Path) -> ActivationManifest:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text())
    return ActivationManifest(
        root=manifest_path.parent,
        dtype=data["dtype"],
        flatten_tokens=data["flatten_tokens"],
        shards=data["shards"],
        num_items=data["num_items"],
        num_tokens=data["num_tokens"],
        metadata=data.get("metadata", {}),
    )


class ActivationDataset(Dataset[Tensor]):
    """Torch Dataset for activation shards written by ActivationDatasetWriter."""

    def __init__(
        self,
        manifest_path: str | Path,
        device: str = "cpu",
        cache_size: int = 2,
        preload: bool = False,
    ) -> None:
        self.manifest = load_manifest(manifest_path)
        self.device = resolve_device(device)
        self.cache_size = cache_size
        self._cache: OrderedDict[int, Tensor] = OrderedDict()
        if preload:
            for shard_idx in range(len(self.manifest.shards)):
                self._cache[shard_idx] = self._load_shard(shard_idx)

    def __len__(self) -> int:
        return self.manifest.num_items

    def __getitem__(self, index: int) -> Tensor:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_idx = self._locate_shard(index)
        shard = self._get_shard(shard_idx)
        shard_offset = self.manifest.shard_offsets[shard_idx]
        inner_idx = index - shard_offset
        vector = shard[inner_idx]
        return vector.to(self.device)

    def _locate_shard(self, index: int) -> int:
        offsets = self.manifest.shard_offsets
        lo, hi = 0, len(offsets)
        while lo < hi:
            mid = (lo + hi) // 2
            if index < offsets[mid]:
                hi = mid
            else:
                lo = mid + 1
        shard_idx = max(0, lo - 1)
        return shard_idx

    def _get_shard(self, shard_idx: int) -> Tensor:
        if shard_idx in self._cache:
            self._cache.move_to_end(shard_idx)
            return self._cache[shard_idx]
        shard_tensor = self._load_shard(shard_idx)
        if len(self._cache) >= self.cache_size and self.cache_size > 0:
            self._cache.popitem(last=False)
        if self.cache_size > 0:
            self._cache[shard_idx] = shard_tensor
        return shard_tensor

    def _load_shard(self, shard_idx: int) -> Tensor:
        shard_info = self.manifest.shards[shard_idx]
        shard_path = self.manifest.root / shard_info["path"]
        return torch.load(shard_path, map_location="cpu")
