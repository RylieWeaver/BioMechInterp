from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List
import pyfastx

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ..utils.devices import resolve_device




class FastaChunkDataset(Dataset):
    """
    DataLoader for a FASTA file with a list of sequences. Loop through the sequences
    in random order (reshuffled each epoch), and for each sequence sample up to
    chunks_per_seq random windows of length context_len.
    """
    def __init__(
        self, 
        fasta_path,
        seq_ids,
        context_len: int,
        stride: int,
        chunks_per_seq=1,
        tokenizer=None,
        regions_dir=None,
    ):
        super().__init__()
        self.fasta_path = fasta_path
        self.regions_dir = regions_dir
        self.seq_ids = seq_ids
        self._worker_fasta = None
        self._region_store = None
        self.tokenizer = tokenizer if tokenizer is not None else DNATokenizer()
        self.seq_ids = [sid for sid in self.seq_ids if len(pyfastx.Fasta(self.fasta_path)[sid]) >= context_len]  # Filter out too short sequences
        self.context_len = context_len
        self.stride = stride
        self.k = chunks_per_seq

    def __len__(self):
        return len(self.seq_ids)
    
    def __getitem__(self, idx):
        # Lazy init per worker
        if self._worker_fasta is None:
            self._worker_fasta = pyfastx.Fasta(self.fasta_path)

        # Get the sequence at that index
        seq_id = self.seq_ids[idx]
        seq = str(self._worker_fasta[seq_id])

        # Fill regions
        default = self.tokenizer.reg2id["[UNSPECIFIED]"]
        if self._region_store is not None:
            seq_regions = torch.tensor(self._region_store.get_region_labels(seq_id), dtype=torch.int32)
            # Make sure regions length matches sequence length
            if len(seq_regions) < len(seq):
                diff = len(seq) - len(seq_regions)
                seq_regions = torch.cat([seq_regions, default * torch.ones(diff, dtype=torch.int32)])
        else:
            seq_regions = default * torch.ones(len(seq), dtype=torch.int32)

        # Get up to k random chunks in that sequence
        n_chunks = 1 + (len(seq) - self.context_len) // self.stride
        chunk_idx = random.sample(range(n_chunks), min(self.k, n_chunks))
        starts = [w * self.stride for w in chunk_idx]
        chunks = [seq[st:st+self.context_len] for st in starts]
        bp_ids = [self.tokenizer.encode(ch) for ch in chunks]
        region_ids = [seq_regions[st:st+self.context_len] for st in starts]

        return {"seq_id": seq_id, "starts": starts, "sequences": chunks, "bp_ids": bp_ids, "region_ids": region_ids}
