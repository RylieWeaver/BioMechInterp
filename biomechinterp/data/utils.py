# General

# Torch
import torch
from torch.utils.data import random_split

# biomechinterp



def move_to(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, dict):
        return {k: move_to(v, device) for k, v in batch.items()}
    elif isinstance(batch, list):
        return [move_to(v, device) for v in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to(v, device) for v in batch)
    else:
        return batch
    

def shuffle_split(dataset, split_ratio=0.8, seed=42):
    total_size = len(dataset)
    train_size = int(split_ratio * total_size)
    val_size = int((1 - split_ratio)/2 * total_size)
    test_size = total_size - val_size - train_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    return train_set, val_set, test_set
