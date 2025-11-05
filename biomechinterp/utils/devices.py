# General
from typing import Literal

# Torch
import torch

# biomechinterp



DeviceType = Literal["auto", "cpu", "cuda", "mps"]
def resolve_device(device: DeviceType | str) -> torch.device:
    """Pick the best available torch.device given a preference string."""
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)
