# General
from dataclasses import dataclass
from typing import Literal, Tuple

# Torch
import torch
import torch.nn as nn
import torch.nn.functional as F

# biomechinterp
from activation import get_activation



@dataclass
class SparseAutoencoderConfig:
    input_dim: int
    latent_dim: int
    hidden_dim: int
    activation: str = "relu"
    dropout: float = 0.0
    tied_weights: bool = False


class SparseAutoencoder(nn.Module):
    def __init__(self, config: SparseAutoencoderConfig) -> None:
        super().__init__()
        self.cfg = config
        act = get_activation(config.activation)

        # Define architecture
        self.encoder = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            act(),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(config.hidden_dim, config.latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            act(),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(config.hidden_dim, config.input_dim)
        )

    def forward(self, inputs):
        codes = self.encoder(inputs)
        reconstruction = self.decoder(codes)
        return reconstruction, codes
