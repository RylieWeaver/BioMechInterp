# General
from dataclasses import dataclass

# Torch
import torch.nn as nn

# biomechinterp
from .activation import get_activation



@dataclass
class SparseAutoencoderConfig:
    input_dim: int
    hidden_dim: int
    latent_dim: int
    activation: str = "relu"
    dropout: float = 0.0


class SparseAutoencoder(nn.Module):
    def __init__(self, config: SparseAutoencoderConfig):
        super().__init__()
        self.cfg = config
        act = get_activation(config.activation)
        dropout = config.dropout
        input_dim = config.input_dim
        hidden_dim = config.hidden_dim
        latent_dim = config.latent_dim

        # Define architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            act(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, inputs):
        codes = self.encoder(inputs)
        reconstruction = self.decoder(codes)
        return reconstruction, codes
