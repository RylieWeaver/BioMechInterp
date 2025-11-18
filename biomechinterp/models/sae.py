# General
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union

# Torch
import torch
import torch.nn as nn

# biomechinterp
from biomechinterp.utils import Config
from .activation import get_activation



class SparseAutoencoderConfig(Config):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim

    @staticmethod
    def load(path: Path) -> "SparseAutoencoderConfig":
        with open(path) as f:
            data = json.load(f)
        return SparseAutoencoderConfig(**data)


class SparseAutoencoder(nn.Module):
    def __init__(self, config: SparseAutoencoderConfig):
        super().__init__()
        self.cfg = config
        self.act = get_activation("relu")()
        input_dim = config.input_dim
        latent_dim = config.latent_dim

        # Define architecture
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def encode(self, x):
        x = x - self.decoder.bias
        codes = self.act(self.encoder(x))
        return codes

    def forward(self, inputs):
        codes = self.encode(inputs)
        reconstruction = self.decoder(codes)
        return reconstruction, codes

    @classmethod
    def load(cls, dir: Union[Path, str]):
        # Setup path
        dir = Path(dir)

        # Init model with config
        model_cfg = SparseAutoencoderConfig.load(dir / "model_config.json")
        model = cls(model_cfg)

        # Load state dict
        state_dict = torch.load(dir / "model.pt", weights_only=True)
        model.load_state_dict(state_dict)
        return model
