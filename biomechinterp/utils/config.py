# General
import json, inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# Torch

# biomechinterp



def filtered_kwargs(cls, source: Mapping[str, Any]) -> dict[str, Any]:
    params = inspect.signature(cls).parameters
    return {k: v for k, v in source.items() if k in params}


@dataclass
class Config:
    def to_dict(self):
        out = self.__dict__.copy()
        for k, v in list(out.items()):
            if hasattr(v, "to_dict"):
                out[k] = v.to_dict()
            elif isinstance(v, Path):
                out[k] = str(v)
        return out
    
    def save(self, path: Path):
        cfg = self.to_dict()
        with path.open("w") as f:
            json.dump(cfg, f, indent=4)
