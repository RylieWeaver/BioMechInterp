# General
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

# Torch

# biomechinterp



def filtered_kwargs(cls, source: Mapping[str, Any]) -> dict[str, Any]:
    params = inspect.signature(cls).parameters
    return {k: v for k, v in source.items() if k in params}


@dataclass(frozen=True)
class Config:
    def to_dict(self):
        out = self.__dict__.copy()
        for k, v in list(out.items()):
            if hasattr(v, "to_dict"):
                out[k] = v.to_dict()
            elif isinstance(v, Path):
                out[k] = str(v)
        return out
