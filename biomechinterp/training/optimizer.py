# General
from dataclasses import dataclass
from typing import Dict, Any, Literal, Optional

# Torch
import torch
from torch.optim import Adam, AdamW, SGD, RMSprop, Adagrad, Adadelta, Rprop, LBFGS

# biomechinterp
from biomechinterp.utils import Config, filtered_kwargs



STR2OPT = {
    "adam": Adam,
    "adamw": AdamW,
    "sgd": SGD,
    "rmsprop": RMSprop,
    "adagrad": Adagrad,
    "adadelta": Adadelta,
    "rprop": Rprop,
    "lbfgs": LBFGS,
}

@dataclass
class OptHandler(Config):
    name: Literal["adam","adamw","sgd","rmsprop","adagrad","adadelta","rprop","lbfgs"] = "adamw"
    lr: float = 1e-3
    extras: Optional[Dict[str, Any]] = None

    def build(self, params) -> torch.optim.Optimizer:
        opt_cls = STR2OPT[self.name]
        kwargs: Dict[str, Any] = {"lr": self.lr}
        if self.extras:
            kwargs.update(filtered_kwargs(opt_cls, self.extras))
        return opt_cls(params, **kwargs)
