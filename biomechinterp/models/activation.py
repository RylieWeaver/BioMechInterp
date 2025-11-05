# General
from functools import partial

# Torch
import torch.nn as nn

# biomechinterp
from biomechinterp.utils import filtered_kwargs



str2act = {
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "gelu_new": nn.GELU,
    "relu": nn.ReLU,
    "leakyrelu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,    
}

def get_activation(act_config):
    # allow string or dict
    if isinstance(act_config, str):
        act_config = {"name": act_config}

    # Choose activation class
    name = act_config.get("name", "silu").lower()
    if name not in str2act:
        raise ValueError(f"unsupported activation '{name}'")
    activation_class = str2act[name]

    # Read activation-specific kwargs
    kwargs = kwargs.update(filtered_kwargs(activation_class, act_config))

    return partial(activation_class, **kwargs)
