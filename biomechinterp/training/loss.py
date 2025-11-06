# General

# Torch
import torch
import torch.nn.functional as F

# biomechinterp



def sparse_autoencoder_loss(
    inputs: torch.Tensor,
    reconstruction: torch.Tensor,
    sparse_rep: torch.Tensor,
    l1_coefficient: float,
    mse_coefficient: float = 1.0,
) -> torch.Tensor:
    """Combined reconstruction + sparsity penalty."""
    recon_loss = F.mse_loss(reconstruction, inputs)
    sparsity_loss = sparse_rep.abs().mean()
    return mse_coefficient * recon_loss + l1_coefficient * sparsity_loss
