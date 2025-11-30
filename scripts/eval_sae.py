# General
import os, argparse, sys
from tqdm import tqdm
from pathlib import Path

# Torch
import torch
from torch.utils.data import DataLoader

# add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# biomechinterp
from biomechinterp.models import SparseAutoencoder
from biomechinterp.utils import resolve_device



def main():
    # Setup
    parser = argparse.ArgumentParser()
    default_dir = os.getcwd()
    parser.add_argument("--pos_data_path", type=str, default=f"{default_dir}/pos_activations.pt", help="Path to positive samples data file.")
    parser.add_argument("--neg_data_path", type=str, default=f"{default_dir}/neg_activations.pt", help="Path to negative samples data file.")
    parser.add_argument("--pos_save_path", type=str, default=f"{default_dir}/pos_latents.pt", help="Path to save positive latent representations.")
    parser.add_argument("--neg_save_path", type=str, default=f"{default_dir}/neg_latents.pt", help="Path to save negative latent representations.")
    args = parser.parse_args()
    args.pos_data_path = Path(args.pos_data_path)
    args.neg_data_path = Path(args.neg_data_path)
    args.pos_save_path = Path(args.pos_save_path)
    args.neg_save_path = Path(args.neg_save_path)
    device = resolve_device("auto")

    # Get model
    model = SparseAutoencoder.load(f"{default_dir}/checkpoints/epoch_615").to(device)  # Hard-coded for now

    # Get data
    pos_dataset = torch.load(args.pos_data_path, weights_only=False)
    neg_dataset = torch.load(args.neg_data_path, weights_only=False)
    pos_loader = DataLoader(pos_dataset["activations"], batch_size=64, shuffle=False)
    neg_loader = DataLoader(neg_dataset["activations"], batch_size=64, shuffle=False)

    # Evaluate and collect
    pos_latents = []
    neg_latents = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(pos_loader, desc="Extracting Positive Latents"):
            _, codes = model(batch)
            pos_latents.append(codes.cpu())
        for batch in tqdm(neg_loader, desc="Extracting Negative Latents"):
            _, codes = model(batch)
            neg_latents.append(codes.cpu())
    pos_latents = torch.cat(pos_latents, dim=0)
    neg_latents = torch.cat(neg_latents, dim=0)

    # Save latents
    torch.save({"pos_latents": pos_latents}, args.pos_save_path)
    torch.save({"neg_latents": neg_latents}, args.neg_save_path)


if __name__ == "__main__":
    main()
