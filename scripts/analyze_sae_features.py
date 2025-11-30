# General
import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# Torch
import torch

# add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# biomechinterp
from biomechinterp.utils import resolve_device



def latent_ops(latents: torch.Tensor):
    """
    latents: [N, L, D]
    """
    mag_means = latents.mean(dim=1)                 # [N, D]
    mag_mean_stat = mag_means.mean(dim=0)           # [D]
    mag_maxs = latents.max(dim=1).values            # [N, D]
    mag_max_stat = mag_maxs.mean(dim=0)             # [D]
    binary = (latents > 0).float()                  # [N, L, D]
    binary_means = binary.mean(dim=1)               # [N, D]
    binary_mean_stat = binary_means.mean(dim=0)     # [D]
    binary_maxs = binary.max(dim=1).values          # [N, D]
    binary_max_stat = binary_maxs.mean(dim=0)       # [D]
    return {
        "mag": latents,
        "mag_means": mag_means,
        "mag_mean_stat": mag_mean_stat,
        "mag_maxs": mag_maxs,
        "mag_max_stat": mag_max_stat,
        "binary": binary,
        "binary_means": binary_means,
        "binary_mean_stat": binary_mean_stat,
        "binary_maxs": binary_maxs,
        "binary_max_stat": binary_max_stat,
    }


def cdf(x):
    x = np.sort(x)
    y = np.linspace(0, 1, len(x))
    return x, y


def plot_latents(pos_latents, neg_latents, out_path: Path, feature_idx: int, title_prefix: str = ""):
    """
    latents: [N, L, D] or [N, D] if reduced over L
    """
    # Reshape data
    pos_acts = pos_latents[..., feature_idx].reshape(-1).cpu().numpy()      # [-1]
    neg_acts = neg_latents[..., feature_idx].reshape(-1).cpu().numpy()      # [-1]

    # # Filter 0s
    # pos_acts = pos_acts[pos_acts > 0]
    # neg_acts = neg_acts[neg_acts > 0]

    # Get CDF data
    x_pos, y_pos = cdf(pos_acts)
    x_neg, y_neg = cdf(neg_acts)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        x_pos,
        y_pos,
        label=f"Positive (TATA, n={len(pos_acts)})",
        color="red",
    )
    plt.plot(
        x_neg,
        y_neg,
        label=f"Negative (Control, n={len(neg_acts)})",
        color="blue",
    )
    plt.title(f"{title_prefix} SAE Feature {feature_idx} CDF")
    plt.xlabel("Activation Magnitude")
    plt.ylabel("Cumulative Fraction")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    # Setup
    parser = argparse.ArgumentParser()
    default_dir = os.getcwd()
    parser.add_argument("--pos_data_path", type=str, default=f"{default_dir}/pos_latents.pt", help="Path to save positive latent representations.")
    parser.add_argument("--neg_data_path", type=str, default=f"{default_dir}/neg_latents.pt", help="Path to save negative latent representations.")
    parser.add_argument("--output_dir", type=str, default=f"{default_dir}/sae_feature_analysis", help="Directory to save output plots.")
    parser.add_argument("--topk", type=int, default=10, help="Number of top features to analyze.")
    args = parser.parse_args()
    args.pos_data_path = Path(args.pos_data_path)
    args.neg_data_path = Path(args.neg_data_path)
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(exist_ok=True, parents=True)
    device = resolve_device("auto")

    # Get latents
    pos_latents = torch.load(args.pos_data_path, map_location=device, weights_only=False)           # dict
    neg_latents = torch.load(args.neg_data_path, map_location=device, weights_only=False)           # dict

    # Reduce latents over sequence length L
    pos_latents = latent_ops(pos_latents["pos_latents"])                                            # [N1, L, D]
    neg_latents = latent_ops(neg_latents["neg_latents"])                                            # [N2, L, D]

    # Get stats for topk selection
    mean_diff_stat = (pos_latents["mag_mean_stat"] - neg_latents["mag_mean_stat"])                  # [D]
    max_diff_stat = (pos_latents["mag_max_stat"] - neg_latents["mag_max_stat"])                     # [D]
    mean_diff_bin_stat = (pos_latents["binary_mean_stat"] - neg_latents["binary_mean_stat"])        # [D]
    max_diff_bin_stat = (pos_latents["binary_max_stat"] - neg_latents["binary_max_stat"])           # [D]

    # Get topk features by relative differences
    topk_mean_indices = torch.topk(mean_diff_stat.abs(), k=args.topk).indices.tolist()              # [k]
    topk_max_indices = torch.topk(max_diff_stat.abs(), k=args.topk).indices.tolist()                # [k]
    topk_mean_bin_indices = torch.topk(mean_diff_bin_stat.abs(), k=args.topk).indices.tolist()      # [k]
    topk_max_bin_indices = torch.topk(max_diff_bin_stat.abs(), k=args.topk).indices.tolist()        # [k]

    # Save topk summary
    summary_data = {
        "topk_max_mag_indices": topk_max_indices,
        "max_mag_pos": pos_latents["mag_max_stat"][topk_max_indices].cpu().numpy(),
        "max_mag_neg": neg_latents["mag_max_stat"][topk_max_indices].cpu().numpy(),
        "max_mag_stats": max_diff_stat[topk_max_indices].cpu().numpy(),
        "topk_mean_mag_indices": topk_mean_indices,
        "mean_mag_pos": pos_latents["mag_mean_stat"][topk_mean_indices].cpu().numpy(),
        "mean_mag_neg": neg_latents["mag_mean_stat"][topk_mean_indices].cpu().numpy(),
        "mean_mag_stats": mean_diff_stat[topk_mean_indices].cpu().numpy(),
        "topk_max_bin_indices": topk_max_bin_indices,
        "max_bin_pos": pos_latents["binary_max_stat"][topk_max_bin_indices].cpu().numpy(),
        "max_bin_neg": neg_latents["binary_max_stat"][topk_max_bin_indices].cpu().numpy(),
        "max_bin_stats": max_diff_bin_stat[topk_max_bin_indices].cpu().numpy(),
        "topk_mean_bin_indices": topk_mean_bin_indices,
        "mean_bin_pos": pos_latents["binary_mean_stat"][topk_mean_bin_indices].cpu().numpy(),
        "mean_bin_neg": neg_latents["binary_mean_stat"][topk_mean_bin_indices].cpu().numpy(),
        "mean_bin_stats": mean_diff_bin_stat[topk_mean_bin_indices].cpu().numpy(),
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df)
    summary_df.to_csv(args.output_dir / "sae_feature_analysis_summary.csv", index=False)

    # Plot distributions and save
    for idx in set(topk_mean_indices + topk_max_indices + topk_mean_bin_indices + topk_max_bin_indices):
        out_path = args.output_dir / f"sae_feature_{idx}_distribution.png"
        plot_latents(
            pos_latents["mag"],
            neg_latents["mag"],
            out_path,
            feature_idx=idx,
            title_prefix="All",
        )
        plot_latents(
            pos_latents["mag_means"],
            neg_latents["mag_means"],
            args.output_dir / f"sae_feature_{idx}_mean_distribution.png",
            feature_idx=idx,
            title_prefix="Mean Reduced",
        )
        plot_latents(
            pos_latents["mag_maxs"],
            neg_latents["mag_maxs"],
            args.output_dir / f"sae_feature_{idx}_max_distribution.png",
            feature_idx=idx,
            title_prefix="Max Reduced",
        )


if __name__ == "__main__":
    main()
