# General
import os
import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

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


def plot_roc(pos_latents, neg_latents, out_path: Path, feature_idx: int, title_prefix: str = ""):
    """
    pos_latents, neg_latents: [N, L, D] or [N, D] if reduced over L
    AUROC is computed on the flattened activations for a single feature.
    """
    # Flatten feature activations
    pos_acts = pos_latents[..., feature_idx].reshape(-1).cpu().numpy()
    neg_acts = neg_latents[..., feature_idx].reshape(-1).cpu().numpy()
    scores = np.concatenate([pos_acts, neg_acts])

    # Get Score Where Feature Indicates TATA: 1=TATA, 0=Control
    y_up = np.concatenate([
        np.ones_like(pos_acts, dtype=np.int32),
        np.zeros_like(neg_acts, dtype=np.int32),
    ])
    auc_up = roc_auc_score(y_up, scores)

    # Get Score Where Feature Indicates Control: 1=Control, 0=TATA
    y_down = np.concatenate([
        np.zeros_like(pos_acts, dtype=np.int32),
        np.ones_like(neg_acts, dtype=np.int32),
    ])
    auc_down = roc_auc_score(y_down, scores)

    # Dynamically choose best direction
    ## Positive numbers indicate TATA
    if auc_up >= 0.5:
        auc = auc_up
        preferred = "TATA"
        fpr, tpr, _ = roc_curve(y_up, scores, pos_label=1)
    ## Positive numbers indicate Control
    else:
        auc = auc_down
        preferred = "Control"
        fpr, tpr, _ = roc_curve(y_down, scores, pos_label=1)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f} ({preferred})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} SAE Feature {feature_idx} ROC")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _bootstrap_ecdf(vals, n_boot, grid_size, random_seed):
    vals = np.asarray(vals)
    N = len(vals)
    rng = np.random.default_rng(random_seed)

    x_grid = np.linspace(vals.min(), vals.max(), grid_size)         # [G]
    boot_cdfs = np.empty((n_boot, grid_size), dtype=np.float64)     # [B, G]

    for b in range(n_boot):
        sample = vals[rng.integers(0, N, size=N)]                   # [N]
        comparisons = (sample[:, None] <= x_grid[None, :])          # [N, G]
        cdfs = comparisons.mean(axis=0)                             # [G]
        boot_cdfs[b] = cdfs

    ecdf = (vals[:, None] <= x_grid[None, :]).mean(axis=0)          # [G]: observed CDF
    lo, hi = np.percentile(boot_cdfs, [0.5, 99.5], axis=0)          # 99% CI
    return x_grid, ecdf, lo, hi


def plot_latents_bootstrap(
        pos_latents,
        neg_latents,
        out_path: Path,
        feature_idx: int,
        title_prefix: str = "",
        n_boot: int = 10000,
        grid_size: int = 200,
        random_seed: int = 42
    ):
    """
    Bootstrapped ECDFs with 99% CIs for each feature.
    pos_latents, neg_latents: [N, L, D] or [N, D]
    """
    # Flatten feature activations
    pos_acts = pos_latents[..., feature_idx].reshape(-1).cpu().numpy()
    neg_acts = neg_latents[..., feature_idx].reshape(-1).cpu().numpy()

    x_pos, ecdf_pos, lo_pos, hi_pos = _bootstrap_ecdf(
        pos_acts, n_boot=n_boot, grid_size=grid_size, random_seed=random_seed
    )
    x_neg, ecdf_neg, lo_neg, hi_neg = _bootstrap_ecdf(
        neg_acts, n_boot=n_boot, grid_size=grid_size, random_seed=random_seed
    )

    plt.figure(figsize=(10, 6))

    plt.plot(x_pos, ecdf_pos, label=f"TATA (n={len(pos_acts)})")
    plt.fill_between(x_pos, lo_pos, hi_pos, alpha=0.2)

    plt.plot(x_neg, ecdf_neg, label=f"Control (n={len(neg_acts)})")
    plt.fill_between(x_neg, lo_neg, hi_neg, alpha=0.2)

    plt.title(f"{title_prefix} SAE Feature {feature_idx} CDF (bootstrapped 99% CI)")
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
    summary_df.to_csv(args.output_dir / "summary.csv", index=False)

    # Plot distributions and save
    for idx in sorted(set(topk_mean_indices + topk_max_indices + topk_mean_bin_indices + topk_max_bin_indices)):
        out_path = args.output_dir / f"{idx}_cdf.png"
        plot_latents(
            pos_latents["mag"],
            neg_latents["mag"],
            out_path,
            feature_idx=idx,
            title_prefix="All",
        )
        plot_latents_bootstrap(
            pos_latents["mag"],
            neg_latents["mag"],
            args.output_dir / f"{idx}_cdf_bootstrap.png",
            feature_idx=idx,
            title_prefix="All",
        )
        plot_roc(
            pos_latents["mag"],
            neg_latents["mag"],
            args.output_dir / f"{idx}_roc.png",
            feature_idx=idx,
            title_prefix="All",
        )
        plot_latents(
            pos_latents["mag_means"],
            neg_latents["mag_means"],
            args.output_dir / f"{idx}_mean_cdf.png",
            feature_idx=idx,
            title_prefix="Mean Reduced",
        )
        plot_latents_bootstrap(
            pos_latents["mag_means"],
            neg_latents["mag_means"],
            args.output_dir / f"{idx}_mean_cdf_bootstrap.png",
            feature_idx=idx,
            title_prefix="Mean Reduced",
        )
        plot_roc(
            pos_latents["mag_means"],
            neg_latents["mag_means"],
            args.output_dir / f"{idx}_mean_roc.png",
            feature_idx=idx,
            title_prefix="Mean Reduced",
        )
        plot_latents(
            pos_latents["mag_maxs"],
            neg_latents["mag_maxs"],
            args.output_dir / f"{idx}_max_cdf.png",
            feature_idx=idx,
            title_prefix="Max Reduced",
        )
        plot_latents_bootstrap(
            pos_latents["mag_maxs"],
            neg_latents["mag_maxs"],
            args.output_dir / f"{idx}_max_cdf_bootstrap.png",
            feature_idx=idx,
            title_prefix="Max Reduced",
        )
        plot_roc(
            pos_latents["mag_maxs"],
            neg_latents["mag_maxs"],
            args.output_dir / f"{idx}_max_roc.png",
            feature_idx=idx,
            title_prefix="Max Reduced",
        )


if __name__ == "__main__":
    main()
