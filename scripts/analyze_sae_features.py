#!/usr/bin/env python

# General
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch

# add the parent directory (project root) to sys.path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# biomechinterp
from biomechinterp.models import SparseAutoencoder
from biomechinterp.utils import resolve_device


def compute_enrichment_scores(pos_latents: torch.Tensor, neg_latents: torch.Tensor):
    """
    Assumes latents have shape [batch, seq, features] and that the last dim is features.
    """
    if pos_latents.ndim != 3 or neg_latents.ndim != 3:
        raise ValueError(
            f"Expected pos/neg latents to be 3D [batch, seq, feature], "
            f"got {pos_latents.shape=} and {neg_latents.shape=}"
        )

    # mean activation per feature across all tokens (batch + seq)
    # result: [features]
    pos_mean = pos_latents.mean(dim=(0, 1))
    neg_mean = neg_latents.mean(dim=(0, 1))

    # max over seq dim (per token position), then mean across batch
    # pos_latents: [B, T, F]
    # pos_latents.max(dim=1).values: [B, F]
    # .mean(dim=0): [F]
    pos_max_per_seq = pos_latents.max(dim=1).values
    neg_max_per_seq = neg_latents.max(dim=1).values

    pos_max = pos_max_per_seq.mean(dim=0)
    neg_max = neg_max_per_seq.mean(dim=0)

    enrichment_scores_mean = pos_mean - neg_mean
    enrichment_scores_max = pos_max - neg_max

    return (
        enrichment_scores_mean,
        enrichment_scores_max,
        pos_mean,
        neg_mean,
        pos_max_per_seq,
        neg_max_per_seq,
    )


def save_topk_stats(
    enrichment_scores: torch.Tensor,
    pos_stats: torch.Tensor,
    neg_stats: torch.Tensor,
    top_k: int,
    csv_path: Path,
    pos_col_name: str,
    neg_col_name: str,
):
    top_values, top_indices = torch.topk(enrichment_scores, k=top_k)
    df = pd.DataFrame(
        {
            "Feature_Index": top_indices.cpu().numpy(),
            "Enrichment_Score": top_values.cpu().numpy(),
            pos_col_name: pos_stats[top_indices].cpu().numpy(),
            neg_col_name: neg_stats[top_indices].cpu().numpy(),
        }
    )
    df.to_csv(csv_path, index=False)
    return df, top_indices


def plot_feature_distribution(
    pos_latents: torch.Tensor,
    neg_latents: torch.Tensor,
    feature_idx: int,
    out_path: Path,
    title_prefix: str,
):
    """
    Plot activation distribution for a single feature across *all tokens*,
    using only non-zero activations.
    Assumes latents: [B, T, F].
    """
    # shape: [B, T]
    pos_acts = pos_latents[:, :, feature_idx].reshape(-1).cpu().numpy()
    neg_acts = neg_latents[:, :, feature_idx].reshape(-1).cpu().numpy()

    pos_acts_nz = pos_acts[pos_acts > 0]
    neg_acts_nz = neg_acts[neg_acts > 0]

    plt.figure(figsize=(10, 6))
    sns.histplot(
        pos_acts_nz,
        color="blue",
        label="Positive (TATA)",
        kde=True,
        stat="density",
        alpha=0.5,
    )
    sns.histplot(
        neg_acts_nz,
        color="red",
        label="Negative (Control)",
        kde=True,
        stat="density",
        alpha=0.5,
    )

    plt.title(
        f"{title_prefix} for SAE Feature {feature_idx}\n(Non-zero values only)"
    )
    plt.xlabel("Activation Magnitude")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_max_feature_distribution(
    pos_max_per_seq: torch.Tensor,
    neg_max_per_seq: torch.Tensor,
    feature_idx: int,
    out_path: Path,
    title_prefix: str,
):
    """
    Plot distribution of per-sequence max activations for a single feature.
    pos_max_per_seq: [B, F] from pos_latents.max(dim=1).values
    """
    pos_acts = pos_max_per_seq[:, feature_idx].cpu().numpy()
    neg_acts = neg_max_per_seq[:, feature_idx].cpu().numpy()

    pos_acts_nz = pos_acts[pos_acts > 0]
    neg_acts_nz = neg_acts[neg_acts > 0]

    plt.figure(figsize=(10, 6))
    sns.histplot(
        pos_acts_nz,
        color="blue",
        label="Positive (TATA)",
        kde=True,
        stat="density",
        alpha=0.5,
    )
    sns.histplot(
        neg_acts_nz,
        color="red",
        label="Negative (Control)",
        kde=True,
        stat="density",
        alpha=0.5,
    )

    plt.title(
        f"{title_prefix} for SAE Feature {feature_idx}\n(Non-zero values only)"
    )
    plt.xlabel("Activation Magnitude")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def analyze_decoder_feature(
    checkpoint_path: Path, feature_idx: int, out_dir: Path
):
    print("\nLoading SAE Checkpoint for Decoder Analysis...")
    try:
        # Load model to CPU
        sae = SparseAutoencoder.load(checkpoint_path).to("cpu")

        # Assuming: decoder is Linear(hidden_dim, input_dim)
        # In PyTorch: weight shape is [out_features, in_features]
        # For a decoder mapping latent -> input, we usually want columns = features
        decoder_weights = sae.decoder.weight.detach()  # [input_dim, hidden_dim] usually

        if feature_idx >= decoder_weights.shape[1]:
            raise IndexError(
                f"Feature index {feature_idx} is out of bounds for decoder weight "
                f"with shape {decoder_weights.shape}"
            )

        feature_vector = decoder_weights[:, feature_idx]
        out_path = out_dir / f"feature_{feature_idx}_decoder_vec.pt"
        torch.save(feature_vector, out_path)
        print(f"Saved decoder weight vector for feature {feature_idx} to {out_path}")
    except Exception as e:
        print(f"Could not analyze decoder weights: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze SAE feature enrichment between positive and negative latents."
    )
    default_dir = Path.cwd()

    # inputs (Outputs from eval_sae.py)
    parser.add_argument(
        "--pos_latents_path",
        type=Path,
        default=default_dir / "pos_latents.pt",
        help="Path to positive latents file.",
    )
    parser.add_argument(
        "--neg_latents_path",
        type=Path,
        default=default_dir / "neg_latents.pt",
        help="Path to negative latents file.",
    )

    # optional: Checkpoint to analyze decoder weights (Feature -> Vocab)
    parser.add_argument(
        "--checkpoint_path",
        type=Path,
        default=default_dir / "checkpoints" / "epoch_10",
        help="Path to SAE checkpoint.",
    )

    # outputs
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=default_dir / "results",
        help="Directory to save plots and CSVs.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top discriminative features to export.",
    )

    args = parser.parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # If you actually want to use the device later, keep this; otherwise remove.
    _device = resolve_device("auto")

    print("Loading Data...")
    # load latents (force CPU for analysis / plotting)
    pos_data = torch.load(args.pos_latents_path, map_location="cpu", weights_only=False)
    neg_data = torch.load(args.neg_latents_path, map_location="cpu", weights_only=False)

    # dictionary structure from eval_sae.py
    pos_latents = pos_data["pos_latents"] if isinstance(pos_data, dict) else pos_data
    neg_latents = neg_data["neg_latents"] if isinstance(neg_data, dict) else neg_data

    print(f"Loaded Latents - Pos: {pos_latents.shape}, Neg: {neg_latents.shape}")

    print("Calculating Feature Enrichment Scores...")
    (
        enrichment_scores_mean,
        enrichment_scores_max,
        pos_mean,
        neg_mean,
        pos_max_per_seq,
        neg_max_per_seq,
    ) = compute_enrichment_scores(pos_latents, neg_latents)

    # Top-k by mean activation difference
    mean_csv_path = out_dir / "top_tata_features.csv"
    mean_df, top_indices_mean = save_topk_stats(
        enrichment_scores=enrichment_scores_mean,
        pos_stats=pos_mean,
        neg_stats=neg_mean,
        top_k=args.top_k,
        csv_path=mean_csv_path,
        pos_col_name="Pos_Mean_Act",
        neg_col_name="Neg_Mean_Act",
    )

    print(f"\nTop {args.top_k} Discriminative Features (Mean):")
    print(mean_df)
    print(f"Saved statistics to {mean_csv_path}")

    # Top-k by max activation difference (averaged over batch)
    # Here pos_max_per_seq / neg_max_per_seq still contain the per-sequence maxima [B, F]
    max_means_pos = pos_max_per_seq.mean(dim=0)
    max_means_neg = neg_max_per_seq.mean(dim=0)

    max_csv_path = out_dir / "top_max_tata_features.csv"
    max_df, top_indices_max = save_topk_stats(
        enrichment_scores=enrichment_scores_max,
        pos_stats=max_means_pos,
        neg_stats=max_means_neg,
        top_k=args.top_k,
        csv_path=max_csv_path,
        pos_col_name="Pos_Max_Act",
        neg_col_name="Neg_Max_Act",
    )

    print(f"\nTop {args.top_k} Discriminative Features (Per-seq Max):")
    print(max_df)
    print(f"Saved statistics to {max_csv_path}")

    # visualize the best feature to prove it separates the classes
    best_feature_idx = int(top_indices_mean[0].item())
    best_max_feature_idx = int(top_indices_max[0].item())

    # Plot mean-based feature distribution
    mean_plot_path = out_dir / f"feature_{best_feature_idx}_dist.png"
    plot_feature_distribution(
        pos_latents=pos_latents,
        neg_latents=neg_latents,
        feature_idx=best_feature_idx,
        out_path=mean_plot_path,
        title_prefix="Activation Distribution (Mean-based feature)",
    )
    print(f"Saved distribution plot to {mean_plot_path}")

    # Plot max-based feature distribution (over per-seq maxima)
    max_plot_path = out_dir / f"feature_{best_max_feature_idx}_max_dist.png"
    plot_max_feature_distribution(
        pos_max_per_seq=pos_max_per_seq,
        neg_max_per_seq=neg_max_per_seq,
        feature_idx=best_max_feature_idx,
        out_path=max_plot_path,
        title_prefix="Per-sequence Max Activation Distribution",
    )
    print(f"Saved max-distribution plot to {max_plot_path}")

    # If checkpoint exists, we can look at the decoder weights for the best feature
    if args.checkpoint_path.exists():
        analyze_decoder_feature(
            checkpoint_path=args.checkpoint_path,
            feature_idx=best_feature_idx,
            out_dir=out_dir,
        )
    else:
        print(
            f"Checkpoint not found at {args.checkpoint_path}, "
            "skipping decoder analysis."
        )

if __name__ == "__main__":
    main()
