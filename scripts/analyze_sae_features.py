# General
import os, argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Torch
import torch

# add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# biomechinterp
from biomechinterp.models import SparseAutoencoder
from biomechinterp.utils import resolve_device

def main():
    # setup
    parser = argparse.ArgumentParser()
    default_dir = os.getcwd()
    
    # inputs (Outputs from eval_sae.py)
    parser.add_argument("--pos_latents_path", type=str, default=f"{default_dir}/pos_latents.pt", help="Path to positive latents file.")
    parser.add_argument("--neg_latents_path", type=str, default=f"{default_dir}/neg_latents.pt", help="Path to negative latents file.")
    
    # optional: Checkpoint to analyze decoder weights (Feature -> Vocab)
    parser.add_argument("--checkpoint_path", type=str, default=f"{default_dir}/checkpoints/epoch_10", help="Path to SAE checkpoint.")
    
    # outputs
    parser.add_argument("--output_dir", type=str, default=f"{default_dir}/results", help="Directory to save plots and CSVs.")
    
    args = parser.parse_args()
    args.pos_latents_path = Path(args.pos_latents_path)
    args.neg_latents_path = Path(args.neg_latents_path)
    args.checkpoint_path = Path(args.checkpoint_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    device = resolve_device("auto")

    print("Loading Data...")
    # load latents
    pos_data = torch.load(args.pos_latents_path, map_location="cpu", weights_only=False)
    neg_data = torch.load(args.neg_latents_path, map_location="cpu", weights_only=False)
    
    # dictionary structure from eval_sae.py
    pos_latents = pos_data["pos_latents"] if isinstance(pos_data, dict) else pos_data
    neg_latents = neg_data["neg_latents"] if isinstance(neg_data, dict) else neg_data

    print(f"Loaded Latents - Pos: {pos_latents.shape}, Neg: {neg_latents.shape}")

    print("Calculating Feature Enrichment Scores...")
    
    # calculate mean activation per feature across all tokens
    # shape: [hidden_dim] (e.g., 128 features)
    pos_mean = pos_latents.mean(dim=0)
    neg_mean = neg_latents.mean(dim=0)
    
    # score = How much more active is this feature in pos vs neg?
    enrichment_scores = pos_mean - neg_mean
    
    # get top_k features
    top_k = 5
    top_values, top_indices = torch.topk(enrichment_scores, k=top_k)
    
    results_df = pd.DataFrame({
        "Feature_Index": top_indices.numpy(),
        "Enrichment_Score": top_values.numpy(),
        "Pos_Mean_Act": pos_mean[top_indices].numpy(),
        "Neg_Mean_Act": neg_mean[top_indices].numpy()
    })
    
    csv_path = out_dir / "top_tata_features.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nTop {top_k} Discriminative Features:")
    print(results_df)
    print(f"Saved statistics to {csv_path}")

    # visualize the best feature to prove it separates the classes
    best_feature_idx = top_indices[0].item()
    print(f"\nVisualizing distributions for Top Feature: {best_feature_idx}")

    plt.figure(figsize=(10, 6))
    
    # convert to numpy for plotting
    pos_acts = pos_latents[:, best_feature_idx].numpy()
    neg_acts = neg_latents[:, best_feature_idx].numpy()
    
    # remove zero activations to see the "firing" distribution clearly
    pos_acts_nz = pos_acts[pos_acts > 0]
    neg_acts_nz = neg_acts[neg_acts > 0]

    sns.histplot(pos_acts_nz, color="blue", label="Positive (TATA)", kde=True, stat="density", alpha=0.5)
    sns.histplot(neg_acts_nz, color="red", label="Negative (Control)", kde=True, stat="density", alpha=0.5)
    
    plt.title(f"Activation Distribution for SAE Feature {best_feature_idx}\n(Non-zero values only)")
    plt.xlabel("Activation Magnitude")
    plt.ylabel("Density")
    plt.legend()
    
    plot_path = out_dir / f"feature_{best_feature_idx}_dist.png"
    plt.savefig(plot_path)
    print(f"Saved distribution plot to {plot_path}")
    
    # If checkpoint exists, we can look at the feature vector
    if args.checkpoint_path.exists():
        print("\nLoading SAE Checkpoint for Decoder Analysis...")
        try:
            # Load model to CPU
            sae = SparseAutoencoder.load(args.checkpoint_path).to("cpu")
            
            # Assuming: decoder is Linear(hidden_dim, input_dim)
            decoder_weights = sae.decoder.weight.detach() # [input_dim, hidden_dim] usually in PyTorch Linear
            
            # We need the column corresponding to the feature index
            feature_vector = decoder_weights[:, best_feature_idx]
            
            # Save this vector for further analysis
            torch.save(feature_vector, out_dir / f"feature_{best_feature_idx}_decoder_vec.pt")
            print(f"Saved decoder weight vector for feature {best_feature_idx}")
            
        except Exception as e:
            print(f"Could not analyze decoder weights: {e}")
    else:
        print(f"Checkpoint not found at {args.checkpoint_path}, skipping decoder analysis.")

if __name__ == "__main__":
    main()