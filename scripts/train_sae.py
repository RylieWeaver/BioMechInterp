# General
import os, argparse
from pathlib import Path

# Torch
import torch

# biomechinterp
from biomechinterp.models import SparseAutoencoder, SparseAutoencoderConfig
from biomechinterp.data import shuffle_split
from biomechinterp.training import SAETrainer, SAETrainerConfig, OptHandler, LoaderHandler
from biomechinterp.utils import resolve_device



def main():
    # Setup
    parser = argparse.ArgumentParser()
    default_dir = os.getcwd()
    parser.add_argument("--pos_data_path", type=str, default=f"{default_dir}/pos_activations.pt", help="Path to positive activations file.")
    parser.add_argument("--neg_data_path", type=str, default=f"{default_dir}/neg_activations.pt", help="Path to negative activations file.")
    parser.add_argument("--model_id", default="InstaDeepAI/nucleotide-transformer-500m-1000g", help="Hugging Face model id.")
    parser.add_argument("--checkpoint_dir", type=str, default=f"{default_dir}/checkpoints", help="Path to checkpoint model.")
    args = parser.parse_args()
    args.pos_data_path = Path(args.pos_data_path)
    args.neg_data_path = Path(args.neg_data_path)
    args.checkpoint_dir = Path(args.checkpoint_dir)
    device = resolve_device("auto")

    # Get model
    model_cfg = SparseAutoencoderConfig(
        input_dim=1280,  # Currently hard-coded by model
        latent_dim=128,
    )
    model = SparseAutoencoder(model_cfg)

    # Get data
    pos_dataset = torch.load(args.pos_data_path, weights_only=False)
    neg_dataset = torch.load(args.neg_data_path, weights_only=False)
    dataset = {
        "activations": torch.cat([pos_dataset["activations"], neg_dataset["activations"]], dim=0)
    }  # NOTE: We may need to worry about dataset imbalance here
    train_ds, val_ds, test_ds = shuffle_split(dataset["activations"], seed=42)

    # Train from scratch
    opt_handler = OptHandler(name="adamw", lr=1e-3)
    loader_handler = LoaderHandler(batch_size=64)
    trainer_cfg = SAETrainerConfig(
        epochs=10,
        l1_coefficient=1e-3,
        opt_handler=opt_handler,
        loader_handler=loader_handler,
        checkpoint_dir=args.checkpoint_dir,
        save_every=1,
    )
    trainer = SAETrainer(
        config=trainer_cfg,
        model=model,
        device=device,
    )
    trainer.set_loaders(train_ds, val_ds, test_ds)
    trainer.train(epochs=10)  # Train from 0 to 10

    # Train from checkpoint
    # ckpt_epoch = 10  # Hard-coded for now
    # trainer = SAETrainer.load(args.checkpoint_dir / f"epoch_{ckpt_epoch}", device=device)
    # trainer.set_loaders(train_ds, val_ds, test_ds)
    # trainer.train(last_epoch=ckpt_epoch, epochs=20)  # Continue training from epoch 10 to 20


if __name__ == "__main__":
    main()
