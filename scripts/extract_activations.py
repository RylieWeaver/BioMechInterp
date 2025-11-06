# General
import os, argparse
from pathlib import Path
from tqdm import tqdm
import pyfastx
from typing import List
from transformers import AutoTokenizer, AutoModel

# Torch
import torch
from torch import Tensor
from torch.utils.data import DataLoader

# biomechinterp
from biomechinterp.utils import resolve_device



def main():
    # Read args
    parser = argparse.ArgumentParser()
    default_dir = os.getcwd()
    default_fasta_name = "file.fa"
    parser.add_argument("--fasta_path", type=str, default=f"{default_dir}/{default_fasta_name}", help="Path to fasta file")
    parser.add_argument("--model_id", default="InstaDeepAI/nucleotide-transformer-500m-1000g", help="Hugging Face model id.")
    parser.add_argument("--layer", type=int, default=-16, help="Hidden layer index (supports negative indices).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing sequences.")
    parser.add_argument("--save_path", type=str, default=f"{default_dir}/activations.pt", help="Path to save activations.")
    args = parser.parse_args()
    args.fasta_path = Path(args.fasta_path)
    args.save_path = Path(args.save_path)

    # Setup
    device = resolve_device("auto")

    # Get Fasta datasets
    sequences = []
    fasta_file = pyfastx.Fasta(str(args.fasta_path))
    for fasta_seq in fasta_file:
        sequences.append(str(fasta_seq.seq))
    loader = DataLoader(sequences, batch_size=args.batch_size, shuffle=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id).to(device)
    model.eval()

    # Extract activations
    activations: List[Tensor] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Activations"):
            # Tokenize
            encoded = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model.config.max_position_embeddings,                                # Max length is model dependent
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            mask = encoded["attention_mask"].bool()                                             # [B, max_S]

            # Forward pass
            outputs = model(**encoded, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states                                               # List of length L: [B, max_S, D]

            # Get specified layer activations
            layer_index = args.layer if args.layer >= 0 else len(hidden_states) + args.layer
            layer_hidden = hidden_states[layer_index]                                           # [B, max_S, hidden_size]
            layer_hidden = layer_hidden[mask]                                                   # [T, D] where T is total tokens in batch
            activations.append(layer_hidden)

    # Collect and save
    activations = torch.cat(activations, dim=0)
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "num_sequences": len(sequences),
            "model_id": args.model_id,
            "layer": args.layer,
            "activations": activations,
        },
        args.save_path,
    )
    print(f"Saved activations with shape {activations.shape} to {args.save_path}")


if __name__ == "__main__":
    main()
