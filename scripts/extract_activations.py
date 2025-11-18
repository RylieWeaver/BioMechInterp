# General
import os, argparse, sys
from pathlib import Path
from tqdm import tqdm
import pyfastx
from transformers import AutoTokenizer, AutoModel

# Torch
import torch
from torch.utils.data import DataLoader

# add the parent directory (project root) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# biomechinterp
from biomechinterp.utils import resolve_device



def main():
    # Setup
    parser = argparse.ArgumentParser()
    default_dir = os.getcwd()
    parser.add_argument("--pos_fasta_path", type=str, default=f"{default_dir}/pos_file.fa", help="Path to positive fasta file")
    parser.add_argument("--neg_fasta_path", type=str, default=f"{default_dir}/neg_file.fa", help="Path to negative fasta file")
    parser.add_argument("--model_id", default="InstaDeepAI/nucleotide-transformer-500m-1000g", help="Hugging Face model id.")
    parser.add_argument("--layer", type=int, default=-1, help="Hidden layer index (supports negative indices).")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for processing sequences.")
    parser.add_argument("--pos_save_path", type=str, default=f"{default_dir}/pos_activations.pt", help="Path to save positive activations.")
    parser.add_argument("--neg_save_path", type=str, default=f"{default_dir}/neg_activations.pt", help="Path to save negative activations.")
    args = parser.parse_args()
    args.pos_fasta_path = Path(args.pos_fasta_path)
    args.neg_fasta_path = Path(args.neg_fasta_path)
    args.pos_save_path = Path(args.pos_save_path)
    args.neg_save_path = Path(args.neg_save_path)
    args.pos_save_path.parent.mkdir(parents=True, exist_ok=True)
    args.neg_save_path.parent.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")

    # Get Fasta datasets
    pos_sequences = []
    neg_sequences = []
    pos_fasta_file = pyfastx.Fasta(str(args.pos_fasta_path))
    neg_fasta_file = pyfastx.Fasta(str(args.neg_fasta_path))
    for fasta_seq in pos_fasta_file:
        pos_sequences.append(str(fasta_seq.seq))
    for fasta_seq in neg_fasta_file:
        neg_sequences.append(str(fasta_seq.seq))
    pos_loader = DataLoader(pos_sequences, batch_size=args.batch_size, shuffle=False)
    neg_loader = DataLoader(neg_sequences, batch_size=args.batch_size, shuffle=False)

    # Get model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id).to(device)
    model.eval()

    # Extract activations
    with torch.no_grad():
        # Positive
        pos_activations = []
        for batch in tqdm(pos_loader, desc="Extracting Positive Activations"):
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
            pos_activations.append(layer_hidden.unsqueeze(0))

        # Negative
        neg_activations = []
        for batch in tqdm(neg_loader, desc="Extracting Negative Activations"):
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
            neg_activations.append(layer_hidden.unsqueeze(0))

    # Collect and save
    pos_activations = torch.cat(pos_activations, dim=0)
    neg_activations = torch.cat(neg_activations, dim=0)
    torch.save(
        {
            "num_sequences": len(pos_sequences),
            "model_id": args.model_id,
            "layer": args.layer,
            "activations": pos_activations,
        },
        args.pos_save_path,
    )
    torch.save(
        {
            "num_sequences": len(neg_sequences),
            "model_id": args.model_id,
            "layer": args.layer,
            "activations": neg_activations,
        },
        args.neg_save_path,
    )
    print(f"Saved pos and neg activations with shapes {pos_activations.shape} and {neg_activations.shape}")


if __name__ == "__main__":
    main()
