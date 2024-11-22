import os

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling, subgraph
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.datasets import Planetoid

from parser import get_args
from models import get_gnn_model
from prepareDataset import prepare_dataset

torch.manual_seed(42)

def train(args):
    # Set device to GPU if available
    device = args.device
    print(f"Using device: {device}")

    # Prepare datasets
    train_data, val_data, test_data = prepare_dataset(args)

    # Move data to GPU
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    # Initialize model
    model = get_gnn_model(args).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    model.train()
    for epoch in range(args.num_epoch):
        optimizer.zero_grad()
        print(train_data.edge_index, train_data.edge_index.shape)
        train_edge_index, _ = subgraph(
            subset=torch.arange(train_data.num_nodes).to(device),  # Subset specifies nodes to keep
            edge_index=train_data.edge_index,          # Original edge indices
            relabel_nodes=True                         # Relabel nodes to consecutive IDs
        )

        # Negative sampling
        neg_edge_index = negative_sampling(
            edge_index=train_edge_index,
            num_nodes=train_data.num_nodes,
            num_neg_samples=train_edge_index.size(1)
        ).to(device)

        # Compute node embeddings
        node_embeddings = model(train_data.x, train_edge_index)

        # Compute cosine similarity scores
        pos_scores = F.cosine_similarity(
            node_embeddings[train_edge_index[0]],
            node_embeddings[train_edge_index[1]]
        )
        neg_scores = F.cosine_similarity(
            node_embeddings[neg_edge_index[0]],
            node_embeddings[neg_edge_index[1]]
        )

        # Compute AUC surrogate loss (Equation 9)
        loss = torch.clamp(args.margin + neg_scores - pos_scores, min=0).mean()

        # Backpropagation
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch + 1}/{args.num_epoch}, Loss: {loss.item():.4f}")

    # Save the trained model
    saveDir = os.path.join(args.save_dir, f'{args.gnn_method}_{args.dataset}_model.pth')
    torch.save(model.state_dict(), f"{args.gnn_method}_model.pth")
    print(f"Model saved to {args.gnn_method}_model.pth")

def print_run_summary(args):
    """
    Print a summary of the run based on the provided arguments.
    """
    print("\n=========== Run Summary ===========")
    print(f"Dataset: {args.dataset}")
    print(f"Dataset Directory: {args.dataset_dir}")
    print(f"Splits: Train={args.splits[0]:.2f}, Val={args.splits[1]:.2f}, Test={args.splits[2]:.2f}")
    print(f"GNN Method: {args.gnn_method}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Epochs: {args.num_epoch}")
    print(f"Margin for AUC Loss: {args.margin}")
    print(f"Model Save Directory: {args.save_dir}")
    print(f"Device: {'CUDA (GPU)' if args.device.type == 'cuda' else 'CPU'}")
    print("===================================\n")


if __name__ == "__main__":
    args = get_args()
    print_run_summary(args)
    train(args)
