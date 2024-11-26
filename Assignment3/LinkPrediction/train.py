import os
import datetime

from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import negative_sampling, subgraph, to_undirected
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.datasets import Planetoid

from parser import get_args
from models import get_gnn_model
from prepareDataset import prepare_dataset

import wandb
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(42)

def train(args):
    # Set device to GPU if available
    device = args.device
    # Prepare datasets
    train_data, val_data, test_data, Q = prepare_dataset(args)
    print(f'##### Dataset split #####')
    print(f"Number of nodes: {args.num_nodes}")
    print(f"Training edges: {train_data.edge_index.shape[1]}")
    print(f"Validation edges: {val_data.edge_index.shape[1]}")
    print(f"Testing edges: {test_data.edge_index.shape[1]}")
    print(f'#########################')

    # Move data to GPU
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)

    # Initialize the model, optimizer, and dataloader
    model = get_gnn_model(args)
    model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    print("\n=========== Training ===========")
    # Training loop
    with tqdm(range(args.num_epoch), total=args.num_epoch, desc=f"Training...") as pbar:
        for epoch in pbar:
            model.train()
            total_loss = 0
            train_data.to(args.device)
            optimizer.zero_grad()

            node_embeddings = model(train_data.x, train_data.edge_index)
            
            # Compute loss
            
            ## 1. Loss from paper (Currently not implemented correctly)
            # Positive and negative edge sampling
            pos_edge_index = train_data.edge_index
            neg_edge_index = negative_sampling(
                edge_index=train_data.edge_index,
                num_nodes=train_data.num_nodes,
                num_neg_samples=pos_edge_index.size(1)
            )

            # pos_scores = F.cosine_similarity(
            #     node_embeddings[pos_edge_index[0]], node_embeddings[pos_edge_index[1]]
            # )
            # neg_scores = F.cosine_similarity(
            #     node_embeddings[neg_edge_index[0]], node_embeddings[neg_edge_index[1]]
            # )

            ## 2. New Loss (Find for each node {Neighbors and Non-Neighbors} and calculated difference over mean)
            # 2.1 Naive and very time taking for CPU and OOM for CUDA
            edge_index = train_data.edge_index
            pos_scores = 0
            neg_scores = 0
            margin = args.margin

            queryNodes = torch.randperm(train_data.num_nodes)[:5000]
            
            for node in queryNodes:
                # Get all neighbors of the given node
                neighbors = edge_index[1][edge_index[0] == node].unique()
                
                num_neighbors = len(neighbors)
                if num_neighbors:
                    all_nodes = torch.arange(train_data.num_nodes, device=train_data.edge_index.device)
                    non_neighbors = torch.tensor(list(set(all_nodes.tolist()) - set(neighbors.tolist()) - {node}),
                                                device=edge_index.device)


                    # 1.) Naively take all the non-neighbors or some of them
                    # pos_scores += F.cosine_similarity(node_embeddings[node].unsqeeze(), node_embeddings[neighbors]).mean()
                    # neg_scores += F.cosine_similarity(node_embeddings[node], node_embeddings[non_neighbors]).mean()



                    # 2.) Take hard negatives
                    # Compute cosine similarities for non-neighbors
                    non_neighbor_similarities = F.cosine_similarity(
                        node_embeddings[node].unsqueeze(0), 
                        node_embeddings[non_neighbors]
                    )
                    
                    # Select top-K hard negatives based on similarity
                    k = min(len(non_neighbors), math.ceil(num_neighbors/2))
                    hard_negatives_indices = torch.topk(non_neighbor_similarities, k, largest=True).indices
                    hard_negatives = non_neighbors[hard_negatives_indices]
                    ek = min(len(non_neighbors), math.floor(num_neighbors/2))
                    easy_negatives = non_neighbors[~hard_negatives_indices]
                    shuffled_indices = torch.randperm(easy_negatives.size(0))[:ek]
                    easy_negatives = easy_negatives[shuffled_indices]
                    non_neighbors = torch.cat((hard_negatives, easy_negatives))
                    # Calculate mean scores
                    pos_scores += F.cosine_similarity(
                        node_embeddings[node].unsqueeze(0), 
                        node_embeddings[neighbors]
                    ).mean()
                    
                    neg_scores += F.cosine_similarity(
                        node_embeddings[node].unsqueeze(0), 
                        node_embeddings[hard_negatives]
                    ).mean()

            pos_scores /= len(queryNodes)
            neg_scores /= len(queryNodes)
            
            loss = torch.sum(torch.relu(margin + neg_scores - pos_scores))
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() # Total loss is overall loss over an epoch

            # Validation
            if val_data.edge_index is not None:
                model.eval()
                with torch.no_grad():
                    val_embeddings = model(val_data.x.to(args.device), val_data.edge_index.to(args.device))
                    val_pos_scores = F.cosine_similarity(
                        val_embeddings[val_data.edge_index[0]], val_embeddings[val_data.edge_index[1]]
                    )
                    val_loss = (1 - val_pos_scores).mean().item()

            wandb.log({
                "Epoch": epoch,
                "Train Loss": total_loss,
                "Val Loss": val_loss,
                "Pos Score": pos_scores,
                "Neg Score": neg_scores
            }, step=epoch)
            
            pbar.set_postfix({"Epoch": epoch+1, "Train Loss": total_loss, "Val Loss": val_loss})

    # plot_cosinescores_with_neighbors(train_data, model)

    # Save the trained model
    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    modelPath = os.path.join(args.save_dir, f'{args.gnn_method}_{args.dataset}_model_{timestamp}.pth')
    torch.save(model.state_dict(), modelPath)
    
    print(f"Model saved to {modelPath}")
    print("Training complete. Model saved.")
    print("================================\n")


def plot_cosinescores_with_neighbors(train_data, model):
    device = model.device
    # Move data to the appropriate device
    x = train_data.x.to(device)
    edge_index = train_data.edge_index.to(device)

    # Compute embeddings using the model
    model.eval()
    with torch.no_grad():
        node_embeddings = model(x, edge_index)  # Shape: (num_nodes, embedding_dim)

    # Compute cosine similarity
    norm_embeddings = node_embeddings / node_embeddings.norm(dim=1, keepdim=True)
    cosine_scores = torch.mm(norm_embeddings, norm_embeddings.t())  # Shape: (num_nodes, num_nodes)

    # Create adjacency matrix to mark neighbors
    adj_matrix = torch.zeros((train_data.num_nodes, train_data.num_nodes), device=device)
    adj_matrix[edge_index[0], edge_index[1]] = 1

    # Convert to numpy for plotting
    cosine_scores = cosine_scores.cpu().numpy()
    adj_matrix = adj_matrix.cpu().numpy()

    # Create a heatmap with neighbors marked
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cosine_scores,
        cmap="coolwarm",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "Cosine Similarity"}
    )

    # Overlay neighbors
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i, j] == 1:
                plt.text(
                    j + 0.5, i + 0.5, "*",
                    color="black", fontsize=6,
                    ha="center", va="center"
                )

    plt.title("Cosine Similarity Heatmap with Neighbors Marked")
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.savefig("cosine_similarity_heatmap_with_neighbors.png", dpi=300, bbox_inches="tight")


    

def print_run_summary(args):
    """
    Print a summary of the run based on the provided arguments.
    """
    print("\n=========== Run Summary ===========")
    print(f"Dataset: {args.dataset}")
    print(f"Dataset Directory: {args.dataset_dir}")
    print(f"Splits: Train={args.splits[0]:.2f}, Val={args.splits[1]:.2f}, Test={args.splits[2]:.2f}")
    print(f"GNN Method: {args.gnn_method}")
    # print(f"Batch Size: {args.batch_size}")
    print(f"Number of Epochs: {args.num_epoch}")
    print(f"Learning rate: {args.lr}")
    print(f"Margin for AUC Loss: {args.margin}")
    print(f"Model Save Directory: {args.save_dir}")
    print(f"Device: {args.device}")
    print("===================================\n")

if __name__ == "__main__":
    args = get_args()
    print_run_summary(args)
    wandb.init(project='Link Predicition', name=f'delete_{args.dataset}_{args.gnn_method}')
    train(args)
    wandb.finish()