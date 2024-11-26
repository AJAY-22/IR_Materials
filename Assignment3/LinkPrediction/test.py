import os

import torch
import torch.nn.functional as F

from parser import get_args
from models import get_gnn_model
from metrics import adamic_adar_score, common_neighbors_score, compute_metrics, compute_metrics_lsh
from LSH import RandomLSH, TrainableLSH

from tqdm import tqdm

def gnn_inference(model, q, data):
    """
    Use GNN to score node pairs (q, all test nodes).
    """
    node_embeddings = model(data.x, data.edge_index)
    q_embedding = node_embeddings[q]
    scores = F.cosine_similarity(q_embedding.unsqueeze(0), node_embeddings, dim=1)
    return node_embeddings, scores

def test(args):
    model_path = os.path.join(args.save_dir, f'{args.gnn_method}_{args.dataset}_model.pth')
    q_set_path = os.path.join(args.dataset_dir, f'{args.dataset}_q.pth')
    test_data_path = os.path.join(args.dataset_dir, f'{args.dataset}_test_data.pth')

    Q = torch.load(q_set_path)
    data = torch.load(test_data_path)  # Test data
    args.num_features = data.num_features
    data.to(args.device)
    
    edge_index = data.edge_index  # Test edges (ground truth)
    num_test_node = data.x.shape[0] # total number of test nodes
    model = get_gnn_model(args)  # Adjust input_dim as needed
    model.load_state_dict(torch.load(model_path))
    model.to(args.device)
    model.eval()
    
    metrics = {"GNN": [], "Adamic-Adar": [], "Common-Neighbors": []}
    
    with tqdm(Q, desc='Querying Nodes', total=len(Q)) as pbar:
        for q in pbar:
            y_true = torch.zeros(num_test_node, dtype=torch.float32, device=args.device)
            # local_q = (data.global_id == q).nonzero(as_tuple=True)[0].item()
            y_true[edge_index[1][(edge_index[0] == q).nonzero(as_tuple=True)]] = 1  # Mark true edges

            if args.inference_method is None:
                # GNN scores
                _, y_pred_gnn = gnn_inference(model, q, data)
                metrics["GNN"].append(compute_metrics(y_true, y_pred_gnn))
                
                # Adamic-Adar scores
                y_pred_adamic = adamic_adar_score(edge_index, q.item(), num_test_node)
                metrics["Adamic-Adar"].append(compute_metrics(y_true, y_pred_adamic))
                
                # Common-Neighbors scores
                y_pred_common = common_neighbors_score(edge_index, q.item(), num_test_node)
                metrics["Common-Neighbors"].append(compute_metrics(y_true, y_pred_common))

            elif args.inference_method == 'R_LSH':
                # GNN scores
                all_node_embeddings, _ = gnn_inference(model, q, data)
                
                random_lsh = RandomLSH(input_dim=32, num_hashes=8)

                hash_codes = random_lsh.hash(all_node_embeddings)
                buckets = random_lsh.group_by_hash(hash_codes)

                query_embedding = all_node_embeddings[q]  # Example query node embedding
                top_ks_neighbors = []
                for k in [1, 5, 10, -1]: # first 1, 5, 10 and all the nodes in the bucket sorted in cosine similarity score manner
                    top_ks_neighbors.append(random_lsh.infer(all_node_embeddings[q], buckets, all_node_embeddings, top_k=k))
                
                true_neighbors = edge_index[1][(edge_index[0] == q).nonzero(as_tuple=True)]
                metrics["GNN"].append(compute_metrics_lsh(true_neighbors, top_ks_neighbors))
                
                # Adamic-Adar scores
                y_pred_adamic = adamic_adar_score(edge_index, q.item(), num_test_node)
                metrics["Adamic-Adar"].append(compute_metrics(y_true, y_pred_adamic))
                
                # Common-Neighbors scores
                y_pred_common = common_neighbors_score(edge_index, q.item(), num_test_node)
                metrics["Common-Neighbors"].append(compute_metrics(y_true, y_pred_common))

            elif args.inference_method == 'T_LSH':
                # GNN scores
                all_node_embeddings, _ = gnn_inference(model, q, data)
                trained_lsh = TrainableLSH(input_dim=32, num_hashes=8)
                trained_lsh.load_state_dict(torch.load(args.trained_lsh))
                trained_lsh.eval()
                trained_lsh.to(all_node_embeddings.device)
                buckets = trained_lsh.bucketize(all_node_embeddings)
                query_embedding = all_node_embeddings[q]  # Example query node embedding
                top_ks_neighbors = []
                for k in [1, 5, 10, -1]: # first 1, 5, 10 and all the nodes in the bucket sorted in cosine similarity score manner
                    top_ks_neighbors.append(trained_lsh.infer(all_node_embeddings[q], buckets, all_node_embeddings, top_k=k))
                
                true_neighbors = edge_index[1][(edge_index[0] == q).nonzero(as_tuple=True)]
                metrics["GNN"].append(compute_metrics_lsh(true_neighbors, top_ks_neighbors))


                # Adamic-Adar scores
                y_pred_adamic = adamic_adar_score(edge_index, q.item(), num_test_node)
                metrics["Adamic-Adar"].append(compute_metrics(y_true, y_pred_adamic))
                
                # Common-Neighbors scores
                y_pred_common = common_neighbors_score(edge_index, q.item(), num_test_node)
                metrics["Common-Neighbors"].append(compute_metrics(y_true, y_pred_common))

            else:
                print(args.inference_method)
                raise ValueError("Something wrong with argument inference_method, value passed: ", args.inference_method)
    
    # Summarize metrics

    print(f'Faster Inference method: {args.inference_method}')
    for method, results in metrics.items():
        avg_metrics = {key: sum(res[key] for res in results) / len(results) for key in results[0]}
        print(f"{method} Metrics:")
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.4f}")

def print_run_summary(args):
    """
    Print a summary of the run based on the provided arguments.
    """
    print("\n=========== Run Summary ===========")
    print(f"Dataset: {args.dataset}")
    print(f"GNN Method: {args.gnn_method}")
    testModelPath = os.path.join(args.save_dir, f'{args.gnn_method}_{args.dataset}_model.pth')
    print(f'Test Model Path: {testModelPath}')
    print(f'Inference Method: {args.inference_method}')
    print(f"Device: {args.device}")
    print("===================================\n")

if __name__ == '__main__':
    args = get_args()
    print_run_summary(args)
    test(args)