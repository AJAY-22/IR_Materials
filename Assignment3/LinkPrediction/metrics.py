import torch
import networkx as nx
import torch.nn.functional as F

# TODO: Look into pyG for metrics like linkprediction p@k and linkprediction map

def adamic_adar_score(edge_index, q, num_test_node):
    """
    Compute Adamic-Adar scores for a query node q.
    :param edge_index: Edge indices of the test graph.
    :param q: Query node.
    :param test_nodes: All test nodes.
    :return: Scores for the query node with each test node.
    """
    g = nx.Graph()
    g.add_nodes_from(range(num_test_node))
    g.add_edges_from(edge_index.t().tolist())  # Convert edge_index to edge list
    scores = nx.adamic_adar_index(g, [(q, node) for node in range(num_test_node) if q != node])
    return torch.tensor([score[2] for score in scores])  # Return only scores


def common_neighbors_score(edge_index, q, num_test_node):
    """
    Compute Common Neighbors scores for a query node q.
    :param edge_index: Edge indices of the test graph.
    :param q: Query node.
    :param test_nodes: All test nodes.
    :return: Scores for the query node with each test node.
    """
    g = nx.Graph()
    g.add_nodes_from(range(num_test_node))
    g.add_edges_from(edge_index.t().tolist())
    scores = [(len(list(nx.common_neighbors(g, q, node)))) for node in range(num_test_node)]
    return torch.tensor(scores, dtype=torch.float32)


def precision_at_k(y_true, y_pred_scores, k):
    """
    Compute Precision@K.
    :param y_true: Ground truth (binary labels for edges).
    :param y_pred_scores: Predicted scores for edges.
    :param k: The value of K.
    :return: Precision@K
    """
    sorted_indices = torch.argsort(y_pred_scores, descending=True)
    top_k_indices = sorted_indices[:k]
    top_k_truth = y_true[top_k_indices]
    precision = top_k_truth.sum().item() / k
    return precision


def mean_reciprocal_rank(y_true, y_pred_scores):
    """
    Compute Mean Reciprocal Rank (MRR).
    :param y_true: Ground truth (binary labels for edges).
    :param y_pred_scores: Predicted scores for edges.
    :return: MRR
    """
    sorted_indices = torch.argsort(y_pred_scores, descending=True)
    ranks = torch.nonzero(y_true[sorted_indices], as_tuple=False)
    if len(ranks) == 0:
        return 0.0  # No correct predictions
    reciprocal_rank = 1.0 / (ranks[0].item() + 1)  # First relevant item rank
    return reciprocal_rank


def compute_metrics(y_true, y_pred_scores, ks=[1, 5, 10]):
    """
    Compute all metrics: Precision@K and MRR.
    :param y_true: Ground truth (binary labels for edges).
    :param y_pred_scores: Predicted scores for edges.
    :param ks: List of K values for Precision@K.
    :return: Dictionary of metrics.
    """
    metrics = {}
    for k in ks:
        metrics[f"Precision@{k}"] = precision_at_k(y_true, y_pred_scores, k)
    metrics["MRR"] = mean_reciprocal_rank(y_true, y_pred_scores)
    return metrics



def compute_metrics_lsh(true_neighbors, top_ks_neighbors, ks=[1, 5, 10]):
    # Adjacency matrix for the test graph
    metrics = {}
    for i, k in enumerate(ks):
        precision_k = []
        reciprocal_ranks = []
        
        top_k_neighbors = top_ks_neighbors[i] 
        top_k_set = set(top_k_neighbors)
        true_set = set(true_neighbors.tolist())
        intersection = top_k_set & true_set

        precision_k.append(len(intersection) / k)
        
        precision_k = torch.tensor(precision_k).mean().item()
        metrics[f"Precision@{k}"] = precision_k

    # Compute Reciprocal Rank (MRR)
    for rank, node in enumerate(top_ks_neighbors[-1], start=1):
        if node in true_set:
            reciprocal_ranks.append(1.0 / rank)
            break
    else:
        reciprocal_ranks.append(0.0)

    # Compute mean metrics
    mrr = torch.tensor(reciprocal_ranks).mean().item()

    metrics['MRR'] = mrr

    return metrics

# # Example Usage
# query_nodes = torch.arange(100)  # Example query nodes
# metrics = compute_metrics_lsh(test_edge_index, query_nodes, embeddings, buckets, top_k=10)
# print("Metrics:", metrics)
