import os

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import negative_sampling, subgraph, to_dense_adj

from parser import get_args

torch.manual_seed(42)

def prepare_dataset(args):
    dataset_name = args.dataset
    split_ratios = args.splits
    if dataset_name == "CoraFull":
        from torch_geometric.datasets import CoraFull as Dataset
    elif dataset_name == "DeezerEurope":
        from torch_geometric.datasets import DeezerEurope as Dataset
    else:
        raise ValueError("Unsupported dataset")

    dirPath = os.path.join(args.dataset_dir, dataset_name)
    dataset = Dataset(root=dirPath)
    args.num_nodes = dataset[0].num_nodes
    args.num_features = dataset[0].num_features
    train_data, val_data, test_data, Q = manual_split(dataset[0], args)

    dataDir = os.path.join(args.dataset_dir, 'preprocessed_data', f'{args.dataset}')
    if not os.path.exists(dataDir):
        os.makedirs(dataDir)
        torch.save(train_data, os.path.join(dataDir, f'train.pth'))
        torch.save(val_data, os.path.join(dataDir, f'val.pth'))
        torch.save(test_data, os.path.join(dataDir, f'test.pth'))
        torch.save(Q, os.path.join(dataDir, f'q.pth'))
        print(f'Saved prepocessed data at {dataDir}')
     
    return train_data, val_data, test_data, Q


def manual_split(data, args):

    num_nodes = data.num_nodes
    
    train_nodes, temp_nodes = train_test_split(range(num_nodes), test_size=args.splits[1] + args.splits[2], random_state=42)
    val_size = int(len(temp_nodes) * (args.splits[1] / (args.splits[1] + args.splits[2])))
    val_nodes, test_nodes = temp_nodes[:val_size], temp_nodes[val_size:]
    
    train_edge_index, _ = subgraph(train_nodes, data.edge_index, relabel_nodes=True)
    val_edge_index, _ = subgraph(val_nodes, data.edge_index, relabel_nodes=True)
    test_edge_index, _ = subgraph(test_nodes, data.edge_index, relabel_nodes=True)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_nodes] = True
    train_data = data.clone()
    train_data.edge_index = train_edge_index
    train_data.x = data.x[train_mask]

    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask[val_nodes] = True
    val_data = data.clone()
    val_data.edge_index = val_edge_index
    val_data.x = data.x[val_mask]

    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask[test_nodes] = True
    test_data = data.clone()
    test_data.edge_index = test_edge_index
    test_data.x = data.x[test_mask]
    test_data.global_id = torch.tensor(test_nodes, dtype=torch.long)
    
    # Identify 1000 nodes in Q with at least one triangle
    qPath = os.path.join(args.dataset_dir, f'{args.dataset}_q.pth')
    if os.path.exists(qPath):
        Q = torch.load(qPath)
        print(f'Loaded existing Q with {len(Q)} nodes.')
    else:
        Q = nodes_in_triangles(test_data.edge_index, test_data.num_nodes)

    return train_data, val_data, test_data, Q

# def find_triangle_nodes(edge_index, num_nodes, required_count=1000):
#     # Create adjacency matrix
#     adj = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    
#     # Find nodes that are part of at least one triangle
#     triangle_nodes = set()
#     for node in range(num_nodes):
#         neighbors = torch.nonzero(adj[node], as_tuple=False).squeeze(1)
#         for i in range(len(neighbors)):
#             for j in range(i + 1, len(neighbors)):
#                 if adj[neighbors[i], neighbors[j]] == 1:  # Check if neighbors are connected
#                     triangle_nodes.add(node)
#                     break
#             if node in triangle_nodes:
#                 break  # Break early if node already added
    
#     # Sample the required number of nodes if more than required_count
#     triangle_nodes = list(triangle_nodes)
#     print(f'Found {len(triangle_nodes)} triangle nodes in graph !')
#     if len(triangle_nodes) > required_count:
#         triangle_nodes = torch.tensor(triangle_nodes)[:required_count]
#     else:
#         triangle_nodes = torch.tensor(triangle_nodes)

#     return triangle_nodes


def nodes_in_triangles(edge_index, num_nodes, required=1000):
    adj_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    adj_matrix[edge_index[0], edge_index[1]] = 1
    adj_matrix[edge_index[1], edge_index[0]] = 1
    
    # Compute A^2 and A^3
    A2 = torch.mm(adj_matrix, adj_matrix)  # A^2
    A3 = torch.mm(adj_matrix, A2)  # A^3

    # Find nodes with non-zero diagonal in A^3 (part of at least one triangle)
    triangle_nodes = torch.nonzero(torch.diag(A3) > 0, as_tuple=False).squeeze()

    return triangle_nodes

def main():
    args = get_args()
    train_data, val_data, test_data, Q = prepare_dataset(args)

    print(f"Number of nodes: {args.num_nodes}")
    assert args.num_nodes == train_data.num_nodes + val_data.num_nodes + test_data.num_nodes, "Total nodes and split nodes not matching !!!"
    print(f'Training Nodes: {train_data.num_nodes}')
    print(f"Training edges: {train_data.edge_index.shape[1]}")
    print(f'Validation Nodes: {val_data.num_nodes}')
    print(f"Validation edges: {val_data.edge_index.shape[1]}")
    print(f'Test Nodes: {test_data.num_nodes}')
    print(f"Testing edges: {test_data.edge_index.shape[1]}")

if __name__ == '__main__':
    main()