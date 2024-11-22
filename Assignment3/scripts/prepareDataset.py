import os

from parser import get_args
from torch_geometric.transforms import RandomLinkSplit
import torch
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
    args.num_features = dataset.num_features
    graph = dataset[0]

    splitter = RandomLinkSplit(num_val=split_ratios[1], num_test=split_ratios[2], is_undirected=True)
    train_data, val_data, test_data = splitter(graph)

    return train_data, val_data, test_data

def main():
    args = get_args()
    train_data, val_data, test_data = prepare_dataset(args)

    print(f"Number of nodes: {train_data.num_nodes}")
    print(f"Training edges: {train_data.edge_index.shape[1]}")
    print(f"Validation edges: {val_data.edge_index.shape[1]}")
    print(f"Testing edges: {test_data.edge_index.shape[1]}")

if __name__ == '__main__':
    main()