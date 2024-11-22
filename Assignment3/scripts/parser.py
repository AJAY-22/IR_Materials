import argparse

import torch

def get_args():
    parser = argparse.ArgumentParser(description="Link Prediction Task Argument Parser")

    parser.add_argument(
        "--dataset",
        type=str,
        default="CoraFull",
        choices=["CoraFull", "DeezerEurope"],
        help="Name of the dataset to use for the task (default: CoraFull)",
    )
    
    parser.add_argument(
        "--splits",
        type=float,
        nargs=3,
        default=[0.6, 0.2, 0.2],
        help="Train, Validation, and Test splits as a list of three floats (default: [0.6, 0.2, 0.2])",
    )

    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="./data",
        help="Directory path where datasets will be downloaded and processed (default: ./data)",
    )

    parser.add_argument(
        "--gnn_method",
        type=str,
        default="GAT",
        choices=["GCN", "GAT", "GIN"],
        help="GNN model to use for training (default: GCN)",
    )

    parser.add_argument(
        "--batch_size", "-bs",  # Long form and alias
        type=int,
        default=32,
        dest="batch_size",  # Internal variable name
        help="Batch size for training (default: 32)"
    )

    parser.add_argument(
        "--num_epoch", "-e",  # Long form and alias
        type=int,
        default=50,
        dest="num_epoch",  # Internal variable name
        help="Num of epochs for training (default: 50)"
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=1.0,
        help="Tunable margin for AUC surrogate loss (default: 1.0)"
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="./saved_models/",
        help="Directory path where datasets will be downloaded and processed (default: ./data)",
    )
    
    args = parser.parse_args()

    if not abs(sum(args.splits) - 1.0) < 1e-6:
        parser.error("Splits must sum to 1.0")

    return args

if __name__ == "__main__":
    args = get_args()
    print(f"Dataset: {args.dataset}")
    print(f"Splits: Train={args.splits[0]}, Val={args.splits[1]}, Test={args.splits[2]}")
