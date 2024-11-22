import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, GINConv

from parser import get_args

torch.manual_seed(42)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x


class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GIN, self).__init__()
        self.conv1 = GINConv(Linear(input_dim, hidden_dim))
        self.conv2 = GINConv(Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

def get_gnn_model(args):
    if args.gnn_method == "GCN":
        model = GCN(input_dim=args.num_features, hidden_dim=64, output_dim=32)
    elif args.gnn_method == "GAT":
        model = GAT(input_dim=args.num_features, hidden_dim=64, output_dim=32, heads=4)
    elif args.gnn_method == "GIN":
        model = GIN(input_dim=args.num_features, hidden_dim=64, output_dim=32)
    else:
        raise ValueError(f"Unknown GNN method: {args.gnn_method}")

    return model

if __name__ == '__main__':
    args = get_args()
    
    model = get_gnn_model(args)