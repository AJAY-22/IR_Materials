import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv, GATConv, GINConv

from parser import get_args

torch.manual_seed(42)

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        layer_dims = [input_dim] + hidden_dim + [output_dim]
        
        self.layers = torch.nn.ModuleList([
            GCNConv(layer_dims[i], layer_dims[i + 1])
            for i in range(len(layer_dims) - 1)
        ])
        
    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:  # Apply ReLU for all but the last layer
            x = layer(x, edge_index).relu()
        x = self.layers[-1](x, edge_index)  # No activation for the final layer
        return x


class GAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GAT, self).__init__()
        layer_dims = [input_dim] + hidden_dim + [output_dim]
        heads = [1] + heads + [1]
        self.layers = torch.nn.ModuleList([
            GATConv(layer_dims[i] * heads[i], layer_dims[i + 1], heads[i+1])
            for i in range(len(layer_dims) - 1)
        ])
        
    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:  # Apply ReLU for all but the last layer
            x = layer(x, edge_index).relu()
        x = self.layers[-1](x, edge_index)  # No activation for the final layer
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
        model = GCN(input_dim=args.num_features, hidden_dim = [2048, 1024, 512, 256, 128, 64], output_dim=32)
    elif args.gnn_method == "GAT":
        model = GAT(input_dim=args.num_features, hidden_dim = [1024, 128], output_dim=32, heads=[4, 4])
    elif args.gnn_method == "GIN":
        model = GIN(input_dim=args.num_features, hidden_dim = 64, output_dim=32)
    else:
        raise ValueError(f"Unknown GNN method: {args.gnn_method}")

    return model

if __name__ == '__main__':
    args = get_args()
    
    model = get_gnn_model(args)