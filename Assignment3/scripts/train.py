import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.optim import Adam

from parser import get_args
from prepareDataset import prepare_dataset
from models import getModel

def train_model(args):
    datasetSplit = prepare_dataset(args)
    model = getModel(args)

    optimizer = Adam(model.parameters(), lr=0.01)

    train_loader = DataLoader([train_data], batch_size=1, shuffle=True)
    val_loader = DataLoader([val_data], batch_size=1, shuffle=False)
    test_loader = DataLoader([test_data], batch_size=1, shuffle=False)

    # Training loop
    model.train()
    for epoch in range(50):
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.mse_loss(out, data.x)  # Example loss; replace with link prediction loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    # Validation loop
    model.eval()
    for data in val_loader:
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            # Perform validation step


if __name__ == "__main__":
    args = get_args()
    train_model(args)
