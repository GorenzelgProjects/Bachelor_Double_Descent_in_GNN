import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

# Create a custom dataset class if needed
class OGBLoader(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return 1  # Single graph dataset

    def __getitem__(self, idx):
        return self.data