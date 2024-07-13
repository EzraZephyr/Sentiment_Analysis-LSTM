import torch
from torch.utils.data import Dataset

class data_process(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    # Initialize to receive data and target values

    def __len__(self):
        return len(self.X)
    # Return the length of the data, allowing the dataloader to handle it based on batch size

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)
    # Convert data to tensors and return
