import numpy as np
from torch.utils.data import Dataset


class ArxivDataset(Dataset):
    def __init__(self, data: np.ndarray, targets: np.ndarray):
        self.data = data
        self.targets = targets

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]
        return x, y

    def __len__(self):
        return len(self.data)
