import torch
from typing import NamedTuple

class Example(NamedTuple):
    x: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor
    category: torch.Tensor

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, dim=30, n=10):
        super().__init__()
        self.features = torch.rand(n, dim)
        self.category = torch.randint(0, 10, (n,)).long()
        self.features_dim = dim

        self._len = n

    def __len__(self):
        return self._len

    def __getitem__(self, idx) -> Example:
        n_positives = torch.randint(1, 10, (1,)).item()
        n_negatives = torch.randint(1, 10, (1,)).item()
        positives = torch.rand((n_positives,self.features_dim))
        negatives = torch.rand((n_negatives,self.features_dim))
        return Example(
            self.features[idx],
            positives,
            negatives,
            self.category[idx]
        )

class SyntheticGNNDataset(torch.utils.data.Dataset):
    def __init__(self, dim=30, n=1000):
        super().__init__()
        self.edge_index = torch.randint(0,n, (1000,2)).long()
        self.labels = torch.randint(0,2, (1000,)).long()
        self.x = torch.rand(n, dim)
        self.category = torch.randint(0, 10, (n,)).long()

        self._len = n

    def __len__(self):
        return self._len

    def __getitem__(self, idx) -> Example:
        return Example(self.x[idx], self.edge_index[idx], self.category[idx], self.labels[idx])
