import torch
from torch.utils.data import DataLoader
from model import PCompanion
from prod2vec import Prod2Vec
from common_nn import ComplementaryTypeEncoder, ComplementaryItem
from synthetic_data import SyntheticDataset
from loss import ContrastiveHingeLoss
from torch.nn.utils.rnn import pad_sequence
from typing import NamedTuple

class Batch(NamedTuple):
    x: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor
    category: torch.Tensor

dataset = SyntheticDataset()

def custom_collate(batch):
    features, pos, neg, categories = list(zip(*batch))

    return Batch(
        torch.vstack(features),
        pad_sequence(pos, batch_first=True),
        pad_sequence(neg, batch_first=True),
        torch.hstack(categories)
    )

# @hydra.main(config_path="../", config_name="config")
def main():
    loader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate)
    # pcompanion = hydra.utils.instantiate(config)
    pcompanion = PCompanion(
        Prod2Vec(30, 30),
        ComplementaryItem(10, 30),
        ComplementaryTypeEncoder(10, 30)
    )
    criterion = ContrastiveHingeLoss()
    optimizer = torch.optim.Adam(pcompanion.parameters(), lr=0.001)
    for epoch in range(10):
        for batch in loader:
            anchor, positive, negative = pcompanion.prod2vec(batch.x, batch.positive, batch.negative)
            loss = criterion(anchor, positive, negative)
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    main()
