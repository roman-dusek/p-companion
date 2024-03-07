import torch
from torch import nn


class ContrastiveHingeLoss(nn.Module):
    def __init__(self, lamb=1, epsilon=1):
        """
        :param lamb:
        :param epsilon:
        """
        super().__init__()
        self._epsilon = epsilon
        self._lamb = lamb

    def forward(self, anchor, positive, negative):
        """
        :param pos_neg_labels:
        :param distances:
        """
        pos_neg_labels = torch.hstack([torch.ones(positive.shape[0]), -torch.ones(negative.shape[0])])
        distances = torch.vstack([anchor - positive, anchor - negative])

        return torch.sum((self._epsilon - pos_neg_labels * (self._lamb - torch.norm(distances, dim=-1))).clip(0))

    # def forward(self, pos_neg_labels, distances):
    #     """
    #     :param pos_neg_labels:
    #     :param distances:
    #     """
    #
    #     return torch.sum((self._epsilon - pos_neg_labels * (self._lamb - torch.norm(distances, dim=-1))).clip(0))
