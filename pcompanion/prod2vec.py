import torch
import torch.nn as nn
# from torch_geometric.nn import GATConv


# class GNNProd2Vec(nn.Module):
#     def __init__(self, in_dim=300, out_dim=128, fix_gat_parameters=True):
#         super().__init__()
#
#         self.gat = GATConv(out_dim,
#                            out_dim,
#                            add_self_loops=False,
#                            bias=False)
#
#         self.mlp = nn.Sequential(
#             nn.BatchNorm1d(in_dim),
#             nn.Linear(in_dim, in_dim),
#             nn.Tanh(),
#             nn.Linear(in_dim, in_dim),
#             nn.Tanh(),
#             nn.Linear(in_dim, out_dim)
#         )
#
#         if fix_gat_parameters:
#             self._fix_gat_parameters()
#
#     def _fix_gat_parameters(self):
#         """
#         Makes sure that GAT works the same as in paper
#         """
#         self.gat.lin_src.weight = torch.nn.Parameter(torch.zeros_like(self.gat.lin_src.weight).fill_diagonal_(1),
#                                                      requires_grad=False)
#
#         self.gat.att_src = torch.nn.Parameter(torch.ones_like(self.gat.att_src),
#                                               requires_grad=False)
#
#         self.gat.att_dst = torch.nn.Parameter(torch.ones_like(self.gat.att_dst),
#                                               requires_grad=False)
#
#     def forward(self, x, edge_index, labels):
#         x_mlp = self.mlp(x)
#         positive = self.gat(x_mlp, edge_index[:, labels == 1])
#         negative = self.gat(x_mlp, edge_index[:, labels == -1])
#         return x_mlp, positive, negative

class Prod2Vec(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mlp = nn.Linear(input_dim, output_dim)

    def _attention_agg(self, item_i, set_j):
        attention = nn.functional.softmax((item_i.unsqueeze(1) * set_j).sum(2))
        return (set_j * attention.unsqueeze(-1)).sum(1)

    def forward(self, x, positive, negative):
        x = self.mlp(x)
        positive = self.mlp(positive)
        negative = self.mlp(negative)

        return (
            x,
            self._attention_agg(x, positive),
            self._attention_agg(x, negative)
        )
