from torch import nn

from common_nn import ComplementaryItem, ComplementaryTypeEncoder
from prod2vec import Prod2Vec



class PCompanion(nn.Module):
    def __init__(self,
                 prod2vec: Prod2Vec,
                 complementary_item: ComplementaryItem,
                 complementary_type_encoder:ComplementaryTypeEncoder
                 ):
        super().__init__()
        self.complementary_type_encoder = complementary_type_encoder
        self.complementary_item = complementary_item
        self.prod2vec = prod2vec

    def forward(self, x, positive, negative, category):
        item_i, pos_item_j, neg_item_j = self.prod2vec(x, positive, negative)

        # x_main, x_compl = self.complementary_type_encoder(x)
        # item_embds, type_prediction = self.complementary_item(x_main, x_compl)

        return item_i, pos_item_j, neg_item_j
