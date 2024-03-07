from torch import nn

class ComplementaryTypeEncoder(nn.Module):

    def __init__(self, n_categories, n_dim=64, dropout_p=0.2,
                 hidden_dim=32):
        super().__init__()

        self.main_cat_embd = nn.Embedding(n_categories, n_dim)
        self.complementary_cat_embd = nn.Embedding(n_categories, n_dim)

        self.encoder = nn.Sequential(
            nn.Linear(n_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )
        self.decoder = nn.Linear(hidden_dim, n_dim)

    def forward(self, x):
        x_main = self.main_cat_embd(x[:, 0])
        x_compl = self.complementary_cat_embd(x[:, 1:])

        x_main = self.encoder(x_main)
        x_main = self.decoder(x_main)

        return x_main, x_compl


class ComplementaryItem(nn.Module):

    def __init__(self, type_dim=64, prod2vec_dim=128):
        super().__init__()

        self.prod2vec_dim = prod2vec_dim
        self.mlp = nn.Linear(type_dim, prod2vec_dim)

    def forward(self, item_embds, type_prediction):
        return item_embds * self.mlp(type_prediction).view(-1, 1, self.prod2vec_dim)
