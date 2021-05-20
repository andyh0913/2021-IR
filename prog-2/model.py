import torch
from torch import nn

class MF(nn.Module):
    def __init__(self, user_num, item_num, n_factor):
        super().__init__()
        self.user_emb = nn.Embedding(user_num, n_factor)
        self.item_emb = nn.Embedding(item_num, n_factor)
        self.user_bias = nn.Embedding(user_num, 1)
        self.item_bias = nn.Embedding(item_num, 1)

    def forward(self, x):
        # x.shape = (batch_size, 2)

        user_emb = self.user_emb(x[:,0])
        item_emb = self.item_emb(x[:,1])
        return (user_emb * item_emb).sum(axis=1)

class MFBPR(nn.Module):
    def __init__(self, user_num, item_num, n_factor):
        super().__init__()
        self.user_emb = nn.Embedding(user_num, n_factor)
        self.item_emb = nn.Embedding(item_num, n_factor)

    def forward(self, x):
        # x.shape = (batch_size, 3)
        assert x.shape[1] == 3

        user_emb = self.user_emb(x[:,0])
        pos_emb = self.item_emb(x[:,1])
        neg_emb = self.item_emb(x[:,2])
        return (user_emb * (pos_emb-neg_emb)).sum(axis=1)

    def get_ratings(self, users, items):
        user_emb = self.user_emb(users)
        item_emb = self.item_emb(items)
        return torch.mm(user_emb, item_emb.transpose(0,1))