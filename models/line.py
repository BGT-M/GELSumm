import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def line(G, epochs=10, neg_size=5):
    G = G
    dataset = LINEDataset(G, neg_size)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=256, num_workers=8)

    model = LINE(G.number_of_nodes(), 128)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        for i, data in enumerate(dataloader):
            u, v, w = data
            u = u.view(-1)
            v = v.view(-1)
            w = w.view(-1)
            loss = model(u, v, w)
            loss *= (neg_size + 1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return model.emb.numpy()


class LINEDataset(Dataset):
    def __init__(self, G, neg_size=5):
        self.G = G
        self.edges = list(G.edges())
        self.neg_size = neg_size
        degrees = np.array([G.degree(n) for n in G.nodes()])
        self.prob = np.power(degrees, 0.75)
        self.cum_prob = np.cumsum(self.prob)

        self.M = len(self.prob) * 10
        table = []
        idx = 0
        ms = np.linspace(0, 1, self.M)
        for m in ms:
            if m >= self.prob[idx]:
                idx += 1
            table.append(idx)
        self.table = table

    def __len__(self):
        return self.G.number_of_edges()

    def __getitem__(self, idx):
        u, v  = self.edges[idx]
        w = self.G[u][v]['wgt']
        us, vs, ws = [u], [v], [w]

        for _ in range(self.neg_size):
            v = self._neg_sample(u, v)
            us.append(u)
            vs.append(v)
            ws.append(self.G[u][v]['wgt'])
            
        return torch.LongTensor(us), torch.LongTensor(vs), torch.FloatTensor(ws)

    def _neg_sample(self, u, v):
        while True:
            m = random.randint(0, self.M-1)
            node = self.table[m]
            if node != u and node != v:
                break
        return node


class LINE(nn.Module):
    def __init__(self, N, dim=128):
        super(LINEModel, self).__init__()
        self.dim = dim
        self.N = N
        self.emb = nn.Embedding(N, dim)
        self.ctx = nn.Embedding(N, dim)

    def forward(self, u, v, w):
        x = self.emb(u)
        y = self.ctx(v)
        x = w * torch.sum(x1 * x2, dim=1)
        return -F.logsigmoid(x).mean()

