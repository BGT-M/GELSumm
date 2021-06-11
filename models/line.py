import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def run_LINE(G, epochs=50, neg_size=5):
    dataset = LINEDataset(G)
    print(len(dataset))
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1024, num_workers=8)

    model = LINE(G.number_of_nodes(), 128)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    epoch = 0
    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            total_loss = 0.0
            for i, data in enumerate(dataloader):
                u, v, w = data
                u = u.view(-1)
                v = v.view(-1)
                w = w.view(-1)
                loss = model(u, v, w)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss /= len(dataloader)
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch}, loss: {total_loss}")
    embeds1 = model.emb1.weight.detach().numpy()

    model.order = 2
    dataset = LINEDataset(G, neg_size)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1024, num_workers=8)
    epoch = 0
    with tqdm(total=epochs) as pbar:
        for epoch in range(epochs):
            total_loss = 0.0
            for i, data in enumerate(dataloader):
                u, v, w = data
                u = u.view(-1)
                v = v.view(-1)
                w = w.view(-1)
                loss = model(u, v, w)
                # loss *= (neg_size + 1)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss /= len(dataloader)
            pbar.update(1)
            pbar.set_description(f"Epoch {epoch}: loss: {total_loss}")
    embeds2 = model.emb2.weight.detach().numpy()
    embeds = np.hstack([embeds1, embeds2])

    return embeds


class LINEDataset(Dataset):
    def __init__(self, G, neg_size=5):
        super(LINEDataset, self).__init__()
        self.G = G
        self.edges = list(G.edges())

        self.neg_size = neg_size
        degrees = np.array([G.degree(n) for n in range(G.number_of_nodes())])
        self.prob = np.power(degrees, 0.75)
        self.prob /= np.sum(self.prob)
        self.cum_prob = np.cumsum(self.prob)

        self.M = len(self.cum_prob) * 10
        table = []
        idx = 0
        ms = np.linspace(0, 1, self.M)
        for m in ms:
            if m >= self.cum_prob[idx]:
                idx += 1
            table.append(min(idx, self.G.number_of_nodes()-1))
        self.table = table

    def __len__(self):
        return self.G.number_of_edges()

    def __getitem__(self, idx):
        u, v  = self.edges[idx]
        w = self.G[u][v]['weight']
        us, vs, ws = [u], [v], [w]

        for _ in range(self.neg_size):
            v = self._neg_sample(u, v)
            us.append(u)
            vs.append(v)
            ws.append(-1.0)
            
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
        super(LINE, self).__init__()
        self.dim = dim
        self.N = N
        self.emb1 = nn.Embedding(N, dim)
        self.emb2 = nn.Embedding(N, dim)
        self.ctx = nn.Embedding(N, dim)
        self.order = 1

    def forward(self, u, v, w):
        if self.order == 1:
            x1 = self.emb1(u)
            x2 = self.emb1(v)
        else:
            x1 = self.emb2(u)
            x2 = self.ctx(v)
        x = w * torch.sum(x1 * x2, dim=1)
        return -F.logsigmoid(x).mean()

