import torch
import torch.nn as nn
import torch.nn.functional as F


class SummSGC(nn.Module):
    def __init__(self, nfeat, nclass, S):
        super(SummSGC, self).__init__()

        self.S = S
        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return torch.spmm(self.S, self.W(x))
