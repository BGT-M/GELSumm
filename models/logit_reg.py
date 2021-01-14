import torch
import torch.nn as nn
import torch.nn.functional as F

class LogitRegression(nn.Module):
    def __init__(self, nfeat, nclass):
        super(LogitRegression, self).__init__()

        self.W = nn.Linear(nfeat, nclass)

    def forward(self, x):
        return self.W(x)