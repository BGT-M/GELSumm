import torch
import torch.nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class SummGCN(Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5) -> None:
        super(SummGCN, self).__init__()
        self.layer1 = SummGCNLayer(in_dim, hidden_dim)
        self.layer2 = SummGCNLayer(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, inputs):
        x, R = *inputs
        x = F.relu(self.layer1(x, R))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.layer2(x, R)
        return F.log_softmax(x, dim=1)


class SummGCNLayer(Module):
    def __init__(self, in_dim, out_dim) -> None:
        super(SummGCNLayer, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_dim, out_dim))

    def forward(self, x, R):
        x = torch.matmul(x, self.weight)
        x = torch.spmm(R, x)
        return x
