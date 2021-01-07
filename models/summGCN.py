import math

import torch
import torch.nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class SummGCN(Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5, bias=True):
        super(SummGCN, self).__init__()
        self.layer1 = SummGCNLayer(in_dim, hidden_dim, bias=bias)
        self.layer2 = SummGCNLayer(hidden_dim, out_dim, bias=bias)
        self.dropout = dropout

    def forward(self, inputs):
        x, adj = inputs
        output = F.relu(self.layer1(x, adj))
        output = F.dropout(output, p=self.dropout, training=self.training)
        output = self.layer2(output, adj)

        return output


class SummGCNLayer(Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(SummGCNLayer, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_dim, out_dim))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_dim))
        else:
            self.register_parameter('bias', None)
        self.init_parameter()

    def forward(self, x, adj):
        output = torch.matmul(x, self.weight)
        output = torch.spmm(adj, output)
        if self.bias is not None:
            output += self.bias
        return output

    def init_parameter(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
