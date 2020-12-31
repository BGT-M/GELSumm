from functools import total_ordering
import time
from argparse import ArgumentParser
from utils import load_data

import numpy as np
import scipy.sparse as ssp
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import profiler

from summGCN import SummGCN
from utils import load_data, accuracy, to_torch, normalize

parser = ArgumentParser()
parser.add_argument("--dataset", required=True, type=str,
                    help="Dataset name.")
parser.add_argument("--cuda", type=int, default=-1,
                    help="GPU id(-1 to use cpu).")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
parser.add_argument("--epochs", type=int, default=200,
                    help="Number of epochs to train.")
parser.add_argument("--lr", type=float, default=0.01,
                    help="Initial learning rate.")
parser.add_argument("--weight_decay", type=float, default=5e-4,
                    help="Weight decay (L2 loss on parameters).")
parser.add_argument("--hidden", type=int, default=16,
                    help="Number of hidden units.")
parser.add_argument("--dropout", type=float, default=0.5,
                    help="Dropout rate (1 - keep probability).")
parser.add_argument("--type", type=str, choices=["rw", "symm"], default="rw",
                    help="Aggregation type")
parser.add_argument("--log_turn", type=int, default=10,
                    help="Number of turn to log")
args = parser.parse_args()

gpu_id = args.cuda
if not torch.cuda.is_available():
    gpu_id = -1
np.random.seed(args.seed)
torch.manual_seed(args.seed)

R, S, A, A_s, features, labels, idx_train, idx_val, idx_test = load_data(
    args.dataset)
N, d = features.shape
n = S.shape[1]
nclass = labels.max().item() + 1

features = normalize(features)
features_s = S.T @ features
if args.type == "rw":
    adj = normalize(A_s)
else:
    adj = R @ A_s @ R
A += ssp.eye(N)
degs = np.array(A.sum(axis=1)).squeeze()
D_inv = ssp.diags(np.power(np.sqrt(degs), -1))
A = D_inv @ A @ D_inv

S, A, adj, features_s, labels = to_torch(S), to_torch(A), to_torch(
    adj), to_torch(features_s), to_torch(labels)
idx_train, idx_val, idx_test = to_torch(
    idx_train), to_torch(idx_val), to_torch(idx_test)

device = f"cuda:{gpu_id}"
if gpu_id >= 0:
    A = A.cuda(device)
    adj = adj.cuda(device)
    S = S.cuda(device)
    features_s = features_s.cuda(device)
    labels = labels.cuda(device)
    idx_train = idx_train.cuda(device)
    idx_val = idx_val.cuda(device)
    idx_test = idx_test.cuda(device)

model = SummGCN(d, args.hidden, nclass, S, dropout=args.dropout)
if gpu_id >= 0:
    model = model.cuda(device)


if __name__ == "__main__":
    max_val_acc = 0.0
    best_params = None
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    start = time.time()
    with profiler.profile(use_cuda=True) as prof:
        model.train()
        optimizer.zero_grad()
        embeds = model((features_s, adj))
        # embeds = torch.spmm(A, embeds)
        output = F.log_softmax(embeds, dim=1)

        y_, y = output[idx_train], labels[idx_train]
        loss_train = F.nll_loss(y_, y)
        acc_train = accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    end = time.time()
    print(f"{end-start:.2f} seconds")
    print(prof.table(sort_by="self_cpu_time_total"))
    prof.export_chrome_trace('./output/profile.json')
