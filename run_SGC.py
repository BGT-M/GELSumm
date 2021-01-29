import logging
import os
import pickle
import time
from argparse import ArgumentParser

import numpy as np
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
import torch.optim as optim

from models.summSGC import SummSGC
from utils import accuracy, aug_normalized_adjacency, load_dataset, normalize, sgc_precompute, to_torch, f1

logger = logging.getLogger('summSGC')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s %(filename)s %(lineno)d %(levelname)s: %(message)s')

time_str = time.strftime('%Y-%m-%d-%H-%M')
if len(logger.handlers) < 2:
    filename = f'summSGC_{time_str}.log'
    file_handler = logging.FileHandler(filename, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

parser = ArgumentParser()
parser.add_argument("--dataset", required=True, type=str,
                    help="Dataset name.")
parser.add_argument("--cuda", type=int, default=-1,
                    help="GPU id(-1 to use cpu).")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed.")
parser.add_argument("--epochs", type=int, default=2,
                    help="Number of epochs to train.")
parser.add_argument("--lr", type=float, default=1,
                    help="Initial learning rate.")
parser.add_argument("--weight_decay", type=float, default=5e-4,
                    help="Weight decay (L2 loss on parameters).")
parser.add_argument("--degree", type=int, default=2,
                    help="degree of the approximation.")
args = parser.parse_args()
logger.debug("Args:")
logger.debug(args)

gpu_id = args.cuda
if not torch.cuda.is_available():
    gpu_id = -1
np.random.seed(args.seed)
torch.manual_seed(args.seed)

_, S, adj, adj_s, features, labels, idx_train, idx_val, idx_test = load_dataset(
    args.dataset)
N, d = features.shape
n = S.shape[1]
nclass = labels.max().item() + 1
logger.info(f"Dataset loaded. N: {N}, n: {n}, feature: {d}-dim")
logger.info(
    f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

# Preprocessing
adj_train = adj[idx_train, :][:, idx_train]
adj = aug_normalized_adjacency(adj)
adj_train = aug_normalized_adjacency(adj_train)
features = (features - features.mean(axis=0)) / features.std(axis=0)
# Precompute
S, adj, adj_s, features, labels = to_torch(S), to_torch(adj), to_torch(
    adj_s), to_torch(features), to_torch(labels)
idx_train, idx_val, idx_test = to_torch(
    idx_train), to_torch(idx_val), to_torch(idx_test)
features, precompute_time = sgc_precompute(features, adj, args.degree)
features_s = torch.spmm(S.transpose(0, 1), features)

device = f"cuda:{gpu_id}"
if gpu_id >= 0:
    adj = adj.cuda(device)
    adj_s = adj_s.cuda(device)
    S = S.cuda(device)
    features_s = features_s.cuda(device)
    labels = labels.cuda(device)
    idx_train = idx_train.cuda(device)
    idx_val = idx_val.cuda(device)
    idx_test = idx_test.cuda(device)

model = SummSGC(d, nclass, S)
if gpu_id >= 0:
    model = model.cuda(device)


def train(model):
    optimizer = optim.LBFGS(model.parameters(), lr=args.lr)
    model.train()

    def closure():
        optimizer.zero_grad()
        output = model(features_s)
        loss_train = F.cross_entropy(output, labels)
        loss_train.backward()
        return loss_train
    start = time.time()
    for epoch in range(args.epochs):
        loss_train = optimizer.step(closure)
    train_time = time.time() - start
    logger.info(f"Training completes, costs {train_time} seconds.")
    return model, train_time


def test(model):
    model.eval()
    output = model(features_s)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    f1_micro, f1_macro = f1(output[idx_test], labels[idx_test])

    logger.info("Test result:")
    message = "Loss: {:.4f} accuracy: {:.4f} f1 micro= {:.4f} f1 macro= {:.4f}".format(
        loss_test.item(), acc_test.item(), f1_micro, f1_macro)
    logger.info(message)


if __name__ == '__main__':
    train(model)
    test(model)
