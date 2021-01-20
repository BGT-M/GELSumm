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

from models.summGCN import SummGCN
from utils import accuracy, f1, load_data, normalize, to_torch

logger = logging.getLogger('summGCN')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s %(filename)s %(lineno)d %(levelname)s: %(message)s')

time_str = time.strftime('%Y-%m-%d-%H-%M')

parser = ArgumentParser()
parser.add_argument("--dataset", required=True, type=str,
                    help="Dataset name.")
parser.add_argument("--cuda", type=int, default=-1,
                    help="GPU id (-1 to use cpu).")
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
parser.add_argument("--degree", type=int, default=2,
                    help="Degree of graph filter")
parser.add_argument("--log_turn", type=int, default=10,
                    help="Number of turn to log")
args = parser.parse_args()

if len(logger.handlers) < 2:
    filename = f'summGCN_{args.dataset}_{time_str}.log'
    file_handler = logging.FileHandler(filename, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

logger.debug("Args:")
logger.debug(args)

gpu_id = args.cuda
if not torch.cuda.is_available():
    gpu_id = -1
np.random.seed(args.seed)
torch.manual_seed(args.seed)

R, S, adj, adj_s, features, labels, full_labels, indices, full_indices = load_data(
    args.dataset)
idx_train, idx_val, idx_test = indices['train'], indices['val'], indices['test']
N, d = features.shape
n = S.shape[1]
nclass = full_labels.max().item() + 1
logger.info(f"Dataset loaded. N: {N}, n: {n}, feature: {d}-dim")
logger.info(
    f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

if args.type == "rw":
    adj = normalize(adj)
    adj_s = normalize(adj_s)
else:
    adj += ssp.eye(N)
    degs = np.array(adj.sum(axis=1)).flatten()
    D_inv_sqrt = ssp.diags(np.power(degs, -0.5))
    adj = D_inv_sqrt @ adj @ D_inv_sqrt
    adj_s = R @ adj_s @ R
features = (features-features.mean(axis=0)) / features.std(axis=0)
features_s = S.T @ features

S, adj, adj_s, features, features_s, labels, full_labels = to_torch(S), to_torch(adj), to_torch(
    adj_s), to_torch(features), to_torch(features_s), to_torch(labels), to_torch(full_labels)
idx_train, idx_val, idx_test = to_torch(
    idx_train), to_torch(idx_val), to_torch(idx_test)

device = f"cuda:{gpu_id}"
if gpu_id >= 0:
    adj = adj.cuda(device)
    adj_s = adj_s.cuda(device)
    S = S.cuda(device)
    features = features.cuda(device)
    features_s = features_s.cuda(device)
    labels = labels.cuda(device)
    full_labels = full_labels.cuda(device)
    idx_train = idx_train.cuda(device)
    idx_val = idx_val.cuda(device)
    idx_test = idx_test.cuda(device)

model = SummGCN(d, args.hidden, nclass, dropout=args.dropout)
if gpu_id >= 0:
    model = model.cuda(device)


def train(model, epochs):
    max_val_acc = 0.0
    best_params = None
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(epochs):
        start = time.time()
        model.train()
        optimizer.zero_grad()
        output = model((features_s, adj_s))
        output = F.log_softmax(output, dim=1)

        y_, y = output[idx_train], labels[idx_train]
        loss_train = F.nll_loss(y_, y)
        acc_train = accuracy(y_, y)
        loss_train.backward()
        optimizer.step()

        model.eval()
        embeds = model((features_s, adj_s))
        output = F.log_softmax(embeds, dim=1)
        y_, y = output[idx_val], labels[idx_val]
        loss_val = F.nll_loss(y_, y)
        acc_val = accuracy(y_, y)
        f1_micro, f1_macro = f1(y_, y)
        if acc_val.cpu().item() > max_val_acc:
            max_val_acc = acc_val.cpu().item()
            best_params = model.state_dict()
        end = time.time()

        message = "{} {} {} {} {} {} {} {}".format(
            "Epoch: {:04d}".format(epoch+1),
            "loss_train: {:.4f}".format(loss_train.cpu().item()),
            "acc_train: {:.4f}".format(acc_train.cpu().item()),
            "loss_val: {:.4f}".format(loss_val.cpu().item()),
            "acc_val: {:.4f}".format(acc_val.cpu().item()),
            "f1 micro: {:.4f}".format(f1_micro),
            "f1 macro: {:.4f}".format(f1_macro),
            "time: {:.2f} s".format(end - start)
        )
        if args.log_turn <= 0:
            logger.debug(message)
        elif epoch % args.log_turn == args.log_turn - 1:
            logger.info(message)
        else:
            logger.debug(message)

        optimizer.step()
    model.load_state_dict(best_params)


def test(model):
    global S, adj, full_labels
    model.eval()
    output = model((features_s, adj_s))
    output = output.cpu()
    S = S.cpu()
    adj = adj.cpu()
    full_labels = full_labels.cpu()

    output = torch.spmm(S, output)
    for _ in range(args.degree):
        output = torch.spmm(adj, output)
    output = F.log_softmax(output, dim=1)
    f_idx_test = full_indices['test']
    f_idx_test = to_torch(f_idx_test)
    y, y_ = full_labels[f_idx_test], output[f_idx_test]
    loss_test = F.nll_loss(y_, y)
    acc_test = accuracy(y_, y)
    f1_micro, f1_macro = f1(y_, y)
    message = f"Test set results: loss= {loss_test.item():.4f} accuracy= {acc_test.item():.4f} f1 micro= {f1_micro:.4f} f1 macro= {f1_macro:.4f}"
    logger.info(message)


if __name__ == "__main__":
    logger.info("Start training...")
    start_time = time.time()
    train(model, args.epochs)
    logger.info(f"Training completed, costs {time.time()-start_time} seconds.")

    test(model)
    if not os.path.exists(os.path.join('output', args.dataset)):
        os.makedirs(os.path.join('output', args.dataset))
    pickle.dump(model.state_dict(), open(os.path.join(
        'output', args.dataset, f'model_{time_str}.pkl'), 'wb'))
