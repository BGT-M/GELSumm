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

from summGCN import SummGCN
from utils import accuracy, load_data, normalize, to_torch, f1

logger = logging.getLogger('summGCN')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s %(filename)s %(lineno)d %(levelname)s: %(message)s')

time_str = time.strftime('%Y-%m-%d-%H-%M')
if len(logger.handlers) < 2:
    filename = f'summGCN_{time_str}.log'
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
logger.debug("Args:")
logger.debug(args)

gpu_id = args.cuda
if not torch.cuda.is_available():
    gpu_id = -1
np.random.seed(args.seed)
torch.manual_seed(args.seed)

R, S, A, A_s, features, labels, full_labels, idx_train, idx_val, idx_test = load_data(
    args.dataset)
N, d = features.shape
n = S.shape[1]
# nclass = labels.max().item() + 1
nclass = labels.shape[1]
logger.info(f"Dataset loaded. N: {N}, n: {n}, feature: {d}-dim")
logger.info(
    f"Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

features = normalize(features)
features_s = S.T @ features
if args.type == "rw":
    adj = normalize(A_s)
else:
    adj = R @ A_s @ R
    # adj = (S.T @ S) @ adj
A += ssp.eye(N)
degs = np.array(A.sum(axis=1)).squeeze()
D_inv = ssp.diags(np.power(np.sqrt(degs), -1))
A = D_inv @ A @ D_inv

S, A, adj, features_s, labels, full_labels = to_torch(S), to_torch(A), to_torch(
    adj), to_torch(features_s), to_torch(labels), to_torch(full_labels)
idx_train, idx_val, idx_test = to_torch(
    idx_train), to_torch(idx_val), to_torch(idx_test)

device = f"cuda:{gpu_id}"
if gpu_id >= 0:
    A = A.cuda(device)
    adj = adj.cuda(device)
    S = S.cuda(device)
    features_s = features_s.cuda(device)
    labels = labels.type(torch.float).cuda(device)
    full_labels = full_labels.type(torch.float).cuda(device)
    idx_train = idx_train.cuda(device)
    idx_val = idx_val.cuda(device)
    idx_test = idx_test.cuda(device)

model = SummGCN(d, args.hidden, nclass, S, dropout=args.dropout)
if gpu_id >= 0:
    model = model.cuda(device)


def train(model, epochs):
    max_val_f1 = 0.0
    best_params = None
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(epochs):
        start = time.time()
        model.train()
        optimizer.zero_grad()
        embeds = model((features_s, adj))
        output = torch.sigmoid(embeds)

        y_, y = output[idx_train], labels[idx_train]
        loss_train = F.binary_cross_entropy(y_, y)
        loss_train.backward()
        optimizer.step()
        f1_micro_tr, f1_macro_tr = f1(y_, y)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        end = time.time()

        model.eval()
        embeds = model((features_s, adj))
        embeds = torch.spmm(S, embeds)
        output = torch.sigmoid(embeds)
        y_, y = output[idx_val], full_labels[idx_val]
        loss_val = F.binary_cross_entropy(y_, y)
        f1_micro_va, f1_macro_va = f1(y_, y)
        if f1_micro_va > max_val_f1:
            max_val_f1 = f1_micro_va
            best_params = model.state_dict()

        message = "{} {} {} {} {} {} {} {}".format(
            "Epoch: {:04d}".format(epoch+1),
            "loss_train: {:.4f}".format(loss_train.cpu().item()),
            "f1 micro: {:.4f}".format(f1_micro_tr),
            "f1 macro: {:.4f}".format(f1_macro_tr),
            "loss_val: {:.4f}".format(loss_val.cpu().item()),
            "f1 micro: {:.4f}".format(f1_micro_va),
            "f1 macro: {:.4f}".format(f1_macro_va),
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
    model.eval()
    embeds = model((features_s, adj))
    embeds = torch.spmm(S, embeds)
    output = torch.sigmoid(embeds)
    y_, y = output[idx_val], full_labels[idx_val]
    loss_test = F.binary_cross_entropy(y_, y)
    f1_micro, f1_macro = f1(y_, y)
    message = f"Test set results: loss= {loss_test.item():.4f} f1 micro= {f1_micro:.4f} f1 macro= {f1_macro:.4f}"
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
