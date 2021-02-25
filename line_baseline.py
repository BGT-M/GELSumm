import logging
import os
import random
import time
from argparse import ArgumentParser

import networkx as nx
import numpy as np
import scipy.sparse as ssp
import torch
from models.line import run_LINE

from utils import accuracy, f1, normalize, to_torch, aug_normalized_adjacency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

logger = logging.getLogger('line_baseline')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s %(filename)s %(lineno)d %(levelname)s: %(message)s')

parser = ArgumentParser()
parser.add_argument("--dataset", type=str, help="Dataset name")
parser.add_argument("--seed", type=int, default=42, help="RNG seed")
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')

args = parser.parse_args()
if len(logger.handlers) < 2:
    filename = f'line_baseline_{args.dataset}.log'
    file_handler = logging.FileHandler(filename, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
logger.debug(args)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

def learn_embeds():
    adj = ssp.load_npz(os.path.join('/data', args.dataset, 'adj.npz'))
    G = nx.from_scipy_sparse_matrix(adj, edge_attribute='weight', create_using=nx.Graph())
    del adj
    logger.info("Start training LINE...")
    start_time = time.perf_counter()
    embeds = run_LINE(G, args.epochs)
    end_time = time.perf_counter()
    logger.info(f"LINE learning costs {end_time-start_time:.4f} seconds")

    if not os.path.exists(f'output/{args.dataset}'):
        os.mkdir(f'output/{args.dataset}')
    np.save(os.path.join('output', args.dataset, 'line.npy'), embeds)
    return embeds

def load_embeds():
    tmp_embeds = np.genfromtxt(f'/data/{args.dataset}/vec_all.txt', skip_header=1)
    embeds = np.zeros((tmp_embeds.shape[0], tmp_embeds.shape[1]-1))
    for i in range(tmp_embeds.shape[0]):
        idx = int(tmp_embeds[i, 0])
        embeds[i] = tmp_embeds[i, 1:]
    return embeds


def test(dataset, embeds):
    full_labels = np.load(os.path.join('/data', dataset, 'labels.npy'))
    full_indices = np.load(os.path.join('/data', dataset, 'indices.npz'))

    train_idx, val_idx, test_idx = full_indices['train'], full_indices['val'], full_indices['test']
    print(f"train/test: {len(train_idx)}/{len(test_idx)}")

    model = LogisticRegression(solver='lbfgs')
    model.fit(embeds[train_idx], full_labels[train_idx])
    predict = model.predict(embeds[test_idx])
    acc_test = accuracy_score(full_labels[test_idx], predict)
    logger.info(f"Test set results: accuracy= {acc_test:.4f}")

if __name__ == "__main__":
    # embeds = load_embeds()
    embeds = learn_embeds()
    test(args.dataset, embeds)
