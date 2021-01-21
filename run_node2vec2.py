import argparse
import logging
import os
import time

import networkx as nx
import numpy as np
import scipy.sparse as ssp
import torch.nn.functional as F
import torch.optim as optim

from utils import accuracy, f1, load_data, normalize, to_torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from models.node2vec import node2vec

logger = logging.getLogger('node2vec2')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s %(filename)s %(lineno)d %(levelname)s: %(message)s')
time_str = time.strftime('%Y-%m-%d-%H-%M')


def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run node2vec.")
    parser.add_argument('--dataset', required=True, help='Dataset name')
    parser.add_argument('--seed', type=int, default=42, help='RNG seed')
    parser.add_argument('--power', type=int, default=2, help='Power of filter (2)')

    return parser.parse_args()


args = parse_args()
if len(logger.handlers) < 2:
    filename = f'node2vec2_{args.dataset}.log'
    file_handler = logging.FileHandler(filename, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
logger.debug(args)


def learn_embeddings():
    adj = ssp.load_npz(os.path.join('data', args.dataset, 'A.npz'))
    G = nx.from_scipy_sparse_matrix(adj, edge_attribute='wgt', create_using=nx.Graph())

    start_time = time.time()
    embeds = node2vec(G, args.seed)
    end_time = time.time()
    logger.info(f"Node2vec costs {end_time-start_time} seconds")

    if not os.path.exists(f'output/{args.dataset}'):
        os.mkdir(f'output/{args.dataset}')

    np.save(os.path.join('output', args.dataset, 'node2vec.npy'), embeds)
    return embeds


def test(dataset, embeds):
    full_labels = np.load(os.path.join('data', dataset, 'full_labels.npy'))
    full_indices = np.load(os.path.join('data', dataset, 'full_indices.npz'))

    nclass = full_labels.max() + 1
    train_idx, test_idx = full_indices['train'], full_indices['test']

    model = LogisticRegression(solver='lbfgs')
    model.fit(embeds[train_idx], full_labels[train_idx])
    predict = model.predict(embeds[test_idx])
    acc_test = accuracy_score(full_labels[test_idx], predict)
    logger.info(f"Test set results: accuracy= {acc_test:.4f}")


if __name__ == "__main__":
    embeds = learn_embeddings()
    test(args.dataset, embeds)
