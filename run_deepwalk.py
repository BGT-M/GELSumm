import logging
import os
import random
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import networkx as nx
import numpy as np
import scipy.sparse as ssp
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec

from utils import accuracy, f1, load_data, normalize, to_torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from models.deepwalk import deepwalk

logger = logging.getLogger('deepwalk')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s %(filename)s %(lineno)d %(levelname)s: %(message)s')

parser = ArgumentParser("deepwalk",
                        formatter_class=ArgumentDefaultsHelpFormatter,
                        conflict_handler='resolve')
parser.add_argument("--dataset", type=str, help="Dataset name")
parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                    help="drop a debugger if an exception is raised.")
parser.add_argument('--format', default='adjlist',
                    help='File format of input file')
parser.add_argument("-l", "--log", dest="log", default="INFO",
                    help="log verbosity level")
parser.add_argument('--seed', default=0, type=int,
                    help='Seed for random walk generator.')
parser.add_argument('--undirected', default=True, type=bool,
                    help='Treat graph as undirected.')
parser.add_argument("--power", type=int, default=2, help="Power of filter")

args = parser.parse_args()
if len(logger.handlers) < 2:
    filename = f'deepwalk_{args.dataset}.log'
    file_handler = logging.FileHandler(filename, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
logger.debug(args)


def learn_embeds():
    adj = ssp.load_npz(os.path.join('data', args.dataset, 'A_s.npz'))
    G = nx.from_scipy_sparse_matrix(adj, edge_attribute='wgt', create_using=nx.Graph())
    start_time = time.time()
    embeds = deepwalk(G, args.seed)
    end_time = time.time()
    logger.info(f"Deepwalk learning costs {end_time-start_time:.4f} seconds")

    if not os.path.exists(f'output/{args.dataset}'):
        os.mkdir(f'output/{args.dataset}')
    np.save(os.path.join('output', args.dataset, 'deepwalk.npy'), embeds)
    return embeds


def test(dataset, embeds, power):
    R, S, adj, adj_s, features, labels, full_labels, indices, full_indices = load_data(dataset)
    adj = normalize(adj)
    del R, adj_s, features, labels

    print(embeds.shape, S.shape)
    embeds = S @ embeds
    for _ in range(power):
        embeds = adj @ embeds
    nclass = full_labels.max() + 1
    train_idx, test_idx = full_indices['train'], full_indices['test']

    model = LogisticRegression(solver='lbfgs')
    model.fit(embeds[train_idx], full_labels[train_idx])
    predict = model.predict(embeds[test_idx])
    acc_test = accuracy_score(full_labels[test_idx], predict)
    logger.info(f"Test set results: accuracy= {acc_test:.4f}")


if __name__ == "__main__":
    embeds = learn_embeds()
    test(args.dataset, embeds, args.power)
