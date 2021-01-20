import argparse
import logging
import os
import time

import networkx as nx
import numpy as np
import scipy.sparse as ssp
import torch.nn.functional as F
import torch.optim as optim

from node2vec import Node2Vec
from utils import accuracy, f1, load_data, normalize, to_torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logger = logging.getLogger('node2vec')
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

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window-size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=1, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')

    parser.add_argument('--unweighted', dest='unweighted',
                        action='store_false')

    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')

    parser.add_argument('--undirected', dest='undirected',
                        action='store_false')

    parser.set_defaults(directed=False)

    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1, help="Training epochs")

    return parser.parse_args()


args = parse_args()
if len(logger.handlers) < 2:
    filename = f'node2vec_{args.dataset}.log'
    file_handler = logging.FileHandler(filename, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
logger.debug(args)


def read_graph():
    '''
    Reads the input network in networkx.
    '''
    path = os.path.join("data", args.dataset, "A_s.npz")
    adj_s = ssp.load_npz(path)
    G = nx.from_scipy_sparse_matrix(adj_s, create_using=nx.Graph())
    # if args.weighted:
    #     G = nx.read_edgelist(path, nodetype=int, data=(
    #         ('weight', float),), create_using=nx.DiGraph())
    # else:
    #     G = nx.read_edgelist(path, nodetype=int,
    #                          create_using=nx.DiGraph())
    #     for edge in G.edges():
    #         G[edge[0]][edge[1]]['weight'] = 1
    if not args.weighted:
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    # if not args.directed:
    #     G = G.to_undirected()

    return G


def learn_embeddings():
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    nx_G = read_graph()

    start_time = time.time()
    logger.info("Start walking...")
    node2vec = Node2Vec(nx_G, dimensions=args.dimensions, walk_length=args.walk_length, num_walks=args.num_walks, workers=args.workers, seed=args.seed)
    logger.info("Start learning embeddings...")
    model = node2vec.fit(window=args.window_size, min_count=1, batch_words=4)
    end_time = time.time()
    logger.info("Learning completes.")
    logger.info(f"Node2vec costs {end_time-start_time} seconds")

    N = nx_G.number_of_nodes()
    embeds = np.zeros((N, args.dimensions))
    indices = []
    lost_indices = []
    for i in range(N):
        try:
            embeds[i] = model.wv[str(i)]
            indices.append(i)
        except KeyError:
            lost_indices.append(i)
            continue
    avg_embed = embeds[indices].mean(axis=0)
    embeds[lost_indices] = avg_embed
    np.save(os.path.join('output', args.dataset, 'node2vec.npy'), embeds)
    return embeds


def test(dataset, embeds, lr, epochs):
    R, S, adj, adj_s, features, labels, full_labels, indices, full_indices = load_data(dataset)
    adj = normalize(adj)
    del R, adj_s, features, labels

    embeds = S @ embeds
    for _ in range(2):
        embeds = adj @ embeds
    nclass = full_labels.max() + 1
    train_idx, test_idx = full_indices['train'], full_indices['test']

    model = LogisticRegression(solver='lbfgs')
    model.fit(embeds[train_idx], full_labels[train_idx])
    predict = model.predict(embeds[test_idx])
    acc_test = accuracy_score(full_labels[test_idx], predict)
    logger.info(f"Test set results: accuracy= {acc_test:.4f}")
    

if __name__ == "__main__":
    embeds = learn_embeddings()
    test(args.dataset, embeds, args.lr, args.epochs)
