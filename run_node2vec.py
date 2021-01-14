import argparse
import logging
import os
import time

import networkx as nx
import numpy as np
import scipy.sparse as ssp
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec

from models import node2vec
from models.logit_reg import LogitRegression
from utils import accuracy, f1, load_data, normalize, to_torch

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

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--seed', type=int, default=42,
                        help='RNG seed')

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

    return parser.parse_args()


args = parse_args()
if len(logger.handlers) < 2:
    filename = f'node2vec_{args.dataset}_{time_str}.log'
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
    path = os.path.join("data", args.dataset, "adj_s.edgelist")
    if args.weighted:
        G = nx.read_edgelist(path, nodetype=int, data=(
            ('weight', float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(path, nodetype=int,
                             create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G


def learn_embeddings():
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    nx_G = read_graph()
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)

    logger.info("Start generating walks...")
    start_time = time.time()
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    walks = [str(walk) for walk in walks]

    logger.info("Start learning embeddings...")
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size,
                     min_count=0, sg=1, workers=args.workers, iter=args.iter)
    end_time = time.time()
    logger.info("Learning completes.")
    logger.info(f"Node2vec costs {end_time-start_time} seconds")

    N = len(model.wv.index2word)
    rows, cols = [], []
    for i in range(N):
        j = int(model.wv.index2word[i])
        rows.append(j)
        cols.append(i)
    P = ssp.csr_matrix(([1] * len(rows), (rows, cols)), shape=(N, N))
    embeds = P @ model.wv.vectors
    np.save(os.path.join('output', args.dataset, 'node2vec.npy'), embeds)
    return embeds


def test(dataset, embeds, lr, epochs):
    R, S, adj, adj_s, features, labels, full_labels, indices, full_indices = load_data(dataset)
    adj = normalize(adj)
    del R, adj_s, features, labels

    embeds = S @ embeds
    for _ in range(2):
        embeds = adj @ embeds
    embeds = to_torch(embeds)
    nclass = full_labels.max() + 1
    full_labels = to_torch(full_labels)
    train_idx, test_idx = full_indices['train'], full_indices['test']
    train_idx, test_idx = to_torch(train_idx), to_torch(test_idx)

    logit_reg = LogitRegression(embeds.shape[1], nclass)
    optimizer = optim.LBFGS(logit_reg.parameters(), lr=lr)

    logit_reg.train()
    def closure():
        optimizer.zero_grad()
        output = logit_reg(embeds[train_idx])
        loss_train = F.cross_entropy(output, full_labels[train_idx])
        loss_train.backward()
        return loss_train
    for epoch in range(epochs):
        loss_train = optimizer.step(closure)

    predict = logit_reg(embeds[test_idx])
    loss_test = F.cross_entropy(predict, full_labels[test_idx])
    acc_test = accuracy(predict, full_labels[test_idx])
    f1_micro, f1_macro = f1(predict, full_labels[test_idx])
    logger.info(
        f"Test set results: loss= {loss_test.item():.4f} accuracy= {acc_test.item():.4f} f1 micro= {f1_micro:.4f} f1 macro= {f1_macro:.4f}")

if __name__ == "__main__":
    embeds = learn_embeddings()
    test(args.dataset, embeds, args.lr, args.epochs)
