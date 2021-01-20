import logging
import os
import random
import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import numpy as np
import scipy.sparse as ssp
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec

from utils import accuracy, f1, load_data, normalize, to_torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
parser.add_argument('--matfile-variable-name', default='network',
                    help='variable name of adjacency matrix inside a .mat file.')
parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                    help='Size to start dumping walks to disk, instead of keeping them in memory.')
parser.add_argument('--number-walks', default=10, type=int,
                    help='Number of random walks to start at each node')
parser.add_argument('--representation-size', default=64, type=int,
                    help='Number of latent dimensions to learn for each node.')
parser.add_argument('--seed', default=0, type=int,
                    help='Seed for random walk generator.')
parser.add_argument('--undirected', default=True, type=bool,
                    help='Treat graph as undirected.')
parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                    help='Use vertex degree to estimate the frequency of nodes '
                    'in the random walks. This option is faster than '
                    'calculating the vocabulary.')
parser.add_argument('--walk-length', default=40, type=int,
                    help='Length of the random walk started at each node')
parser.add_argument('--window-size', default=5, type=int,
                    help='Window size of skipgram model.')
parser.add_argument('--workers', default=1, type=int,
                    help='Number of parallel processes.')
parser.add_argument("--degree", type=int, default=2,
                    help="Degree of graph filter")
parser.add_argument("--lr", type=float, default=1.0, help="Learning rate")
parser.add_argument("--epochs", type=int, default=2, help="Training epochs")

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
    if args.format == "adjlist":
        path = os.path.join("data", args.dataset, "adj_s.adjlist")
        G = graph.load_adjacencylist(path, undirected=args.undirected)
    elif args.format == "edgelist":
        path = os.path.join("data", args.dataset, "adj_s.edgelist")
        G = graph.load_edgelist(path, undirected=args.undirected)
    elif args.format == "mat":
        path = os.path.join("data", args.dataset, "adj_s.mat")
        G = graph.load_matfile(path, variable_name=args.matfile_variable_name, undirected=args.undirected)
    else:
        raise Exception(
            "Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

    N = len(G.nodes())
    logger.info("Number of nodes: {}".format(len(G.nodes())))

    num_walks = len(G.nodes()) * args.number_walks

    logger.info("Number of walks: {}".format(num_walks))

    data_size = num_walks * args.walk_length

    logger.info("Data size (walks*length): {}".format(data_size))

    start_time = time.time()
    if data_size < args.max_memory_data_size:
        logger.info("Walking...")
        walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                            path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))
        logger.info("Training...")
        model = Word2Vec(walks, size=args.representation_size,
                         window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers)
    else:
        logger.info("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(
            data_size, args.max_memory_data_size))
        logger.info("Walking...")

        walks_filebase = args.dataset + ".walks"
        walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
                                                          path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
                                                          num_workers=args.workers)

        logger.info("Counting vertex frequency...")
        if not args.vertex_freq_degree:
            vertex_counts = serialized_walks.count_textfiles(
                walk_files, args.workers)
        else:
            # use degree distribution for frequency in tree
            vertex_counts = G.degree(nodes=G.iterkeys())

        logger.info("Training...")
        walks_corpus = serialized_walks.WalksCorpus(walk_files)
        model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                         size=args.representation_size,
                         window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)
    end_time = time.time()
    logger.info(f"Deepwalk learning costs {end_time-start_time:.4f} seconds")

    embeds = np.zeros((N, args.representation_size))
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
    if not os.path.exists(f'output/{args.dataset}'):
        os.mkdir(f'output/{args.dataset}')
    np.save(os.path.join('output', args.dataset, 'deepwalk.npy'), embeds)
    return embeds


def test(dataset, embeds, degree):
    R, S, adj, adj_s, features, labels, full_labels, indices, full_indices = load_data(dataset)
    adj = normalize(adj)
    del R, adj_s, features, labels

    print(embeds.shape, S.shape)
    embeds = S @ embeds
    for _ in range(degree):
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
    test(args.dataset, embeds, args.degree)
