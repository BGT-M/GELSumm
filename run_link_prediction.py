import logging
import os
import time
from argparse import ArgumentParser

import networkx as nx
import numpy as np
import scipy.sparse as ssp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold

from models.deepwalk import deepwalk
from models.line import run_LINE
from utils import accuracy, f1, load_dataset, aug_normalized_adjacency, to_torch


logger = logging.getLogger('lp')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s %(filename)s %(lineno)d %(levelname)s: %(message)s')

parser = ArgumentParser()
parser.add_argument('--method', type=str, choices=['deepwalk', 'line'], help='Embed method')
parser.add_argument('--dataset', type=str)
parser.add_argument('--power', type=int, default=8, help='Maximum power of smoothing filter')
parser.add_argument('--embed_path', type=str, default='', help='Pre-trained embedding path')
args = parser.parse_args()

if len(logger.handlers) < 2:
    filename = f'lp_{args.method}_{args.dataset}.log'
    file_handler = logging.FileHandler(filename, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
logger.debug(args)


def learn_embeds_dw():
    adj = ssp.load_npz(os.path.join('data', args.dataset, 'adj_s.npz')).tocsr()
    G = nx.from_scipy_sparse_matrix(adj, edge_attribute='weight', create_using=nx.Graph())
    del adj
    logger.info("Start training DeepWalk...")
    start_time = time.perf_counter()
    embeds = deepwalk(G)
    end_time = time.perf_counter()
    logger.info(f"Deepwalk learning costs {end_time-start_time:.4f} seconds")

    if not os.path.exists(f'output/{args.dataset}'):
        os.mkdir(f'output/{args.dataset}')
    np.save(os.path.join('output', args.dataset, 'deepwalk.npy'), embeds)
    return embeds, end_time - start_time


def learn_embeds_line():
    adj = ssp.load_npz(os.path.join('data', args.dataset, 'adj_s.npz')).tocsr()
    G = nx.from_scipy_sparse_matrix(adj, edge_attribute='weight', create_using=nx.Graph())
    del adj
    logger.info("Start training LINE...")
    start_time = time.perf_counter()
    embeds = run_LINE(G, 100, 5)
    end_time = time.perf_counter()
    logger.info(f"LINE learning costs {end_time-start_time:.4f} seconds")

    if not os.path.exists(f'output/{args.dataset}'):
        os.mkdir(f'output/{args.dataset}')
    np.save(os.path.join('output', args.dataset, 'line.npy'), embeds)
    return embeds, end_time - start_time


def test(embeds, power):
    dataset = args.dataset
    dataset_raw = dataset[:args.dataset.find('_')]
    adj = ssp.load_npz(f'data/{dataset_raw}/adj_lp.npz')
    filter = aug_normalized_adjacency(adj)
    R = ssp.load_npz(f'data/{args.dataset}/R.npz')

    start_time = time.perf_counter()
    print(R.shape, embeds.shape)
    embeds = R @ embeds
    for _ in range(power):
        embeds = filter @ embeds
    end_time = time.perf_counter()
    logger.info(f'Refinement costs {end_time-start_time:.4f} seconds')
    
    positive = np.load(f'/data/{dataset_raw}/positive.npy').astype(np.int)
    negative = np.load(f'/data/{dataset_raw}/negative.npy').astype(np.int)
    pos_embeds = []
    neg_embeds = []
    pos_labels = np.array([1] * len(positive))
    neg_labels = np.array([0] * len(negative))
    for i, j in positive:
        pos_embeds.append(np.multiply(embeds[i], embeds[j]))
    for i, j in negative:
        neg_embeds.append(np.multiply(embeds[i], embeds[j]))
    pos_embeds, neg_embeds = np.array(pos_embeds), np.array(neg_embeds)
    X = np.concatenate([pos_embeds, neg_embeds])
    y = np.concatenate([pos_labels, neg_labels])

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = 0.0
    for train_id, test_id in kfold.split(X):
        logit = LogisticRegression(random_state=42)
        logit.fit(X[train_id], y[train_id])
        probs = logit.predict_proba(X[test_id])[:, 1]
        auc_score = roc_auc_score(y[test_id], probs)
        scores += auc_score
        logger.debug(f'auc score: {auc_score:.4f}')
    logger.info(f'Average auc score: {scores / 5:.4f}')
    
    return end_time - start_time


if __name__ == '__main__':
    embeds, train_time = None, 0.0
    if len(args.embed_path) > 0:
        embeds = np.load(args.embed_path)
    elif args.method == 'deepwalk':
        embeds, train_time = learn_embeds_dw()
    elif args.method == 'line':
        embeds, train_time = learn_embeds_line()
    else:
        raise NotImplementedError(f'Unsupported method: {args.method}')
    for p in range(args.power):
        refine_time = test(embeds, p)
        logger.debug(f"Total time: {train_time+refine_time:.4f}")
