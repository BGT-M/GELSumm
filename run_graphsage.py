import time
from argparse import ArgumentParser

import networkx as nx
import numpy as np
import scipy.sparse as ssp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from models.graphsage import graphsage
from utils import load_dataset_gs, normalize

parser = ArgumentParser()
parser.add_argument("--dataset", required=True, type=str,
                    help="Dataset name.")
args = parser.parse_args()

if __name__ == '__main__':
    f = open(f'graphsage_{args.dataset}.log', mode='a')

    adj, adj_s, feats, full_indices, full_labels = load_dataset_gs(args.dataset)
    N, d = feats.shape
    n = adj_s.shape[0]
    R = ssp.load_npz(f'data/{args.dataset}/R.npz')
    # Feature normalization
    feats = (feats - feats.mean(axis=0)) / feats.std(axis=0)

    deg = np.array(adj.sum(axis=1)).flatten()
    deg_inv_sqrt = np.power(deg, -0.5)
    deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0
    D = ssp.diags(deg)
    D_inv_sqrt = ssp.diags(deg_inv_sqrt)

    with np.errstate(divide='ignore'):
        deg_s = np.array(adj_s.sum(axis=1)).flatten()
        deg_inv = np.power(deg_s, -1)
        deg_inv[np.isinf(deg_inv)] = 0

        feats_s = ssp.diags(deg_inv) @ R.T @ D @ feats
        adj += ssp.eye(N)
        adj = D_inv_sqrt @ adj @ D_inv_sqrt

    G = nx.from_scipy_sparse_matrix(
        adj_s, edge_attribute='wgt', create_using=nx.Graph())
    # print("Not remove self loop.", file=f)
    print("Remove self loop.", file=f)
    G.remove_edges_from(nx.selfloop_edges(G))
    nx.set_node_attributes(G, False, 'test')
    nx.set_node_attributes(G, False, 'val')

    weighted = True
    type_ = 'gcn'
    print(f"Weighted: {weighted}", file=f)
    print(f"Type: {type_}", file=f)
    start_time = time.process_time()
    embeds = graphsage.graphsage(G, feats_s, type_, weighted, int(1000 * n / N))
    end_time = time.process_time()
    print(f"Training GraphSage costs {end_time-start_time} seconds")
    print(f"Training GraphSage costs {end_time-start_time} seconds", file=f)

    embeds = R @ embeds
    np.save(f'./data/{args.dataset}.npy', embeds)

    for k in range(5):
        lr = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=500)
        lr.fit(embeds[full_indices['train']], full_labels[full_indices['train']])
        y_pred = lr.predict(embeds[full_indices['test']])
        y_true = full_labels[full_indices['test']]

        f1_micro = f1_score(y_true, y_pred, average='micro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        print(f"K={k}")
        print(f"K={k}", file=f)
        print(f"Micro f1: {f1_micro:.4f}, Macro f1: {f1_macro:.4f}")
        print(f"Micro f1: {f1_micro:.4f}, Macro f1: {f1_macro:.4f}", file=f)
        embeds = adj @ embeds

    print("-" * 20, file=f)
    if not f.closed:
        f.close()
