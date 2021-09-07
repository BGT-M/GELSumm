import time
from argparse import ArgumentParser

import networkx as nx
import numpy as np
import scipy.sparse as ssp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from models.graphsage import graphsage


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--dataset', type=str, help='Dataset name')
	args = parser.parse_args()

	f = open(f'graphsage_{args.dataset}.log', mode='a')
	adj = ssp.load_npz(f'./data/{args.dataset}/adj.npz')
	indices = np.load(f'./data/{args.dataset}/indices.npz')
	feats = np.load(f'./data/{args.dataset}/feats.npy')
	labels = np.load(f'./data/{args.dataset}/labels.npy')

	G = nx.from_scipy_sparse_matrix(adj, create_using=nx.Graph())
	nx.set_node_attributes(G, False, 'test')
	nx.set_node_attributes(G, False, 'val')
	d_test = {n : True for n in indices['test']}
	d_val = {n : True for n in indices['val']}
	nx.set_node_attributes(G, d_test, 'test')
	nx.set_node_attributes(G, d_val, 'val')

	start_time = time.process_time()
	embeds = graphsage.graphsage(G, feats, 'mean', False, 1000)
	end_time = time.process_time()
	print(f"Training GraphSage costs {end_time-start_time} seconds")
	print(f"Training GraphSage costs {end_time-start_time} seconds", file=f)

	np.save(f'./data/{args.dataset}.npy', embeds)
	lr = LogisticRegression()
	lr.fit(embeds[indices['train']], labels[indices['train']])
	y_pred = lr.predict(embeds[indices['test']])
	y_true = labels[indices['test']]

	f1_micro = f1_score(y_true, y_pred, average='micro')
	f1_macro = f1_score(y_true, y_pred, average='macro')
	print(f"Micro f1: {f1_micro:.4f}, Macro f1: {f1_macro:.4f}")
	print(f"Micro f1: {f1_micro:.4f}, Macro f1: {f1_macro:.4f}", file=f)

	if not f.closed:
		f.close()
