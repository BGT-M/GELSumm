import numbers
import os
import time

import numpy as np
from numpy.core.numeric import _full_like_dispatcher
import scipy.sparse as ssp
import torch
from sklearn.metrics import f1_score


def load_dataset(dataset):
    dir_ = os.path.join("data", dataset)
    A = ssp.load_npz(os.path.join(dir_, "adj.npz"))
    A_s = ssp.load_npz(os.path.join(dir_, "adj_s.npz"))
    features = np.load(os.path.join(dir_, "feats.npy"))

    labels = np.load(os.path.join(dir_, "labels.npy"))
    full_labels = np.load(os.path.join(dir_, "full_labels.npy"))
    indices = np.load(os.path.join(dir_, "indices.npz"))
    full_indices = np.load(os.path.join(dir_, "full_indices.npz"))

    return A, A_s, features, labels, full_labels, indices, full_indices


def load_dataset_gs(dataset):
    dir_ = os.path.join("data", dataset)
    adj = ssp.load_npz(os.path.join(dir_, "adj.npz"))
    adj_s = ssp.load_npz(os.path.join(dir_, "adj_s.npz"))
    features = np.load(os.path.join(dir_, "feats.npy"))
    full_indices = np.load(os.path.join(dir_, "full_indices.npz"))
    full_labels = np.load(os.path.join(dir_, "full_labels.npy"))

    return adj, adj_s, features, full_indices, full_labels


def to_torch(x):
    if isinstance(x, np.ndarray):
        ret = torch.from_numpy(x).type(torch.float32)
        if issubclass(x.dtype.type, numbers.Integral):
            ret = ret.type(torch.long)
        return ret
    elif isinstance(x, ssp.spmatrix):
        x = x.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((x.row, x.col)).astype(np.int64))
        values = torch.from_numpy(x.data)
        shape = torch.Size(x.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    y_ = preds.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    f1_micro = f1_score(y, y_, average='micro')
    f1_macro = f1_score(y, y_, average='macro')
    return f1_micro, f1_macro


def normalize(mx):
    rowsum = np.array(mx.sum(1), dtype=float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = ssp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def aug_normalized_adjacency(adj, lda=1.0):
    adj = adj + lda * ssp.identity(adj.shape[0])
    adj = adj.tocsr()
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = ssp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt @ (adj @ d_mat_inv_sqrt)


def sgc_precompute(features, adj, degree):
    t = time.time()
    for i in range(degree):
        features = torch.spmm(adj, features)
    precompute_time = time.time() - t
    return features, precompute_time
