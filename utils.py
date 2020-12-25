import os
import numbers

import numpy as np
import scipy.sparse as ssp
import torch


def load_data(dataset):
    dir_ = os.path.join("./data", dataset)
    R = ssp.load_npz(os.path.join(dir_, "R.npz"))
    S = ssp.load_npz(os.path.join(dir_, "S.npz"))
    A = ssp.load_npz(os.path.join(dir_, "A.npz"))
    A_s = ssp.load_npz(os.path.join(dir_, "A_s.npz"))
    features = np.load(os.path.join(dir_, "feats.npy"))
    labels = np.load(os.path.join(dir_, "full_labels.npy"))

    # idxs = np.load(os.path.join(dir_, "full_indices.npz"))
    idxs = np.load(os.path.join(dir_, "few_indices.npz"))
    idx_train = idxs["train"]
    idx_val = idxs["val"]
    idx_test = idxs["test"]

    return R, S, A, A_s, features, labels, idx_train, idx_val, idx_test


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


def normalize(mx):
    rowsum = np.array(mx.sum(1), dtype=float)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = ssp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
