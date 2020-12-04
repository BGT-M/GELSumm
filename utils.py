import os

import numpy as np
import scipy.sparse as ssp

def load_data(dataset):
    dir_ = os.path.join("./data", dataset)
    R = ssp.load_npz(os.path.join(dir_, "R.npz"))
    S = ssp.load_npz(os.path.join(dir_, "S.npz"))
    features = np.load(os.path.join(dir_, "features.npy"))
    labels = np.load_npz(os.path.join(dir_, "labels.npy"))

    idxs = np.load(os.path.join(dir_, "indices.npz"))
    idx_train = idxs["train"]
    idx_val = idxs["val"]
    idx_test = idxs["test"]

    return R, S, features, labels, idx_train, idx_val, idx_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)