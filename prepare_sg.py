import os
from argparse import ArgumentParser

import numpy as np
import scipy.sparse as ssp

parser = ArgumentParser()
parser.add_argument("dataset", type=str, help="Dataset name")
args = parser.parse_args()

if __name__ == "__main__":
    dataset = args.dataset
    adj_s = ssp.load_npz(os.path.join("data", dataset, "A.npz"))
    adj_s = adj_s.tolil()
    with open(os.path.join("data", dataset, "adj_s.adjlist"), "w") as f1, open(os.path.join("data", dataset, "adj_s.edgelist"), "w") as f2:
        for i in range(adj_s.shape[0]):
            f1.write(f"{i}")
            for j in adj_s.rows[i]:
                v = int(adj_s[i, j])
                f1.write(f" {j}")
                f2.write(f"{i} {j} {v}\n")
            f1.write("\n")
