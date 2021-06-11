# GELSumm

## DeepWalk & LINE
`dw_baselin.py` & `LINE_baseline.py`: Base embedding methods on original graphs, node classification task.
`dw_lp_baselin.py` & `LINE_lp_baseline.py`: Base embedding methods on original graphs, link prediction task.

**Input**:
* `data/${dataset}/adj.npz`: original adjacency matrix (scipy sparse matrix).
* `data/${dataset}/labels.npy`: label array (numpy).
* `data/${dataset}/indices.npz`: split indices (numpy, containing keys 'train' and 'test').
* `data/${dataset}_${ratio}/adj_s.npz`: summary adjacency matrix (scipy sparse matrix).
* `data/${dataset}_${ratio}/R.npz`: restore matrix (scipy sparse matrix).

**Output**:
* `output/${dataset}/deepwalk.npy`: DeepWalk's embedding matrix (numpy).
* `output/${dataset}/line.npy`: LINE's embedding matrix (numpy).

## GCN

