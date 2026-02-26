from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import igraph as ig
from typing import Tuple


def build_knn_graph(
    X: np.ndarray,
    k: int = 30,
    metric: str = "euclidean",
) -> ig.Graph:
    """
    Build an undirected kNN graph with edge weights = 1 / (1 + distance),
    symmetrized by keeping the maximum weight for each undirected pair.
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D [N, D].")
    N = X.shape[0]

    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(X)
    dists, nbrs = nn.kneighbors(X, return_distance=True)  # (N, k)

    rows = np.repeat(np.arange(N), k)
    cols = nbrs.ravel()
    weights = 1.0 / (1.0 + dists.ravel())

    mask = rows != cols
    rows, cols, weights = rows[mask], cols[mask], weights[mask]

    u = np.minimum(rows, cols)
    v = np.maximum(rows, cols)
    edges_df = pd.DataFrame({"u": u, "v": v, "w": weights})
    edges_df = edges_df.sort_values(["u", "v", "w"], ascending=[True, True, False])
    edges_df = edges_df.drop_duplicates(["u", "v"], keep="first")

    g = ig.Graph(n=N, edges=list(zip(edges_df.u.values, edges_df.v.values)), directed=False)
    g.es["weight"] = edges_df.w.values
    return g
