from __future__ import annotations
import numpy as np
import igraph as ig
import leidenalg as la
from typing import Tuple


def leiden(
    g: ig.Graph,
    resolution: float = 1.0,
    seed: int = 42,
) -> Tuple[np.ndarray, int]:
    """
    Run Leiden clustering using RBConfigurationVertexPartition with given resolution.
    Returns membership array and number of clusters.
    """
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights=g.es["weight"] if "weight" in g.es.attributes() else None,
        resolution_parameter=resolution,
        seed=seed,
    )
    labels = np.asarray(part.membership, dtype=int)
    n_clusters = int(labels.max() + 1) if labels.size else 0
    return labels, n_clusters
