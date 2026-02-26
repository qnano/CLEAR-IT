#!/usr/bin/env python3
"""
Scan kNN-k and Leiden resolution; report cluster count, stability (mean ARI), and quality.

Outputs:
  - CSV with all runs: <OUTPUTS_DIR>/leiden/<dataset>_<split>_scan.csv
  - Console summary of settings closest to target cluster count (default: 14)

Dependencies:
  clearit.leiden.{io, preprocess, graph}
  numpy, pandas, scikit-learn, igraph, leidenalg
"""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import itertools
import numpy as np
import pandas as pd

import igraph as ig
import leidenalg as la
from sklearn.metrics import adjusted_rand_score

from clearit.config import EMBEDDINGS_DIR, OUTPUTS_DIR
from clearit.leiden.io import list_patients, load_hdf5_split
from clearit.leiden.preprocess import standardize_and_pca
from clearit.leiden.graph import build_knn_graph


# ----------------------------- Configuration -----------------------------

# Dataset selection
DATASET = "TNBC1"  # "TNBC1" or "TNBC2"
SPLIT = "train"    # "train" or "test"

# HDF5 paths
H5_TNBC1 = EMBEDDINGS_DIR / "TNBC1-MxIF8"  / "inForm_MC7"    / "01_features-expressions" / "tnbc1-mxif8.hdf5"
H5_TNBC2 = EMBEDDINGS_DIR / "TNBC2-MIBI44" / "DeepCell_MC17" / "01_features-expressions" / "tnbc2-mibi8.hdf5"

# Embedding + graph defaults (centered on your current working point)
PCA_DIMS_DEFAULT = 64
K_GRID = [40, 50, 60, 70, 80]                  # includes 60
RES_GRID = [0.5, 0.6, 0.7, 0.8, 0.9]           # includes 0.7
SEEDS = [0, 1, 2, 3, 4]                        # used for stability
TARGET_CLUSTERS = 14

# Subsampling caps (set to None for full data)
PER_PATIENT_CAP = 10_000
GLOBAL_CAP = 150_000

# Reproducibility
SEED = 42

# Output
OUT_DIR = OUTPUTS_DIR / "leiden"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------- Utilities -------------------------------

@dataclass
class PartitionStats:
    labels: np.ndarray
    quality: float
    n_clusters: int


def run_leiden_partition(
    g: ig.Graph,
    resolution: float,
    seed: int,
) -> PartitionStats:
    """
    Run Leiden with RBConfigurationVertexPartition; return labels, quality, and cluster count.
    """
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights=g.es["weight"] if "weight" in g.es.attributes() else None,
        resolution_parameter=resolution,
        seed=seed,
    )
    labels = np.asarray(part.membership, dtype=int)
    quality = float(part.quality())  # objective value for the chosen partition type
    n_clusters = int(labels.max() + 1) if labels.size else 0
    return PartitionStats(labels=labels, quality=quality, n_clusters=n_clusters)


def mean_pairwise_ari(labels_list: List[np.ndarray]) -> float:
    """
    Compute mean Adjusted Rand Index over all unique pairs in a list of labelings.
    """
    if len(labels_list) < 2:
        return 1.0
    pairs = [(i, j) for i in range(len(labels_list)) for j in range(i + 1, len(labels_list))]
    aris = [adjusted_rand_score(labels_list[i], labels_list[j]) for i, j in pairs]
    return float(np.mean(aris))


def choose_split_patients(h5_path: Path) -> Tuple[list[str], list[str]]:
    """
    Build a simple train/test split over available patient groups.
    For TNBC1-like counts, the first 47 patients are train; remainder are test.
    For other counts, split 75/25 by index.
    """
    pts = list_patients(h5_path)
    if len(pts) >= 63:
        train = [p for p in pts if int(p[1:]) <= 47]
        test = [p for p in pts if p not in train]
        return train, test
    # Generic fallback split
    n_train = max(1, int(0.75 * len(pts)))
    return pts[:n_train], pts[n_train:]


# --------------------------------- Main ----------------------------------

def main():
    # Resolve dataset path
    h5_path = H5_TNBC1 if DATASET.upper() == "TNBC1" else H5_TNBC2
    assert h5_path.exists(), f"Missing file: {h5_path}"

    # Determine patients for chosen split
    train_pts, test_pts = choose_split_patients(h5_path)
    patients = train_pts if SPLIT == "train" else test_pts
    print(f"{DATASET} | {SPLIT}: {len(patients)} patients")

    # Load and embed once; graph is rebuilt per k
    X, meta, _, _ = load_hdf5_split(
        h5_path=h5_path,
        patients=patients,
        load_expressions=False,
        load_labels=False,
        per_patient_cap=PER_PATIENT_CAP,
        global_cap=GLOBAL_CAP,
        seed=SEED,
    )
    print(f"Features loaded: X = {X.shape}, meta rows = {len(meta)}")

    X_pca, scaler, pca = standardize_and_pca(X, n_components=PCA_DIMS_DEFAULT, seed=SEED)
    cum_var = float(pca.explained_variance_ratio_[:PCA_DIMS_DEFAULT].sum())
    print(f"PCA dims = {PCA_DIMS_DEFAULT}, cumulative explained variance = {cum_var:.4f}")

    # Scan grid
    rows = []
    for k, res in itertools.product(K_GRID, RES_GRID):
        # Build kNN graph for this k
        g = build_knn_graph(X_pca, k=k, metric="euclidean")

        # Precompute some graph stats for reference
        n = g.vcount()
        m = g.ecount()
        mean_deg = float(np.mean(g.degree())) if n > 0 else 0.0
        n_components = len(g.components())

        # Multiple seeds for stability
        label_runs = []
        qualities = []
        ncls = []
        for s in SEEDS:
            stats = run_leiden_partition(g, resolution=res, seed=s)
            label_runs.append(stats.labels)
            qualities.append(stats.quality)
            ncls.append(stats.n_clusters)

        # Stability and summary metrics
        mean_ari = mean_pairwise_ari(label_runs)
        mean_clusters = float(np.mean(ncls))
        std_clusters = float(np.std(ncls))
        mean_quality = float(np.mean(qualities))
        std_quality = float(np.std(qualities))

        rows.append(
            dict(
                dataset=DATASET,
                split=SPLIT,
                pca_dims=PCA_DIMS_DEFAULT,
                k=k,
                resolution=res,
                seeds=len(SEEDS),
                mean_clusters=mean_clusters,
                std_clusters=std_clusters,
                mean_ari=mean_ari,
                mean_quality=mean_quality,
                std_quality=std_quality,
                n_vertices=n,
                n_edges=m,
                mean_degree=mean_deg,
                n_components=n_components,
                target=TARGET_CLUSTERS,
                abs_diff_from_target=abs(mean_clusters - TARGET_CLUSTERS),
            )
        )
        print(
            f"k={k:>2}, res={res:>3.1f} -> "
            f"clusters {mean_clusters:.1f}±{std_clusters:.1f}, "
            f"ARI={mean_ari:.3f}, quality={mean_quality:.4f}, "
            f"|Δ|={abs(mean_clusters - TARGET_CLUSTERS):.1f}"
        )

    df = pd.DataFrame(rows).sort_values(
        ["abs_diff_from_target", "mean_ari", "mean_quality"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    # Save results
    out_csv = OUT_DIR / f"{DATASET.lower()}_{SPLIT}_scan.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved scan results: {out_csv}")

    # Console summary: top candidates near target cluster count
    top = df.head(10)[
        [
            "k",
            "resolution",
            "mean_clusters",
            "std_clusters",
            "mean_ari",
            "mean_quality",
            "mean_degree",
            "n_components",
        ]
    ]
    print("\nTop candidates near target cluster count:")
    print(top.to_string(index=False))


if __name__ == "__main__":
    main()
