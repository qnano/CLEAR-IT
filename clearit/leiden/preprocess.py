from __future__ import annotations
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple


def standardize_and_pca(
    X: np.ndarray,
    n_components: int = 64,
    seed: int = 42,
) -> Tuple[np.ndarray, StandardScaler, PCA]:
    """
    Standardize features and compute PCA embedding.

    Returns
    -------
    X_pca : ndarray [N, n_components]
    scaler : fitted StandardScaler
    pca : fitted PCA object
    """
    if X.ndim != 2:
        raise ValueError("X must be 2D [N, D].")

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xz = scaler.fit_transform(X)

    pca = PCA(n_components=n_components, random_state=seed, svd_solver="auto")
    X_pca = pca.fit_transform(Xz)
    return X_pca, scaler, pca
