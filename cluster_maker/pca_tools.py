###
## cluster_maker – PCA tools
## MA52109 – Mock Exam Extension
## November 2025
###

from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA


def apply_pca(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Apply PCA dimensionality reduction.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input feature matrix.
    n_components : int, default 2
        Number of PCA components to keep.

    Returns
    -------
    X_reduced : ndarray of shape (n_samples, n_components)
        PCA-transformed data.
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    if n_components <= 0:
        raise ValueError("n_components must be a positive integer.")

    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)
