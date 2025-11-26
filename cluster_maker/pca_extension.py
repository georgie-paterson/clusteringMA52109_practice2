from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA


def run_pca(X: np.ndarray, n_components: int = 2, random_state: int | None = None):
    """
    Simple PCA wrapper returning transformed data and components.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    transformed = pca.fit_transform(X)
    return transformed, pca.components_, pca.explained_variance_ratio_
