###
## cluster_maker
## Yas Akilakulasingam - University of Bath
## November 2025
###


"""
Cluster stability diagnostic for evaluating robustness of clustering.

This module introduces a statistically meaningful extension that measures
how consistent the clustering output is under repeated perturbations.
"""

from __future__ import annotations
import numpy as np
from .algorithms import kmeans


def cluster_stability_score(
    X: np.ndarray,
    k: int,
    n_runs: int = 20,
    noise_scale: float = 0.05,
    random_state: int | None = None,
) -> float:
    """
    Compute clustering stability while correcting for label switching.

    Cluster labels are aligned across runs by sorting cluster centroids,
    ensuring that label 0 always refers to the same cluster structure,
    label 1 to the next cluster, etc.

    This produces a meaningful stability score in [0,1].
    """

    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]

    # Co-occurrence matrix
    co_matrix = np.zeros((n_samples, n_samples))

    # --- Base clustering to determine stable label order ---
    base_labels, base_centroids = kmeans(X, k=k, random_state=random_state)

    # Sort centroids by their position (e.g. first feature)
    order = np.lexsort((base_centroids[:, 1], base_centroids[:, 0]))
    label_map = {old: new for new, old in enumerate(order)}

    # Align base labels
    aligned_base = np.array([label_map[l] for l in base_labels])

    # Update co-occurrence for the base run
    for i in range(n_samples):
        for j in range(n_samples):
            if aligned_base[i] == aligned_base[j]:
                co_matrix[i, j] += 1

    # --- Perturbation runs ---
    for _ in range(n_runs - 1):
        noise = rng.normal(scale=noise_scale, size=X.shape)
        X_noisy = X + noise

        labels, centroids = kmeans(X_noisy, k=k, random_state=random_state)

        # Align labels using sorted centroids
        order = np.lexsort((centroids[:, 1], centroids[:, 0]))
        label_map = {old: new for new, old in enumerate(order)}
        aligned = np.array([label_map[l] for l in labels])

        for i in range(n_samples):
            for j in range(n_samples):
                if aligned[i] == aligned[j]:
                    co_matrix[i, j] += 1

    # Final stability matrix
    stability_matrix = co_matrix / n_runs

    # Compute mean off-diagonal stability
    stability = (
        np.sum(stability_matrix) - np.trace(stability_matrix)
    ) / (n_samples * (n_samples - 1))

    return float(stability)