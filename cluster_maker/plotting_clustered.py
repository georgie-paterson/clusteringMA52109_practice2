###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_clusters_2d(X, labels, centroids=None, title="Cluster plot"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", alpha=0.7, s=35)

    if centroids is not None:
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            marker="X",
            s=200,
            c="black",
            label="Centroids"
        )

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster label")

    fig.tight_layout()
    plt.close()

    return fig, ax




def plot_elbow(k_values, inertias, title="Elbow Curve"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(k_values, inertias, marker="o", markersize=7, linewidth=2)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Number of clusters (k)", fontsize=12)
    ax.set_ylabel("Inertia", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

    fig.tight_layout()
    
    plt.close()
    return fig, ax
