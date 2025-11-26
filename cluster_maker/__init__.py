"""
cluster_maker

An educational Python package for generating synthetic clustered data,
running clustering algorithms, evaluating results, and producing
user-friendly plots. Designed for practicals and exams where students
work with an incomplete or faulty version of the package and must fix it.

Allowed libraries:
- Python standard library
- numpy
- pandas
- matplotlib
- scipy
- scikit-learn
"""

"""
cluster_maker

An educational Python package for generating synthetic clustered data,
running clustering algorithms, evaluating results, and producing
user-friendly plots.
"""

# --- Data generation ---
from .dataframe_builder import define_dataframe_structure, simulate_data

# --- Data analysis (Task 3 will add more functions later) ---
from .data_analyser import numeric_summary

# --- Export utilities ---
from .data_exporter import export_to_csv, export_formatted

# --- Preprocessing ---
from .preprocessing import select_features, standardise_features

# --- Clustering algorithms ---
from .algorithms import (
    kmeans,
    sklearn_kmeans,
    init_centroids,
    assign_clusters,
    update_centroids,
)

# --- Evaluation ---
from .evaluation import (
    compute_inertia,
    silhouette_score_sklearn,
    elbow_curve,
)

# --- Plotting ---
from .plotting_clustered import plot_clusters_2d, plot_elbow

# --- High-level interface ---
from .interface import run_clustering


__all__ = [
    # Data generation
    "define_dataframe_structure",
    "simulate_data",

    # Analysis
    "numeric_summary",

    # Export
    "export_to_csv",
    "export_formatted",

    # Preprocessing
    "select_features",
    "standardise_features",

    # Algorithms
    "kmeans",
    "sklearn_kmeans",
    "init_centroids",
    "assign_clusters",
    "update_centroids",

    # Evaluation
    "compute_inertia",
    "silhouette_score_sklearn",
    "elbow_curve",

    # Plotting
    "plot_clusters_2d",
    "plot_elbow",

    # High-level orchestration
    "run_clustering",
]
