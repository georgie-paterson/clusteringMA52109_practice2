###
## cluster_maker - test pca extension
## Georgie Paterson - University of Bath
## November 2025
###

import unittest
import numpy as np
import pandas as pd
import tempfile
import os

from cluster_maker.preprocessing import apply_pca
from cluster_maker.interface import run_clustering


class TestPCAExtension(unittest.TestCase):
    """
    Tests for the PCA preprocessing extension.

    These tests go beyond simple checks and include:
    - dimensionality validation
    - variance ordering checks
    - behaviour under correlated inputs
    - integration with run_clustering
    """

    # ---------------------------------------------------------
    # A: PCA reduces dimensions correctly
    # ---------------------------------------------------------
    def test_pca_reduces_dimensions(self):
        """PCA should reduce a 4D dataset to 2D."""
        X = np.random.rand(30, 4)

        X_reduced = apply_pca(X, n_components=2)

        self.assertEqual(X_reduced.shape, (30, 2))

    # ---------------------------------------------------------
    # B: PCA preserves variance ordering
    # ---------------------------------------------------------
    def test_pca_variance_ordering(self):
        """
        PCA components must be ordered by descending variance.
        Component 1 should explain at least as much variance as Component 2.
        """
        X = np.random.rand(100, 3)
        X_reduced = apply_pca(X, n_components=3)

        variances = np.var(X_reduced, axis=0)

        self.assertGreaterEqual(variances[0], variances[1])
        self.assertGreaterEqual(variances[1], variances[2])

    # ---------------------------------------------------------
    # C: PCA responds correctly to correlated variables
    # ---------------------------------------------------------
    def test_pca_captures_correlation_structure(self):
        """
        PCA should capture the main direction of variance if two variables
        are highly correlated (eigenvalue gap should be large).
        """
        x = np.linspace(0, 10, 200)
        y = 3 * x + np.random.normal(0, 0.01, size=200)  # strongly correlated
        z = np.random.normal(0, 1, size=200)

        X = np.column_stack([x, y, z])
        X_reduced = apply_pca(X, n_components=2)

        # First component should dominate the second due to correlation
        var_pc1 = np.var(X_reduced[:, 0])
        var_pc2 = np.var(X_reduced[:, 1])

        self.assertGreater(var_pc1, 5 * var_pc2)

    # ---------------------------------------------------------
    # D: PCA should error on invalid n_components
    # ---------------------------------------------------------
    def test_pca_invalid_component_numbers(self):
        """Zero or too-many components should raise ValueError."""

        X = np.random.rand(10, 3)

        with self.assertRaises(ValueError):
            apply_pca(X, n_components=0)

        with self.assertRaises(ValueError):
            apply_pca(X, n_components=10)

    # ---------------------------------------------------------
    # E: Integration test — PCA inside run_clustering
    # ---------------------------------------------------------
    def test_run_clustering_with_pca(self):
        """
        Integration test to ensure PCA correctly plugs into run_clustering.
        The clustering output's centroids must respect the PCA dimension.
        """

        df = pd.DataFrame({
            "a": np.random.rand(40),
            "b": np.random.rand(40),
            "c": np.random.rand(40),
            "d": np.random.rand(40),
        })

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "data.csv")
            df.to_csv(csv_path, index=False)

            result = run_clustering(
                input_path=csv_path,
                feature_cols=["a", "b", "c", "d"],
                k=3,
                standardise=True,
                use_pca=True,
                pca_components=2,
            )

            centroids = result["centroids"]

            # Expect centroids with 2 PCA components
            self.assertEqual(centroids.shape[1], 2)

    # ---------------------------------------------------------
    # F: PCA output should be centered (empirical mean ~ 0)
    # ---------------------------------------------------------
    def test_pca_output_mean_is_zero(self):
        """
        PCA-transformed data should have mean approximately zero
        along each principal component (floating-point tolerance allowed).
        """
        X = np.random.rand(200, 5)

        X_reduced = apply_pca(X, n_components=3)

        means = np.mean(X_reduced, axis=0)

        for m in means:
            self.assertAlmostEqual(m, 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
