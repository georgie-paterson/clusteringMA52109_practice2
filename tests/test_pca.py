###
## cluster_maker - PCA tests
## Georgie Paterson - University of Bath
## November 2025
###

###
## cluster_maker – PCA tests
## University of Bath – November 2025
###

import unittest
import numpy as np

from cluster_maker.preprocessing import pca_transform


class TestPCA(unittest.TestCase):

    def test_pca_dimension_reduction(self):
        """
        PCA should reduce dimensions correctly when n_components < original dims.
        """
        rng = np.random.RandomState(123)
        X = rng.randn(200, 5)     # 200 samples, 5 features

        X_pca = pca_transform(X, n_components=2)

        # Check reduced dimensionality
        self.assertEqual(X_pca.shape, (200, 2))

    def test_pca_variance_ordering(self):
        """
        The first principal component should have higher variance than the second.
        This is a core mathematical property of PCA.
        """
        rng = np.random.RandomState(42)
        X = rng.randn(500, 4)     # More robust random dataset

        X_pca = pca_transform(X, n_components=2)

        var_pc1 = np.var(X_pca[:, 0])
        var_pc2 = np.var(X_pca[:, 1])

        # First component MUST explain more variance than second
        self.assertGreater(var_pc1, var_pc2)

    def test_pca_stability_under_scaling(self):
        """
        PCA should be invariant to uniform scaling of the entire dataset.
        Only the scaling of components should change, not their ordering.
        """
        rng = np.random.RandomState(10)
        X = rng.randn(300, 3)
        X_scaled = 5 * X  # scale the entire dataset uniformly

        X_pca = pca_transform(X, n_components=2)
        X_scaled_pca = pca_transform(X_scaled, n_components=2)

        # Check ordering of variances remains the same
        var_original = np.var(X_pca, axis=0)
        var_scaled = np.var(X_scaled_pca, axis=0)

        # Variances should scale by factor**2, but ordering should remain
        self.assertGreater(var_original[0], var_original[1])
        self.assertGreater(var_scaled[0], var_scaled[1])


if __name__ == "__main__":
    unittest.main()

