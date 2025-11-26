import unittest
import numpy as np
from cluster_maker.pca_extension import run_pca


class TestPCAExtension(unittest.TestCase):

    def test_pca_output_shapes(self):
        X = np.random.randn(20, 5)
        transformed, components, var = run_pca(X, n_components=2)

        self.assertEqual(transformed.shape, (20, 2))
        self.assertEqual(components.shape, (2, 5))
        self.assertEqual(len(var), 2)

    def test_pca_reduces_dimension(self):
        X = np.random.randn(10, 4)
        transformed, _, _ = run_pca(X, n_components=3)
        self.assertEqual(transformed.shape[1], 3)
