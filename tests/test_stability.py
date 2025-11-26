import unittest
import numpy as np
from cluster_maker import cluster_stability_score


class TestClusterStability(unittest.TestCase):

    def test_stability_returns_valid_value(self):
        """
        Basic test: stability score should be between 0 and 1,
        and the function should run without errors.
        """
        X = np.array([
            [1, 2],
            [3, 4],
            [5.0, 5.2],
            [5.1, 5.3],
        ])

        score = cluster_stability_score(X, k=2, n_runs=5, noise_scale=0.01)

        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

if __name__ == "__main__":
    unittest.main()
