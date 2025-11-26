###
## cluster_maker - test file 2
## Georgie Paterson - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data


class TestDataFrameBuilder(unittest.TestCase):
    def test_define_dataframe_structure_basic(self):
        column_specs = [
            {"name": "x", "reps": [0.0, 1.0, 2.0]},
            {"name": "y", "reps": [10.0, 11.0, 12.0]},
        ]
        seed_df = define_dataframe_structure(column_specs)
        self.assertEqual(seed_df.shape, (3, 2))
        self.assertListEqual(list(seed_df.columns), ["x", "y"])
        self.assertTrue(np.allclose(seed_df["x"].values, [0.0, 1.0, 2.0]))

    def test_simulate_data_shape(self):
        column_specs = [
            {"name": "x", "reps": [0.0, 5.0]},
            {"name": "y", "reps": [2.0, 4.0]},
        ]
        seed_df = define_dataframe_structure(column_specs)
        data = simulate_data(seed_df, n_points=100, random_state=1)
        self.assertEqual(data.shape[0], 100)
        self.assertIn("true_cluster", data.columns)

    def test_column_summary(self):
        from cluster_maker.data_analyser import column_summary

        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [10, None, 30],
            "c": [5.0, 5.0, 5.0],
            "label": ["x", "y", "z"]
        })

        summary = column_summary(df)

        self.assertIn("a", summary.index)
        self.assertIn("b", summary.index)
        self.assertIn("c", summary.index)
        self.assertIn("label", summary.index)

        self.assertEqual(summary.loc["a", "type"], "numeric")
        self.assertEqual(summary.loc["b", "type"], "numeric")
        self.assertEqual(summary.loc["c", "type"], "numeric")
        self.assertEqual(summary.loc["label", "type"], "non-numeric")

        self.assertEqual(summary.loc["b", "n_missing"], 1)
        self.assertEqual(summary.loc["a", "mean"], 2)
        self.assertEqual(summary.loc["c", "std"], 0)


if __name__ == "__main__":
    unittest.main()
