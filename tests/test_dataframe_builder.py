###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd

from cluster_maker.dataframe_builder import define_dataframe_structure, simulate_data
from cluster_maker.data_exporter import export_summary_reports


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
    
    def test_summarise_numeric_columns_handles_non_numeric_and_missing(self):
        # create DataFrame with 3 numeric cols, 1 non-numeric and one missing value
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, None, 6.0],
            "c": [0.0, 0.0, 1.0],
            "label": ["x", "y", "z"],
        })

        summary = export_summary_reports(df)

        # three numeric columns
        self.assertEqual(summary.shape[0], 3)
        # expected columns present
        self.assertListEqual(list(summary.columns), ["mean", "std", "min", "max", "n_missing"])
        # check values (sample std, mean, and missing counts)
        self.assertTrue(np.allclose(summary.loc["a", "mean"], 2.0))
        self.assertTrue(np.allclose(summary.loc["a", "std"], 1.0))
        self.assertTrue(np.allclose(summary.loc["b", "mean"], 5.0))
        self.assertEqual(int(summary.loc["b", "n_missing"]), 1)
        self.assertAlmostEqual(float(summary.loc["c", "std"]), 0.577350269, places=6)


if __name__ == "__main__":
    unittest.main()