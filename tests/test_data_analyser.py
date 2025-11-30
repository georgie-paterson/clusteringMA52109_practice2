###
## cluster_maker - test file for data_analyser.py
## Georgie Paterson - University of Bath
## November 2025
###

import unittest
import pandas as pd
import numpy as np

from cluster_maker.data_analyser import column_summary


class TestColumnSummary(unittest.TestCase):
    """
    Tests for the column_summary function introduced in Task 3(a).
    """

    def test_column_summary_mixed_dataframe(self):
        """
        Check that the function correctly handles:
        - three numeric columns,
        - one non-numeric column,
        - at least one missing value.
        """

        # Create a DataFrame with mixed types and a missing value
        df = pd.DataFrame({
            "height": [150, 160, 170, np.nan],   # numeric with a missing value
            "weight": [50.0, 55.5, 60.0, 65.0],  # numeric
            "age": [20, 25, 30, 35],             # numeric
            "name": ["Alice", "Bob", "Cara", "Dan"]  # non-numeric
        })

        summary = column_summary(df)

        # ---- Structural checks ----
        self.assertEqual(len(summary), 4)  # 4 columns → 4 summary rows

        expected_columns = [
            "column", "mean", "std", "min", "max", "n_missing", "note"
        ]
        for col in expected_columns:
            self.assertIn(col, summary.columns)

        # ---- Numeric column checks ----
        row_height = summary[summary["column"] == "height"].iloc[0]
        self.assertEqual(row_height["n_missing"], 1)
        self.assertAlmostEqual(row_height["mean"], df["height"].mean(), places=6)
        self.assertEqual(row_height["note"], "numeric")

        # ---- Non-numeric column checks ----
        row_name = summary[summary["column"] == "name"].iloc[0]
        self.assertTrue(np.isnan(row_name["mean"]))
        self.assertEqual(row_name["note"], "non-numeric (ignored)")


if __name__ == "__main__":
    unittest.main()
