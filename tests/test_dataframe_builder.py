###
## cluster_maker - test file
## James Foadi - University of Bath
## November 2025
###

import unittest

import numpy as np
import pandas as pd
import cluster_maker.data_analyser as analyser

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

class TestDataAnalyser(unittest.TestCase):
    """
    New test class to verify the functionality of the data_analyser module.
    (You may integrate this method into your existing TestDataFrameBuilder 
    class if that is the only test class in your file.)
    """
    
    def test_get_numeric_summary_robustness(self):
        """
        Tests the get_numeric_summary function for correct output, robustness 
        to non-numeric columns, and accurate handling of missing values.

        Requirement Checklist:
        - at least 3 numeric columns: A_num, B_num, C_num (Check)
        - at least 1 non-numeric column: D_str (Check)
        - at least 1 missing value: np.nan in A_num (Check)
        """
        # Create a test DataFrame
        test_data = {
            'A_num': [10.0, 20.0, 30.0, np.nan, 50.0],  # 1 NaN, Mean=27.5
            'B_num': [5.0, 5.0, 5.0, 5.0, 5.0],        # All same, Std=0
            'C_num': [1.0, 2.0, 3.0, 4.0, 5.0],
            'D_str': ['a', 'b', 'c', 'd', 'e'],        # Non-numeric
            'E_bool': [True, False, True, False, True] # Non-numeric (bool)
        }
        df = pd.DataFrame(test_data)
        
        # Calculate the summary using the function from Task 3a
        summary = analyser.get_numeric_summary(df)
        
        # 1. Check shape and content (should only contain the 3 numeric columns)
        self.assertEqual(summary.shape, (3, 5), "Summary should have 3 rows (numeric columns) and 5 stats columns.")
        self.assertTrue('A_num' in summary.index, "A_num must be included in the summary index.")
        self.assertFalse('D_str' in summary.index, "D_str (non-numeric) must be ignored.")

        # 2. Check correctness for 'A_num'
        # Mean of [10, 20, 30, 50] = 110 / 4 = 27.5
        self.assertAlmostEqual(summary.loc['A_num', 'mean'], 27.5, places=5)
        self.assertEqual(summary.loc['A_num', 'count_na'], 1, "Should correctly count 1 missing value in A_num.")
        
        # 3. Check correctness for 'B_num' (standard deviation should be 0.0)
        self.assertEqual(summary.loc['B_num', 'std'], 0.0)
        
        # 4. Check correctness for 'C_num' (min and max)
        self.assertEqual(summary.loc['C_num', 'min'], 1.0)
        self.assertEqual(summary.loc['C_num', 'max'], 5.0)

if __name__ == "__main__":
    unittest.main()
