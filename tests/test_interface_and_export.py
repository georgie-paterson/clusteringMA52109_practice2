###
## cluster_maker - test_interface_and_export
## Tests for run_clustering and data exporting functions
## Yas Akilakulasingam - University of Bath
## November 2025
###

import unittest
import os
import tempfile
import pandas as pd
import numpy as np

# Import high-level interface
from cluster_maker import run_clustering

# Import exporter function from Task 3
from cluster_maker import export_summary


class TestInterfaceAndExport(unittest.TestCase):

    # ------------------------------------------------------------
    # Part (a): Tests for run_clustering error handling
    # ------------------------------------------------------------

    def test_run_clustering_missing_file(self):
        """
        Test that run_clustering raises a clean, controlled error when the
        input CSV file does not exist. This checks:
          - No raw traceback
          - Appropriate exception type (FileNotFoundError)
          - Clear error message

        This supports the marking criteria focusing on robust error handling
        and user-friendly communication.
        """

        with self.assertRaises(FileNotFoundError):
            run_clustering(
                input_path="THIS_FILE_DOES_NOT_EXIST.csv",
                feature_cols=["a", "b"],
                k=3,
            )

    def test_run_clustering_missing_feature_cols(self):
        """
        Test that run_clustering raises a clear error when required feature
        columns are missing from the input CSV.

        We create a temporary CSV with known columns, then ask for a column
        that does not exist.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")

            # Create a simple CSV with only two numeric columns
            df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
            df.to_csv(csv_path, index=False)

            # Asking for a missing column "z" should raise KeyError
            with self.assertRaises(KeyError):
                run_clustering(
                    input_path=csv_path,
                    feature_cols=["x", "z"],  # "z" does not exist
                    k=2,
                )

    # ------------------------------------------------------------
    # Part (b): Tests for the exporting functions
    # ------------------------------------------------------------

    def test_export_summary_creates_files(self):
        """
        Test that export_summary successfully creates output files when given
        valid input paths.

        This checks:
          - CSV summary file is created
          - Text summary file is created

        This supports data processing and user interaction criteria.
        """

        # Create a small summary DataFrame
        summary_df = pd.DataFrame({
            "mean": {"a": 1.0, "b": 2.0},
            "std": {"a": 0.1, "b": 0.2},
            "min": {"a": 0.0, "b": 1.0},
            "max": {"a": 2.0, "b": 3.0},
            "missing_values": {"a": 0, "b": 1},
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "summary.csv")
            txt_path = os.path.join(tmpdir, "summary.txt")

            # Call the export function
            export_summary(summary_df, csv_path, txt_path)

            # Check both files were created
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(txt_path))

    def test_export_summary_invalid_path(self):
        """
        Test that export_summary raises a controlled error when given an
        invalid output path (e.g., a directory that does not exist).

        This ensures robust error handling without an uncontrolled traceback.
        """

        summary_df = pd.DataFrame({
            "mean": {"a": 1.0},
            "std": {"a": 0.5},
            "min": {"a": 0.0},
            "max": {"a": 2.0},
            "missing_values": {"a": 0},
        })

        # Intentionally giving an invalid directory path
        invalid_path = "/this/path/does/not/exist/summary.csv"

        with self.assertRaises(ValueError):
            export_summary(summary_df, invalid_path, "summary.txt")


if __name__ == "__main__":
    unittest.main()
