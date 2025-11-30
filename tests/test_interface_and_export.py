###
## cluster_maker - test file - task 5
## Georgie Paterson - University of Bath
## November 2025
###

import unittest
import os
import pandas as pd
import numpy as np
import tempfile

from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_summary
from cluster_maker.data_analyser import column_summary


class TestInterfaceAndExport(unittest.TestCase):
    """
    Tests for high-level interface behaviour (run_clustering)
    and exporting functions (export_summary), as required for Task 5.
    """

    # -------------------------------------------------------------
    # 5a(i) Missing input file → clean, controlled error
    # -------------------------------------------------------------
    def test_run_clustering_missing_file(self):
        """run_clustering should raise FileNotFoundError for a missing CSV file."""

        missing_path = "this/path/does/not/exist.csv"

        with self.assertRaises(FileNotFoundError):
            run_clustering(
                input_path=missing_path,
                feature_cols=["x", "y"],
                algorithm="kmeans",
                k=3,
                standardise=True,
                output_path="dummy.csv",
                random_state=0,
                compute_elbow=False,
            )

    # -------------------------------------------------------------
    # 5a(ii) Missing required feature columns → clean, controlled error
    # -------------------------------------------------------------
    def test_run_clustering_missing_required_columns(self):
        """run_clustering should raise KeyError when required feature columns are missing."""

        # Create a temporary CSV missing the expected feature columns
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "data.csv")
            df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})  # wrong columns
            df.to_csv(csv_path, index=False)

            # run_clustering will call select_features → KeyError is expected
            with self.assertRaises(KeyError):
                run_clustering(
                    input_path=csv_path,
                    feature_cols=["x", "y"],   # required but not present
                    algorithm="kmeans",
                    k=3,
                    standardise=True,
                    output_path=os.path.join(tmpdir, "out.csv"),
                    random_state=0,
                    compute_elbow=False,
                )

    # -------------------------------------------------------------
    # 5b(i) export_summary should create both CSV and TXT files
    # -------------------------------------------------------------
    def test_export_summary_creates_files(self):
        """export_summary should successfully create CSV and TXT output files."""

        # Small valid summary DataFrame
        df = pd.DataFrame({
            "column": ["x", "y"],
            "mean": [0.0, 1.0],
            "std": [1.0, 2.0],
            "min": [0.0, -1.0],
            "max": [2.0, 3.0],
            "n_missing": [0, 0],
            "note": ["numeric", "numeric"],
        })

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_out = os.path.join(tmpdir, "summary.csv")
            txt_out = os.path.join(tmpdir, "summary.txt")

            export_summary(df, csv_out, txt_out)

            self.assertTrue(os.path.exists(csv_out))
            self.assertTrue(os.path.exists(txt_out))

    # -------------------------------------------------------------
    # 5b(ii) export_summary should raise a clean error on invalid path
    # -------------------------------------------------------------
    def test_export_summary_invalid_path(self):
        """export_summary should raise an error when given an invalid output path."""

        summary = pd.DataFrame({
            "column": ["x"],
            "mean": [1.0],
            "std": [0.5],
            "min": [0.0],
            "max": [2.0],
            "n_missing": [0],
            "note": ["numeric"],
        })

        bad_csv_path = "/this/does/not/exist/output.csv"
        bad_txt_path = "/this/does/not/exist/output.txt"

        with self.assertRaises(Exception):  # IOError/OSError acceptable
            export_summary(summary, bad_csv_path, bad_txt_path)


if __name__ == "__main__":
    unittest.main()
