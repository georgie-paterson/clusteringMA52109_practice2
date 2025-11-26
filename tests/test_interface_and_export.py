import os
import tempfile
import unittest

import pandas as pd

from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import (
    export_to_csv,
    export_formatted,
    export_summary_reports,
)
from cluster_maker.data_analyser import summarise_numeric_columns


class TestInterfaceAndExport(unittest.TestCase):
    def test_run_clustering_missing_input_file_raises_file_not_found(self):
        missing = "this_file_should_not_exist_99999.csv"
        with self.assertRaises(FileNotFoundError) as cm:
            run_clustering(missing)
        self.assertTrue(str(cm.exception))

    def test_run_clustering_missing_required_features_raises_value_error(self):
        # create a CSV with only a non-feature column
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as fh:
            fh.write("only_col\nx\ny\nz\n")
            tmp = fh.name
        try:
            with self.assertRaises(ValueError) as cm:
                run_clustering(tmp)
            self.assertTrue(str(cm.exception))
        finally:
            os.unlink(tmp)

    def test_export_functions_create_files_and_fail_on_invalid_paths(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "out.csv")
            txt_path = os.path.join(td, "out.txt")

            # should create files
            export_to_csv(df, csv_path)
            export_formatted(df, txt_path)
            self.assertTrue(os.path.exists(csv_path))
            self.assertTrue(os.path.exists(txt_path))

            # summary exporter
            summary = summarise_numeric_columns(df)
            csv2 = os.path.join(td, "summary.csv")
            txt2 = os.path.join(td, "summary.txt")
            export_summary_reports(summary, csv2, txt2)
            self.assertTrue(os.path.exists(csv2))
            self.assertTrue(os.path.exists(txt2))

            # invalid path: make a regular file where a directory is expected
            bad_file = os.path.join(td, "somefile")
            with open(bad_file, "w", encoding="utf-8") as fh:
                fh.write("not a directory")

            bad_dest = os.path.join(bad_file, "will_fail.csv")
            with self.assertRaises(Exception):
                export_to_csv(df, bad_dest)


if __name__ == "__main__":
    unittest.main()