import unittest
import pandas as pd
import os

from cluster_maker.interface import run_clustering
from cluster_maker.data_exporter import export_numeric_summary
from cluster_maker.data_analyser import numeric_summary


class TestInterfaceAndExport(unittest.TestCase):

    def test_missing_input_file(self):
        with self.assertRaises(FileNotFoundError):
            run_clustering("no_such_file.csv", ["a", "b"])

    def test_missing_required_columns(self):
        df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        df.to_csv("temp.csv", index=False)

        with self.assertRaises(ValueError):
            run_clustering("temp.csv", ["nope"])

        os.remove("temp.csv")

    def test_exporter_creates_files(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [5, 6, 7]})
        summary = numeric_summary(df)

        os.makedirs("tmp_out", exist_ok=True)
        csv_path = "tmp_out/test.csv"
        txt_path = "tmp_out/test.txt"

        export_numeric_summary(summary, csv_path, txt_path)

        self.assertTrue(os.path.exists(csv_path))
        self.assertTrue(os.path.exists(txt_path))

        # Cleanup
        os.remove(csv_path)
        os.remove(txt_path)
        os.rmdir("tmp_out")
