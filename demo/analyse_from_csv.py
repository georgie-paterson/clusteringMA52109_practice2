from __future__ import annotations
import sys
import os
import pandas as pd

from cluster_maker.data_analyser import numeric_summary
from cluster_maker.data_exporter import export_numeric_summary


OUTPUT_DIR = "demo_output"


def main(args: list[str]) -> None:
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments.")
        print("Usage: python demo/analyse_from_csv.py path/to/data.csv")
        sys.exit(1)

    input_path = args[1]

    if not os.path.exists(input_path):
        print(f"ERROR: File '{input_path}' not found.")
        sys.exit(1)

    print(f"Reading CSV: {input_path}")
    df = pd.read_csv(input_path)

    print("Computing numeric summary...")
    summary = numeric_summary(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_out = os.path.join(OUTPUT_DIR, "summary.csv")
    txt_out = os.path.join(OUTPUT_DIR, "summary.txt")

    print("Saving output files...")
    export_numeric_summary(summary, csv_out, txt_out)

    print("Done. Files saved in demo_output/")


if __name__ == "__main__":
    main(sys.argv)
