###
## cluster_maker: analysis from CSV
## MA52109 – Mock Exam
## November 2025
###

from __future__ import annotations

import sys
import os
import pandas as pd

from cluster_maker.data_analyser import column_summary
from cluster_maker.data_exporter import export_summary


OUTPUT_DIR = "demo_output"


def main(args: list[str]) -> None:
    print("=== cluster_maker: CSV analysis tool ===\n")

    # Require exactly 1 argument: input CSV file
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments.")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv\n")
        sys.exit(1)

    input_path = args[1]
    print(f"Input CSV file: {input_path}")

    # Check file existence
    if not os.path.exists(input_path):
        print(f"\nERROR: The file '{input_path}' does not exist.")
        sys.exit(1)

    # Load CSV
    print("\nLoading data...")
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        print(f"ERROR: Unable to read CSV file:\n{exc}")
        sys.exit(1)

    print("Data loaded successfully.")
    print(f"Number of rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("-" * 60)

    # Ensure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Run numeric summary
    print("Running numeric summary using column_summary()...")
    summary = column_summary(df)
    print("Summary computed.")

    # Save outputs
    csv_out = os.path.join(OUTPUT_DIR, "summary.csv")
    txt_out = os.path.join(OUTPUT_DIR, "summary.txt")

    print(f"\nSaving CSV summary to:\n  {csv_out}")
    print(f"Saving human-readable summary to:\n  {txt_out}")

    export_summary(summary, csv_out, txt_out)

    print("\nAnalysis completed.")
    print("=== End of analysis ===")


if __name__ == "__main__":
    main(sys.argv)
