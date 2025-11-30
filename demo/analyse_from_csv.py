###
## cluster_maker: demo for analysing data from a CSV file
## James Foadi - University of Bath
## November 2025
###


from __future__ import annotations

import os
import sys
import pandas as pd

from cluster_maker.data_analyser import column_summary
from cluster_maker.data_exporter import export_summary

OUTPUT_DIR = "demo_output"


def main(args: list[str]) -> None:
    print("\n=== cluster_maker: CSV Analysis Tool ===\n")

    # ------------------------------------------------------------
    # (a) Check number of arguments
    # ------------------------------------------------------------
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments.")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv")
        return  # must exit gracefully, no traceback

    input_path = args[1]
    print(f"> Input file provided: {input_path}\n")

    # ------------------------------------------------------------
    # (b) Read the input CSV
    # ------------------------------------------------------------
    print("Step 1: Loading CSV file...")

    if not os.path.exists(input_path):
        print(f"ERROR: The file '{input_path}' does not exist.")
        return

    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        print(f"ERROR: Failed to read the CSV file.\nDetails: {exc}")
        return

    print("✓ CSV loaded successfully.")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}\n")

    # ------------------------------------------------------------
    # (c) Compute summary using column_summary()
    # ------------------------------------------------------------
    print("Step 2: Computing numeric column summary...\n")

    try:
        summary_df = column_summary(df)
    except Exception as exc:
        print(f"ERROR: Analysis failed.\nDetails: {exc}")
        return

    print("✓ Summary created.")
    print("Preview:")
    print(summary_df.head(), "\n")

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_output_path = os.path.join(OUTPUT_DIR, "column_summary.csv")
    txt_output_path = os.path.join(OUTPUT_DIR, "column_summary.txt")

    # ------------------------------------------------------------
    # (c) Export results using export_summary()
    # ------------------------------------------------------------
    print("Step 3: Saving results...")

    try:
        export_summary(summary_df, csv_output_path, txt_output_path)
    except Exception as exc:
        print(f"ERROR: Could not save output files.\nDetails: {exc}")
        return

    print("✓ Results saved successfully.")
    print(f"  CSV summary file: {csv_output_path}")
    print(f"  Text summary file: {txt_output_path}\n")

    # ------------------------------------------------------------
    # (d) Final message
    # ------------------------------------------------------------
    print("=== Analysis Complete ===")
    print("You can now open the summary files in the 'demo_output' directory.\n")


if __name__ == "__main__":
    main(sys.argv)
