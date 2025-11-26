###
## cluster_maker: demo for cluster analysis
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import os
import sys
import pandas as pd

from cluster_maker import run_clustering, select_features

OUTPUT_DIR = "demo_output"


def main(args):
    print("\n=== Numeric Summary Demo ===\n")

    # Require exactly 2 arguments:
    if len(args) != 2:
        print("ERROR: Wrong number of arguments.")
        print("Usage: python demo/analyse_from_csv.py path/to/input.csv")
        sys.exit(1)

    input_path = args[1]

    if not os.path.exists(input_path):
        print(f"ERROR: File '{input_path}' does not exist.")
        sys.exit(1)

    print(f"Reading CSV file: {input_path}")
    df = pd.read_csv(input_path)
    print("File loaded successfully.\n")

    print("Computing summary for numeric columns...")
    summary = numeric_summary(df)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_out = os.path.join(OUTPUT_DIR, "summary.csv")
    txt_out = os.path.join(OUTPUT_DIR, "summary.txt")

    print("Saving results...")
    export_numeric_summary(summary, csv_out, txt_out)

    print("\nSummary saved to:")
    print(f"  - {csv_out}")
    print(f"  - {txt_out}")
    print("\n=== End of demo ===\n")


if __name__ == "__main__":
    main(sys.argv)