from __future__ import annotations

import sys
import os
import pandas as pd

# Ensure the package is importable if run from root without installation
try:
    from cluster_maker.data_analyser import calculate_extended_statistics
    from cluster_maker.data_exporter import export_summary_report
except ImportError:
    # Fallback: add parent directory to path to find cluster_maker
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from cluster_maker.data_analyser import calculate_extended_statistics
    from cluster_maker.data_exporter import export_summary_report

OUTPUT_DIR = "demo_output"

def main() -> None:
    # --- Part (a): Argument Validation ---
    # We expect 2 arguments: [script_name, input_csv_path]
    if len(sys.argv) != 2:
        print("ERROR: Invalid number of arguments.")
        print("Usage: python demo/analyse_from_csv.py <path/to/input.csv>")
        sys.exit(1)  # Exit cleanly without traceback

    input_path = sys.argv[1]
    
    # Check if file exists before trying to read it
    if not os.path.exists(input_path):
        print(f"ERROR: The file '{input_path}' was not found.")
        sys.exit(1)

    # --- Part (d): Progress Message ---
    print(f"Reading data from '{input_path}'...")

    # --- Part (b): Read CSV ---
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"ERROR: Could not read the CSV file. Reason: {e}")
        sys.exit(1)
        
    print(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns.")

    # --- Part (c): Compute Summary ---
    print("Calculating extended statistics for numeric columns...")
    try:
        summary_df = calculate_extended_statistics(df)
    except Exception as e:
        print(f"ERROR during analysis: {e}")
        sys.exit(1)

    if summary_df.empty:
        print("WARNING: No numeric columns found to analyse.")
    else:
        print(f"Computed statistics for {len(summary_df)} numeric columns.")

    # --- Part (c): Export Results ---
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    csv_output_path = os.path.join(OUTPUT_DIR, "analysis_summary.csv")
    txt_output_path = os.path.join(OUTPUT_DIR, "analysis_summary.txt")

    print(f"Exporting results to directory '{OUTPUT_DIR}'...")
    
    try:
        export_summary_report(summary_df, csv_output_path, txt_output_path)
    except Exception as e:
        print(f"ERROR during export: {e}")
        sys.exit(1)

    print("Done!")
    print(f"  -> CSV Summary: {csv_output_path}")
    print(f"  -> Text Report: {txt_output_path}")


if __name__ == "__main__":
    main()