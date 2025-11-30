###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Union, TextIO

import pandas as pd


### Task 3b #######
def export_summary_report(
    summary_df: pd.DataFrame,
    base_filename: str,
) -> None:
    """
    Exports the descriptive statistics summary DataFrame into two files:
    1. A machine-readable CSV file.
    2. A human-readable, neatly formatted text report.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summary DataFrame (index=statistics, columns=features).
    base_filename : str
        The base name for the output files (e.g., "report" will produce
        "report_summary.csv" and "report_report.txt").
    """
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")

    print("\n--- Starting Summary Report Export ---")

    # --- 1. Export to CSV ---
    csv_filename = f"{base_filename}_summary.csv"
    print(f"Step 1: Exporting Summary DataFrame to CSV: {csv_filename}")
    
    # We include the index (statistics names) in the CSV for clarity
    export_to_csv(summary_df, csv_filename, include_index=True)
    print("  -> CSV export complete.")

    # --- 2. Generate Human-Readable Text Report ---
    report_filename = f"{base_filename}_report.txt"
    print(f"Step 2: Generating human-readable text report: {report_filename}")

    report_lines = ["\n*** Data Summary Report ***\n"]
    
    # The summary_df has features as columns, so we iterate over them
    for feature_name in summary_df.columns:
        # Access the series of statistics for the current feature
        stats = summary_df[feature_name]

        # Format the output line by line for neatness
        report_lines.append(f"--- Feature: {feature_name} ---")       
  
        # Get statistics with default 0 if not present, and format to 2 decimal places
        mean_val = stats.get('mean', 0.0)
        std_val = stats.get('std', 0.0)
        min_val = stats.get('min', 0.0)
        max_val = stats.get('max', 0.0)
        
        # Missing values should be integers
        missing_val = int(stats.get('missing_values', 0))

        report_lines.append(f"  Mean:            {mean_val:,.2f}")
        report_lines.append(f"  Standard Dev:    {std_val:,.2f}")
        report_lines.append(f"  Minimum:         {min_val:,.2f}")
        report_lines.append(f"  Maximum:         {max_val:,.2f}")
        report_lines.append(f"  Missing Values:  {missing_val:,d}")
        report_lines.append("\n") # Add extra line for separation

    final_report = "\n".join(report_lines)

    # --- 3. Write to Text File ---
    print(f"Step 3: Writing formatted report content to text file: {report_filename}")
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(final_report)

    print("  -> Text report export complete.")
    print("--- Summary Report Export Finished ---")




def export_to_csv(
    data: pd.DataFrame,
    filename: str,
    delimiter: str = ",",
    include_index: bool = False,
) -> None:
    """
    Export a DataFrame to CSV.

    Parameters
    ----------
    data : pandas.DataFrame
    filename : str
        Output filename.
    delimiter : str, default ","
    include_index : bool, default False
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    data.to_csv(filename, sep=delimiter, index=include_index)


def export_formatted(
    data: pd.DataFrame,
    file: Union[str, TextIO],
    include_index: bool = False,
) -> None:
    """
    Export a DataFrame as a formatted text table.

    Parameters
    ----------
    data : pandas.DataFrame
    file : str or file-like
        Filename or open file handle.
    include_index : bool, default False
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")

    table_str = data.to_string(index=include_index)

    if isinstance(file, str):
        with open(file, "w", encoding="utf-8") as f:
            f.write(table_str)
    else:
        file.write(table_str)