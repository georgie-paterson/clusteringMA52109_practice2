###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Union, TextIO

import pandas as pd


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

def export_summary(summary_df: pd.DataFrame, csv_path: str, txt_path: str) -> None:
    """
    Export a summary DataFrame to both CSV and plain-text formats.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summary statistics as produced by column_summary().
        Expected to contain columns such as:
        ['column', 'mean', 'std', 'min', 'max', 'n_missing', 'note']

    csv_path : str
        File path where the CSV version will be saved.

    txt_path : str
        File path where the human-readable text summary will be saved.

    Notes
    -----
    - The CSV file contains the raw table of summary statistics.
    - The text file contains a neatly formatted, readable summary with
      one section per column.
    """
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")

    # Save the CSV directly
    summary_df.to_csv(csv_path, index=False)

    # Build human-readable text
    lines = []
    for _, row in summary_df.iterrows():
        col_name = row["column"]
        note = row.get("note", "")

        # Format numeric values (NaN-safe)
        mean = row.get("mean")
        std = row.get("std")
        min_val = row.get("min")
        max_val = row.get("max")
        missing = row.get("n_missing", 0)

        lines.append(f"Column: {col_name}")
        lines.append(f"  Note: {note}")
        lines.append(f"  Mean: {mean}")
        lines.append(f"  Std: {std}")
        lines.append(f"  Min: {min_val}")
        lines.append(f"  Max: {max_val}")
        lines.append(f"  Missing values: {missing}")
        lines.append("")  # Blank line between columns

    # Write text summary
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))

