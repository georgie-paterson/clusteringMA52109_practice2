###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Union, TextIO

import pandas as pd

# import the analyser function so we can compute a summary if a raw DataFrame is passed
from .data_analyser import summarise_numeric_columns

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

def export_summary_reports(
    summary_df: pd.DataFrame,
    csv_path: str,
    text_path: str,
    delimiter: str = ",",
    include_index: bool = False,
    float_format: str = ".6g",
) -> None:
    """
    Export a numeric-summary DataFrame to CSV and to a human-readable text file.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Expected to contain per-column statistics (e.g. index=column name,
        columns include 'mean','std','min','max','n_missing'). Non-present
        fields are ignored when formatting.
    csv_path : str
        Destination CSV file path.
    text_path : str
        Destination human-readable text file path.
    delimiter : str
        CSV delimiter (default ",").
    include_index : bool
        Whether to include the DataFrame index in the CSV.
    float_format : str
        Format specification for floating values when writing the text report.
    """
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")
    
    # Detect if input is already a summary (contains expected statistic columns).
    required_cols = {"mean", "std", "min", "max", "n_missing"}
    if not required_cols.issubset(set(summary_df.columns)):
        # treat input as raw data and compute the summary automatically
        summary = summarise_numeric_columns(summary_df)
    else:
        summary = summary_df

    # Write CSV via the existing helper
    export_to_csv(summary_df, csv_path, delimiter=delimiter, include_index=include_index)

    # Build a readable one-line-per-column summary
    def fmt(v):
        if pd.isna(v):
            return "NA"
        try:
            return format(float(v), float_format)
        except Exception:
            return str(v)

    lines = []
    idx_name = (
        summary_df.index.name if summary_df.index.name else "column"
    )
    header = f"Summary report written from DataFrame (index = {idx_name})\n"
    header += "-" * 80 + "\n"
    lines.append(header)

    for col_name, row in summary_df.iterrows():
        parts = []
        for key in ["mean", "std", "min", "max", "n_missing"]:
            if key in row:
                parts.append(f"{key}={fmt(row[key])}")
        line = f"{col_name}: " + ", ".join(parts)
        lines.append(line + "\n")

    # If summary_df is empty produce a helpful message
    if summary_df.shape[0] == 0:
        lines.append("No numeric columns found — nothing to summarise.\n")

    with open(text_path, "w", encoding="utf-8") as f:
        f.writelines(lines)