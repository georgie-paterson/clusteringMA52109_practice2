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
    Export a summary DataFrame (created by summarise_numeric_columns)
    to both a CSV file and a neatly formatted text file.

    Parameters
    ----------
    summary_df : pandas.DataFrame
        Summary statistics DataFrame from data_analyser.summarise_numeric_columns.
    csv_path : str
        Output CSV file path.
    txt_path : str
        Output plain-text file path.

    Notes
    -----
    - Includes friendly, clear error messages.
    """

    # --- Type checking for robustness ---
    if not isinstance(summary_df, pd.DataFrame):
        raise TypeError("summary_df must be a pandas DataFrame.")

    # Write CSV
    try:
        summary_df.to_csv(csv_path, index=True)
    except Exception as exc:
        raise ValueError(f"Could not write CSV file to '{csv_path}': {exc}")

    # Write human-readable text file
    try:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("Summary of Numeric Columns\n")
            f.write("=" * 40 + "\n\n")
            for col in summary_df.index:
                stats = summary_df.loc[col]
                line = (
                    f"{col}: "
                    f"mean={stats['mean']:.3f}, "
                    f"std={stats['std']:.3f}, "
                    f"min={stats['min']}, "
                    f"max={stats['max']}, "
                    f"missing={stats['missing_values']}"
                )
                f.write(line + "\n")

    except Exception as exc:
        raise ValueError(f"Could not write text summary to '{txt_path}': {exc}")
