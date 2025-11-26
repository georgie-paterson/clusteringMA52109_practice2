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
    Export the summary DataFrame to CSV, and also create a human-readable
    text summary with one line per column.
    """

    # Write CSV summary
    summary_df.to_csv(csv_path, index=True)

    # Write readable text summary
    with open(txt_path, "w") as f:
        for col, row in summary_df.iterrows():
            f.write(
                f"Column: {col} | "
                f"mean={row['mean']:.4f}, "
                f"std={row['std']:.4f}, "
                f"min={row['min']}, "
                f"max={row['max']}, "
                f"missing={row['n_missing']}\n"
            )