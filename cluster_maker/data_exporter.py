###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import os
import pandas as pd


def export_to_csv(df: pd.DataFrame, path: str, delimiter: str = ",", include_index: bool = False) -> None:
    """
    Export a DataFrame to CSV. Used throughout the package.
    """
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    df.to_csv(path, sep=delimiter, index=include_index)


def export_numeric_summary(summary_df: pd.DataFrame, csv_path: str, txt_path: str) -> None:
    """
    Export the numeric summary both as:
        - CSV file
        - human-readable text file
    """
    # CSV export
    export_to_csv(summary_df, csv_path, delimiter=",", include_index=True)

    # Text export
    directory = os.path.dirname(txt_path)
    if directory and not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    with open(txt_path, "w") as f:
        for col in summary_df.index:
            row = summary_df.loc[col]
            f.write(
                f"{col}: mean={row['mean']:.3f}, std={row['std']:.3f}, "
                f"min={row['min']}, max={row['max']}, missing={row['missing']}\n"
            )

def export_summary(summary_df, csv_path, txt_path):
    """
    Save numeric summary to CSV and a clean text file.
    """

    # Save CSV
    summary_df.to_csv(csv_path)

    # Save text summary
    with open(txt_path, "w") as f:
        for col in summary_df.index:
            stats = summary_df.loc[col]
            line = (
                f"{col}: mean={stats['mean']:.3f}, "
                f"std={stats['std']:.3f}, "
                f"min={stats['min']:.3f}, "
                f"max={stats['max']:.3f}, "
                f"missing={int(stats['missing'])}"
            )
            f.write(line + "\n")