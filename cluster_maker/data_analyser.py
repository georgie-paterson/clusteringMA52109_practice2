###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import pandas as pd


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for all numeric columns in a DataFrame.

    For each numeric column, return:
        - mean
        - standard deviation
        - minimum
        - maximum
        - count of missing values

    Non-numeric columns are ignored.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    summary : pandas.DataFrame
        Summary statistics indexed by column name.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # Select only numeric columns
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        raise ValueError("No numeric columns found in DataFrame.")

    summary = pd.DataFrame({
        "mean": numeric_df.mean(),
        "std": numeric_df.std(),
        "min": numeric_df.min(),
        "max": numeric_df.max(),
        "missing": numeric_df.isnull().sum()
    })

    return summary