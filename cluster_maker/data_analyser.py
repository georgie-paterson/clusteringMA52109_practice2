###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

import pandas as pd
import numpy as np  


def calculate_descriptive_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics for each numeric column in the DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    stats : pandas.DataFrame
        Result of `data.describe()` including count, mean, std, etc.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    return data.describe()


def calculate_correlation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the correlation matrix for numeric columns in the DataFrame.

    Parameters
    ----------
    data : pandas.DataFrame

    Returns
    -------
    corr : pandas.DataFrame
        Correlation matrix.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    return data.corr(numeric_only=True)

def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics for all columns in the input DataFrame.

    For each numeric column, the function returns:
    - mean
    - standard deviation
    - minimum value
    - maximum value
    - number of missing values

    Non-numeric columns are clearly reported: their summary statistics
    are returned as NaN, and a descriptive note is included so users can
    see which columns were excluded from the numeric analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame potentially containing mixed column types.

    Returns
    -------
    summary_df : pandas.DataFrame
        A DataFrame with one row per column and the following fields:
        ['column', 'mean', 'std', 'min', 'max', 'n_missing', 'note']
    """
    summary_records = []

    for col in df.columns:
        series = df[col]

        # Numeric columns: compute statistics
        if pd.api.types.is_numeric_dtype(series):
            summary_records.append({
                "column": col,
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "n_missing": series.isna().sum(),
                "note": "numeric",
            })

        # Non-numeric columns: clearly reported but not analysed
        else:
            summary_records.append({
                "column": col,
                "mean": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "n_missing": series.isna().sum(),
                "note": "non-numeric (ignored)",
            })

    return pd.DataFrame(summary_records)
