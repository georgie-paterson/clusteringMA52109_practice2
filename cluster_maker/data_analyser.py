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
    Compute summary statistics for each column in the DataFrame.
    Numeric columns receive full statistics.
    Non-numeric columns are included and clearly reported.
    """

    rows = []

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_numeric_dtype(series):
            rows.append({
                "column": col,
                "type": "numeric",
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "max": series.max(),
                "n_missing": series.isna().sum(),
            })
        else:
            # clearly report non-numeric
            rows.append({
                "column": col,
                "type": "non-numeric",
                "mean": np.nan,
                "std": np.nan,
                "min": None,
                "max": None,
                "n_missing": series.isna().sum(),
            })

    summary_df = pd.DataFrame(rows).set_index("column")
    return summary_df