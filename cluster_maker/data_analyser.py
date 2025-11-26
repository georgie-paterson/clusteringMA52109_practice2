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

def get_numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates key descriptive statistics for all numeric columns in a DataFrame,
    including mean, std, min, max, and the count of missing (NaN) values.

    The function is robust and ignores non-numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame with mixed data types.

    Returns
    -------
    summary : pandas.DataFrame
        A summary DataFrame with features as the index and columns: 
        mean, std, min, max, count_na.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")
        
    # Select only numeric columns (Requirement: robust to non-numeric)
    numeric_df = df.select_dtypes(include=np.number)

    if numeric_df.empty:
        # User Interaction: Inform the user if nothing was found
        print("INFO: No numeric columns found in the DataFrame to summarize.")
        return pd.DataFrame()
        
     
    # 1. Get standard descriptive statistics (mean, std, min, max)
    summary_stats = numeric_df.describe().loc[['mean', 'std', 'min', 'max']].T
    
    # 2. Calculate the count of missing (NaN) values
    missing_count = numeric_df.isnull().sum().rename('count_na')
    
    # 3. Combine descriptive stats with missing count
    final_summary = pd.concat([summary_stats, missing_count], axis=1)
    
    final_summary.columns = ['mean', 'std', 'min', 'max', 'count_na']

    # Report ignored columns
    ignored_cols = [col for col in df.columns if col not in numeric_df.columns]
    if ignored_cols:
        print(f"INFO: Ignored non-numeric columns in summary: {', '.join(ignored_cols)}")

    return final_summary