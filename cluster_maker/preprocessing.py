###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def select_features(data: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Select a subset of columns to use as features, ensuring they are numeric.

    Parameters
    ----------
    data : pandas.DataFrame
    feature_cols : list of str
        Column names to select.

    Returns
    -------
    X_df : pandas.DataFrame
        DataFrame containing only the selected feature columns.

    Raises
    ------
    KeyError
        If any requested column is missing.
    TypeError
        If any selected column is non-numeric.
    """
    missing = [col for col in feature_cols if col not in data.columns]
    if missing:
        raise KeyError(f"The following feature columns are missing: {missing}")

    X_df = data[feature_cols].copy()

    non_numeric = [
        col for col in X_df.columns
        if not pd.api.types.is_numeric_dtype(X_df[col])
    ]
    if non_numeric:
        raise TypeError(f"The following feature columns are not numeric: {non_numeric}")

    return X_df


def standardise_features(X: np.ndarray) -> np.ndarray:
    """
    Standardise features to zero mean and unit variance.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)

    Returns
    -------
    X_scaled : ndarray of shape (n_samples, n_features)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    scaler = StandardScaler()
    return scaler.fit_transform(X)

def pca_transform(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Perform PCA using SVD and return the data projected onto the first
    n_components principal components.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input numeric data.

    n_components : int
        Number of principal components to keep.

    Returns
    -------
    X_pca : array, shape (n_samples, n_components)
        Transformed data.
    """

    # Step 1: centre the data
    X_centered = X - np.mean(X, axis=0)

    # Step 2: SVD (numerically stable PCA)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Step 3: project onto first principal components
    components = Vt[:n_components]      # shape (n_components, n_features)
    X_pca = X_centered @ components.T   # shape (n_samples, n_components)

    return X_pca