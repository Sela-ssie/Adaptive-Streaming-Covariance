"""Reference covariance-estimation functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sample_covariance(X: ArrayLike) -> NDArray[np.float64]:
    """
    Compute the unbiased sample covariance matrix.

    Parameters
    ----------
    X:
        Two-dimensional array-like object with shape
        (n_observations, n_features).

        Each row represents one observation.
        Each column represents one variable or asset.

    Returns
    -------
    numpy.ndarray
        Covariance matrix with shape (n_features, n_features).

    Raises
    ------
    ValueError
        If X is not two-dimensional or contains fewer than two observations.
    """
    observations = np.asarray(X, dtype=np.float64)

    if observations.ndim != 2:
        raise ValueError(
            "X must be two-dimensional with shape "
            "(n_observations, n_features)."
        )

    n_observations, n_variables = observations.shape

    if n_observations < 2:
        raise ValueError(
            "At least two observations are required to compute "
            "an unbiased sample covariance."
        )

    if n_variables == 0:
        raise ValueError("X must have at least one variable (column).")
    
    mean = observations.mean(axis=0)
    centered = observations - mean

    scatter = centered.T @ centered

    return scatter / (n_observations - 1)