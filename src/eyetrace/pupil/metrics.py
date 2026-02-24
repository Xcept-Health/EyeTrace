"""
Basic statistical metrics for pupil diameter time series.
"""

import numpy as np
from typing import Union, List

def variance(diameters: Union[np.ndarray, List[float]]) -> float:
    """
    Compute the variance of pupil diameters.

    Parameters
    ----------
    diameters : array-like
        Sequence of pupil diameter measurements.

    Returns
    -------
    float
        Variance (unbiased, i.e., divided by N-1).
    """

    arr = np.asarray(diameters, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    return np.var(arr, ddof=1)

def std_dev(diameters: Union[np.ndarray, List[float]]) -> float:
    """
    Compute the standard deviation of pupil diameters.

    Parameters
    ----------
    diameters : array-like
        Sequence of pupil diameter measurements.

    Returns
    -------
    float
        Standard deviation (unbiased).
    """
    arr = np.asarray(diameters, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    return np.std(arr, ddof=1)


def coefficient_variation(diameters: Union[List[float], np.ndarray]) -> float:
    """
    Compute the coefficient of variation (CV) of pupil diameters.

    CV = (std / mean) * 100, expressed as a percentage.

    Parameters
    ----------
    diameters : array-like
        Sequence of pupil diameter measurements.

    Returns
    -------
    float
        Coefficient of variation in percent.
    """
    
    arr = np.asarray(diameters, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    mean = np.mean(arr)
    if mean == 0:
        return 0.0
    std = np.std(arr, ddof=1)
    return (std / mean) * 100.0


def normalized_diameter(
    diameters: Union[List[float], np.ndarray],
    reference_diameter: float
) -> np.ndarray:
    """
    Normalize pupil diameter by a reference diameter (e.g., baseline).

    Parameters
    ----------
    diameters : array-like
        Sequence of pupil diameter measurements.
    reference_diameter : float
        Reference diameter (e.g., mean under controlled lighting).

    Returns
    -------
    np.ndarray
        Normalized diameters (d / reference).
    """
    
    arr = np.asarray(diameters, dtype=np.float64)
    if reference_diameter == 0:
        raise ValueError("reference_diameter cannot be zero")
    return arr / reference_diameter



def zscore(diameters: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Compute the Z-score normalization of pupil diameters.

    Z = (d - mean) / std

    Parameters
    ----------
    diameters : array-like
        Sequence of pupil diameter measurements.

    Returns
    -------
    np.ndarray
        Z-scores.
    """
    arr = np.asarray(diameters, dtype=np.float64)
    mean = np.mean(arr)
    std = np.std(arr, ddof=1)
    if std == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std