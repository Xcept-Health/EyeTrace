"""
General mathematical helper functions.
"""

import numpy as np
from collections import deque


def mad_outlier_removal(data, threshold=3.5):
    """
    Remove outliers based on Median Absolute Deviation (MAD).

    Parameters
    ----------
    data : np.ndarray
        1D array.
    threshold : float
        Threshold for outlier detection (commonly 3.5).

    Returns
    -------
    mask : np.ndarray (bool)
        False for outliers.
    cleaned : np.ndarray
        Data with outliers replaced by NaN (or original with mask).
    """
    # Convertir en float pour permettre l'assignation de NaN
    data = np.asarray(data, dtype=np.float64)
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    if mad == 0:
        return np.ones_like(data, dtype=bool), data.copy()
    modified_z_scores = 0.6745 * (data - median) / mad
    mask = np.abs(modified_z_scores) < threshold
    cleaned = data.copy()
    cleaned[~mask] = np.nan
    return mask, cleaned


def sliding_window_view(data, window_size):
    """
    Generate a sliding window view of a 1D array.

    Parameters
    ----------
    data : np.ndarray
        1D input.
    window_size : int
        Size of each window.

    Returns
    -------
    np.ndarray, shape (n_windows, window_size)
        Each row is a window.
    """
    if len(data) < window_size:
        return np.array([]).reshape(0, window_size)
    shape = (len(data) - window_size + 1, window_size)
    strides = (data.strides[0], data.strides[0])
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def find_peaks(data, height=None, distance=1):
    """
    Simple peak detection (local maxima).

    Parameters
    ----------
    data : np.ndarray
        1D array.
    height : float, optional
        Minimum peak height.
    distance : int
        Minimum number of samples between peaks.

    Returns
    -------
    peaks : np.ndarray
        Indices of detected peaks.
    """
    if len(data) < 3:
        return np.array([])
    # Find local maxima (greater than both neighbors)
    peaks = np.where((data[1:-1] > data[:-2]) & (data[1:-1] > data[2:]))[0] + 1
    if height is not None:
        peaks = peaks[data[peaks] > height]
    if distance > 1:
        # Sort peaks by height and filter by distance (greedy)
        sorted_idx = np.argsort(data[peaks])[::-1]
        filtered = []
        for idx in sorted_idx:
            p = peaks[idx]
            if not filtered or all(abs(p - fp) >= distance for fp in filtered):
                filtered.append(p)
        peaks = np.array(sorted(filtered))
    return peaks