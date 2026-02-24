"""
Signal filtering utilities.
"""

import numpy as np
from scipy.signal import savgol_filter as scipy_savgol


def moving_average(data, window_size):
    """
    Apply moving average filter.

    Parameters
    ----------
    data : np.ndarray
        1D array of data.
    window_size : int
        Size of the averaging window (must be odd).

    Returns
    -------
    np.ndarray
        Filtered data (same length as input).
    """
    if window_size % 2 == 0:
        window_size += 1  # ensure odd
    kernel = np.ones(window_size) / window_size
    padded = np.pad(data, window_size//2, mode='edge')
    filtered = np.convolve(padded, kernel, mode='valid')
    return filtered


def savgol_filter(data, window_length, polyorder):
    """
    Wrapper for scipy's Savitzky-Golay filter.

    Parameters
    ----------
    data : np.ndarray
        1D array.
    window_length : int
        Must be odd and > polyorder.
    polyorder : int
        Polynomial order.

    Returns
    -------
    np.ndarray
        Smoothed data.
    """
    if len(data) < window_length:
        return data.copy()
    return scipy_savgol(data, window_length, polyorder)


def kalman_filter_1d(data, Q=1e-3, R=1e-1):
    """
    Simple 1D Kalman filter for smoothing.

    Parameters
    ----------
    data : np.ndarray
        1D array of measurements.
    Q : float
        Process noise covariance.
    R : float
        Measurement noise covariance.

    Returns
    -------
    np.ndarray
        Filtered estimates.
    """
    n = len(data)
    x_est = np.zeros(n)
    p_est = np.zeros(n)

    # Initialization
    x_est[0] = data[0]
    p_est[0] = 1.0

    for k in range(1, n):
        # Prediction
        x_pred = x_est[k-1]
        p_pred = p_est[k-1] + Q

        # Update
        K = p_pred / (p_pred + R)
        x_est[k] = x_pred + K * (data[k] - x_pred)
        p_est[k] = (1 - K) * p_pred

    return x_est