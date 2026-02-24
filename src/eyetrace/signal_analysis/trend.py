"""
Trend slope estimation.
"""

import numpy as np

def trend_slope(signal, times=None):
    """
    Estimate linear trend slope using least squares.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.
    times : np.ndarray, optional
        Time indices. If None, uses uniform indices.

    Returns
    -------
    float
        Slope (units of signal per unit time).
    """
    signal = np.asarray(signal)
    n = len(signal)
    if times is None:
        x = np.arange(n)
    else:
        x = np.asarray(times)
    A = np.vstack([x, np.ones(n)]).T
    slope, intercept = np.linalg.lstsq(A, signal, rcond=None)[0]
    return slope