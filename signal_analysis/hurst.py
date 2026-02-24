"""
Hurst exponent estimation using rescaled range (R/S) method.
"""

import numpy as np

def hurst_exponent(signal):
    """
    Estimate the Hurst exponent using the R/S method.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.

    Returns
    -------
    float
        Hurst exponent (between 0 and 1). H > 0.5 indicates persistence,
        H < 0.5 anti-persistence, H = 0.5 random walk.
    """
    signal = np.asarray(signal)
    n = len(signal)
    if n < 100:
        return np.nan

    # Compute cumulative deviations from mean
    mean = np.mean(signal)
    cumdev = np.cumsum(signal - mean)

    # Compute R/S for different lags
    lags = np.unique(np.logspace(0, np.log10(n//2), 20).astype(int))
    rs_values = []

    for lag in lags:
        if lag < 2:
            continue
        n_segments = n // lag
        rs_seg = []
        for i in range(n_segments):
            segment = cumdev[i*lag:(i+1)*lag]
            R = np.max(segment) - np.min(segment)
            S = np.std(signal[i*lag:(i+1)*lag], ddof=1)
            if S > 0:
                rs_seg.append(R / S)
        if rs_seg:
            rs_values.append(np.mean(rs_seg))

    if len(rs_values) < 2:
        return np.nan

    # Fit log(RS) ~ H * log(lag)
    log_lags = np.log(lags[:len(rs_values)])
    log_rs = np.log(rs_values)
    H, _ = np.polyfit(log_lags, log_rs, 1)
    return H