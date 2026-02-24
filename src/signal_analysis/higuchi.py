"""
Higuchi fractal dimension.
"""

import numpy as np

def higuchi_fractal_dimension(signal, kmax=None):
    """
    Compute the Higuchi fractal dimension of a time series.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.
    kmax : int, optional
        Maximum interval (default = len(signal)//4).

    Returns
    -------
    float
        Fractal dimension (between 1 and 2).
    """
    signal = np.asarray(signal)
    n = len(signal)
    if kmax is None:
        kmax = n // 4
    if kmax < 2:
        return np.nan

    L = []
    k_values = range(1, kmax+1)
    for k in k_values:
        Lk = 0
        for m in range(k):
            # Subsequence starting at m with step k
            sub = signal[m:n:k]
            if len(sub) < 2:
                continue
            # Length of the curve
            length = np.sum(np.abs(np.diff(sub))) * (n - 1) / (k * (len(sub) - 1))
            Lk += length
        L.append(Lk / k)

    log_k = np.log(k_values)
    log_L = np.log(L)
    # Linear fit excluding possibly the first point (optional)
    coeffs = np.polyfit(log_k, log_L, 1)
    return -coeffs[0]  # Higuchi FD = -slope