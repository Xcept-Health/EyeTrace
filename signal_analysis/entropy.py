"""
Entropy measures: sample entropy and approximate entropy.
"""

import numpy as np

def sample_entropy(signal, m=2, r=None):
    """
    Compute Sample Entropy (SampEn) of a time series.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.
    m : int
        Embedding dimension (pattern length).
    r : float, optional
        Tolerance (default is 0.2 * std(signal)). Usually 0.1-0.25.

    Returns
    -------
    float
        Sample Entropy value.
    """
    signal = np.asarray(signal)
    n = len(signal)
    if n < m + 1:
        return np.nan

    if r is None:
        r = 0.2 * np.std(signal, ddof=1)

    def _maxdist(xi, xj):
        return np.max(np.abs(xi - xj))

    def _phi(m):
        # Build templates of length m
        templates = [signal[i:i+m] for i in range(n - m + 1)]
        B = 0
        for i in range(len(templates)):
            for j in range(len(templates)):
                if i != j and _maxdist(templates[i], templates[j]) <= r:
                    B += 1
        return B / (len(templates) * (len(templates) - 1))

    Bm = _phi(m)
    Am = _phi(m+1)
    if Bm == 0 or Am == 0:
        return np.nan
    return -np.log(Am / Bm)


def approximate_entropy(signal, m=2, r=None):
    """
    Compute Approximate Entropy (ApEn) of a time series.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.
    m : int
        Embedding dimension.
    r : float, optional
        Tolerance (default 0.2 * std).

    Returns
    -------
    float
        Approximate Entropy.
    """
    signal = np.asarray(signal)
    n = len(signal)
    if n < m + 1:
        return np.nan

    if r is None:
        r = 0.2 * np.std(signal, ddof=1)

    def _phi(m):
        templates = [signal[i:i+m] for i in range(n - m + 1)]
        C = []
        for i in range(len(templates)):
            count = 0
            for j in range(len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
            C.append(count / (n - m + 1))
        return np.mean(np.log(C))

    return _phi(m) - _phi(m+1)