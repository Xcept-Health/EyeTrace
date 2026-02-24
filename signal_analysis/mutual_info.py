"""
Mutual information between two signals.
"""

import numpy as np
from sklearn.metrics import mutual_info_score

def mutual_information(x, y, bins=10):
    """
    Compute mutual information between two signals.

    Parameters
    ----------
    x, y : np.ndarray
        Input signals (same length).
    bins : int or array-like
        Number of bins for histogram.

    Returns
    -------
    float
        Mutual information in nats.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    # Discretize
    c_xy = np.histogram2d(x, y, bins)[0]
    return mutual_info_score(None, None, contingency=c_xy)