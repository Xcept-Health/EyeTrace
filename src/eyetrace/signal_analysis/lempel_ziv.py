"""
Lempel-Ziv complexity.
"""

import numpy as np

def lempel_ziv_complexity(signal, normalize=True):
    """
    Compute Lempel-Ziv complexity of a binary sequence.

    The signal is first binarized (e.g., by thresholding at median).

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.
    normalize : bool
        If True, normalize by n / log2(n) (typical for binary sequences).

    Returns
    -------
    float
        Lempel-Ziv complexity.
    """
    # Binarize: values above median -> 1, below -> 0
    median = np.median(signal)
    binary = (signal > median).astype(int)
    n = len(binary)

    c = 1
    u = 1
    v = 1
    v_max = 1
    for i in range(1, n):
        if binary[i] == binary[v-1]:
            v += 1
        else:
            v_max = max(v, v_max)
            v = i - u + 1
            u = i + 1
            c += 1
    v_max = max(v, v_max)
    c += 1

    if normalize:
        return c * np.log2(n) / n
    else:
        return c