"""
Eyelid symmetry: correlation between left and right EAR.
"""

import numpy as np

def eyelid_symmetry(left_ear: np.ndarray, right_ear: np.ndarray) -> float:
    """
    Compute the correlation coefficient between left and right EAR.

    Parameters
    ----------
    left_ear, right_ear : np.ndarray
        EAR time series for left and right eyes (same length).

    Returns
    -------
    float
        Pearson correlation coefficient.
    """
    if len(left_ear) != len(right_ear) or len(left_ear) < 2:
        return np.nan
    return np.corrcoef(left_ear, right_ear)[0, 1]