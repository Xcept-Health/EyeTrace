"""
Inter-pupillary distance (IPD).
"""

import numpy as np

def inter_pupillary_distance(left_pupil, right_pupil):
    """
    Compute distance between pupils.

    Parameters
    ----------
    left_pupil, right_pupil : np.ndarray, shape (2,)
        Pupil centers (pixels).

    Returns
    -------
    float
        Distance in pixels.
    """
    return np.linalg.norm(left_pupil - right_pupil)