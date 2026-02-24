"""
EAR jerk (rate of change of EAR velocity).
"""

import numpy as np
from ..pupil.dynamics import first_derivative

def ear_jerk(ear_sequence: np.ndarray, times: np.ndarray,
             smooth: bool = True) -> np.ndarray:
    """
    Compute the jerk (second derivative) of EAR.

    Jerk is the derivative of velocity, i.e., the rate of change of
    eyelid movement speed. It can be used to detect abrupt movements.

    Parameters
    ----------
    ear_sequence : np.ndarray
        EAR time series.
    times : np.ndarray
        Corresponding timestamps.
    smooth : bool
        Whether to smooth before differentiation.

    Returns
    -------
    np.ndarray
        Jerk values (same length as input).
    """
    vel = first_derivative(ear_sequence, times, smooth=smooth)
    jerk = first_derivative(vel, times, smooth=smooth)
    return jerk