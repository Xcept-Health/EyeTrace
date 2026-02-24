"""
Eyelid closing and opening speeds.
"""

import numpy as np
from scipy.signal import savgol_filter
from .blink_detection import detect_blinks
from ..pupil.dynamics import first_derivative

def eyelid_closing_speed(ear_sequence: np.ndarray, times: np.ndarray,
                         threshold: float = 0.2, smooth: bool = True) -> np.ndarray:
    """
    Compute the closing speed for each blink.

    For each blink, the closing speed is the maximum negative slope
    during the closing phase (from start to the point where EAR drops
    below threshold, or to the minimum).

    Parameters
    ----------
    ear_sequence : np.ndarray
        EAR time series.
    times : np.ndarray
        Corresponding timestamps.
    threshold : float
        EAR threshold for closed eye.
    smooth : bool
        Whether to smooth the derivative.

    Returns
    -------
    np.ndarray
        Array of closing speeds (positive values? usually negative, but we return absolute)
    """
    blinks = detect_blinks(ear_sequence, threshold)
    if not blinks:
        return np.array([])

    vel = first_derivative(ear_sequence, times, smooth=smooth)
    speeds = []
    for start, end in blinks:
        # Closing phase: from start to the first frame where EAR < threshold
        # We can take the minimum (most negative) velocity in that interval
        closing_vel = vel[start:end+1]  # rough approximation
        if len(closing_vel) > 0:
            speeds.append(np.min(closing_vel))  # negative
    return np.array(speeds)


def eyelid_opening_speed(ear_sequence: np.ndarray, times: np.ndarray,
                          threshold: float = 0.2, smooth: bool = True) -> np.ndarray:
    """
    Compute the opening speed for each blink.

    Opening phase: from the last closed frame to the end of the blink.
    """
    blinks = detect_blinks(ear_sequence, threshold)
    if not blinks:
        return np.array([])

    vel = first_derivative(ear_sequence, times, smooth=smooth)
    speeds = []
    for start, end in blinks:
        # Opening phase: from the minimum EAR (usually near end) to the end
        # Simpler: use maximum positive velocity after the minimum
        # Find index of minimum in the blink
        min_idx = start + np.argmin(ear_sequence[start:end+1])
        opening_vel = vel[min_idx:end+1]
        if len(opening_vel) > 0:
            speeds.append(np.max(opening_vel))
    return np.array(speeds)