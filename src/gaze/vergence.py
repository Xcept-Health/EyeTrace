"""
Vergence speed: rate of change of the angle between the two eyes' gaze axes.
"""

import numpy as np
from .utils import angular_velocity
from ..pupil.dynamics import first_derivative

def vergence_angle(left_gaze: np.ndarray, right_gaze: np.ndarray) -> np.ndarray:
    """
    Compute the vergence angle between left and right gaze vectors.
    
    Parameters
    ----------
    left_gaze, right_gaze : np.ndarray, shape (n, 3)
        3D gaze vectors for each eye.
    
    Returns
    -------
    np.ndarray, shape (n,)
        Vergence angle in radians.
    """
    n = len(left_gaze)
    angle = np.zeros(n)
    for i in range(n):
        v1 = left_gaze[i] / np.linalg.norm(left_gaze[i])
        v2 = right_gaze[i] / np.linalg.norm(right_gaze[i])
        dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle[i] = np.arccos(dot)
    return angle


def vergence_speed(left_gaze: np.ndarray, right_gaze: np.ndarray,
                   times: np.ndarray, smooth: bool = True) -> np.ndarray:
    """
    Compute the speed of vergence angle change.
    
    Returns
    -------
    np.ndarray
        Vergence speed (rad/s) at each time.
    """
    angle = vergence_angle(left_gaze, right_gaze)
    speed = first_derivative(angle, times, smooth=smooth)
    return speed