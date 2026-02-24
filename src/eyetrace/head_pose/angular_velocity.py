"""
Head angular velocity.
"""

import numpy as np
from ..pupil.dynamics import first_derivative

def head_angular_velocity(pitch, roll, yaw, times):
    """
    Compute angular velocity of the head.

    Parameters
    ----------
    pitch, roll, yaw : np.ndarray
        Time series of angles (radians).
    times : np.ndarray
        Corresponding timestamps.

    Returns
    -------
    tuple of np.ndarray
        (pitch_vel, roll_vel, yaw_vel) each same length as input.
    """
    pitch_vel = first_derivative(pitch, times, smooth=True)
    roll_vel = first_derivative(roll, times, smooth=True)
    yaw_vel = first_derivative(yaw, times, smooth=True)
    return pitch_vel, roll_vel, yaw_vel