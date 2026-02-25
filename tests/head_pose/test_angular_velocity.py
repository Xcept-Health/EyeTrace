"""
Tests unitaires pour eyetrace.head_pose.angular_velocity
"""

import pytest
import numpy as np
from eyetrace.head_pose.angular_velocity import head_angular_velocity


def test_head_angular_velocity():
    """Test calcul de la vitesse angulaire de la tête."""
    times = np.arange(0, 0.3, 0.033)  # ~9 frames
    pitch = np.sin(times * 8) * 0.25
    roll = np.zeros_like(times)
    yaw = np.cos(times * 5) * 0.18

    p_vel, r_vel, y_vel = head_angular_velocity(pitch, roll, yaw, times)

    assert len(p_vel) == len(times)
    assert isinstance(p_vel, np.ndarray)
    assert isinstance(r_vel, np.ndarray)
    assert isinstance(y_vel, np.ndarray)