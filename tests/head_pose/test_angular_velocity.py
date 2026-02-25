"""
Tests for head_pose.angular_velocity module.
"""

import pytest
import numpy as np
from eyetrace.head_pose.angular_velocity import head_angular_velocity

def test_head_angular_velocity():
    """Test angular velocity calculation."""
    t = np.array([0, 0.1, 0.2, 0.3])
    pitch = np.array([0, 0.1, 0.2, 0.3])
    roll = np.array([0, 0, 0, 0])
    yaw = np.array([0, -0.1, -0.2, -0.3])
    pv, rv, yv = head_angular_velocity(pitch, roll, yaw, t)
    assert len(pv) == len(t)
    assert len(rv) == len(t)
    assert len(yv) == len(t)
    # Vitesse attendue : pitch ~1 rad/s, yaw ~ -1 rad/s
    np.testing.assert_allclose(pv[1:-1], 1.0, rtol=0.1)
    np.testing.assert_allclose(yv[1:-1], -1.0, rtol=0.1)