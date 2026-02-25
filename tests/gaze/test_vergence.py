"""
Tests for gaze.vergence module.
"""

import pytest
import numpy as np
from eyetrace.gaze.vergence import vergence_speed

def test_vergence_speed():
    """Test vergence speed (rate of change of angle)."""
    t = np.array([0, 0.1, 0.2])
    # Simulate gaze vectors for left and right eyes
    left_gaze = np.array([
        [0, 0, 1],
        [0.1, 0, 0.99],
        [0.2, 0, 0.98]
    ])
    right_gaze = np.array([
        [0, 0, 1],
        [-0.1, 0, 0.99],
        [-0.2, 0, 0.98]
    ])
    speed = vergence_speed(left_gaze, right_gaze, t, smooth=False)
    # Vergence is increasing, so speed should be positive
    assert np.all(speed >= 0)
    assert len(speed) == len(t)
