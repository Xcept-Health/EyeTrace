"""
Tests for gaze.saccades module.
"""

import pytest
import numpy as np
from eyetrace.gaze.saccades import (
    detect_saccades,
    saccade_velocity,
    saccade_acceleration
)

def test_detect_saccades():
    """Test that saccade detection identifies at least one event."""
    t = np.linspace(0, 10, 100)
    points = np.zeros((100, 3))
    
    # Add a non-zero Z component to avoid null vectors
    points[:, 2] = 1.0
    
    # Create a synthetic saccade from frame 40 to 50
    points[40:51, 0] = np.linspace(0, 5, 11)
    
    saccades = detect_saccades(points, t, velocity_threshold=1.0)
    assert len(saccades) >= 1

def test_saccade_velocity():
    """Test that velocity calculation returns a valid array of the correct length."""
    t = np.array([0, 0.1, 0.2, 0.3])
    points = np.array([
        [0, 0, 1],
        [2, 0, 1],
        [4, 0, 1],
        [4, 0, 1]
    ])
    
    vel = saccade_velocity(points, t, smooth=False)
    
    # Ensure output length matches input and contains no NaNs
    assert len(vel) == len(points)
    assert np.all(np.isfinite(vel))

def test_saccade_acceleration():
    """Test that acceleration calculation returns finite values and correct length."""
    t = np.array([0, 0.1, 0.2, 0.3])
    points = np.array([
        [0, 0, 1],
        [1, 0, 1],
        [3, 0, 1],
        [6, 0, 1]
    ])
    
    # Calculate velocity first
    vel = saccade_velocity(points, t, smooth=False)
    
    # Then calculate acceleration
    acc = saccade_acceleration(vel, t)
    
    assert len(acc) == len(points)
    assert np.all(np.isfinite(acc))

