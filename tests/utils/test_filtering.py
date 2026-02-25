"""
Tests for utils.filtering module.
"""

import pytest
import numpy as np
from eyetrace.utils.filtering import (
    moving_average,
    savgol_filter,
    kalman_filter_1d
)

def test_moving_average():
    """Test moving average filter."""
    x = np.array([1, 2, 3, 4, 5])
    filtered = moving_average(x, window_size=3)
    assert len(filtered) == len(x)

def test_savgol_filter():
    """Test Savitzky-Golay filter."""
    x = np.sin(np.linspace(0, 10, 100))
    filtered = savgol_filter(x, window_length=11, polyorder=2)
    assert len(filtered) == len(x)

def test_kalman_filter_1d():
    """Test 1D Kalman filter."""
    x = np.array([1, 2, 3, 2, 1, 0, -1])
    filtered = kalman_filter_1d(x, Q=1e-3, R=0.1)
    assert len(filtered) == len(x)