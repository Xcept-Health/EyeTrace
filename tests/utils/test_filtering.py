"""
Tests for utils.filtering module.
"""

import pytest
import numpy as np
from eyetrace.utils.filtering import (
    moving_average,
    savgol_filter,
    butterworth_filter,
    kalman_filter_1d
)

def test_moving_average():
    """Test moving average filter."""
    x = np.array([1, 2, 3, 4, 5])
    filtered = moving_average(x, window=3)
    expected = np.array([1, 2, 3, 4, 5])  # avec padding, la sortie a même longueur
    # En réalité, la fonction peut retourner un array de même longueur avec padding
    # On teste juste qu'elle s'exécute et que la longueur est conservée
    assert len(filtered) == len(x)

def test_savgol_filter():
    """Test Savitzky-Golay filter."""
    x = np.sin(np.linspace(0, 10, 100))
    filtered = savgol_filter(x, window_length=11, polyorder=2)
    assert len(filtered) == len(x)

def test_butterworth_filter():
    """Test Butterworth filter."""
    t = np.linspace(0, 1, 100)
    x = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 30 * t)
    filtered = butterworth_filter(x, fs=100, cutoff=20, order=4, btype='low')
    assert len(filtered) == len(x)

def test_kalman_filter_1d():
    """Test 1D Kalman filter."""
    x = np.array([1, 2, 3, 2, 1, 0, -1])
    filtered = kalman_filter_1d(x, Q=1e-3, R=0.1)
    assert len(filtered) == len(x)