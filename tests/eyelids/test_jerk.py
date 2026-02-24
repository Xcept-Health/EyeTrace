"""
Tests for eyelids.jerk module.
"""

import pytest
import numpy as np
from eyetrace.eyelids import ear_jerk


def test_ear_jerk_constant():
    """Test jerk for constant EAR (should be zero)."""
    ear = np.ones(100) * 0.3
    t = np.arange(100) / 30.0  # 30 fps
    jerk = ear_jerk(ear, t, smooth=False)
    # Derivative of constant is zero, so jerk should be near zero
    np.testing.assert_allclose(jerk, 0.0, atol=1e-6)


def test_ear_jerk_linear():
    """Test jerk for linearly increasing EAR (should be zero)."""
    ear = np.linspace(0.2, 0.4, 100)
    t = np.arange(100) / 30.0
    jerk = ear_jerk(ear, t, smooth=False)
    # First derivative constant, second derivative zero
    np.testing.assert_allclose(jerk, 0.0, atol=1e-5)


def test_ear_jerk_parabolic():
    """Test jerk for quadratic EAR (second derivative constant)."""
    t = np.arange(100) / 30.0
    ear = 0.1 + 0.1 * t**2   # quadratic
    jerk = ear_jerk(ear, t, smooth=False)
    # Second derivative should be constant 0.2
    expected_jerk = np.ones_like(t) * 0.2
    # Ignorer les 2 premiers et 2 derniers points à cause des effets de bord
    np.testing.assert_allclose(jerk[2:-2], expected_jerk[2:-2], rtol=0.1)