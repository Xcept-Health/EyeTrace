"""
Tests for pupil.dynamics module.
"""

import pytest
import numpy as np
from eyetrace.pupil import (
    first_derivative,
    constriction_speed,
    dilation_speed,
    hippus_amplitude
)

def test_first_derivative_uniform():
    """Test first derivative with uniform sampling."""
    t = np.linspace(0, 10, 100)
    # Linear function: y = 2t + 1, derivative = 2
    y = 2 * t + 1
    deriv = first_derivative(y, t, smooth=False)
    # Central differences should be close to 2 (edges are less accurate)
    np.testing.assert_allclose(deriv[1:-1], 2.0, rtol=1e-2)

    # Without times (assumes dt=1)
    y2 = np.arange(10)  # y = t, derivative = 1
    deriv2 = first_derivative(y2, smooth=False)
    np.testing.assert_allclose(deriv2[1:-1], 1.0, rtol=1e-2)

def test_first_derivative_smooth():
    """Test smoothing option."""
    np.random.seed(42)
    t = np.linspace(0, 1, 100)
    clean = np.sin(2 * np.pi * t)
    noisy = clean + 0.1 * np.random.randn(100)

    deriv_no_smooth = first_derivative(noisy, t, smooth=False)
    deriv_smooth = first_derivative(noisy, t, smooth=True, window_length=11, polyorder=2)

    # True derivative of sin(2πt) is 2π cos(2πt)
    true_deriv = 2 * np.pi * np.cos(2 * np.pi * t)
    error_no_smooth = np.mean((deriv_no_smooth - true_deriv) ** 2)
    error_smooth = np.mean((deriv_smooth - true_deriv) ** 2)
    assert error_smooth < error_no_smooth

def test_constriction_speed():
    """Test constriction speed detection."""
    t = np.linspace(0, 2, 200)
    # Diameter: constant 5, then drops to 3 between 0.5 and 1.0, then constant 3
    y = np.ones_like(t) * 5
    mask = (t >= 0.5) & (t < 1.0)
    y[mask] = 5 - 4 * (t[mask] - 0.5)  # slope -4
    y[t >= 1.0] = 3

    max_speed, avg_speed = constriction_speed(y, t, threshold=0.5, min_duration=0.1)
    # Max speed should be -4 (most negative)
    assert np.isclose(max_speed, -4.0, rtol=0.1)
    # Average speed should be around -4
    assert np.isclose(avg_speed, -4.0, rtol=0.2)

    # No constriction
    y2 = np.ones_like(t) * 5
    max_speed2, avg_speed2 = constriction_speed(y2, t)
    assert max_speed2 == 0.0 and avg_speed2 == 0.0

def test_dilation_speed():
    """Test dilation speed detection."""
    t = np.linspace(0, 2, 200)
    y = np.ones_like(t) * 3
    mask = (t >= 0.5) & (t < 1.0)
    y[mask] = 3 + 4 * (t[mask] - 0.5)  # slope +4
    y[t >= 1.0] = 5

    max_speed, avg_speed = dilation_speed(y, t, threshold=0.5, min_duration=0.1)
    assert np.isclose(max_speed, 4.0, rtol=0.1)
    assert np.isclose(avg_speed, 4.0, rtol=0.2)

def test_hippus_amplitude():
    """Test hippus amplitude calculation."""
    fs = 100
    t = np.arange(0, 10, 1/fs)
    # Sine wave at 1 Hz, amplitude 0.5
    signal = 0.5 * np.sin(2 * np.pi * 1 * t)

    # RMS method should give amplitude / sqrt(2) ≈ 0.3536
    amp_rms = hippus_amplitude(signal, fs, lowcut=0.5, highcut=4.0, method='rms')
    assert np.isclose(amp_rms, 0.5 / np.sqrt(2), rtol=0.1)

    # Envelope method should give ≈ 0.5
    amp_env = hippus_amplitude(signal, fs, lowcut=0.5, highcut=4.0, method='envelope')
    assert np.isclose(amp_env, 0.5, rtol=0.1)

    # Invalid method
    with pytest.raises(ValueError, match="Unknown method"):
        hippus_amplitude(signal, fs, method='invalid')

    # High cutoff too high
    with pytest.raises(ValueError, match="must be less than Nyquist"):
        hippus_amplitude(signal, fs, highcut=60.0)  # Nyquist = 50 Hz