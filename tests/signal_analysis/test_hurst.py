"""
Tests for signal_analysis.hurst module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.hurst import hurst_exponent

def test_hurst_exponent():
    """Test Hurst exponent calculation."""
    # Brownian motion
    np.random.seed(42)
    steps = np.random.randn(1000)
    brownian = np.cumsum(steps)
    h = hurst_exponent(brownian)
    # Check that the exponent is a finite number
    assert np.isfinite(h), f"h={h}"

    # Strong trend
    t = np.linspace(0, 10, 1000)
    trend = 5.0 * t + 0.1 * np.cumsum(steps)
    h2 = hurst_exponent(trend)
    # Check that the exponent for a trended signal is finite
    assert np.isfinite(h2), f"h2={h2}"
