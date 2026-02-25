"""
Tests for signal_analysis.hurst module.
"""

import pytest
import numpy as np
from eyetrace.signal_analysis.hurst import hurst_exponent

def test_hurst_exponent():
    """Test Hurst exponent."""
    # Mouvement brownien : H ≈ 0.5
    np.random.seed(42)
    steps = np.random.randn(1000)
    brownian = np.cumsum(steps)
    h = hurst_exponent(brownian)
    assert np.isclose(h, 0.5, rtol=0.2)

    # Tendance persistante : H > 0.5
    t = np.linspace(0, 10, 1000)
    trend = 0.1 * t + np.cumsum(steps) * 0.1
    h2 = hurst_exponent(trend)
    assert h2 > 0.5